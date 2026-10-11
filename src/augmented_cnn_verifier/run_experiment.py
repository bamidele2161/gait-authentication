"""Participant-disjoint evaluation of the paper-inspired augmented CNN verifier."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.augmented_cnn_verifier.config import EVALUATION_CONDITIONS, MODELS_DIR, RESULTS_DIR
from src.augmented_cnn_verifier.train import (
    VerifierTrainingConfig,
    predict_legitimate_probability,
    train_verifier,
)
from src.condition_invariant.dataset import load_all_windows
from src.condition_invariant.folds import create_outer_folds, prepare_development_fold_data
from src.condition_invariant.metrics import (
    calculate_authentication_rates,
    macro_average_rates,
    select_global_threshold,
)
from src.condition_invariant.scoring import ComparisonScore, causal_mean_fusion


def _group(windows):
    grouped = {}
    for window in windows:
        grouped.setdefault(window.participant_id, {}).setdefault(window.condition, []).append(window)
    return {
        participant: {
            condition: tuple(sorted(items, key=lambda item: item.start_sample))
            for condition, items in conditions.items()
        }
        for participant, conditions in grouped.items()
    }


def _other_control_windows(groups, claimed):
    return tuple(
        window
        for participant, conditions in groups.items()
        if participant != claimed
        for window in conditions["st_control"]
    )


def _score(model, normalizer, claimed, probe_groups, conditions, batch_size, device):
    scores = []
    for condition in conditions:
        for probe_participant, condition_windows in probe_groups[condition].items():
            probabilities = predict_legitimate_probability(
                model, normalizer, condition_windows, batch_size, device
            )
            scores.extend(
                ComparisonScore(
                    claimed_participant_id=claimed,
                    probe_participant_id=probe_participant,
                    condition=condition,
                    window_index=window.window_index,
                    start_sample=window.start_sample,
                    # Existing biometric metrics use lower-is-better distances.
                    distance=float(1.0 - probability),
                )
                for window, probability in zip(condition_windows, probabilities)
            )
    return tuple(scores)


def _condition_groups(dataset, participants, conditions):
    return {
        condition: {
            participant: tuple(sorted(dataset[participant][condition], key=lambda w: w.start_sample))
            for participant in participants
        }
        for condition in conditions
    }


def _rows(scores, fold_index, threshold):
    return [
        {
            "fold_index": fold_index,
            "claimed_participant_id": score.claimed_participant_id,
            "probe_participant_id": score.probe_participant_id,
            "condition": score.condition,
            "window_index": score.window_index,
            "start_sample": score.start_sample,
            "is_genuine": score.is_genuine,
            "legitimate_probability": 1.0 - score.distance,
            "distance": score.distance,
            "threshold": threshold,
            "accepted": score.distance <= threshold,
        }
        for score in scores
    ]


def run_fold(fold_number, config, fusion_window=1, device="cpu", windows_dir=None):
    dataset = load_all_windows(windows_dir) if windows_dir is not None else load_all_windows()
    participants = tuple(sorted(dataset))
    fold = create_outer_folds(participants, 4)[fold_number - 1]
    development = prepare_development_fold_data(dataset, fold, learning_fraction=0.80)
    learning = _group(development.learning_windows)
    validation = _group(development.validation_windows)

    output_name = f"paper_augmentation_fold_{fold_number}_fusion{fusion_window}"
    model_dir = MODELS_DIR / output_name
    result_dir = RESULTS_DIR / output_name
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Fold {fold_number}/4: development={','.join(fold.development_participants)} "
        f"evaluation={','.join(fold.evaluation_participants)}"
    )

    # Simulate enrollment for every development identity. Only that person's
    # ST-control learning windows are positive enrollment data. ST-fatigue is
    # held out for validation and never used by the optimizer.
    development_scores = []
    best_epochs = []
    for participant_index, claimed in enumerate(fold.development_participants):
        print(f"  development verifier: {claimed}")
        verifier_config = replace(config, seed=config.seed + participant_index)
        trained = train_verifier(
            positive_learning=learning[claimed]["st_control"],
            negative_learning=_other_control_windows(learning, claimed),
            positive_validation=tuple(
                (*validation[claimed]["st_control"], *validation[claimed]["st_fatigue"])
            ),
            negative_validation=tuple(
                window
                for participant, conditions in validation.items()
                if participant != claimed
                for condition in ("st_control", "st_fatigue")
                for window in conditions[condition]
            ),
            config=verifier_config,
            device=device,
        )
        best_epochs.append(trained.best_epoch)
        validation_probes = {
            condition: {
                participant: validation[participant][condition]
                for participant in fold.development_participants
            }
            for condition in ("st_control", "st_fatigue")
        }
        development_scores.extend(_score(
            trained.model, trained.normalizer, claimed, validation_probes,
            ("st_control", "st_fatigue"), config.batch_size, device,
        ))

    development_scores = tuple(development_scores)
    threshold = select_global_threshold(
        development_scores, fold.development_participants, target_far=0.01
    )
    enrollment_epochs = max(1, int(round(float(np.median(best_epochs)))))
    print(
        f"  global development threshold={threshold.threshold:.6f} "
        f"FRR/FAR={threshold.validation_frr:.2%}/{threshold.validation_far:.2%}; "
        f"unseen-user enrollment epochs={enrollment_epochs}"
    )

    # Train one small verifier for each unseen user from normal walking only.
    # Generic impostor examples come exclusively from development identities.
    evaluation_probes = _condition_groups(
        dataset, fold.evaluation_participants, EVALUATION_CONDITIONS
    )
    generic_impostors = tuple(
        window
        for participant in fold.development_participants
        for window in dataset[participant]["st_control"]
    )
    evaluation_scores = []
    for participant_index, claimed in enumerate(fold.evaluation_participants):
        print(f"  unseen-user enrollment verifier: {claimed}")
        verifier_config = replace(config, seed=config.seed + 10_000 + participant_index)
        trained = train_verifier(
            positive_learning=tuple(dataset[claimed]["st_control"]),
            negative_learning=generic_impostors,
            config=verifier_config,
            device=device,
            fixed_epochs=enrollment_epochs,
        )
        torch.save(
            {
                "state_dict": trained.model.state_dict(),
                "claimed_participant": claimed,
                "epochs": enrollment_epochs,
                "training_config": asdict(verifier_config),
            },
            model_dir / f"{claimed}_verifier.pt",
        )
        np.savez(
            model_dir / f"{claimed}_normalizer.npz",
            mean=trained.normalizer.mean,
            standard_deviation=trained.normalizer.standard_deviation,
        )
        evaluation_scores.extend(_score(
            trained.model, trained.normalizer, claimed, evaluation_probes,
            EVALUATION_CONDITIONS, config.batch_size, device,
        ))

    evaluation_scores = tuple(evaluation_scores)
    if fusion_window > 1:
        evaluation_scores = causal_mean_fusion(evaluation_scores, fusion_window)

    rates = []
    for claimed in fold.evaluation_participants:
        for condition in EVALUATION_CONDITIONS:
            selected = tuple(
                score for score in evaluation_scores
                if score.claimed_participant_id == claimed and score.condition == condition
            )
            rate = calculate_authentication_rates(selected, threshold.threshold)
            rates.append(rate)
            print(
                f"  participant={claimed:8s} condition={condition:12s} "
                f"FRR={rate.frr:.2%} FAR={rate.far:.2%} EER={rate.eer:.2%}"
            )

    summaries = tuple(
        macro_average_rates(tuple(rate for rate in rates if rate.condition == condition))
        for condition in EVALUATION_CONDITIONS
    )
    pd.DataFrame(_rows(development_scores, fold.fold_index, threshold.threshold)).to_csv(
        result_dir / "development_threshold_scores.csv", index=False
    )
    pd.DataFrame(_rows(evaluation_scores, fold.fold_index, threshold.threshold)).to_csv(
        result_dir / "evaluation_scores.csv", index=False
    )
    pd.DataFrame(asdict(rate) for rate in rates).to_csv(
        result_dir / "participant_metrics.csv", index=False
    )
    pd.DataFrame(asdict(summary) for summary in summaries).to_csv(
        result_dir / "macro_summary.csv", index=False
    )
    with (result_dir / "protocol.json").open("w") as handle:
        json.dump({
            "fold": fold_number,
            "development_participants": fold.development_participants,
            "evaluation_participants": fold.evaluation_participants,
            "best_development_epochs": best_epochs,
            "enrollment_epochs": enrollment_epochs,
            "threshold": asdict(threshold),
            "augmentation": {
                "methods": ["noise", "temporal_shift", "amplitude_stretch", "time_stretch"],
                "maximum_distortion": config.maximum_distortion,
                "training_only": True,
            },
        }, handle, indent=2)

    print(f"\nMacro-average ({fusion_window} window(s) = {fusion_window + 1}s per decision)")
    for summary in summaries:
        print(
            f"{summary.condition:12s} FRR={summary.mean_frr:.2%} "
            f"(SD {summary.standard_deviation_frr:.2%})  "
            f"FAR={summary.mean_far:.2%} "
            f"(SD {summary.standard_deviation_far:.2%})  "
            f"EER={summary.mean_eer:.2%} "
            f"(SD {summary.standard_deviation_eer:.2%})"
        )
    print(f"Results saved under {result_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run paper-inspired augmented CNN verification")
    parser.add_argument("--fold", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batches-per-epoch", type=int, default=50)
    parser.add_argument("--validation-batches", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--augmentation-probability", type=float, default=0.80)
    parser.add_argument("--maximum-distortion", type=float, default=0.15)
    parser.add_argument("--fusion-window", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--windows-dir", type=Path,
        help="sacrum_time directory; useful when running from an isolated worktree",
    )
    args = parser.parse_args()
    if not 0 <= args.augmentation_probability <= 1:
        parser.error("--augmentation-probability must be in [0, 1]")
    if not 0 < args.maximum_distortion <= 0.15:
        parser.error("--maximum-distortion must be in (0, 0.15]")
    config = VerifierTrainingConfig(
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        validation_batches=args.validation_batches,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        augmentation_probability=args.augmentation_probability,
        maximum_distortion=args.maximum_distortion,
    )
    run_fold(args.fold, config, args.fusion_window, args.device, args.windows_dir)


if __name__ == "__main__":
    main()
