"""Run and save the complete four-fold condition-invariant experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

from src.condition_invariant.config import (
    CONDITIONS,
    MODELS_DIR,
    RESULTS_DIR,
    ensure_output_directories,
)
from src.condition_invariant.dataset import GaitDataset, load_all_windows
from src.condition_invariant.enrollment import (
    UserTemplate,
    create_user_template,
    split_evaluation_st_control,
)
from src.condition_invariant.folds import (
    OuterFold,
    create_outer_folds,
    prepare_development_fold_data,
)
from src.condition_invariant.metrics import (
    AuthenticationRates,
    MacroAverage,
    ThresholdSelection,
    calculate_authentication_rates,
    macro_average_rates,
    select_global_threshold,
)
from src.condition_invariant.normalization import ChannelNormalizer
from src.condition_invariant.records import GaitWindow
from src.condition_invariant.scoring import ComparisonScore, score_windows
from src.condition_invariant.train import TrainingConfig, TrainingResult, train_encoder
from src.condition_invariant.verifier import (
    EnrollmentVerifier,
    fit_enrollment_verifier,
    score_with_enrollment_verifier,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings shared by all four outer folds."""

    training: TrainingConfig = TrainingConfig()
    number_of_folds: int = 4
    development_learning_fraction: float = 0.80
    enrollment_fraction: float = 0.60
    baseline_test_start_fraction: float = 0.80
    target_far: float = 0.01
    scoring_batch_size: int = 64
    fusion_window: int = 5
    device: str = "cpu"
    selected_fold_index: int | None = None


def _group_windows(
    windows: tuple[GaitWindow, ...],
) -> dict[str, dict[str, tuple[GaitWindow, ...]]]:
    """Group windows by condition and participant in chronological order."""

    grouped: dict[str, dict[str, list[GaitWindow]]] = {}
    for window in windows:
        grouped.setdefault(window.condition, {}).setdefault(
            window.participant_id, []
        ).append(window)
    return {
        condition: {
            participant_id: tuple(
                sorted(participant_windows, key=lambda item: item.start_sample)
            )
            for participant_id, participant_windows in participants.items()
        }
        for condition, participants in grouped.items()
    }


def _create_templates(
    participants: tuple[str, ...],
    st_control_windows: dict[str, tuple[GaitWindow, ...]],
    model: torch.nn.Module,
    normalizer: ChannelNormalizer,
    batch_size: int,
) -> dict[str, UserTemplate]:
    """Create one normal-walking template for each requested participant."""

    return {
        participant_id: create_user_template(
            participant_id=participant_id,
            enrollment_windows=st_control_windows[participant_id],
            model=model,
            normalizer=normalizer,
            batch_size=batch_size,
        )
        for participant_id in participants
    }


def _score_population(
    participants: tuple[str, ...],
    templates: dict[str, UserTemplate],
    probes: dict[str, dict[str, tuple[GaitWindow, ...]]],
    model: torch.nn.Module,
    normalizer: ChannelNormalizer,
    batch_size: int,
) -> tuple[ComparisonScore, ...]:
    """Compare every participant's probes with every claimed template."""

    scores = []
    for claimed_participant in participants:
        template = templates[claimed_participant]
        for condition in CONDITIONS:
            if condition not in probes:
                raise ValueError(f"Missing probe condition: {condition}")
            for probe_participant in participants:
                if probe_participant not in probes[condition]:
                    raise ValueError(
                        f"Missing {condition} probes for {probe_participant}"
                    )
                scores.extend(
                    score_windows(
                        template=template,
                        windows=probes[condition][probe_participant],
                        model=model,
                        normalizer=normalizer,
                        batch_size=batch_size,
                    )
                )
    return tuple(scores)


def _evaluation_inputs(
    dataset: GaitDataset,
    fold: OuterFold,
    config: ExperimentConfig,
) -> tuple[
    dict[str, tuple[GaitWindow, ...]],
    dict[str, dict[str, tuple[GaitWindow, ...]]],
]:
    """Prepare unseen-user enrolment and probe data without changed-condition enrolment."""

    enrollment_windows = {}
    probes = {condition: {} for condition in CONDITIONS}
    for participant_id in fold.evaluation_participants:
        participant_data = dataset[participant_id]
        split = split_evaluation_st_control(
            tuple(participant_data["st_control"]),
            enrollment_fraction=config.enrollment_fraction,
            test_start_fraction=config.baseline_test_start_fraction,
        )
        enrollment_windows[participant_id] = split.enrollment_windows
        probes["st_control"][participant_id] = split.test_windows
        for condition in CONDITIONS[1:]:
            probes[condition][participant_id] = tuple(
                sorted(
                    participant_data[condition],
                    key=lambda window: window.start_sample,
                )
            )
    return enrollment_windows, probes


def _participant_metrics(
    scores: tuple[ComparisonScore, ...],
    participants: tuple[str, ...],
    thresholds: float | dict[str, float],
) -> tuple[AuthenticationRates, ...]:
    """Calculate one FRR/FAR pair per unseen participant and condition."""

    results = []
    for participant_id in participants:
        threshold = (
            thresholds[participant_id]
            if isinstance(thresholds, dict)
            else thresholds
        )
        for condition in CONDITIONS:
            selected = tuple(
                score
                for score in scores
                if score.claimed_participant_id == participant_id
                and score.condition == condition
            )
            results.append(calculate_authentication_rates(selected, threshold))
    return tuple(results)


def _score_rows(
    scores: tuple[ComparisonScore, ...],
    fold_index: int,
    thresholds: float | dict[str, float],
) -> list[dict[str, object]]:
    """Create reproducible, ISO-friendly window-level score rows."""

    rows = []
    for score in scores:
        threshold = (
            thresholds[score.claimed_participant_id]
            if isinstance(thresholds, dict)
            else thresholds
        )
        rows.append({
            "fold_index": fold_index,
            "claimed_participant_id": score.claimed_participant_id,
            "probe_participant_id": score.probe_participant_id,
            "condition": score.condition,
            "window_index": score.window_index,
            "start_sample": score.start_sample,
            "is_genuine": score.is_genuine,
            "distance": score.distance,
            "threshold": threshold,
            "accepted": score.distance <= threshold,
        })
    return rows


def _fit_evaluation_verifiers(
    fold: OuterFold,
    enrollment_windows: dict[str, tuple[GaitWindow, ...]],
    development_learning_windows: tuple[GaitWindow, ...],
    development_validation_windows: tuple[GaitWindow, ...],
    training: TrainingResult,
    config: ExperimentConfig,
) -> dict[str, EnrollmentVerifier]:
    """Fit one normal-only verifier for every unseen evaluation participant."""

    return {
        participant_id: fit_enrollment_verifier(
            participant_id=participant_id,
            enrollment_windows=enrollment_windows[participant_id],
            cohort_learning_windows=development_learning_windows,
            cohort_validation_windows=development_validation_windows,
            model=training.model,
            normalizer=training.normalizer,
            target_far=config.target_far,
            seed=config.training.seed + fold.fold_index * 100 + index,
            batch_size=config.scoring_batch_size,
            fusion_window=config.fusion_window,
        )
        for index, participant_id in enumerate(fold.evaluation_participants)
    }


def _score_with_verifiers(
    participants: tuple[str, ...],
    verifiers: dict[str, EnrollmentVerifier],
    probes: dict[str, dict[str, tuple[GaitWindow, ...]]],
    training: TrainingResult,
    batch_size: int,
) -> tuple[ComparisonScore, ...]:
    """Score all genuine and impostor probes with enrolment-conditioned models."""

    scores = []
    for claimed_participant in participants:
        verifier = verifiers[claimed_participant]
        for condition in CONDITIONS:
            for probe_participant in participants:
                scores.extend(
                    score_with_enrollment_verifier(
                        verifier,
                        probes[condition][probe_participant],
                        training.model,
                        training.normalizer,
                        batch_size,
                    )
                )
    return tuple(scores)


def _save_fold_artifacts(
    fold: OuterFold,
    training: TrainingResult,
    threshold: ThresholdSelection,
    evaluation_verifiers: dict[str, EnrollmentVerifier],
    fold_training_config: TrainingConfig,
    models_dir: Path,
    results_dir: Path,
) -> None:
    """Save enough fold-specific state to reproduce scores and inspect training."""

    fold_name = f"fold_{fold.fold_index}"
    torch.save(
        {
            "model_state_dict": training.model.state_dict(),
            "training_config": asdict(fold_training_config),
            "development_participants": fold.development_participants,
            "evaluation_participants": fold.evaluation_participants,
            "best_epoch": training.best_epoch,
        },
        models_dir / f"{fold_name}_encoder.pt",
    )
    np.savez(
        models_dir / f"{fold_name}_normalizer.npz",
        mean=training.normalizer.mean,
        standard_deviation=training.normalizer.standard_deviation,
    )
    for participant_id, verifier in evaluation_verifiers.items():
        joblib.dump(
            verifier,
            models_dir / f"{fold_name}_{participant_id}_verifier.joblib",
        )
    with (results_dir / f"{fold_name}_threshold.json").open("w") as handle:
        json.dump(asdict(threshold), handle, indent=2)
    pd.DataFrame(asdict(epoch) for epoch in training.history).to_csv(
        results_dir / f"{fold_name}_training_history.csv", index=False
    )


def run_experiment(
    config: ExperimentConfig = ExperimentConfig(),
    dataset: GaitDataset | None = None,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
) -> tuple[tuple[AuthenticationRates, ...], tuple[MacroAverage, ...]]:
    """Train, calibrate, enrol, and evaluate every participant exactly once."""

    if dataset is None:
        dataset = load_all_windows()
    participants = tuple(sorted(dataset))
    all_folds = create_outer_folds(participants, config.number_of_folds)
    if config.selected_fold_index is None:
        folds = all_folds
    else:
        if not 0 <= config.selected_fold_index < len(all_folds):
            raise ValueError("selected_fold_index is outside the available folds")
        folds = (all_folds[config.selected_fold_index],)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_evaluation_scores = []
    all_threshold_scores = []
    all_rates = []

    for fold in folds:
        print(
            f"Fold {fold.fold_index + 1}/{len(folds)}: "
            f"evaluation={','.join(fold.evaluation_participants)}"
        )
        development = prepare_development_fold_data(
            dataset,
            fold,
            learning_fraction=config.development_learning_fraction,
        )
        fold_training_config = replace(
            config.training,
            seed=config.training.seed + fold.fold_index,
            operating_target_far=config.target_far,
        )
        training = train_encoder(
            learning_windows=development.learning_windows,
            validation_windows=development.validation_windows,
            config=fold_training_config,
            device=config.device,
        )

        learning_groups = _group_windows(development.learning_windows)
        validation_groups = _group_windows(development.validation_windows)
        development_templates = _create_templates(
            participants=fold.development_participants,
            st_control_windows=learning_groups["st_control"],
            model=training.model,
            normalizer=training.normalizer,
            batch_size=config.scoring_batch_size,
        )
        threshold_scores = _score_population(
            participants=fold.development_participants,
            templates=development_templates,
            probes=validation_groups,
            model=training.model,
            normalizer=training.normalizer,
            batch_size=config.scoring_batch_size,
        )
        threshold = select_global_threshold(
            threshold_scores,
            development_participants=fold.development_participants,
            target_far=config.target_far,
        )

        enrollment_windows, evaluation_probes = _evaluation_inputs(
            dataset, fold, config
        )
        evaluation_verifiers = _fit_evaluation_verifiers(
            fold,
            enrollment_windows,
            development.learning_windows,
            development.validation_windows,
            training,
            config,
        )
        evaluation_scores = _score_with_verifiers(
            fold.evaluation_participants,
            evaluation_verifiers,
            evaluation_probes,
            training,
            config.scoring_batch_size,
        )
        verifier_thresholds = {
            participant_id: verifier.threshold
            for participant_id, verifier in evaluation_verifiers.items()
        }
        fold_rates = _participant_metrics(
            evaluation_scores,
            fold.evaluation_participants,
            verifier_thresholds,
        )

        _save_fold_artifacts(
            fold=fold,
            training=training,
            threshold=threshold,
            evaluation_verifiers=evaluation_verifiers,
            fold_training_config=fold_training_config,
            models_dir=models_dir,
            results_dir=results_dir,
        )
        all_threshold_scores.extend(
            _score_rows(threshold_scores, fold.fold_index, threshold.threshold)
        )
        all_evaluation_scores.extend(
            _score_rows(evaluation_scores, fold.fold_index, verifier_thresholds)
        )
        all_rates.extend(fold_rates)
        print(
            f"  best epoch={training.best_epoch}, "
            f"threshold={threshold.threshold:.6f}, "
            f"validation FRR/FAR={threshold.validation_frr:.2%}/"
            f"{threshold.validation_far:.2%}"
        )

    rates = tuple(all_rates)
    summaries = tuple(
        macro_average_rates(
            tuple(rate for rate in rates if rate.condition == condition)
        )
        for condition in CONDITIONS
    )
    pd.DataFrame(all_threshold_scores).to_csv(
        results_dir / "development_threshold_scores.csv", index=False
    )
    pd.DataFrame(all_evaluation_scores).to_csv(
        results_dir / "evaluation_scores.csv", index=False
    )
    pd.DataFrame(asdict(rate) for rate in rates).to_csv(
        results_dir / "participant_metrics.csv", index=False
    )
    pd.DataFrame(asdict(summary) for summary in summaries).to_csv(
        results_dir / "macro_summary.csv", index=False
    )

    print("\nMacro-average across participants (each participant has equal weight)")
    for summary in summaries:
        print(
            f"{summary.condition:12s} "
            f"FRR={summary.mean_frr:.2%} "
            f"(SD {summary.standard_deviation_frr:.2%})  "
            f"FAR={summary.mean_far:.2%} "
            f"(SD {summary.standard_deviation_far:.2%})"
        )
    return rates, summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run condition-invariant gait authentication"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-far", type=float, default=0.01)
    parser.add_argument("--fusion-window", type=int, default=5)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4))
    args = parser.parse_args()

    ensure_output_directories()
    training = TrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        batches_per_epoch=args.batches_per_epoch,
    )
    experiment = ExperimentConfig(
        training=training,
        target_far=args.target_far,
        device=args.device,
        selected_fold_index=None if args.fold is None else args.fold - 1,
        fusion_window=args.fusion_window,
    )
    method_name = f"contrastive_fusion_{args.fusion_window}_verifier"
    output_name = (
        method_name if args.fold is None else f"{method_name}_fold_{args.fold}"
    )
    run_experiment(
        experiment,
        models_dir=MODELS_DIR / output_name,
        results_dir=RESULTS_DIR / output_name,
    )


if __name__ == "__main__":
    main()
