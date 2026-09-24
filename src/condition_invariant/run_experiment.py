"""Run and save the complete four-fold condition-invariant experiment.

Evaluation path (cross-session design)
---------------------------------------
- Enrolment  : ST-control windows only.
- Training    : session-1 ST-control and ST-fatigue only.
- Test probes : session-2 DT-control and DT-fatigue only.
- Scoring     : cosine / Euclidean distance from the enrollment template
                (unit-sphere embeddings → equivalent).
- Threshold   : one global threshold calibrated on held-out session-1
                development probes only.
- No per-user SVM verifier is used; the frozen encoder + global threshold
  is the complete authenticator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.condition_invariant.config import (
    EVALUATION_CONDITIONS,
    MODELS_DIR,
    RESULTS_DIR,
    ensure_output_directories,
)
from src.condition_invariant.dataset import GaitDataset, load_all_windows
from src.condition_invariant.enrollment import (
    UserTemplate,
    create_user_template,
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
from src.condition_invariant.scoring import (
    ComparisonScore, causal_mean_fusion, score_windows,
)
from src.condition_invariant.train import TrainingConfig, TrainingResult, train_encoder


# ---------------------------------------------------------------------------
# Experiment-level config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    """Settings shared by all four outer folds."""

    training: TrainingConfig = TrainingConfig()
    number_of_folds: int = 2
    development_learning_fraction: float = 0.80
    target_far: float = 0.01
    scoring_batch_size: int = 64
    device: str = "cpu"
    selected_fold_index: int | None = None
    # Consecutive probe windows averaged into one decision. Windows are 2 s with
    # a 1 s hop, so n windows span n + 1 seconds of walking. A value of 1 keeps
    # the original per-window behaviour.
    fusion_window: int = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            pid: tuple(sorted(ws, key=lambda w: w.start_sample))
            for pid, ws in participants.items()
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
    """Create one ST-control template for each requested participant."""

    return {
        pid: create_user_template(
            participant_id=pid,
            enrollment_windows=st_control_windows[pid],
            model=model,
            normalizer=normalizer,
            batch_size=batch_size,
        )
        for pid in participants
    }


def _score_population(
    participants: tuple[str, ...],
    templates: dict[str, UserTemplate],
    probes: dict[str, dict[str, tuple[GaitWindow, ...]]],
    model: torch.nn.Module,
    normalizer: ChannelNormalizer,
    batch_size: int,
    conditions: tuple[str, ...],
) -> tuple[ComparisonScore, ...]:
    """Compare every participant's probes with every claimed template.

    Generates both genuine scores (probe owner == claimed user) and
    impostor scores (probe owner != claimed user).
    """

    scores = []
    for claimed_participant in participants:
        template = templates[claimed_participant]
        for condition in conditions:
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
    """Prepare unseen-user enrolment and probe data.

    Enrolment : complete ST-control recording only.
    Probes    : session-2 DT-control and DT-fatigue only.
    """

    enrollment_windows: dict[str, tuple[GaitWindow, ...]] = {}
    probes: dict[str, dict[str, tuple[GaitWindow, ...]]] = {
        condition: {} for condition in EVALUATION_CONDITIONS
    }
    for pid in fold.evaluation_participants:
        participant_data = dataset[pid]
        # Evaluation participants are completely unseen during development.
        # Their full normal-walking recording is therefore enrollment data:
        # no ST-control window is reserved for model fitting or evaluation.
        enrollment_windows[pid] = tuple(
            sorted(
                participant_data["st_control"],
                key=lambda window: window.start_sample,
            )
        )
        for condition in EVALUATION_CONDITIONS:
            probes[condition][pid] = tuple(
                sorted(participant_data[condition], key=lambda w: w.start_sample)
            )
    return enrollment_windows, probes


def _participant_metrics(
    scores: tuple[ComparisonScore, ...],
    participants: tuple[str, ...],
    threshold: float,
) -> tuple[AuthenticationRates, ...]:
    """Calculate one FRR/FAR pair per unseen participant and condition."""

    results = []
    for pid in participants:
        for condition in EVALUATION_CONDITIONS:
            selected = tuple(
                s for s in scores
                if s.claimed_participant_id == pid and s.condition == condition
            )

            rates = calculate_authentication_rates(selected, threshold)

            print(
                f"  participant={pid:8s} "
                f"condition={condition:12s} "
                f"FRR={rates.frr:.2%} "
                f"FAR={rates.far:.2%} "
                f"EER={rates.eer:.2%}"
            )

            results.append(rates)

            # results.append(calculate_authentication_rates(selected, threshold))

    return tuple(results)


def _score_rows(
    scores: tuple[ComparisonScore, ...],
    fold_index: int,
    thresholds: float,
) -> list[dict[str, object]]:
    """Create reproducible, ISO-friendly window-level score rows."""

    return [
        {
            "fold_index": fold_index,
            "claimed_participant_id": s.claimed_participant_id,
            "probe_participant_id":   s.probe_participant_id,
            "condition":              s.condition,
            "window_index":           s.window_index,
            "start_sample":           s.start_sample,
            "is_genuine":             s.is_genuine,
            "distance":               s.distance,
            "threshold":              thresholds,
            "accepted":               s.distance <= thresholds,
        }
        for s in scores
    ]


def _save_fold_artifacts(
    fold: OuterFold,
    training: TrainingResult,
    threshold: ThresholdSelection,
    fold_training_config: TrainingConfig,
    models_dir: Path,
    results_dir: Path,
) -> None:
    """Save enough fold-specific state to reproduce scores and inspect training."""

    fold_name = f"fold_{fold.fold_index}"
    torch.save(
        {
            "model_state_dict":         training.model.state_dict(),
            "training_config":          asdict(fold_training_config),
            "development_participants": fold.development_participants,
            "evaluation_participants":  fold.evaluation_participants,
            "best_epoch":               training.best_epoch,
        },
        models_dir / f"{fold_name}_encoder.pt",
    )
    np.savez(
        models_dir / f"{fold_name}_normalizer.npz",
        mean=training.normalizer.mean,
        standard_deviation=training.normalizer.standard_deviation,
    )
    with (results_dir / f"{fold_name}_threshold.json").open("w") as handle:
        json.dump(asdict(threshold), handle, indent=2)
    pd.DataFrame(asdict(epoch) for epoch in training.history).to_csv(
        results_dir / f"{fold_name}_training_history.csv", index=False
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    config: ExperimentConfig = ExperimentConfig(),
    dataset: GaitDataset | None = None,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
) -> tuple[tuple[AuthenticationRates, ...], tuple[MacroAverage, ...]]:
    """Train, calibrate, enrol, and evaluate every participant exactly once.

    Cross-session authentication flow per fold
    ------------------------------------------
    1. Train encoder on development participants' ST-control/ST-fatigue data.
    2. Calibrate one global threshold from held-out development
       ST-control/ST-fatigue scores.
    3. Enrol each evaluation participant from their ST-control only.
    4. Score each evaluation participant's DT-control/DT-fatigue probes
       using Euclidean distance to their enrollment template.
    5. Apply the global threshold → accept / reject.
    """

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

    all_evaluation_scores: list[dict] = []
    all_threshold_scores:  list[dict] = []
    all_rates: list[AuthenticationRates] = []

    for fold in folds:
        print(
            f"Fold {fold.fold_index + 1}/{len(all_folds)}: "
            f"evaluation={','.join(fold.evaluation_participants)}"
        )

        # ── 1. Train encoder ──────────────────────────────────────────────
        development = prepare_development_fold_data(
            dataset, fold,
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

        # ── 2. Calibrate on held-out session-1 development windows only ──
        learning_groups   = _group_windows(development.learning_windows)
        validation_groups = _group_windows(development.validation_windows)

        # Templates from development participants' ST-control learning windows.
        development_templates = _create_templates(
            participants=fold.development_participants,
            st_control_windows=learning_groups["st_control"],
            model=training.model,
            normalizer=training.normalizer,
            batch_size=config.scoring_batch_size,
        )
        # DT data is deliberately absent from threshold calibration.
        threshold_scores = _score_population(
            participants=fold.development_participants,
            templates=development_templates,
            probes=validation_groups,
            model=training.model,
            normalizer=training.normalizer,
            batch_size=config.scoring_batch_size,
            conditions=("st_control", "st_fatigue"),
        )
        threshold = select_global_threshold(
            threshold_scores,
            development_participants=fold.development_participants,
            target_far=config.target_far,
        )

        # ── 3 & 4. Enrol from ST-control; test on session-2 DT only ───────
        enrollment_windows, evaluation_probes = _evaluation_inputs(
            dataset, fold, config
        )
        evaluation_templates = _create_templates(
            participants=fold.evaluation_participants,
            st_control_windows=enrollment_windows,
            model=training.model,
            normalizer=training.normalizer,
            batch_size=config.scoring_batch_size,
        )
        evaluation_scores = _score_population(
            participants=fold.evaluation_participants,
            templates=evaluation_templates,
            probes=evaluation_probes,
            model=training.model,
            normalizer=training.normalizer,
            batch_size=config.scoring_batch_size,
            conditions=EVALUATION_CONDITIONS,
        )
        # One decision per run of consecutive windows instead of one per window.
        #
        # The threshold above is deliberately left calibrated on UNFUSED
        # development scores. Fusing those too drives the development genuine
        # and impostor distributions almost perfectly apart, which collapses the
        # threshold far below where unseen session-2 scores fall and sends FRR
        # to ~100%. Fusing only the probe side is what the measured gain rests on.
        if config.fusion_window > 1:
            evaluation_scores = causal_mean_fusion(
                evaluation_scores, config.fusion_window
            )

        # ── 5. Apply threshold, compute per-participant metrics ───────────
        fold_rates = _participant_metrics(
            evaluation_scores,
            fold.evaluation_participants,
            threshold.threshold,
        )

        _save_fold_artifacts(
            fold=fold,
            training=training,
            threshold=threshold,
            fold_training_config=fold_training_config,
            models_dir=models_dir,
            results_dir=results_dir,
        )
        all_threshold_scores.extend(
            _score_rows(threshold_scores, fold.fold_index, threshold.threshold)
        )
        all_evaluation_scores.extend(
            _score_rows(evaluation_scores, fold.fold_index, threshold.threshold)
        )
        all_rates.extend(fold_rates)
        print(
            f"  best epoch={training.best_epoch}, "
            f"threshold={threshold.threshold:.6f}, "
            f"val FRR/FAR={threshold.validation_frr:.2%}/{threshold.validation_far:.2%}"
        )

    rates    = tuple(all_rates)
    summaries = tuple(
        macro_average_rates(
            tuple(r for r in rates if r.condition == condition)
        )
        for condition in EVALUATION_CONDITIONS
    )

    pd.DataFrame(all_threshold_scores).to_csv(
        results_dir / "development_threshold_scores.csv", index=False
    )
    pd.DataFrame(all_evaluation_scores).to_csv(
        results_dir / "evaluation_scores.csv", index=False
    )
    pd.DataFrame(asdict(r) for r in rates).to_csv(
        results_dir / "participant_metrics.csv", index=False
    )
    pd.DataFrame(asdict(s) for s in summaries).to_csv(
        results_dir / "macro_summary.csv", index=False
    )

    decision_seconds = config.fusion_window + 1
    print(
        f"\nMacro-average across participants (each participant has equal weight, "
        f"{config.fusion_window} window(s) = {decision_seconds}s per decision)"
    )
    for summary in summaries:
        print(
            f"{summary.condition:12s} "
            f"FRR={summary.mean_frr:.2%} "
            f"(SD {summary.standard_deviation_frr:.2%})  "
            f"FAR={summary.mean_far:.2%} "
            f"(SD {summary.standard_deviation_far:.2%})  "
            f"EER={summary.mean_eer:.2%} "
            f"(SD {summary.standard_deviation_eer:.2%})"
        )
    return rates, summaries


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cross-session condition-invariant gait authentication"
    )
    parser.add_argument("--epochs",            type=int,   default=40)
    parser.add_argument("--patience",          type=int,   default=7)
    parser.add_argument("--batches-per-epoch", type=int,   default=100)
    parser.add_argument("--participants-per-batch", type=int, default=8)
    parser.add_argument("--windows-per-condition", type=int, default=3)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--identity-loss-weight", type=float, default=0.3)
    parser.add_argument("--device",            default="cpu")
    parser.add_argument("--target-far",        type=float, default=0.01)
    parser.add_argument("--fold",              type=int,   choices=(1, 2))
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument(
        "--soft-margin", action="store_true",
        help="use softplus(d_p - d_n) instead of the hinge, so the triplet "
             "objective never saturates",
    )
    parser.add_argument(
        "--fusion-window", type=int, default=10,
        help="consecutive 2s probe windows averaged per decision; n spans "
             "n+1 seconds of walking. Use 1 for the original per-window system.",
    )
    args = parser.parse_args()

    ensure_output_directories()
    training = TrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        batches_per_epoch=args.batches_per_epoch,
        participants_per_batch=args.participants_per_batch,
        windows_per_condition=args.windows_per_condition,
        validation_batches=args.validation_batches,
        identity_loss_weight=args.identity_loss_weight,
        margin=args.margin,
        soft_margin=args.soft_margin,
    )
    experiment = ExperimentConfig(
        training=training,
        target_far=args.target_far,
        device=args.device,
        selected_fold_index=None if args.fold is None else args.fold - 1,
        fusion_window=args.fusion_window,
    )
    method_name = "session1_cnn_bilstm_hard"
    if args.soft_margin:
        method_name = f"{method_name}_softmargin"
    elif args.margin != 0.2:
        method_name = f"{method_name}_margin{args.margin}"
    if args.fusion_window > 1:
        method_name = f"{method_name}_fusion{args.fusion_window}"
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
