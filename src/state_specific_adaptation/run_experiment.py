"""Evaluate temporary current-state verifiers with causal score fusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.condition_augmentation.run_experiment import (
    load_features,
    select_similarity_threshold,
    split_development_frame,
)
from src.condition_invariant.folds import create_outer_folds
from src.secure_adaptation.protocol import split_normal_enrollment, split_trusted_session
from src.secure_adaptation.run_experiment import (
    _balanced_cohort,
    _fit_verifier,
    _refit_adapted_verifier,
)
from src.utils import FEATURE_COLS, ORIGINAL_RATE, SESSIONS, WINDOW_SAMPLES


CHANGED_CONDITIONS = tuple(condition for condition in SESSIONS if condition != "st_control")


def causal_median_scores(model, frame: pd.DataFrame, fusion_window: int):
    """Return causal median scores and their corresponding final windows."""

    if fusion_window < 1:
        raise ValueError("fusion_window must be at least 1")
    ordered = frame.sort_values("start_sample").reset_index(drop=True)
    if len(ordered) < fusion_window:
        raise ValueError("Probe stream is shorter than the fusion window")
    raw_scores = model.decision_function(ordered[FEATURE_COLS])
    scores = np.asarray([
        np.median(raw_scores[index - fusion_window + 1 : index + 1])
        for index in range(fusion_window - 1, len(raw_scores))
    ])
    return scores, ordered.iloc[fusion_window - 1 :].reset_index(drop=True)


def _cohort_fused_scores(model, frame: pd.DataFrame, fusion_window: int):
    """Fuse each participant recording independently, then concatenate."""

    score_parts = []
    for _, stream in frame.groupby(["participant_id", "session_type"], sort=True):
        scores, _ = causal_median_scores(model, stream, fusion_window)
        score_parts.append(scores)
    if not score_parts:
        raise ValueError("Cannot score an empty cohort")
    return np.concatenate(score_parts)


def _calibrate_state_model(
    model,
    positive: pd.DataFrame,
    negative: pd.DataFrame,
    target_far: float,
    fusion_window: int = 1,
):
    positive_scores, _ = causal_median_scores(model, positive, fusion_window)
    negative_scores = _cohort_fused_scores(model, negative, fusion_window)
    return select_similarity_threshold(positive_scores, negative_scores, target_far)


def mine_hard_negatives(model, frames, maximum_per_participant: int = 80):
    """Select the most genuine-looking development impostor windows per user."""

    if maximum_per_participant < 1:
        raise ValueError("maximum_per_participant must be at least 1")
    cohort = pd.concat(frames, ignore_index=True).copy()
    if cohort.empty:
        raise ValueError("Cannot mine an empty impostor cohort")
    cohort["_hardness_score"] = model.decision_function(cohort[FEATURE_COLS])
    selected = []
    for _, participant_frame in cohort.groupby("participant_id", sort=True):
        count = min(maximum_per_participant, len(participant_frame))
        selected.append(participant_frame.nlargest(count, "_hardness_score"))
    hard = pd.concat(selected, ignore_index=True)
    return hard.drop(columns="_hardness_score")


def trusted_prefix(update: pd.DataFrame, seconds: float) -> pd.DataFrame:
    """Take a chronological prefix from an already trusted update segment."""

    if seconds <= 0 or update.empty:
        raise ValueError("A positive duration and non-empty update are required")
    ordered = update.sort_values("start_sample")
    boundary = ordered.start_sample.iloc[0] + round(seconds * ORIGINAL_RATE)
    prefix = ordered[ordered.start_sample < boundary].copy()
    if prefix.empty:
        raise ValueError("Trusted prefix is empty")
    return prefix


def select_candidate(candidate_rows: list[dict], target_far: float) -> dict:
    """Choose low-FRR calibration performance without exceeding target FAR."""

    eligible = [
        row for row in candidate_rows
        if row["calibration_far"] <= target_far + 1e-12
    ]
    if not eligible:
        eligible = candidate_rows
    return min(
        eligible,
        key=lambda row: (
            row["calibration_frr"],
            row["calibration_far"],
            row["update_seconds"],
            row["hard_negatives_per_participant"],
        ),
    )


def fit_selected_state_model(
    static_model,
    trusted_update: pd.DataFrame,
    calibration: pd.DataFrame,
    development_learning_frames,
    base_negatives: pd.DataFrame,
    validation_negatives: pd.DataFrame,
    target_far: float,
    update_candidates=(20.0, 30.0, 60.0),
    hard_negative_candidates=(0, 20, 40, 80),
):
    """Select update length and boundary strength without touching test data."""

    available_seconds = (
        trusted_update.start_sample.max()
        - trusted_update.start_sample.min()
        + WINDOW_SAMPLES
    ) / ORIGINAL_RATE
    if max(update_candidates) > available_seconds + 1e-9:
        raise ValueError("Trusted update is shorter than the largest candidate duration")

    candidates = []
    for duration in update_candidates:
        positive = trusted_prefix(trusted_update, duration)
        base_model = _refit_adapted_verifier(positive, base_negatives, static_model)
        hard_ranked = None
        for hard_count in hard_negative_candidates:
            if hard_count == 0:
                model = base_model
            else:
                if hard_ranked is None:
                    hard_ranked = mine_hard_negatives(
                        base_model,
                        development_learning_frames,
                        maximum_per_participant=max(hard_negative_candidates),
                    )
                # ``mine_hard_negatives`` returns each participant's rows in
                # descending hardness order, so smaller candidates are nested
                # prefixes of the same ranked set.
                selected_hard = hard_ranked.groupby(
                    "participant_id", sort=True, group_keys=False
                ).head(hard_count)
                negatives = pd.concat((base_negatives, selected_hard), ignore_index=True)
                model = _refit_adapted_verifier(positive, negatives, static_model)
            threshold, frr, far = _calibrate_state_model(
                model, calibration, validation_negatives, target_far, fusion_window=1
            )
            candidates.append({
                "model": model,
                "threshold": threshold,
                "calibration_frr": frr,
                "calibration_far": far,
                "update_seconds": duration,
                "update_windows": len(positive),
                "hard_negatives_per_participant": hard_count,
            })
    return select_candidate(candidates, target_far), candidates


def _evaluate(
    model,
    threshold: float,
    claimed_id: str,
    probes: dict[str, pd.DataFrame],
    fusion_window: int,
):
    genuine_count = impostor_count = false_rejections = false_acceptances = 0
    rows = []
    for probe_id, frame in probes.items():
        scores, aligned = causal_median_scores(model, frame, fusion_window)
        accepted = scores >= threshold
        genuine = probe_id == claimed_id
        if genuine:
            genuine_count += len(scores)
            false_rejections += int((~accepted).sum())
        else:
            impostor_count += len(scores)
            false_acceptances += int(accepted.sum())
        for window, score, decision in zip(aligned.itertuples(), scores, accepted):
            rows.append({
                "probe_participant_id": probe_id,
                "window_index": window.window_index,
                "start_sample": window.start_sample,
                "is_genuine": genuine,
                "score": float(score),
                "threshold": threshold,
                "accepted": bool(decision),
            })
    return (
        false_rejections / genuine_count,
        false_acceptances / impostor_count,
        rows,
    )


def run_fold(
    fold_number: int = 1,
    update_seconds: float = 30.0,
    calibration_seconds: float = 20.0,
    fusion_window: int = 5,
    target_far: float = 0.005,
    hard_negatives_per_participant: int = 80,
    adaptive_search: bool = False,
    seed: int = 42,
):
    """Run one participant-disjoint temporary-verifier experiment."""

    if not 0 <= target_far < 1:
        raise ValueError("target_far must be between 0 and 1")
    if adaptive_search and update_seconds < 60:
        raise ValueError("adaptive_search requires update_seconds of at least 60")
    dataset = load_features()
    fold = create_outer_folds(tuple(sorted(dataset)), 4)[fold_number - 1]
    tag = (
        f"fold_{fold_number}_update_{update_seconds:g}s_"
        f"fusion_{fusion_window}_far_{target_far:g}_"
        f"hard_{hard_negatives_per_participant}"
    )
    result_dir = Path("src/results/state_specific_adaptation") / tag
    model_dir = Path("src/models/state_specific_adaptation") / tag
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    development_learning = {condition: [] for condition in SESSIONS}
    development_validation = {condition: [] for condition in SESSIONS}
    for participant_id in fold.development_participants:
        for condition in SESSIONS:
            learning, validation = split_development_frame(dataset[participant_id][condition])
            development_learning[condition].append(learning)
            development_validation[condition].append(validation)

    normal_negative_training = _balanced_cohort(
        development_learning["st_control"], 80, seed
    )
    normal_negative_validation = _balanced_cohort(
        development_validation["st_control"], 40, seed + 1
    )
    condition_negative_training = {
        condition: _balanced_cohort(development_learning[condition], 80, seed + index + 10)
        for index, condition in enumerate(CHANGED_CONDITIONS)
    }
    condition_negative_validation = {
        condition: _balanced_cohort(development_validation[condition], 40, seed + index + 20)
        for index, condition in enumerate(CHANGED_CONDITIONS)
    }

    evaluation_splits = {}
    for participant_id in fold.evaluation_participants:
        normal = split_normal_enrollment(dataset[participant_id]["st_control"])
        evaluation_splits[participant_id] = {"st_control": normal}
        for condition in CHANGED_CONDITIONS:
            changed = split_trusted_session(
                dataset[participant_id][condition], update_seconds, calibration_seconds
            )
            evaluation_splits[participant_id][condition] = (
                changed.update, changed.calibration, changed.test
            )

    metric_rows = []
    score_rows = []
    for participant_index, claimed_id in enumerate(fold.evaluation_participants):
        normal_train, normal_calibration, _ = evaluation_splits[claimed_id]["st_control"]
        static_model = _fit_verifier(
            normal_train, normal_negative_training, seed + participant_index
        )
        static_operating_points = {}
        normal_probes = {
            probe_id: evaluation_splits[probe_id]["st_control"][2]
            for probe_id in fold.evaluation_participants
        }
        for window_size, method in (
            (1, "static"),
            (fusion_window, f"static_fused_{fusion_window}"),
        ):
            threshold, calibration_frr, calibration_far = _calibrate_state_model(
                static_model,
                normal_calibration,
                normal_negative_validation,
                target_far,
                fusion_window=window_size,
            )
            static_operating_points[window_size] = threshold
            frr, far, rows = _evaluate(
                static_model, threshold, claimed_id, normal_probes, window_size
            )
            metric_rows.append({
                "claimed_participant_id": claimed_id,
                "condition": "st_control",
                "method": method,
                "frr": frr,
                "far": far,
                "calibration_frr": calibration_frr,
                "calibration_far": calibration_far,
                "update_seconds": 0,
                "update_windows": 0,
                "calibration_seconds": 0,
                "fusion_window": window_size,
                "target_far": target_far,
            })
            for row in rows:
                score_rows.append({
                    "fold": fold_number,
                    "claimed_participant_id": claimed_id,
                    "condition": "st_control",
                    "method": method,
                    **row,
                })

        for condition in CHANGED_CONDITIONS:
            update, calibration, _ = evaluation_splits[claimed_id][condition]
            probes = {
                probe_id: evaluation_splits[probe_id][condition][2]
                for probe_id in fold.evaluation_participants
            }

            for window_size, method in (
                (1, "static"),
                (fusion_window, f"static_fused_{fusion_window}"),
            ):
                frr, far, rows = _evaluate(
                    static_model,
                    static_operating_points[window_size],
                    claimed_id,
                    probes,
                    window_size,
                )
                metric_rows.append({
                    "claimed_participant_id": claimed_id,
                    "condition": condition,
                    "method": method,
                    "frr": frr,
                    "far": far,
                    "calibration_frr": np.nan,
                    "calibration_far": np.nan,
                    "update_seconds": 0,
                    "update_windows": 0,
                    "calibration_seconds": 0,
                    "fusion_window": window_size,
                    "target_far": target_far,
                })
                for row in rows:
                    score_rows.append({
                        "fold": fold_number,
                        "claimed_participant_id": claimed_id,
                        "condition": condition,
                        "method": method,
                        **row,
                    })

            # Unlike mixed adaptation, only verified current-state gait is a
            # genuine class here. Normal gait remains in the permanent model.
            state_model = _refit_adapted_verifier(
                update,
                condition_negative_training[condition],
                static_model,
            )
            joblib.dump(state_model, model_dir / f"{claimed_id}_{condition}_state.joblib")

            for window_size, method in (
                (1, "state_specific"),
                (fusion_window, f"state_specific_fused_{fusion_window}"),
            ):
                threshold, calibration_frr, calibration_far = _calibrate_state_model(
                    state_model,
                    calibration,
                    condition_negative_validation[condition],
                    target_far,
                    fusion_window=window_size,
                )
                frr, far, rows = _evaluate(
                    state_model, threshold, claimed_id, probes, window_size
                )
                metric_rows.append({
                    "claimed_participant_id": claimed_id,
                    "condition": condition,
                    "method": method,
                    "frr": frr,
                    "far": far,
                    "calibration_frr": calibration_frr,
                    "calibration_far": calibration_far,
                    "update_seconds": update_seconds,
                    "update_windows": len(update),
                    "calibration_seconds": calibration_seconds,
                    "fusion_window": window_size,
                    "target_far": target_far,
                })
                for row in rows:
                    score_rows.append({
                        "fold": fold_number,
                        "claimed_participant_id": claimed_id,
                        "condition": condition,
                        "method": method,
                        **row,
                    })

            hard_negatives = mine_hard_negatives(
                state_model,
                development_learning[condition],
                maximum_per_participant=hard_negatives_per_participant,
            )
            robust_negatives = pd.concat(
                (condition_negative_training[condition], hard_negatives),
                ignore_index=True,
            )
            hard_model = _refit_adapted_verifier(
                update,
                robust_negatives,
                static_model,
            )
            joblib.dump(
                hard_model,
                model_dir / f"{claimed_id}_{condition}_hard_negative.joblib",
            )
            threshold, calibration_frr, calibration_far = _calibrate_state_model(
                hard_model,
                calibration,
                condition_negative_validation[condition],
                target_far,
                fusion_window=1,
            )
            frr, far, rows = _evaluate(
                hard_model, threshold, claimed_id, probes, fusion_window=1
            )
            metric_rows.append({
                "claimed_participant_id": claimed_id,
                "condition": condition,
                "method": "state_specific_hard_negative",
                "frr": frr,
                "far": far,
                "calibration_frr": calibration_frr,
                "calibration_far": calibration_far,
                "update_seconds": update_seconds,
                "update_windows": len(update),
                "calibration_seconds": calibration_seconds,
                "fusion_window": 1,
                "target_far": target_far,
                "hard_negatives_per_participant": hard_negatives_per_participant,
            })
            for row in rows:
                score_rows.append({
                    "fold": fold_number,
                    "claimed_participant_id": claimed_id,
                    "condition": condition,
                    "method": "state_specific_hard_negative",
                    **row,
                })

            if adaptive_search:
                selected, candidates = fit_selected_state_model(
                    static_model=static_model,
                    trusted_update=update,
                    calibration=calibration,
                    development_learning_frames=development_learning[condition],
                    base_negatives=condition_negative_training[condition],
                    validation_negatives=condition_negative_validation[condition],
                    target_far=target_far,
                )
                selected_model = selected["model"]
                joblib.dump(
                    selected_model,
                    model_dir / f"{claimed_id}_{condition}_selected.joblib",
                )
                frr, far, rows = _evaluate(
                    selected_model,
                    selected["threshold"],
                    claimed_id,
                    probes,
                    fusion_window=1,
                )
                metric_rows.append({
                    "claimed_participant_id": claimed_id,
                    "condition": condition,
                    "method": "state_specific_selected",
                    "frr": frr,
                    "far": far,
                    "calibration_frr": selected["calibration_frr"],
                    "calibration_far": selected["calibration_far"],
                    "update_seconds": selected["update_seconds"],
                    "update_windows": selected["update_windows"],
                    "calibration_seconds": calibration_seconds,
                    "fusion_window": 1,
                    "target_far": target_far,
                    "hard_negatives_per_participant": selected[
                        "hard_negatives_per_participant"
                    ],
                    "candidate_count": len(candidates),
                })
                for row in rows:
                    score_rows.append({
                        "fold": fold_number,
                        "claimed_participant_id": claimed_id,
                        "condition": condition,
                        "method": "state_specific_selected",
                        **row,
                    })

    metrics = pd.DataFrame(metric_rows)
    scores = pd.DataFrame(score_rows)
    summary = metrics.groupby(["condition", "method"])[["frr", "far"]].agg(["mean", "std"])
    metrics.to_csv(result_dir / "participant_metrics.csv", index=False)
    scores.to_csv(result_dir / "evaluation_scores.csv", index=False)
    summary.to_csv(result_dir / "macro_summary.csv")
    print(summary.to_string())
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run temporary state-specific gait adaptation")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--update-seconds", type=float, default=30.0)
    parser.add_argument("--calibration-seconds", type=float, default=20.0)
    parser.add_argument("--fusion-window", type=int, default=5)
    parser.add_argument("--target-far", type=float, default=0.005)
    parser.add_argument("--hard-negatives-per-participant", type=int, default=80)
    parser.add_argument("--adaptive-search", action="store_true")
    args = parser.parse_args()
    run_fold(
        fold_number=args.fold,
        update_seconds=args.update_seconds,
        calibration_seconds=args.calibration_seconds,
        fusion_window=args.fusion_window,
        target_far=args.target_far,
        hard_negatives_per_participant=args.hard_negatives_per_participant,
        adaptive_search=args.adaptive_search,
    )


if __name__ == "__main__":
    main()
