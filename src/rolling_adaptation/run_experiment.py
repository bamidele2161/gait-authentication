"""Evaluate secure rolling adaptation on the evolving ST-fatigue session."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.condition_augmentation.run_experiment import split_development_frame
from src.condition_invariant.folds import create_outer_folds
from src.rolling_adaptation.protocol import create_rolling_schedule, windows_inside
from src.secure_adaptation.protocol import split_normal_enrollment
from src.secure_adaptation.run_experiment import (
    _balanced_cohort, _fit_verifier, _refit_adapted_verifier,
)
from src.two_stage_verification.decision import and_decision
from src.two_stage_verification.features import SPECTRAL_COLS
from src.two_stage_verification.run_experiment import (
    calibrate_group_robust, fit_model, load_combined_features,
)
from src.utils import FEATURE_COLS


def select_group_robust_threshold(model, positive, negative, target_far):
    """Select a similarity threshold satisfying FAR for every cohort identity."""
    positive_scores = model.decision_function(positive[FEATURE_COLS])
    negative_scores = model.decision_function(negative[FEATURE_COLS])
    groups = negative.participant_id.astype(str).to_numpy()
    best = None
    for threshold in np.r_[np.unique(positive_scores), np.inf]:
        frr = float(np.mean(positive_scores < threshold))
        accepted = negative_scores >= threshold
        group_fars = [
            float(accepted[groups == group].mean()) for group in np.unique(groups)
        ]
        worst_far = max(group_fars)
        pooled_far = float(accepted.mean())
        if worst_far <= target_far + 1e-12:
            candidate = (frr, worst_far, pooled_far, -threshold)
            if best is None or candidate < best[0]:
                best = (candidate, threshold)
    if best is None:
        raise ValueError("No robust threshold satisfies target FAR")
    (frr, worst_far, pooled_far, _), threshold = best
    return float(threshold), frr, pooled_far, worst_far


def score_period(model, threshold, claimed_id, probes):
    genuine = impostor = false_reject = false_accept = 0
    rows = []
    for probe_id, frame in probes.items():
        if frame.empty:
            continue
        scores = model.decision_function(frame[FEATURE_COLS])
        accepted = scores >= threshold
        is_genuine = probe_id == claimed_id
        if is_genuine:
            genuine += len(frame)
            false_reject += int((~accepted).sum())
        else:
            impostor += len(frame)
            false_accept += int(accepted.sum())
        for window, score, decision in zip(frame.itertuples(), scores, accepted):
            rows.append({
                "probe_participant_id": probe_id,
                "window_index": window.window_index,
                "start_sample": window.start_sample,
                "is_genuine": is_genuine,
                "score": float(score),
                "threshold": threshold,
                "accepted": bool(decision),
            })
    return genuine, impostor, false_reject, false_accept, rows


def score_two_stage_period(primary, secondary, thresholds, claimed_id, probes):
    """Accept only windows approved by rolling time and spectral stages."""
    genuine = impostor = false_reject = false_accept = 0
    rows = []
    for probe_id, frame in probes.items():
        if frame.empty:
            continue
        first = primary.decision_function(frame[FEATURE_COLS])
        second = secondary.decision_function(frame[SPECTRAL_COLS])
        accepted = and_decision(first, second, thresholds[0], thresholds[1])
        is_genuine = probe_id == claimed_id
        if is_genuine:
            genuine += len(frame)
            false_reject += int((~accepted).sum())
        else:
            impostor += len(frame)
            false_accept += int(accepted.sum())
        for window, score, other_score, decision in zip(
            frame.itertuples(), first, second, accepted
        ):
            rows.append({
                "probe_participant_id": probe_id,
                "window_index": window.window_index,
                "start_sample": window.start_sample,
                "is_genuine": is_genuine,
                "score": float(score),
                "secondary_score": float(other_score),
                "threshold": float(thresholds[0]),
                "secondary_threshold": float(thresholds[1]),
                "accepted": bool(decision),
            })
    return genuine, impostor, false_reject, false_accept, rows


def score_rescue_period(
    fixed_model, fixed_threshold, rolling_model, secondary, rolling_thresholds,
    claimed_id, probes,
):
    """Let rolling-plus-spectral agreement rescue fixed-model rejections."""
    genuine = impostor = false_reject = false_accept = 0
    rows = []
    for probe_id, frame in probes.items():
        if frame.empty:
            continue
        fixed_scores = fixed_model.decision_function(frame[FEATURE_COLS])
        rolling_scores = rolling_model.decision_function(frame[FEATURE_COLS])
        spectral_scores = secondary.decision_function(frame[SPECTRAL_COLS])
        fixed_accept = fixed_scores >= fixed_threshold
        rescue_accept = and_decision(
            rolling_scores, spectral_scores,
            rolling_thresholds[0], rolling_thresholds[1],
        )
        accepted = fixed_accept | rescue_accept
        is_genuine = probe_id == claimed_id
        if is_genuine:
            genuine += len(frame)
            false_reject += int((~accepted).sum())
        else:
            impostor += len(frame)
            false_accept += int(accepted.sum())
        for window, score, other_score, decision in zip(
            frame.itertuples(), rolling_scores, spectral_scores, accepted
        ):
            rows.append({
                "probe_participant_id": probe_id,
                "window_index": window.window_index,
                "start_sample": window.start_sample,
                "is_genuine": is_genuine,
                "score": float(score),
                "secondary_score": float(other_score),
                "threshold": float(rolling_thresholds[0]),
                "secondary_threshold": float(rolling_thresholds[1]),
                "accepted": bool(decision),
            })
    return genuine, impostor, false_reject, false_accept, rows


def run_fold(fold_number=1, target_far=0.005, buffer_windows=60, seed=42):
    dataset = load_combined_features()
    fold = create_outer_folds(tuple(sorted(dataset)), 4)[fold_number - 1]
    result_dir = Path("src/results/rolling_adaptation") / f"fold_{fold_number}"
    model_dir = Path("src/models/rolling_adaptation") / f"fold_{fold_number}"
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    fatigue_learning, fatigue_validation = [], []
    normal_learning = []
    for participant_id in fold.development_participants:
        learn, valid = split_development_frame(dataset[participant_id]["st_fatigue"])
        fatigue_learning.append(learn)
        fatigue_validation.append(valid)
        normal_learn, _ = split_development_frame(dataset[participant_id]["st_control"])
        normal_learning.append(normal_learn)
    fatigue_negative_train = _balanced_cohort(fatigue_learning, 80, seed)
    fatigue_negative_validation = _balanced_cohort(fatigue_validation, 40, seed + 1)
    normal_negative_train = _balanced_cohort(normal_learning, 80, seed + 2)

    schedules = {
        participant_id: create_rolling_schedule(dataset[participant_id]["st_fatigue"])
        for participant_id in fold.evaluation_participants
    }
    metric_rows, score_rows = [], []
    for participant_index, claimed_id in enumerate(fold.evaluation_participants):
        normal_train = split_normal_enrollment(dataset[claimed_id]["st_control"])[0]
        reference = _fit_verifier(normal_train, normal_negative_train, seed + participant_index)
        schedule = schedules[claimed_id]
        trusted_buffer = schedule.initial_update.copy()
        rolling_model = _refit_adapted_verifier(
            trusted_buffer, fatigue_negative_train, reference
        )
        rolling_threshold, _, _, _ = select_group_robust_threshold(
            rolling_model, schedule.initial_calibration,
            fatigue_negative_validation, target_far,
        )
        fixed_model = rolling_model
        fixed_threshold = rolling_threshold
        rolling_secondary = fit_model(
            trusted_buffer, fatigue_negative_train, SPECTRAL_COLS, reference
        )
        rolling_two_stage_thresholds = calibrate_group_robust(
            rolling_model, rolling_secondary, schedule.initial_calibration,
            fatigue_negative_validation, target_far,
        )[:2]
        fixed_secondary = rolling_secondary
        fixed_two_stage_thresholds = rolling_two_stage_thresholds
        totals = {
            "fixed_state": [0, 0, 0, 0],
            "rolling_state": [0, 0, 0, 0],
            "fixed_two_stage": [0, 0, 0, 0],
            "rolling_two_stage": [0, 0, 0, 0],
            "rolling_spectral_rescue": [0, 0, 0, 0],
        }

        for period in schedule.periods:
            if not period.refresh_update.empty:
                trusted_buffer = pd.concat(
                    (trusted_buffer, period.refresh_update), ignore_index=True
                ).sort_values("start_sample").tail(buffer_windows)
                rolling_model = _refit_adapted_verifier(
                    trusted_buffer, fatigue_negative_train, reference
                )
                rolling_threshold, _, _, _ = select_group_robust_threshold(
                    rolling_model, period.refresh_calibration,
                    fatigue_negative_validation, target_far,
                )
                rolling_secondary = fit_model(
                    trusted_buffer, fatigue_negative_train, SPECTRAL_COLS, reference
                )
                rolling_two_stage_thresholds = calibrate_group_robust(
                    rolling_model, rolling_secondary, period.refresh_calibration,
                    fatigue_negative_validation, target_far,
                )[:2]
            probes = {
                probe_id: windows_inside(
                    dataset[probe_id]["st_fatigue"],
                    period.evaluation_start_seconds,
                    period.evaluation_end_seconds,
                )
                for probe_id in fold.evaluation_participants
            }
            for method, model, threshold in (
                ("fixed_state", fixed_model, fixed_threshold),
                ("rolling_state", rolling_model, rolling_threshold),
            ):
                result = score_period(model, threshold, claimed_id, probes)
                for index in range(4):
                    totals[method][index] += result[index]
                for row in result[4]:
                    score_rows.append({
                        "fold": fold_number,
                        "claimed_participant_id": claimed_id,
                        "condition": "st_fatigue",
                        "method": method,
                        "cycle": period.cycle,
                        **row,
                    })

            for method, primary, secondary, thresholds in (
                ("fixed_two_stage", fixed_model, fixed_secondary,
                 fixed_two_stage_thresholds),
                ("rolling_two_stage", rolling_model, rolling_secondary,
                 rolling_two_stage_thresholds),
            ):
                result = score_two_stage_period(
                    primary, secondary, thresholds, claimed_id, probes
                )
                for index in range(4):
                    totals[method][index] += result[index]
                for row in result[4]:
                    score_rows.append({
                        "fold": fold_number,
                        "claimed_participant_id": claimed_id,
                        "condition": "st_fatigue",
                        "method": method,
                        "cycle": period.cycle,
                        **row,
                    })

            result = score_rescue_period(
                fixed_model, fixed_threshold,
                rolling_model, rolling_secondary, rolling_two_stage_thresholds,
                claimed_id, probes,
            )
            for index in range(4):
                totals["rolling_spectral_rescue"][index] += result[index]
            for row in result[4]:
                score_rows.append({
                    "fold": fold_number,
                    "claimed_participant_id": claimed_id,
                    "condition": "st_fatigue",
                    "method": "rolling_spectral_rescue",
                    "cycle": period.cycle,
                    **row,
                })

        for method, (genuine, impostor, false_reject, false_accept) in totals.items():
            metric_rows.append({
                "claimed_participant_id": claimed_id,
                "condition": "st_fatigue",
                "method": method,
                "frr": false_reject / genuine,
                "far": false_accept / impostor,
                "genuine_decisions": genuine,
                "impostor_decisions": impostor,
                "buffer_windows": buffer_windows,
                "target_far": target_far,
            })
        joblib.dump(rolling_model, model_dir / f"{claimed_id}_rolling_final.joblib")

    metrics = pd.DataFrame(metric_rows)
    scores = pd.DataFrame(score_rows)
    summary = metrics.groupby("method")[["frr", "far"]].agg(["mean", "std"])
    metrics.to_csv(result_dir / "participant_metrics.csv", index=False)
    scores.to_csv(result_dir / "evaluation_scores.csv", index=False)
    summary.to_csv(result_dir / "macro_summary.csv")
    print(summary.to_string())
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run secure rolling ST-fatigue adaptation")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--target-far", type=float, default=0.005)
    parser.add_argument("--buffer-windows", type=int, default=60)
    args = parser.parse_args()
    run_fold(args.fold, args.target_far, args.buffer_windows)


if __name__ == "__main__":
    main()
