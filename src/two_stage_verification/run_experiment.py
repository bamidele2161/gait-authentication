"""Evaluate a statistical-plus-periodicity two-stage gait verifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.condition_augmentation.run_experiment import load_features, split_development_frame
from src.condition_invariant.folds import create_outer_folds
from src.secure_adaptation.protocol import split_normal_enrollment, split_trusted_session
from src.secure_adaptation.run_experiment import _balanced_cohort, _fit_verifier
from src.two_stage_verification.decision import (
    and_decision,
    select_and_thresholds,
    select_group_robust_and_thresholds,
)
from src.two_stage_verification.features import SPECTRAL_COLS, load_spectral_dataset
from src.utils import FEATURE_COLS, SESSIONS


CHANGED_CONDITIONS = tuple(condition for condition in SESSIONS if condition != "st_control")
KEY_COLS = ["participant_id", "session_type", "window_index", "start_sample", "block_id"]


def load_combined_features():
    """Align existing statistical features with complementary features."""
    statistical = load_features()
    spectral = load_spectral_dataset(SESSIONS)
    combined = {}
    for participant_id, conditions in statistical.items():
        combined[participant_id] = {}
        for condition, time_frame in conditions.items():
            other = spectral[participant_id][condition]
            frame = time_frame.merge(
                other[KEY_COLS + SPECTRAL_COLS], on=KEY_COLS,
                how="inner", validate="one_to_one",
            ).sort_values("start_sample").reset_index(drop=True)
            if len(frame) != len(time_frame):
                raise ValueError(f"Feature alignment failed for {participant_id} {condition}")
            combined[participant_id][condition] = frame
    return combined


def fit_model(positive, negative, columns, reference_model):
    """Fit one stage using enrollment-selected SVM settings."""
    reference_svm = reference_model.named_steps["svm"]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced",
                    C=reference_svm.C, gamma=reference_svm.gamma)),
    ])
    data = pd.concat((positive, negative), ignore_index=True)
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    model.fit(data[columns], labels)
    return model


def paired_scores(primary, secondary, frame):
    return (
        primary.decision_function(frame[FEATURE_COLS]),
        secondary.decision_function(frame[SPECTRAL_COLS]),
    )


def calibrate_two_stage(primary, secondary, positive, negative, target_far):
    pp, ps = paired_scores(primary, secondary, positive)
    np_, ns = paired_scores(primary, secondary, negative)
    return select_and_thresholds(pp, ps, np_, ns, target_far)


def calibrate_group_robust(primary, secondary, positive, negative, target_far):
    """Calibrate against the worst development impostor, not pooled scores."""
    pp, ps = paired_scores(primary, secondary, positive)
    np_, ns = paired_scores(primary, secondary, negative)
    return select_group_robust_and_thresholds(
        pp, ps, np_, ns,
        negative.participant_id.astype(str).to_numpy(),
        target_far,
    )


def evaluate_two_stage(primary, secondary, thresholds, claimed_id, probes):
    primary_threshold, secondary_threshold = thresholds
    genuine_count = impostor_count = false_rejections = false_acceptances = 0
    rows = []
    for probe_id, frame in probes.items():
        primary_scores, secondary_scores = paired_scores(primary, secondary, frame)
        accepted = and_decision(
            primary_scores, secondary_scores, primary_threshold, secondary_threshold
        )
        genuine = probe_id == claimed_id
        if genuine:
            genuine_count += len(frame)
            false_rejections += int((~accepted).sum())
        else:
            impostor_count += len(frame)
            false_acceptances += int(accepted.sum())
        for window, first, second, decision in zip(
            frame.itertuples(), primary_scores, secondary_scores, accepted
        ):
            rows.append({
                "probe_participant_id": probe_id,
                "window_index": window.window_index,
                "start_sample": window.start_sample,
                "is_genuine": genuine,
                "primary_score": float(first),
                "secondary_score": float(second),
                "primary_threshold": primary_threshold,
                "secondary_threshold": secondary_threshold,
                "accepted": bool(decision),
            })
    return false_rejections / genuine_count, false_acceptances / impostor_count, rows


def run_fold(fold_number=1, update_seconds=30.0, calibration_seconds=20.0,
             target_far=0.005, seed=42):
    dataset = load_combined_features()
    fold = create_outer_folds(tuple(sorted(dataset)), 4)[fold_number - 1]
    tag = f"fold_{fold_number}_update_{update_seconds:g}s_far_{target_far:g}"
    result_dir = Path("src/results/two_stage_verification") / tag
    model_dir = Path("src/models/two_stage_verification") / tag
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    learning = {condition: [] for condition in SESSIONS}
    validation = {condition: [] for condition in SESSIONS}
    for participant_id in fold.development_participants:
        for condition in SESSIONS:
            learn, valid = split_development_frame(dataset[participant_id][condition])
            learning[condition].append(learn)
            validation[condition].append(valid)
    normal_negatives = _balanced_cohort(learning["st_control"], 80, seed)
    condition_training = {
        condition: _balanced_cohort(learning[condition], 80, seed + index + 10)
        for index, condition in enumerate(CHANGED_CONDITIONS)
    }
    condition_validation = {
        condition: _balanced_cohort(validation[condition], 40, seed + index + 20)
        for index, condition in enumerate(CHANGED_CONDITIONS)
    }

    splits = {}
    for participant_id in fold.evaluation_participants:
        splits[participant_id] = {
            "st_control": split_normal_enrollment(dataset[participant_id]["st_control"])
        }
        for condition in CHANGED_CONDITIONS:
            changed = split_trusted_session(
                dataset[participant_id][condition], update_seconds, calibration_seconds
            )
            splits[participant_id][condition] = (
                changed.update, changed.calibration, changed.test
            )

    metrics, scores = [], []
    for participant_index, claimed_id in enumerate(fold.evaluation_participants):
        normal_train = splits[claimed_id]["st_control"][0]
        reference = _fit_verifier(normal_train, normal_negatives, seed + participant_index)
        for condition in CHANGED_CONDITIONS:
            update, calibration, _ = splits[claimed_id][condition]
            negatives = condition_training[condition]
            primary = fit_model(update, negatives, FEATURE_COLS, reference)
            secondary = fit_model(update, negatives, SPECTRAL_COLS, reference)
            thresholds = calibrate_two_stage(
                primary, secondary, calibration, condition_validation[condition], target_far
            )
            probes = {
                probe_id: splits[probe_id][condition][2]
                for probe_id in fold.evaluation_participants
            }
            frr, far, rows = evaluate_two_stage(
                primary, secondary, thresholds[:2], claimed_id, probes
            )
            metrics.append({
                "claimed_participant_id": claimed_id,
                "condition": condition,
                "method": "two_stage_and",
                "frr": frr,
                "far": far,
                "calibration_frr": thresholds[2],
                "calibration_far": thresholds[3],
                "update_seconds": update_seconds,
                "target_far": target_far,
            })
            for row in rows:
                scores.append({"fold": fold_number, "claimed_participant_id": claimed_id,
                               "condition": condition, **row})
            joblib.dump(primary, model_dir / f"{claimed_id}_{condition}_primary.joblib")
            joblib.dump(secondary, model_dir / f"{claimed_id}_{condition}_secondary.joblib")

            robust = calibrate_group_robust(
                primary, secondary, calibration,
                condition_validation[condition], target_far
            )
            robust_frr, robust_far, robust_rows = evaluate_two_stage(
                primary, secondary, robust[:2], claimed_id, probes
            )
            metrics.append({
                "claimed_participant_id": claimed_id,
                "condition": condition,
                "method": "two_stage_group_robust",
                "frr": robust_frr,
                "far": robust_far,
                "calibration_frr": robust[2],
                "calibration_far": robust[3],
                "calibration_worst_group_far": robust[4],
                "update_seconds": update_seconds,
                "target_far": target_far,
            })
            for row in robust_rows:
                scores.append({
                    "fold": fold_number,
                    "claimed_participant_id": claimed_id,
                    "condition": condition,
                    "method": "two_stage_group_robust",
                    **row,
                })

    metrics = pd.DataFrame(metrics)
    scores = pd.DataFrame(scores)
    summary = metrics.groupby(["condition", "method"])[["frr", "far"]].agg(["mean", "std"])
    metrics.to_csv(result_dir / "participant_metrics.csv", index=False)
    scores.to_csv(result_dir / "evaluation_scores.csv", index=False)
    summary.to_csv(result_dir / "macro_summary.csv")
    print(summary.to_string())
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run two-stage gait verification")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--update-seconds", type=float, default=30.0)
    parser.add_argument("--calibration-seconds", type=float, default=20.0)
    parser.add_argument("--target-far", type=float, default=0.005)
    args = parser.parse_args()
    run_fold(args.fold, args.update_seconds, args.calibration_seconds, args.target_far)


if __name__ == "__main__":
    main()
