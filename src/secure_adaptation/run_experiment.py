"""Compare static normal-only verification with trusted online adaptation."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.condition_augmentation.run_experiment import (
    load_features,
    select_similarity_threshold,
    split_development_frame,
)
from src.condition_invariant.folds import create_outer_folds
from src.secure_adaptation.protocol import (
    split_normal_enrollment,
    split_trusted_session,
)
from src.utils import FEATURE_COLS, PARAM_GRID, SESSIONS, TARGET_FAR


CHANGED_CONDITIONS = tuple(condition for condition in SESSIONS if condition != "st_control")


def _class_balanced_group_cv(labels, groups, folds: int = 5, seed: int = 42):
    """Create group-disjoint folds containing both verification classes."""

    labels = np.asarray(labels)
    groups = np.asarray(groups)
    generator = np.random.default_rng(seed)
    validation_group_sets = [set() for _ in range(folds)]
    for class_label in (0, 1):
        class_groups = np.unique(groups[labels == class_label])
        if len(class_groups) < folds:
            raise ValueError(
                f"Class {class_label} has only {len(class_groups)} groups; "
                f"at least {folds} are required"
            )
        class_groups = generator.permutation(class_groups)
        for fold_index, fold_groups in enumerate(np.array_split(class_groups, folds)):
            validation_group_sets[fold_index].update(fold_groups.tolist())

    splits = []
    all_indices = np.arange(len(labels))
    for validation_groups in validation_group_sets:
        validation_mask = np.isin(groups, list(validation_groups))
        training_indices = all_indices[~validation_mask]
        validation_indices = all_indices[validation_mask]
        if set(labels[training_indices]) != {0, 1} or set(labels[validation_indices]) != {0, 1}:
            raise AssertionError("A model-selection fold is missing a class")
        splits.append((training_indices, validation_indices))
    return splits


def _balanced_cohort(frames, maximum_per_group: int, seed: int) -> pd.DataFrame:
    """Limit every participant-condition group to the same maximum size."""

    data = pd.concat(frames, ignore_index=True)
    generator = np.random.default_rng(seed)
    selected = []
    for _, group in data.groupby(["participant_id", "session_type"], sort=True):
        take = min(maximum_per_group, len(group))
        positions = np.sort(generator.choice(len(group), size=take, replace=False))
        selected.append(group.iloc[positions])
    return pd.concat(selected, ignore_index=True)


def _fit_verifier(positive: pd.DataFrame, negative: pd.DataFrame, seed: int):
    """Fit one participant-specific RBF-SVM with leakage-aware grouped CV."""

    data = pd.concat((positive, negative), ignore_index=True)
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    positive_groups = (
        "genuine:" + positive.session_type.astype(str)
        + ":" + positive.block_id.astype(str)
    ).to_numpy()
    negative_groups = (
        "cohort:" + negative.participant_id.astype(str)
        + ":" + negative.session_type.astype(str)
        + ":" + negative.block_id.astype(str)
    ).to_numpy()
    groups = np.concatenate((positive_groups, negative_groups))
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced")),
    ])
    cross_validation = _class_balanced_group_cv(labels, groups, folds=5, seed=seed)
    search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        scoring="roc_auc",
        cv=cross_validation,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(data[FEATURE_COLS], labels, groups=groups)
    return search.best_estimator_


def _refit_adapted_verifier(
    positive: pd.DataFrame,
    negative: pd.DataFrame,
    static_model: Pipeline,
):
    """Refit using trusted gait while preserving enrollment-selected settings.

    A short trusted update can belong to only one chronological block. Running
    grouped cross-validation on that block can produce single-class folds, for
    which ROC AUC is undefined. More importantly, deployment adaptation should
    update the identity evidence rather than tune model hyperparameters using a
    person's new condition. C and gamma are therefore frozen from the normal-
    enrollment verifier.
    """

    data = pd.concat((positive, negative), ignore_index=True)
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    static_svm = static_model.named_steps["svm"]
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                class_weight="balanced",
                C=static_svm.C,
                gamma=static_svm.gamma,
            ),
        ),
    ])
    model.fit(data[FEATURE_COLS], labels)
    return model


def _calibrate(model, positive: pd.DataFrame, negative: pd.DataFrame):
    positive_scores = model.decision_function(positive[FEATURE_COLS])
    negative_scores = model.decision_function(negative[FEATURE_COLS])
    return select_similarity_threshold(positive_scores, negative_scores, TARGET_FAR)


def _rates_on_calibration(model, threshold, positive, negative):
    """Measure current-state FRR/FAR before deciding whether to adapt."""

    positive_scores = model.decision_function(positive[FEATURE_COLS])
    negative_scores = model.decision_function(negative[FEATURE_COLS])
    frr = float(np.mean(positive_scores < threshold))
    far = float(np.mean(negative_scores >= threshold))
    return frr, far


def should_activate_adaptation(
    static_frr: float,
    adapted_frr: float,
    adapted_far: float,
    target_far: float = TARGET_FAR,
) -> bool:
    """Activate only a demonstrably more usable, security-compliant model."""

    return adapted_frr < static_frr and adapted_far <= target_far + 1e-12


def _rates(model, threshold: float, claimed_id: str, probes: dict[str, pd.DataFrame]):
    genuine_count = impostor_count = false_rejections = false_acceptances = 0
    score_rows = []
    for probe_id, frame in probes.items():
        scores = model.decision_function(frame[FEATURE_COLS])
        accepted = scores >= threshold
        genuine = probe_id == claimed_id
        if genuine:
            genuine_count += len(scores)
            false_rejections += int((~accepted).sum())
        else:
            impostor_count += len(scores)
            false_acceptances += int(accepted.sum())
        for row, score, decision in zip(frame.itertuples(), scores, accepted):
            score_rows.append({
                "probe_participant_id": probe_id,
                "window_index": row.window_index,
                "start_sample": row.start_sample,
                "is_genuine": genuine,
                "score": float(score),
                "threshold": threshold,
                "accepted": bool(decision),
            })
    return (
        false_rejections / genuine_count,
        false_acceptances / impostor_count,
        score_rows,
    )


def run_fold(
    fold_number: int = 1,
    update_seconds: float = 20.0,
    calibration_seconds: float = 10.0,
    seed: int = 42,
):
    """Run static and securely adapted verifiers on one participant fold."""

    dataset = load_features()
    fold = create_outer_folds(tuple(sorted(dataset)), 4)[fold_number - 1]
    tag = f"fold_{fold_number}_update_{update_seconds:g}s"
    result_dir = Path("src/results/secure_adaptation") / tag
    model_dir = Path("src/models/secure_adaptation") / tag
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
    changed_negative_training = {
        condition: _balanced_cohort(development_learning[condition], 80, seed + index + 10)
        for index, condition in enumerate(CHANGED_CONDITIONS)
    }
    changed_negative_validation = {
        condition: _balanced_cohort(development_validation[condition], 40, seed + index + 20)
        for index, condition in enumerate(CHANGED_CONDITIONS)
    }

    evaluation_splits = {}
    for participant_id in fold.evaluation_participants:
        normal_train, normal_calibration, normal_test = split_normal_enrollment(
            dataset[participant_id]["st_control"]
        )
        evaluation_splits[participant_id] = {
            "st_control": (normal_train, normal_calibration, normal_test)
        }
        for condition in CHANGED_CONDITIONS:
            split = split_trusted_session(
                dataset[participant_id][condition], update_seconds, calibration_seconds
            )
            evaluation_splits[participant_id][condition] = (
                split.update, split.calibration, split.test
            )

    metric_rows = []
    score_rows = []
    for participant_index, claimed_id in enumerate(fold.evaluation_participants):
        normal_train, normal_calibration, normal_test = evaluation_splits[claimed_id]["st_control"]
        static_model = _fit_verifier(
            normal_train, normal_negative_training, seed + participant_index
        )
        static_threshold, static_calibration_frr, static_calibration_far = _calibrate(
            static_model, normal_calibration, normal_negative_validation
        )
        joblib.dump(static_model, model_dir / f"{claimed_id}_static.joblib")

        normal_probes = {
            probe_id: evaluation_splits[probe_id]["st_control"][2]
            for probe_id in fold.evaluation_participants
        }
        frr, far, rows = _rates(static_model, static_threshold, claimed_id, normal_probes)
        metric_rows.append({
            "claimed_participant_id": claimed_id,
            "condition": "st_control",
            "method": "static",
            "frr": frr,
            "far": far,
            "calibration_frr": static_calibration_frr,
            "calibration_far": static_calibration_far,
            "update_windows": 0,
        })
        for row in rows:
            score_rows.append({"fold": fold_number, "claimed_participant_id": claimed_id,
                               "condition": "st_control", "method": "static", **row})

        for condition in CHANGED_CONDITIONS:
            update, adapted_calibration, _ = evaluation_splits[claimed_id][condition]
            probes = {
                probe_id: evaluation_splits[probe_id][condition][2]
                for probe_id in fold.evaluation_participants
            }

            # Static result establishes what happens before the trusted update.
            frr, far, rows = _rates(static_model, static_threshold, claimed_id, probes)
            metric_rows.append({
                "claimed_participant_id": claimed_id,
                "condition": condition,
                "method": "static",
                "frr": frr,
                "far": far,
                "calibration_frr": static_calibration_frr,
                "calibration_far": static_calibration_far,
                "update_windows": 0,
            })
            for row in rows:
                score_rows.append({"fold": fold_number, "claimed_participant_id": claimed_id,
                                   "condition": condition, "method": "static", **row})

            # The target's changed-condition samples enter training only because
            # a trusted secondary factor has already verified their identity.
            adapted_positive = pd.concat((normal_train, update), ignore_index=True)
            adapted_negative = pd.concat(
                (normal_negative_training, changed_negative_training[condition]),
                ignore_index=True,
            )
            adapted_model = _refit_adapted_verifier(
                adapted_positive, adapted_negative, static_model
            )
            adapted_threshold, calibration_frr, calibration_far = _calibrate(
                adapted_model,
                adapted_calibration,
                changed_negative_validation[condition],
            )
            joblib.dump(adapted_model, model_dir / f"{claimed_id}_{condition}_adapted.joblib")
            frr, far, rows = _rates(adapted_model, adapted_threshold, claimed_id, probes)
            metric_rows.append({
                "claimed_participant_id": claimed_id,
                "condition": condition,
                "method": "adapted",
                "frr": frr,
                "far": far,
                "calibration_frr": calibration_frr,
                "calibration_far": calibration_far,
                "update_windows": len(update),
            })
            for row in rows:
                score_rows.append({"fold": fold_number, "claimed_participant_id": claimed_id,
                                   "condition": condition, "method": "adapted", **row})

            static_current_frr, static_current_far = _rates_on_calibration(
                static_model,
                static_threshold,
                adapted_calibration,
                changed_negative_validation[condition],
            )
            adaptation_activated = should_activate_adaptation(
                static_current_frr, calibration_frr, calibration_far
            )
            gated_model = adapted_model if adaptation_activated else static_model
            gated_threshold = adapted_threshold if adaptation_activated else static_threshold
            gated_calibration_frr = calibration_frr if adaptation_activated else static_current_frr
            gated_calibration_far = calibration_far if adaptation_activated else static_current_far
            frr, far, rows = _rates(gated_model, gated_threshold, claimed_id, probes)
            metric_rows.append({
                "claimed_participant_id": claimed_id,
                "condition": condition,
                "method": "gated",
                "frr": frr,
                "far": far,
                "calibration_frr": gated_calibration_frr,
                "calibration_far": gated_calibration_far,
                "static_current_calibration_frr": static_current_frr,
                "static_current_calibration_far": static_current_far,
                "update_windows": len(update),
                "adaptation_activated": adaptation_activated,
            })
            for row in rows:
                score_rows.append({"fold": fold_number, "claimed_participant_id": claimed_id,
                                   "condition": condition, "method": "gated", **row})

    metrics = pd.DataFrame(metric_rows)
    scores = pd.DataFrame(score_rows)
    summary = metrics.groupby(["condition", "method"])[["frr", "far"]].agg(["mean", "std"])
    metrics.to_csv(result_dir / "participant_metrics.csv", index=False)
    scores.to_csv(result_dir / "evaluation_scores.csv", index=False)
    summary.to_csv(result_dir / "macro_summary.csv")
    print(summary.to_string())
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run secure online gait adaptation")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--update-seconds", type=float, default=20.0)
    parser.add_argument("--calibration-seconds", type=float, default=10.0)
    args = parser.parse_args()
    run_fold(args.fold, args.update_seconds, args.calibration_seconds)


if __name__ == "__main__":
    main()
