"""Run development-derived condition augmentation on one outer fold."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.condition_augmentation.augmentation import (
    generate_condition_augmented_features,
    learn_condition_transforms,
)
from src.condition_invariant.folds import create_outer_folds
from src.utils import FEATURE_COLS, FEATURE_DIR, PARAM_GRID, SESSIONS, TARGET_FAR


def load_features() -> dict[str, dict[str, pd.DataFrame]]:
    """Load all sacrum time-domain feature files."""

    dataset: dict[str, dict[str, pd.DataFrame]] = {}
    for condition in SESSIONS:
        for path in sorted((FEATURE_DIR / condition).glob("*_features.csv")):
            participant_id = path.stem.replace("_features", "")
            frame = pd.read_csv(path).sort_values("start_sample").reset_index(drop=True)
            if not np.isfinite(frame[FEATURE_COLS].to_numpy()).all():
                raise ValueError(f"Non-finite features in {path}")
            dataset.setdefault(participant_id, {})[condition] = frame
    if not dataset or any(set(values) != set(SESSIONS) for values in dataset.values()):
        raise ValueError("Feature dataset is empty or incomplete")
    return dataset


def split_development_frame(frame: pd.DataFrame, learning_fraction: float = 0.8):
    """Split complete chronological blocks into learning and validation."""

    blocks = sorted(frame.block_id.unique())
    boundary = min(max(int(len(blocks) * learning_fraction), 1), len(blocks) - 1)
    learning_blocks = set(blocks[:boundary])
    validation_blocks = set(blocks[boundary:])
    learning = frame[frame.block_id.isin(learning_blocks)].copy()
    validation = frame[frame.block_id.isin(validation_blocks)].copy()
    first_validation = validation.start_sample.min()
    learning = learning[learning.start_sample + 256 <= first_validation].copy()
    if learning.empty or validation.empty:
        raise ValueError("Development split produced an empty partition")
    return learning, validation


def split_evaluation_control(frame: pd.DataFrame):
    """Use chronological 60/20/20 normal-walking enrolment protocol."""

    ordered = frame.sort_values("start_sample").reset_index(drop=True)
    train_end = int(0.60 * len(ordered))
    validation_end = int(0.80 * len(ordered))
    training = ordered.iloc[:train_end].copy()
    calibration = ordered.iloc[train_end + 1 : validation_end].copy()
    testing = ordered.iloc[validation_end + 1 :].copy()
    if training.empty or calibration.empty or testing.empty:
        raise ValueError("Evaluation ST-control split produced an empty partition")
    return training, calibration, testing


def sample_cohort(frame: pd.DataFrame, maximum_per_group: int, seed: int):
    """Sample equal maximum counts from each participant-condition group."""

    random_generator = np.random.default_rng(seed)
    selected = []
    for _, group in frame.groupby(["participant_id", "session_type"], sort=True):
        take = min(maximum_per_group, len(group))
        positions = np.sort(random_generator.choice(len(group), size=take, replace=False))
        selected.append(group.iloc[positions])
    return pd.concat(selected, ignore_index=True)


def select_similarity_threshold(positive_scores, negative_scores, target_far):
    """Select the least-rejecting similarity threshold within the FAR target."""

    positive = np.sort(np.asarray(positive_scores, dtype=np.float64))
    negative = np.sort(np.asarray(negative_scores, dtype=np.float64))
    observed = np.concatenate((positive, negative))
    candidates = np.concatenate((np.unique(observed), [np.nextafter(observed.max(), np.inf)]))
    accepted_positive = positive.size - np.searchsorted(positive, candidates, side="left")
    accepted_negative = negative.size - np.searchsorted(negative, candidates, side="left")
    frr = 1 - accepted_positive / positive.size
    far = accepted_negative / negative.size
    valid = np.flatnonzero(far <= target_far)
    order = np.lexsort((-candidates[valid], far[valid], frr[valid]))
    index = valid[int(order[0])]
    return float(candidates[index]), float(frr[index]), float(far[index])


def run_fold(fold_number: int = 1, seed: int = 42):
    """Run one participant-disjoint augmentation experiment."""

    dataset = load_features()
    participants = tuple(sorted(dataset))
    fold = create_outer_folds(participants, 4)[fold_number - 1]
    output_dir = Path("src/results/condition_augmentation") / f"compatible_fold_{fold_number}"
    model_dir = Path("src/models/condition_augmentation") / f"compatible_fold_{fold_number}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    development_learning = []
    development_validation = []
    learning_by_participant = {}
    for participant_id in fold.development_participants:
        learning_by_participant[participant_id] = {}
        for condition in SESSIONS:
            learning, validation = split_development_frame(dataset[participant_id][condition])
            development_learning.append(learning)
            development_validation.append(validation)
            learning_by_participant[participant_id][condition] = learning
    development_learning = pd.concat(development_learning, ignore_index=True)
    development_validation = pd.concat(development_validation, ignore_index=True)

    scaler = StandardScaler().fit(development_learning[FEATURE_COLS])
    standardized = {
        participant_id: {
            condition: scaler.transform(frame[FEATURE_COLS])
            for condition, frame in conditions.items()
        }
        for participant_id, conditions in learning_by_participant.items()
    }
    transforms = learn_condition_transforms(standardized)
    development_scaled = scaler.transform(development_learning[FEATURE_COLS])
    lower_clip = np.quantile(development_scaled, 0.005, axis=0)
    upper_clip = np.quantile(development_scaled, 0.995, axis=0)
    negative_learning = sample_cohort(development_learning, 40, seed)
    negative_validation = sample_cohort(development_validation, 40, seed + 1)
    negative_learning_x = scaler.transform(negative_learning[FEATURE_COLS])
    negative_validation_x = scaler.transform(negative_validation[FEATURE_COLS])

    metric_rows = []
    score_rows = []
    for participant_index, claimed_id in enumerate(fold.evaluation_participants):
        normal_training, normal_calibration, normal_testing = split_evaluation_control(
            dataset[claimed_id]["st_control"]
        )
        normal_training_x = scaler.transform(normal_training[FEATURE_COLS])
        synthetic_x, _, source_rows = generate_condition_augmented_features(
            normal_training_x,
            transforms,
            seed=seed + participant_index,
            repetitions_per_condition=2,
            lower_clip=lower_clip,
            upper_clip=upper_clip,
        )
        positive_x = np.concatenate((normal_training_x, synthetic_x))
        training_x = np.concatenate((positive_x, negative_learning_x))
        training_y = np.concatenate((np.ones(len(positive_x)), np.zeros(len(negative_learning_x))))
        original_groups = np.asarray([f"user:{index}" for index in range(len(normal_training_x))])
        synthetic_groups = original_groups[source_rows]
        negative_groups = (
            "cohort:" + negative_learning.participant_id.astype(str)
            + ":" + negative_learning.block_id.astype(str)
        ).to_numpy()
        groups = np.concatenate((original_groups, synthetic_groups, negative_groups))

        search = GridSearchCV(
            SVC(kernel="rbf", class_weight="balanced"),
            param_grid={"C": PARAM_GRID["svm__C"], "gamma": PARAM_GRID["svm__gamma"]},
            scoring="balanced_accuracy",
            cv=StratifiedGroupKFold(5, shuffle=True, random_state=seed),
            n_jobs=-1,
        )
        search.fit(training_x, training_y, groups=groups)
        classifier = search.best_estimator_
        positive_calibration = classifier.decision_function(
            scaler.transform(normal_calibration[FEATURE_COLS])
        )
        negative_calibration = classifier.decision_function(negative_validation_x)
        threshold, calibration_frr, calibration_far = select_similarity_threshold(
            positive_calibration, negative_calibration, TARGET_FAR
        )
        joblib.dump(classifier, model_dir / f"{claimed_id}_svm.joblib")

        condition_frames = {"st_control": {claimed_id: normal_testing}}
        for condition in SESSIONS:
            condition_frames.setdefault(condition, {})
            for probe_id in fold.evaluation_participants:
                if condition == "st_control":
                    _, _, probe_frame = split_evaluation_control(dataset[probe_id][condition])
                else:
                    probe_frame = dataset[probe_id][condition]
                condition_frames[condition][probe_id] = probe_frame

        for condition in SESSIONS:
            genuine_count = impostor_count = false_rejections = false_acceptances = 0
            for probe_id in fold.evaluation_participants:
                probe_frame = condition_frames[condition][probe_id]
                scores = classifier.decision_function(scaler.transform(probe_frame[FEATURE_COLS]))
                genuine = probe_id == claimed_id
                accepted = scores >= threshold
                if genuine:
                    genuine_count += len(scores)
                    false_rejections += int((~accepted).sum())
                else:
                    impostor_count += len(scores)
                    false_acceptances += int(accepted.sum())
                for row, score, decision in zip(probe_frame.itertuples(), scores, accepted):
                    score_rows.append({
                        "fold": fold_number,
                        "claimed_participant_id": claimed_id,
                        "probe_participant_id": probe_id,
                        "condition": condition,
                        "window_index": row.window_index,
                        "start_sample": row.start_sample,
                        "is_genuine": genuine,
                        "score": float(score),
                        "threshold": threshold,
                        "accepted": bool(decision),
                    })
            metric_rows.append({
                "claimed_participant_id": claimed_id,
                "condition": condition,
                "frr": false_rejections / genuine_count,
                "far": false_acceptances / impostor_count,
                "calibration_frr": calibration_frr,
                "calibration_far": calibration_far,
                "C": classifier.C,
                "gamma": classifier.gamma,
            })

    metrics = pd.DataFrame(metric_rows)
    scores = pd.DataFrame(score_rows)
    summary = metrics.groupby("condition")[["frr", "far"]].agg(["mean", "std"])
    metrics.to_csv(output_dir / "participant_metrics.csv", index=False)
    scores.to_csv(output_dir / "evaluation_scores.csv", index=False)
    summary.to_csv(output_dir / "macro_summary.csv")
    joblib.dump(scaler, model_dir / "development_scaler.joblib")
    print(summary.to_string())
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run statistical condition augmentation")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    run_fold(args.fold)


if __name__ == "__main__":
    main()
