"""Train leakage-resistant, participant-specific sacrum authenticators."""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (  # noqa: E402
    CV_FOLDS, EXCLUDED_PARTICIPANTS, FEATURE_COLS, FEATURES_DIR, MODELS_DIR,
    PARAM_GRID, TARGET_FAR,
)


def load_all_features():
    frames = [pd.read_csv(path) for path in sorted(FEATURES_DIR.glob("*_features.csv"))]
    if not frames:
        raise FileNotFoundError(f"No feature files found in {FEATURES_DIR}")
    data = pd.concat(frames, ignore_index=True)
    data = data[~data["participant_id"].isin(EXCLUDED_PARTICIPANTS)].copy()
    if not np.isfinite(data[FEATURE_COLS].to_numpy()).all():
        raise ValueError("Non-finite feature values found")
    return data


def chronological_split(data):
    """Split every participant into 60/20/20 contiguous portions with guard gaps.

    One window is discarded at each boundary. Because adjacent windows overlap by
    50%, this prevents any raw samples appearing in two different partitions.
    """
    partitions = {"train": [], "validation": [], "test": []}
    for _, participant_data in data.groupby("participant_id", sort=True):
        participant_data = participant_data.sort_values("start_sample").reset_index(drop=True)
        n_rows = len(participant_data)
        train_end = int(0.60 * n_rows)
        validation_end = int(0.80 * n_rows)
        partitions["train"].append(participant_data.iloc[:train_end])
        partitions["validation"].append(participant_data.iloc[train_end + 1:validation_end])
        partitions["test"].append(participant_data.iloc[validation_end + 1:])
    return {name: pd.concat(frames, ignore_index=True) for name, frames in partitions.items()}


def balanced_binary_partition(data, target, seed):
    """Give every impostor equal representation without changing chronology."""
    positive = data[data["participant_id"] == target]
    if positive.empty:
        raise ValueError(f"No positive samples for {target}")
    count = len(positive)
    rng = np.random.default_rng(seed)
    negative_parts = []
    for participant, group in data[data["participant_id"] != target].groupby("participant_id"):
        take = min(count, len(group))
        chosen = np.sort(rng.choice(len(group), size=take, replace=False))
        negative_parts.append(group.iloc[chosen])
    combined = pd.concat([positive, *negative_parts], ignore_index=True)
    X = combined[FEATURE_COLS].to_numpy()
    y = (combined["participant_id"] == target).astype(int).to_numpy()
    groups = (combined["participant_id"].astype(str) + ":" + combined["block_id"].astype(str)).to_numpy()
    return X, y, groups


def threshold_at_target_far(y_true, scores, target_far):
    """Lowest validation threshold whose empirical FAR is within the target."""
    negative_scores = scores[y_true == 0]
    candidates = np.r_[np.inf, np.unique(scores)[::-1], -np.inf]
    valid = []
    for threshold in candidates:
        far = np.mean(negative_scores >= threshold)
        if far <= target_far:
            frr = np.mean(scores[y_true == 1] < threshold)
            valid.append((frr, threshold, far))
    frr, threshold, far = min(valid, key=lambda row: (row[0], row[1]))
    return float(threshold), float(frr), float(far)


def equal_error_rate(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    index = np.argmin(np.abs(fpr - fnr))
    return float((fpr[index] + fnr[index]) / 2), float(thresholds[index])


def train_all_participants(data):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    splits = chronological_split(data)
    participants = sorted(data["participant_id"].unique())
    summary_rows = []

    for participant_index, participant in enumerate(participants):
        X_train, y_train, groups = balanced_binary_partition(
            splits["train"], participant, seed=1000 + participant_index
        )
        X_validation, y_validation, _ = balanced_binary_partition(
            splits["validation"], participant, seed=2000 + participant_index
        )
        X_test, y_test, _ = balanced_binary_partition(
            splits["test"], participant, seed=3000 + participant_index
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced")),
        ])
        cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        search = GridSearchCV(
            # n_jobs=1 is slower but works reliably on macOS and restricted
            # environments where process-level semaphore access is unavailable.
            pipeline, PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=1,
            error_score="raise",
        )
        search.fit(X_train, y_train, groups=groups)

        best_pipeline = search.best_estimator_
        validation_scores = best_pipeline.decision_function(X_validation)
        threshold, validation_frr, validation_far = threshold_at_target_far(
            y_validation, validation_scores, TARGET_FAR
        )
        validation_eer, eer_threshold = equal_error_rate(y_validation, validation_scores)

        scaler = best_pipeline.named_steps["scaler"]
        svm = best_pipeline.named_steps["svm"]
        joblib.dump(scaler, MODELS_DIR / f"{participant}_scaler.pkl")
        joblib.dump(svm, MODELS_DIR / f"{participant}_svm.pkl")
        np.savez(
            MODELS_DIR / f"{participant}_holdout.npz",
            X=scaler.transform(X_test), y=y_test,
        )
        with open(MODELS_DIR / f"{participant}_threshold.json", "w") as handle:
            json.dump({
                "threshold": threshold,
                "target_far": TARGET_FAR,
                "validation_frr": validation_frr,
                "validation_far": validation_far,
                "validation_eer": validation_eer,
                "eer_threshold": eer_threshold,
            }, handle, indent=2)

        summary_rows.append({
            "participant_id": participant,
            "n_train_positive": int(y_train.sum()),
            "n_train_negative": int((y_train == 0).sum()),
            "n_validation_positive": int(y_validation.sum()),
            "n_validation_negative": int((y_validation == 0).sum()),
            "n_test_positive": int(y_test.sum()),
            "n_test_negative": int((y_test == 0).sum()),
            "best_C": search.best_params_["svm__C"],
            "best_gamma": search.best_params_["svm__gamma"],
            "best_cv_auc": search.best_score_,
            "threshold": threshold,
            "validation_frr": validation_frr,
            "validation_far": validation_far,
            "validation_eer": validation_eer,
        })
        print(
            f"{participant}: CV AUC={search.best_score_:.3f}, "
            f"validation EER={validation_eer:.3f}, "
            f"FRR@FAR<={TARGET_FAR:.1%}={validation_frr:.3f}"
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(MODELS_DIR / "training_summary.csv", index=False)
    return summary


def main():
    data = load_all_features()
    print(f"Training on {len(data)} ST-control windows from {data.participant_id.nunique()} participants")
    summary = train_all_participants(data)
    print(f"Training summary saved to {MODELS_DIR / 'training_summary.csv'}")
    print(summary[["participant_id", "best_cv_auc", "validation_eer", "validation_frr"]].to_string(index=False))


if __name__ == "__main__":
    main()
