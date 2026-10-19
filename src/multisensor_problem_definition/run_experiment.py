"""Run the original statistical-SVM protocol using all nine IMU locations.

This is a diagnostic/problem-definition baseline, not the robust stride
verifier.  It deliberately retains the original methodology:

* fixed two-second windows with 50% overlap;
* time-domain statistics from raw XYZ axes;
* one RBF-SVM verifier and one ST-control threshold per participant;
* ST-control-only training and calibration;
* frozen evaluation on ST-fatigue, DT-control, and DT-fatigue.

Both per-window decisions and causal fusion of consecutive window scores are
reported.  The fused operating threshold is calibrated from fused validation
streams, so its stated FAR refers to the same decision unit used at test time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.robust_gait.data import CONDITIONS, LOCATIONS, SRC_DIR, _raw
from src.utils import BLOCK_SAMPLES, PARAM_GRID, SENSOR_COLS, TARGET_FAR

SAMPLE_RATE = 128
WINDOW_SAMPLES = 256
WINDOW_STEP = 128
STATISTICS = (
    "mean", "std", "median", "min", "max", "range", "iqr", "mad",
    "rms", "skewness", "kurtosis",
)
FEATURE_COUNT = len(LOCATIONS) * len(SENSOR_COLS) * len(STATISTICS)
CACHE_DIR = SRC_DIR / "data" / "processed" / "features" / "multisensor_9imu_raw_axes"
RESULTS_DIR = SRC_DIR / "results" / "multisensor_problem_definition"
MODELS_DIR = SRC_DIR / "models" / "multisensor_problem_definition"


@dataclass(frozen=True)
class Config:
    target_far: float = TARGET_FAR
    cv_folds: int = 5
    fusion_windows: int = 10
    seed: int = 42


def _feature_names() -> tuple[str, ...]:
    return tuple(
        f"{location}_{axis}_{statistic}"
        for location in LOCATIONS
        for axis in SENSOR_COLS
        for statistic in STATISTICS
    )


FEATURE_NAMES = _feature_names()


def _synchronized_axes(participant: str, condition: str) -> np.ndarray:
    """Load the same nine recordings used by the robust system.

    DUO-GAIT recordings are synchronized and sampled at the same rate.  The
    shortest stream defines the shared usable interval, matching robust_gait.
    """

    streams = [_raw(participant, condition, code) for code in LOCATIONS.values()]
    length = min(map(len, streams))
    return np.concatenate([stream[:length] for stream in streams], axis=1)


def _window_view(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, len(signal) - WINDOW_SAMPLES + 1, WINDOW_STEP)
    if not len(starts):
        raise ValueError("Recording is too short for one two-second window")
    return np.stack([signal[s:s + WINDOW_SAMPLES] for s in starts]), starts


def statistical_features(windows: np.ndarray) -> np.ndarray:
    """Vectorized eleven-statistic representation of `(n, 256, 54)` windows."""

    mean = windows.mean(axis=1)
    standard_deviation = windows.std(axis=1)
    q25, median, q75 = np.percentile(windows, [25, 50, 75], axis=1)
    centered = windows - mean[:, None, :]
    safe = np.where(standard_deviation > 1e-12, standard_deviation, 1.0)
    standardized = centered / safe[:, None, :]
    skewness = np.mean(standardized ** 3, axis=1)
    kurtosis = np.mean(standardized ** 4, axis=1) - 3.0
    skewness[standard_deviation <= 1e-12] = 0.0
    kurtosis[standard_deviation <= 1e-12] = 0.0
    values = np.stack((
        mean,
        standard_deviation,
        median,
        windows.min(axis=1),
        windows.max(axis=1),
        np.ptp(windows, axis=1),
        q75 - q25,
        np.median(np.abs(windows - median[:, None, :]), axis=1),
        np.sqrt(np.mean(np.square(windows), axis=1)),
        skewness,
        kurtosis,
    ), axis=2)
    result = values.reshape(len(windows), -1).astype(np.float32)
    if result.shape[1] != FEATURE_COUNT or not np.isfinite(result).all():
        raise ValueError("Invalid nine-sensor statistical feature matrix")
    return result


def load_features(participant: str, condition: str) -> pd.DataFrame:
    """Load or create a compact binary feature cache for one recording."""

    path = CACHE_DIR / condition / f"{participant}.npz"
    if path.exists():
        saved = np.load(path)
        features, starts = saved["features"], saved["start_sample"]
    else:
        windows, starts = _window_view(_synchronized_axes(participant, condition))
        features = statistical_features(windows)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=features, start_sample=starts)
    frame = pd.DataFrame(features, columns=FEATURE_NAMES)
    frame["participant_id"] = participant
    frame["condition"] = condition
    frame["start_sample"] = starts
    frame["block_id"] = starts // BLOCK_SAMPLES
    return frame


def participants() -> tuple[str, ...]:
    root = SRC_DIR.parent / "raw_data" / "interim" / "OG_st_control"
    return tuple(sorted(path.name for path in root.glob("sub_*") if path.is_dir()))


def chronological_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Original 60/20/20 split with one overlapping window removed per boundary."""

    ordered = frame.sort_values("start_sample").reset_index(drop=True)
    first, second = int(0.60 * len(ordered)), int(0.80 * len(ordered))
    return {
        "train": ordered.iloc[:first].copy(),
        "validation": ordered.iloc[first + 1:second].copy(),
        "test": ordered.iloc[second + 1:].copy(),
    }


def _purge_cross_block_windows(frame: pd.DataFrame) -> pd.DataFrame:
    within = frame.start_sample.to_numpy() % BLOCK_SAMPLES
    return frame.loc[within + WINDOW_SAMPLES <= BLOCK_SAMPLES].copy()


def _training_partition(
    split_by_person: dict[str, dict[str, pd.DataFrame]], target: str, rng,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = _purge_cross_block_windows(split_by_person[target]["train"])
    parts = [positive]
    for person, splits in split_by_person.items():
        if person == target:
            continue
        candidate = _purge_cross_block_windows(splits["train"])
        take = min(len(positive), len(candidate))
        indices = np.sort(rng.choice(len(candidate), take, replace=False))
        parts.append(candidate.iloc[indices])
    combined = pd.concat(parts, ignore_index=True)
    labels = (combined.participant_id == target).astype(int).to_numpy()
    groups = (
        combined.participant_id.astype(str) + ":" + combined.block_id.astype(str)
    ).to_numpy()
    return combined[list(FEATURE_NAMES)].to_numpy(), labels, groups


def _streams(
    model, scaler, frames: dict[str, pd.DataFrame], target: str,
) -> dict[str, np.ndarray]:
    return {
        person: model.decision_function(
            scaler.transform(frame[list(FEATURE_NAMES)].to_numpy())
        )
        for person, frame in frames.items()
    }


def fuse(scores: np.ndarray, count: int) -> np.ndarray:
    if count <= 1:
        return scores
    if len(scores) < count:
        return np.empty(0, dtype=np.float64)
    cumulative = np.cumsum(np.insert(scores, 0, 0.0))
    return (cumulative[count:] - cumulative[:-count]) / count


def _threshold(streams, target, target_far, fusion_windows):
    genuine = fuse(streams[target], fusion_windows)
    impostor = np.concatenate([
        fuse(scores, fusion_windows)
        for person, scores in streams.items() if person != target
    ])
    candidates = np.r_[np.inf, np.unique(np.concatenate((genuine, impostor)))[::-1], -np.inf]
    valid = []
    for threshold in candidates:
        far = np.mean(impostor >= threshold)
        if far <= target_far:
            valid.append((np.mean(genuine < threshold), threshold, far))
    return min(valid, key=lambda row: (row[0], row[1]))


def _eer(genuine, impostor):
    labels = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    scores = np.r_[genuine, impostor]
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


def _rates(streams, target, threshold, fusion_windows):
    genuine = fuse(streams[target], fusion_windows)
    impostor = np.concatenate([
        fuse(scores, fusion_windows)
        for person, scores in streams.items() if person != target
    ])
    return (
        float(np.mean(genuine < threshold)),
        float(np.mean(impostor >= threshold)),
        _eer(genuine, impostor),
        len(genuine),
        len(impostor),
    )


def run(config: Config = Config()) -> pd.DataFrame:
    people = participants()
    data = {
        (person, condition): load_features(person, condition)
        for person in people for condition in CONDITIONS
    }
    splits = {person: chronological_split(data[person, "st_control"]) for person in people}
    rows = []
    output = RESULTS_DIR / f"raw_axes_9imu_targetfar{config.target_far:g}"
    model_output = MODELS_DIR / f"raw_axes_9imu_targetfar{config.target_far:g}"
    output.mkdir(parents=True, exist_ok=True)
    model_output.mkdir(parents=True, exist_ok=True)

    for target_index, target in enumerate(people):
        model_path = model_output / f"{target}.joblib"
        if model_path.exists():
            best_pipeline = joblib.load(model_path)
            cv_message = "reused cached model"
        else:
            x, y, groups = _training_partition(
                splits, target, np.random.default_rng(config.seed + target_index)
            )
            pipeline = Pipeline((
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", class_weight="balanced")),
            ))
            search = GridSearchCV(
                pipeline, PARAM_GRID, scoring="roc_auc",
                cv=StratifiedGroupKFold(
                    n_splits=config.cv_folds, shuffle=True, random_state=config.seed
                ),
                n_jobs=1, error_score="raise",
            ).fit(x, y, groups=groups)
            best_pipeline = search.best_estimator_
            cv_message = f"CV AUC={search.best_score_:.3f}"
            joblib.dump(best_pipeline, model_path)
        model = best_pipeline.named_steps["svm"]
        scaler = best_pipeline.named_steps["scaler"]
        validation = _streams(
            model, scaler,
            {person: splits[person]["validation"] for person in people}, target,
        )
        thresholds = {}
        for decision, fusion_windows in (("2s", 1), ("11s", config.fusion_windows)):
            validation_frr, threshold, validation_far = _threshold(
                validation, target, config.target_far, fusion_windows
            )
            thresholds[decision] = threshold
            for condition in CONDITIONS:
                frames = (
                    {person: splits[person]["test"] for person in people}
                    if condition == "st_control"
                    else {person: data[person, condition] for person in people}
                )
                rates = _rates(
                    _streams(model, scaler, frames, target), target,
                    threshold, fusion_windows,
                )
                rows.append({
                    "participant_id": target,
                    "decision": decision,
                    "condition": condition,
                    "frr": rates[0], "far": rates[1], "eer": rates[2],
                    "threshold": threshold,
                    "validation_frr": validation_frr,
                    "validation_far": validation_far,
                    "genuine_count": rates[3], "impostor_count": rates[4],
                })
        (model_output / f"{target}_thresholds.json").write_text(
            json.dumps(thresholds, indent=2)
        )
        print(
            f"{target}: {cv_message}, "
            f"2s threshold={thresholds['2s']:.4f}, "
            f"11s threshold={thresholds['11s']:.4f}", flush=True,
        )

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(["decision", "condition"])[["frr", "far", "eer"]].agg(
        ["mean", "std"]
    )
    metrics.to_csv(output / "participant_metrics.csv", index=False)
    summary.to_csv(output / "macro_summary.csv")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    print("\nMacro-average across participants")
    for (decision, condition), row in summary.iterrows():
        print(
            f"{decision:3s} {condition:12s} "
            f"FRR={row[('frr', 'mean')]:.2%} (SD {row[('frr', 'std')]:.2%})  "
            f"FAR={row[('far', 'mean')]:.2%} (SD {row[('far', 'std')]:.2%})  "
            f"EER={row[('eer', 'mean')]:.2%} (SD {row[('eer', 'std')]:.2%})"
        )
    print(f"Results saved under {output}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target-far", type=float, default=TARGET_FAR)
    parser.add_argument("--fusion-windows", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    run(Config(
        target_far=args.target_far,
        fusion_windows=args.fusion_windows,
        seed=args.seed,
        cv_folds=args.cv_folds,
    ))


if __name__ == "__main__":
    main()
