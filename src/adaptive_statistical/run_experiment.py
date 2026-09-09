"""Run enrollment-centered cosine verification with causal EMA adaptation."""

import argparse
from dataclasses import asdict, replace
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.adaptive_statistical.config import (
    CONDITIONS, FEATURES, MODEL_DIR, RESULT_DIR, STATISTICS, Config,
)
from src.fisher_metric.data import participant_folds
from src.utils import WINDOW_DIR


def load_windows():
    dataset = {}
    for condition in CONDITIONS:
        for path in sorted((WINDOW_DIR / condition).glob("*_windows.npy")):
            participant = path.stem.removesuffix("_windows")
            values = np.load(path).astype(np.float32)
            if values.shape[1:] != (256, 6) or not np.isfinite(values).all():
                raise ValueError(f"Invalid windows in {path}")
            dataset.setdefault(participant, {})[condition] = values
    if not dataset or any(set(row) != set(CONDITIONS) for row in dataset.values()):
        raise ValueError("Every participant must contain all four conditions")
    return dataset


def enrollment_parts(windows):
    """Chronological 60/20/20 template/calibration/test split with guards."""
    first = int(len(windows) * 0.60)
    second = int(len(windows) * 0.80)
    template = windows[:first]
    calibration = windows[first + 1:second]
    test = windows[second + 1:]
    if not len(template) or not len(calibration) or not len(test):
        raise ValueError("ST-control recording is too short")
    return template, calibration, test


def raw_channel_center(windows):
    """Six offsets learned from template-enrollment samples only."""
    return np.asarray(windows, dtype=np.float64).mean(axis=(0, 1))


def statistical_features(windows, center):
    """Extract 88 features after applying the claimed user's raw-channel center."""
    centered = np.asarray(windows, dtype=np.float64) - np.asarray(center)[None, None, :]
    signals = [centered[..., index] for index in range(6)]
    signals.extend((
        np.linalg.norm(centered[..., 3:6], axis=2),
        np.linalg.norm(centered[..., 0:3], axis=2),
    ))
    columns = []
    for signal in signals:
        mean = signal.mean(1)
        std = signal.std(1)
        median = np.median(signal, axis=1)
        minimum = signal.min(1)
        maximum = signal.max(1)
        q25, q75 = np.percentile(signal, [25, 75], axis=1)
        safe_std = np.where(std > 1e-12, std, 1.0)
        standardized = (signal - mean[:, None]) / safe_std[:, None]
        values = {
            "mean": mean, "std": std, "median": median, "min": minimum,
            "max": maximum, "range": maximum - minimum, "iqr": q75 - q25,
            "mad": np.median(np.abs(signal - median[:, None]), axis=1),
            "rms": np.sqrt(np.mean(np.square(signal), axis=1)),
            "skewness": np.where(std > 1e-12, np.mean(standardized ** 3, axis=1), 0),
            "kurtosis": np.where(
                std > 1e-12, np.mean(standardized ** 4, axis=1) - 3, 0
            ),
        }
        columns.extend(values[name] for name in STATISTICS)
    return np.column_stack(columns)


def fit_development_scaler(dataset, participants, maximum, seed):
    """Fit one feature scaler without using evaluation participants."""
    rng = np.random.default_rng(seed)
    rows = []
    for participant in participants:
        template, _, _ = enrollment_parts(dataset[participant]["st_control"])
        center = raw_channel_center(template)
        for condition in CONDITIONS:
            windows = dataset[participant][condition]
            take = min(maximum, len(windows))
            indices = np.sort(rng.choice(len(windows), take, replace=False))
            rows.append(statistical_features(windows[indices], center))
    return StandardScaler().fit(np.vstack(rows))


def embeddings(windows, center, scaler):
    values = scaler.transform(statistical_features(windows, center))
    norms = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)
    return values / norms


def make_template(values):
    values = np.asarray(values)
    result = values if values.ndim == 1 else values.mean(axis=0)
    return result / max(np.linalg.norm(result), 1e-9)


def cosine_scores(values, template):
    return np.asarray(values) @ np.asarray(template)


def causal_fusion(values, size):
    values = np.asarray(values, dtype=np.float64)
    cumulative = np.r_[0.0, np.cumsum(values)]
    positions = np.arange(len(values))
    starts = np.maximum(0, positions - size + 1)
    return (cumulative[positions + 1] - cumulative[starts]) / (positions - starts + 1)


def threshold_at_target_far(genuine, impostor, target_far):
    candidates = np.r_[-np.inf, np.unique(np.concatenate((genuine, impostor))), np.inf]
    valid = []
    for threshold in candidates:
        far = float((impostor >= threshold).mean())
        if far <= target_far + 1e-12:
            valid.append((float((genuine < threshold).mean()), far, threshold))
    frr, far, threshold = min(valid)
    return float(threshold), frr, far


def calibration_impostors(dataset, development, center, scaler, maximum, seed):
    rng = np.random.default_rng(seed)
    scores = []
    for participant in development:
        windows = dataset[participant]["st_control"]
        take = min(maximum, len(windows))
        indices = np.sort(rng.choice(len(windows), take, replace=False))
        scores.append(embeddings(windows[indices], center, scaler))
    return np.vstack(scores)


def adaptive_genuine_stream(values, initial_template, threshold, alpha, fusion_window):
    """Score and causally update after accepted windows only."""
    current = initial_template.copy()
    history, raw_scores, decisions = [], [], []
    for value in values:
        raw = float(value @ current)
        raw_scores.append(raw)
        fused = float(np.mean(raw_scores[-fusion_window:]))
        accepted = fused >= threshold
        decisions.append(accepted)
        history.append(fused)
        if accepted:
            current = make_template((1 - alpha) * current + alpha * value)
    return np.asarray(history), np.asarray(decisions), current


def run_fold(fold_number=1, config=Config()):
    dataset = load_windows()
    development, evaluation = participant_folds(dataset, 4)[fold_number - 1]
    scaler = fit_development_scaler(
        dataset, development, config.maximum_cohort_windows, config.seed + fold_number
    )
    metric_rows, score_rows, user_models = [], [], {}
    for user_index, claim in enumerate(evaluation):
        template_windows, calibration_windows, test_windows = enrollment_parts(
            dataset[claim]["st_control"]
        )
        center = raw_channel_center(template_windows)
        initial_template = make_template(embeddings(template_windows, center, scaler))
        genuine_calibration = causal_fusion(
            cosine_scores(embeddings(calibration_windows, center, scaler), initial_template),
            config.fusion_window,
        )
        impostor_vectors = calibration_impostors(
            dataset, development, center, scaler, config.maximum_cohort_windows,
            config.seed + fold_number * 100 + user_index,
        )
        impostor_calibration = causal_fusion(
            cosine_scores(impostor_vectors, initial_template), config.fusion_window
        )
        threshold, calibration_frr, calibration_far = threshold_at_target_far(
            genuine_calibration, impostor_calibration, config.target_far
        )
        current_template = initial_template.copy()
        condition_windows = {"st_control": test_windows}
        condition_windows.update({
            condition: dataset[claim][condition] for condition in CONDITIONS[1:]
        })
        for condition in CONDITIONS:
            condition_start_template = current_template.copy()
            genuine_vectors = embeddings(condition_windows[condition], center, scaler)
            genuine_scores, decisions, current_template = adaptive_genuine_stream(
                genuine_vectors, current_template, threshold, config.ema_alpha,
                config.fusion_window,
            )
            impostor_scores = []
            for probe in evaluation:
                if probe == claim:
                    continue
                probe_vectors = embeddings(dataset[probe][condition], center, scaler)
                impostor_scores.extend(causal_fusion(
                    cosine_scores(probe_vectors, condition_start_template),
                    config.fusion_window,
                ))
            impostor_scores = np.asarray(impostor_scores)
            metric_rows.append({
                "fold": fold_number, "participant": claim, "condition": condition,
                "threshold": threshold, "ema_alpha": config.ema_alpha,
                "frr": float((~decisions).mean()),
                "far": float((impostor_scores >= threshold).mean()),
                "calibration_frr": calibration_frr,
                "calibration_far": calibration_far,
            })
            score_rows.extend({
                "fold": fold_number, "claim": claim, "condition": condition,
                "is_genuine": True, "score_index": index, "similarity": float(value),
                "accepted": bool(decisions[index]),
            } for index, value in enumerate(genuine_scores))
        user_models[claim] = {
            "center": center, "initial_template": initial_template,
            "threshold": threshold,
        }
    metrics = pd.DataFrame(metric_rows)
    output = RESULT_DIR / f"fold_{fold_number}"
    models = MODEL_DIR / f"fold_{fold_number}"
    output.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "participant_metrics.csv", index=False)
    pd.DataFrame(score_rows).to_csv(output / "genuine_scores.csv", index=False)
    summary = metrics.groupby("condition")[["frr", "far"]].agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")
    joblib.dump({"scaler": scaler, "users": user_models}, models / "model.joblib")
    (output / "protocol.json").write_text(json.dumps({
        "development": development, "evaluation": evaluation, "config": asdict(config),
        "threshold_policy": "per-user ST-control calibration at target FAR",
        "update_policy": "causal accepted-window EMA",
    }, indent=2))
    print(f"Fold {fold_number}/4: evaluation={','.join(evaluation)}")
    print(summary.to_string())
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--ema-alpha", type=float, default=0.02)
    parser.add_argument("--fusion-window", type=int, default=5)
    parser.add_argument("--target-far", type=float, default=0.01)
    args = parser.parse_args()
    if not 0 < args.ema_alpha <= 1:
        parser.error("--ema-alpha must be in (0, 1]")
    run_fold(args.fold, replace(
        Config(), ema_alpha=args.ema_alpha, fusion_window=args.fusion_window,
        target_far=args.target_far,
    ))


if __name__ == "__main__":
    main()
