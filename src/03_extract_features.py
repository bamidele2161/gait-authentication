"""Extract time-domain statistical features from IMU windows.

Sacrum: six axes plus acceleration and gyroscope magnitude, 11 statistics each
(88 features). Multi-sensor sets: the six raw axes of every sensor, 11
statistics each (66 features per sensor).  Statistics are computed for all
windows of a recording at once.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (  # noqa: E402
    FEATURE_COLS, FEATURE_DIR, FEATURE_SIGNALS, SENSOR_COLS, SESSIONS,
    SIGNAL_PREFIXES, WINDOW_DIR,
)


def statistical_features(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    centered = signal - mean
    safe_std = std if std > 1e-12 else 1.0
    q25, median, q75 = np.percentile(signal, [25, 50, 75])
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "min": np.min(signal),
        "max": np.max(signal),
        "range": np.ptp(signal),
        "iqr": q75 - q25,
        "mad": np.median(np.abs(signal - median)),
        "rms": np.sqrt(np.mean(np.square(signal))),
        "skewness": np.mean((centered / safe_std) ** 3) if std > 1e-12 else 0.0,
        "kurtosis": np.mean((centered / safe_std) ** 4) - 3 if std > 1e-12 else 0.0,
    }


def extract_features_window(window):
    features = {}
    width = len(SENSOR_COLS)
    expected_channels = len(SIGNAL_PREFIXES) * width
    if window.ndim != 2 or window.shape[1] != expected_channels:
        raise ValueError(
            f"Expected window with {expected_channels} channels, got {window.shape}"
        )
    for sensor, prefix in enumerate(SIGNAL_PREFIXES):
        block = window[:, sensor * width:(sensor + 1) * width]
        signals = {name: block[:, index] for index, name in enumerate(SENSOR_COLS)}
        if "AccMag" in FEATURE_SIGNALS:
            signals["AccMag"] = np.linalg.norm(block[:, 3:6], axis=1)
        if "GyrMag" in FEATURE_SIGNALS:
            signals["GyrMag"] = np.linalg.norm(block[:, 0:3], axis=1)
        for name, signal in signals.items():
            for stat, value in statistical_features(signal).items():
                features[f"{prefix}{name}_{stat}"] = value
    return features


def statistical_features_batch(signals):
    """Same statistics as ``statistical_features`` for (windows, samples) arrays."""
    mean = signals.mean(axis=1)
    std = signals.std(axis=1)
    ok = std > 1e-12
    safe_std = np.where(ok, std, 1.0)
    z = (signals - mean[:, None]) / safe_std[:, None]
    q25, median, q75 = np.percentile(signals, [25, 50, 75], axis=1)
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "min": signals.min(axis=1),
        "max": signals.max(axis=1),
        "range": np.ptp(signals, axis=1),
        "iqr": q75 - q25,
        "mad": np.median(np.abs(signals - median[:, None]), axis=1),
        "rms": np.sqrt(np.mean(np.square(signals), axis=1)),
        "skewness": np.where(ok, np.mean(z ** 3, axis=1), 0.0),
        "kurtosis": np.where(ok, np.mean(z ** 4, axis=1) - 3, 0.0),
    }


def extract_features_batch(windows):
    """(windows, samples, channels) -> DataFrame with FEATURE_COLS."""
    columns = {}
    width = len(SENSOR_COLS)
    expected_channels = len(SIGNAL_PREFIXES) * width
    if windows.ndim != 3 or windows.shape[2] != expected_channels:
        raise ValueError(
            f"Expected windows with {expected_channels} channels, got {windows.shape}"
        )
    for sensor, prefix in enumerate(SIGNAL_PREFIXES):
        block = windows[:, :, sensor * width:(sensor + 1) * width].astype(np.float64)
        signals = {name: block[:, :, index] for index, name in enumerate(SENSOR_COLS)}
        if "AccMag" in FEATURE_SIGNALS:
            signals["AccMag"] = np.linalg.norm(block[:, :, 3:6], axis=2)
        if "GyrMag" in FEATURE_SIGNALS:
            signals["GyrMag"] = np.linalg.norm(block[:, :, 0:3], axis=2)
        for name in FEATURE_SIGNALS:
            for stat, values in statistical_features_batch(signals[name]).items():
                columns[f"{prefix}{name}_{stat}"] = values
    features = pd.DataFrame(columns, columns=FEATURE_COLS)
    if features.shape[1] != len(FEATURE_COLS):
        raise ValueError(
            f"Expected {len(FEATURE_COLS)} features, found {features.shape[1]}"
        )
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Extracted feature matrix contains NaN or infinite values")
    return features


def process_session(session_name):
    windows_dir = WINDOW_DIR / session_name
    output_dir = FEATURE_DIR / session_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for npy_path in sorted(windows_dir.glob("*_windows.npy")):
        participant_id = npy_path.stem.replace("_windows", "")
        labels_path = windows_dir / f"{participant_id}_labels.csv"
        windows = np.load(npy_path)
        labels = pd.read_csv(labels_path)
        if len(windows) != len(labels):
            raise ValueError(f"Window/label mismatch for {participant_id}")

        features = extract_features_batch(windows)
        features = pd.concat([features, labels.reset_index(drop=True)], axis=1)
        features.to_csv(output_dir / f"{participant_id}_features.csv", index=False)
        print(f"{session_name:12s} {participant_id}: {len(features)} feature rows")


def main():
    print(f"Extracting {len(FEATURE_COLS)} time-domain features")
    for session in SESSIONS:
        process_session(session)
    print(f"Features saved under {FEATURE_DIR}")


if __name__ == "__main__":
    main()
