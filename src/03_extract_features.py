"""Extract time-domain statistical features from sacrum IMU windows."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (  # noqa: E402
    FEATURE_COLS, FEATURE_DIR, SENSOR_COLS, SESSIONS, WINDOW_DIR,
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
    signals = {name: window[:, index] for index, name in enumerate(SENSOR_COLS)}
    signals["AccMag"] = np.linalg.norm(window[:, 3:6], axis=1)
    signals["GyrMag"] = np.linalg.norm(window[:, 0:3], axis=1)

    features = {}
    for name, signal in signals.items():
        for stat, value in statistical_features(signal).items():
            features[f"{name}_{stat}"] = value
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

        feature_rows = [extract_features_window(window) for window in windows]
        features = pd.DataFrame(feature_rows, columns=FEATURE_COLS)
        features = pd.concat([features, labels.reset_index(drop=True)], axis=1)
        features.to_csv(output_dir / f"{participant_id}_features.csv", index=False)
        print(f"{session_name:12s} {participant_id}: {len(features)} feature rows")


def main():
    print(f"Extracting {len(FEATURE_COLS)} sacrum time-domain features")
    for session in SESSIONS:
        process_session(session)
    print(f"Features saved under {FEATURE_DIR}")


if __name__ == "__main__":
    main()
