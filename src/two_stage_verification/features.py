"""Complementary spectral and periodicity features for the second verifier."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import FEATURE_SIGNALS, ORIGINAL_RATE, SENSOR_COLS, WINDOW_DIR


SPECTRAL_STATS = (
    "dominant_frequency",
    "spectral_centroid",
    "spectral_entropy",
    "walking_band_ratio",
    "high_band_ratio",
    "autocorrelation_peak",
)
SPECTRAL_COLS = [
    f"spectral_{signal}_{stat}"
    for signal in FEATURE_SIGNALS
    for stat in SPECTRAL_STATS
]


def spectral_features(signal: np.ndarray, sampling_rate: int = ORIGINAL_RATE):
    """Describe cadence, spectral spread, and repetition in one signal."""

    signal = np.asarray(signal, dtype=np.float64)
    centered = signal - signal.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(centered), d=1 / sampling_rate)
    power[0] = 0.0
    total = power.sum()
    if total <= 1e-12:
        return {name: 0.0 for name in SPECTRAL_STATS}
    probability = power / total
    positive = probability > 0
    walking = (frequencies >= 0.5) & (frequencies <= 3.0)
    high = (frequencies > 3.0) & (frequencies <= 10.0)

    autocorrelation = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    zero_lag = autocorrelation[0]
    minimum_lag = max(1, round(sampling_rate / 3.0))
    maximum_lag = min(len(centered), round(sampling_rate / 0.7))
    periodicity = 0.0
    if zero_lag > 1e-12 and maximum_lag > minimum_lag:
        periodicity = float(
            np.max(autocorrelation[minimum_lag:maximum_lag]) / zero_lag
        )
    return {
        "dominant_frequency": float(frequencies[np.argmax(power)]),
        "spectral_centroid": float(np.sum(frequencies * probability)),
        "spectral_entropy": float(-np.sum(probability[positive] * np.log(probability[positive]))),
        "walking_band_ratio": float(probability[walking].sum()),
        "high_band_ratio": float(probability[high].sum()),
        "autocorrelation_peak": periodicity,
    }


def extract_spectral_window(window: np.ndarray) -> dict[str, float]:
    """Extract complementary features from all six axes and two magnitudes."""

    signals = {name: window[:, index] for index, name in enumerate(SENSOR_COLS)}
    signals["AccMag"] = np.linalg.norm(window[:, 3:6], axis=1)
    signals["GyrMag"] = np.linalg.norm(window[:, 0:3], axis=1)
    result = {}
    for signal_name, signal in signals.items():
        for statistic, value in spectral_features(signal).items():
            result[f"spectral_{signal_name}_{statistic}"] = value
    return result


def load_spectral_dataset(sessions) -> dict[str, dict[str, pd.DataFrame]]:
    """Load aligned window arrays and produce second-stage feature frames."""

    dataset = {}
    for condition in sessions:
        directory = WINDOW_DIR / condition
        for path in sorted(directory.glob("*_windows.npy")):
            participant_id = path.stem.replace("_windows", "")
            windows = np.load(path)
            labels = pd.read_csv(directory / f"{participant_id}_labels.csv")
            if len(windows) != len(labels):
                raise ValueError(f"Window/label mismatch for {participant_id} {condition}")
            features = pd.DataFrame(
                [extract_spectral_window(window) for window in windows],
                columns=SPECTRAL_COLS,
            )
            frame = pd.concat((features, labels.reset_index(drop=True)), axis=1)
            if not np.isfinite(frame[SPECTRAL_COLS].to_numpy()).all():
                raise ValueError(f"Non-finite spectral features for {participant_id} {condition}")
            dataset.setdefault(participant_id, {})[condition] = frame
    return dataset
