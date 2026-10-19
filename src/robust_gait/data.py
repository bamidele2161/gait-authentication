"""Multi-location stride cycles for DUO-GAIT.

All nine IMUs (sacrum, shanks, wrists, feet, head, sternum) are read from the
synchronized raw recordings in ``raw_data/interim`` with the project's known
recording trims (``src.utils.KNOWN_TIME_TRIMS``); for the sacrum this
reproduces the saved ``sacrum_time`` windows sample for sample.  Every location
is cut at the same stride boundaries.

Stride boundaries come from the left-shank gyroscope magnitude: its mid-swing
peak occurs exactly once per stride, so every cycle starts at the same gait
phase for every sensor (sacrum minima alternate between left and right steps).
Each stride is resampled to a fixed length, which removes cadence differences
between walking speeds while keeping the shape of the stride.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

SRC_DIR = Path(__file__).resolve().parents[1]
RAW_ROOT = SRC_DIR.parent / "raw_data" / "interim"
WINDOWS_ROOT = SRC_DIR / "data" / "processed" / "windows"
CACHE_DIR = SRC_DIR / "data" / "processed" / "strides" / "robust_gait_9imu_v2"
# Name -> raw file code.  The order fixes the channel layout of the stride
# arrays: channels 0..17 are (AccMagnitude, GyrMagnitude) per location, and
# channels 18 + 6*location + (0..5) are that location's raw
# GyrX, GyrY, GyrZ, AccX, AccY, AccZ.
LOCATIONS = {
    "sacrum": "SA",
    "left_leg": "LL",
    "right_leg": "RL",
    "left_wrist": "LW",
    "right_wrist": "RW",
    "left_foot": "LF",
    "right_foot": "RF",
    "head": "HE",
    "sternum": "ST",
}
SENSOR_COLUMNS = ["GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]
CONDITIONS = ("st_control", "st_fatigue", "dt_control", "dt_fatigue")
SAMPLE_RATE = 128
STEP = 128
STRIDE_SAMPLES = 128
MIN_STRIDE, MAX_STRIDE = 0.7, 1.8  # seconds


@dataclass(frozen=True)
class StrideRecording:
    """Strides of one participant/condition: (n, STRIDE_SAMPLES, locations*2)."""

    strides: np.ndarray
    start: np.ndarray
    end: np.ndarray
    duration: np.ndarray
    locations: tuple[str, ...]


def _raw(participant: str, condition: str, code: str) -> np.ndarray:
    import pandas as pd

    from src.utils import KNOWN_TIME_TRIMS

    frame = pd.read_csv(RAW_ROOT / f"OG_{condition}" / participant / f"{code}.csv")
    trim = KNOWN_TIME_TRIMS.get((condition, participant))
    if trim is not None:
        timestamps = pd.to_numeric(frame["timestamp"])
        frame = frame[timestamps.between(trim["start"], trim["end"], inclusive="both")]
        if len(frame) != trim["expected_rows"]:
            raise ValueError(f"Unexpected trimmed length for {participant} {condition}")
    signal = frame[SENSOR_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy()
    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite samples in {participant} {condition} {code}")
    return signal


def _continuous(path: Path) -> np.ndarray:
    windows = np.load(path)
    return np.concatenate(
        (windows[:-1, :STEP].reshape(-1, windows.shape[2]), windows[-1])
    ).astype(np.float64)


def _magnitudes(signal: np.ndarray) -> np.ndarray:
    """(n, 6) Gyr XYZ + Acc XYZ -> (n, 2) AccMagnitude, GyrMagnitude."""

    return np.column_stack((
        np.linalg.norm(signal[:, 3:6], axis=1), np.linalg.norm(signal[:, 0:3], axis=1)
    ))


def stride_boundaries(left_shank_gyro_magnitude: np.ndarray) -> np.ndarray:
    sos = butter(4, 6.0, btype="low", fs=SAMPLE_RATE, output="sos")
    smooth = sosfiltfilt(sos, left_shank_gyro_magnitude)
    peaks, _ = find_peaks(
        smooth, distance=int(0.6 * SAMPLE_RATE),
        prominence=max(0.5 * np.std(smooth), 1e-6),
    )
    return peaks


def _resample(segment: np.ndarray, samples: int) -> np.ndarray:
    old = np.linspace(0.0, 1.0, len(segment))
    new = np.linspace(0.0, 1.0, samples)
    return np.column_stack([np.interp(new, old, segment[:, c]) for c in range(segment.shape[1])])


def load_strides(
    participant: str, condition: str, locations=tuple(LOCATIONS),
) -> StrideRecording:
    cache = CACHE_DIR / f"{participant}_{condition}.npz"
    if locations == tuple(LOCATIONS) and cache.exists():
        stored = np.load(cache)
        return StrideRecording(
            stored["strides"], stored["start"], stored["end"], stored["duration"],
            tuple(LOCATIONS),
        )
    axes = {name: _raw(participant, condition, code) for name, code in LOCATIONS.items()}
    signals = {name: _magnitudes(value) for name, value in axes.items()}
    length = min(len(s) for s in signals.values())
    stacked = np.concatenate(
        [signals[name][:length] for name in locations]
        + [axes[name][:length] for name in locations],
        axis=1,
    )
    peaks = stride_boundaries(signals["left_leg"][:length, 1])
    strides, starts, ends = [], [], []
    for start, end in zip(peaks[:-1], peaks[1:]):
        if MIN_STRIDE <= (end - start) / SAMPLE_RATE <= MAX_STRIDE:
            strides.append(_resample(stacked[start:end + 1], STRIDE_SAMPLES))
            starts.append(start)
            ends.append(end)
    starts, ends = np.asarray(starts), np.asarray(ends)
    recording = StrideRecording(
        np.stack(strides).astype(np.float32), starts, ends,
        (ends - starts) / SAMPLE_RATE, tuple(locations),
    )
    if locations == tuple(LOCATIONS):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, strides=recording.strides, start=starts, end=ends,
            duration=recording.duration,
        )
    return recording
