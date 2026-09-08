"""Load the existing two-second, six-channel sacrum windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from src.paper_triplet.config import (
    CONDITIONS, WINDOWS_DIR, WINDOW_SAMPLES,
)


@dataclass(frozen=True)
class WindowSet:
    participant_id: str
    condition: str
    values: np.ndarray
    starts: np.ndarray


def load_dataset() -> dict[str, dict[str, WindowSet]]:
    dataset: dict[str, dict[str, WindowSet]] = {}
    for condition in CONDITIONS:
        paths = sorted((WINDOWS_DIR / condition).glob("*_windows.npy"))
        if not paths:
            raise FileNotFoundError(f"No sacrum files for {condition}")
        for path in paths:
            participant_id = path.stem.replace("_windows", "")
            windows = np.load(path).astype(np.float32)
            labels_path = path.with_name(f"{participant_id}_labels.csv")
            import pandas as pd
            labels = pd.read_csv(labels_path)
            starts = labels.start_sample.to_numpy(dtype=np.int64)
            if windows.shape[1:] != (WINDOW_SAMPLES, 6):
                raise ValueError(f"Unexpected window shape in {path}: {windows.shape}")
            if len(windows) != len(starts) or not np.isfinite(windows).all():
                raise ValueError(f"Invalid windows or labels in {path}")
            dataset.setdefault(participant_id, {})[condition] = WindowSet(
                participant_id, condition, windows, starts
            )
    if any(set(recordings) != set(CONDITIONS) for recordings in dataset.values()):
        raise ValueError("Every participant must contain all four conditions")
    return dataset


def chronological_development_split(recording: WindowSet, fraction=0.8):
    boundary = min(max(int(len(recording.values) * fraction), 1), len(recording.values) - 1)
    # Discard the boundary-adjacent window because stored windows overlap 50%.
    return recording.values[:boundary], recording.values[boundary + 1:]


def paper_enrollment_split(recording: WindowSet):
    """Reserve paper-style enrolment, then split it into template/calibration."""
    count = min(400, len(recording.values) // 2)
    template_end = int(count * 0.8)
    if template_end < 1 or count + 1 >= len(recording.values):
        raise ValueError("Recording cannot be split into enrollment and test")
    template = recording.values[:template_end]
    calibration = recording.values[template_end + 1:count]
    test = recording.values[count + 1:]
    if not len(calibration) or not len(test):
        raise ValueError("Enrollment calibration or test is empty")
    return template, calibration, test
