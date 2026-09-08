"""Leakage-aware loading and chronological splitting."""

from dataclasses import dataclass
import numpy as np

from src.unified_gait.config import CONDITIONS, WINDOWS_DIR


@dataclass(frozen=True)
class Recording:
    participant: str
    condition: str
    windows: np.ndarray


def load_dataset():
    dataset = {}
    for condition in CONDITIONS:
        for path in sorted((WINDOWS_DIR / condition).glob("*_windows.npy")):
            participant = path.stem.removesuffix("_windows")
            windows = np.load(path).astype(np.float32)
            if windows.ndim != 3 or windows.shape[1:] != (256, 6):
                raise ValueError(f"Unexpected shape {windows.shape} in {path}")
            if not np.isfinite(windows).all():
                raise ValueError(f"Non-finite data in {path}")
            dataset.setdefault(participant, {})[condition] = Recording(
                participant, condition, windows
            )
    if not dataset or any(set(row) != set(CONDITIONS) for row in dataset.values()):
        raise ValueError("Every participant must have all four conditions")
    return dataset


def chronological_split(windows, fraction=0.8):
    """Split in time and discard one boundary window because overlap is 50%."""
    boundary = min(max(int(len(windows) * fraction), 1), len(windows) - 2)
    learning = windows[:boundary]
    validation = windows[boundary + 1:]
    if not len(learning) or not len(validation):
        raise ValueError("Recording is too short")
    return learning, validation


def enrollment_split(windows):
    """Use the first half for normal enrollment and the second half for testing."""
    boundary = len(windows) // 2
    enrollment = windows[:boundary]
    test = windows[boundary + 1:]
    if not len(enrollment) or not len(test):
        raise ValueError("ST-control recording is too short")
    return enrollment, test


def outer_folds(participants, folds=4):
    """Deterministic participant-disjoint folds."""
    names = tuple(sorted(participants))
    return tuple(
        (
            tuple(name for index, name in enumerate(names) if index % folds != fold),
            tuple(name for index, name in enumerate(names) if index % folds == fold),
        )
        for fold in range(folds)
    )
