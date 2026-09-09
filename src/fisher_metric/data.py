"""Load feature rows and make chronological, leakage-aware partitions."""

import numpy as np
import pandas as pd

from src.fisher_metric.config import CONDITIONS, FEATURES
from src.utils import FEATURE_DIR


def load_features():
    dataset = {}
    for condition in CONDITIONS:
        for path in sorted((FEATURE_DIR / condition).glob("*_features.csv")):
            participant = path.stem.removesuffix("_features")
            frame = pd.read_csv(path).sort_values("start_sample").reset_index(drop=True)
            if not np.isfinite(frame[list(FEATURES)].to_numpy()).all():
                raise ValueError(f"Non-finite feature in {path}")
            dataset.setdefault(participant, {})[condition] = frame
    if not dataset or any(set(row) != set(CONDITIONS) for row in dataset.values()):
        raise ValueError("Every participant must contain all four conditions")
    return dataset


def chronological_split(frame, fraction=0.8):
    """Split complete time blocks and remove overlap across the boundary."""
    blocks = sorted(frame.block_id.unique())
    boundary = min(max(int(len(blocks) * fraction), 1), len(blocks) - 1)
    learning = frame[frame.block_id.isin(blocks[:boundary])].copy()
    validation = frame[frame.block_id.isin(blocks[boundary:])].copy()
    first_validation = validation.start_sample.min()
    learning = learning[learning.start_sample + 256 <= first_validation].copy()
    if learning.empty or validation.empty:
        raise ValueError("Chronological split created an empty partition")
    return learning, validation


def evaluation_enrollment_split(frame):
    """Split normal walking into template, calibration and untouched test rows."""
    frame = frame.sort_values("start_sample").reset_index(drop=True)
    enrollment_end = len(frame) // 2
    template_end = int(enrollment_end * 0.8)
    template_rows = frame.iloc[:template_end].copy()
    calibration = frame.iloc[template_end + 1:enrollment_end].copy()
    test = frame.iloc[enrollment_end + 1:].copy()
    if template_rows.empty or calibration.empty or test.empty:
        raise ValueError("Evaluation recording is too short")
    return template_rows, calibration, test


def participant_folds(participants, folds):
    names = tuple(sorted(participants))
    return tuple(
        (
            tuple(name for index, name in enumerate(names) if index % folds != fold),
            tuple(name for index, name in enumerate(names) if index % folds == fold),
        )
        for fold in range(folds)
    )


def prepare_development(dataset, participants, fraction):
    learning, validation = {}, {}
    for participant in participants:
        learning[participant], validation[participant] = {}, {}
        for condition in CONDITIONS:
            first, second = chronological_split(dataset[participant][condition], fraction)
            learning[participant][condition] = first
            validation[participant][condition] = second
    return learning, validation


def balanced_learning_frame(learning, maximum, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for participant in sorted(learning):
        for condition in CONDITIONS:
            frame = learning[participant][condition]
            take = min(maximum, len(frame))
            positions = np.sort(rng.choice(len(frame), take, replace=False))
            selected = frame.iloc[positions].copy()
            selected["metric_identity"] = participant
            rows.append(selected)
    return pd.concat(rows, ignore_index=True)
