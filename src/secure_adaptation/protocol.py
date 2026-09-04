"""Chronological data partitions for trusted gait-template adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils import ORIGINAL_RATE, WINDOW_SAMPLES


@dataclass(frozen=True)
class AdaptationSplit:
    """Non-overlapping trusted-update, calibration, and evaluation windows."""

    update: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def split_normal_enrollment(frame: pd.DataFrame):
    """Split normal gait into chronological 60/20/20 partitions with guards."""

    ordered = frame.sort_values("start_sample").reset_index(drop=True)
    train_end = int(0.60 * len(ordered))
    validation_end = int(0.80 * len(ordered))
    training = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end + 1 : validation_end].copy()
    testing = ordered.iloc[validation_end + 1 :].copy()
    if training.empty or validation.empty or testing.empty:
        raise ValueError("Normal enrollment split produced an empty partition")
    return training, validation, testing


def split_trusted_session(
    frame: pd.DataFrame,
    update_seconds: float = 20.0,
    calibration_seconds: float = 10.0,
) -> AdaptationSplit:
    """Reserve the start of a changed session for secure online adaptation.

    The secondary authentication factor makes both ``update`` and
    ``calibration`` genuine samples. One overlapping window is removed at each
    boundary so that raw IMU samples cannot occur in two partitions.
    """

    if update_seconds <= 0 or calibration_seconds <= 0:
        raise ValueError("Adaptation durations must be positive")
    ordered = frame.sort_values("start_sample").reset_index(drop=True)
    if ordered.empty:
        raise ValueError("Cannot split an empty recording")

    origin = int(ordered.start_sample.iloc[0])
    update_boundary = origin + round(update_seconds * ORIGINAL_RATE)
    calibration_boundary = update_boundary + round(calibration_seconds * ORIGINAL_RATE)

    update = ordered[ordered.start_sample < update_boundary].copy()
    calibration = ordered[
        (ordered.start_sample >= update_boundary)
        & (ordered.start_sample < calibration_boundary)
    ].copy()
    test = ordered[ordered.start_sample >= calibration_boundary].copy()

    # Windows are two seconds long and begin every second. Remove any window
    # whose raw samples cross the next partition's first starting sample.
    if not calibration.empty:
        update = update[
            update.start_sample + WINDOW_SAMPLES <= calibration.start_sample.min()
        ].copy()
    if not test.empty:
        calibration = calibration[
            calibration.start_sample + WINDOW_SAMPLES <= test.start_sample.min()
        ].copy()

    if update.empty or calibration.empty or test.empty:
        raise ValueError(
            "Recording is too short for the requested update/calibration durations"
        )
    if update.start_sample.max() + WINDOW_SAMPLES > calibration.start_sample.min():
        raise AssertionError("Update and calibration partitions overlap")
    if calibration.start_sample.max() + WINDOW_SAMPLES > test.start_sample.min():
        raise AssertionError("Calibration and test partitions overlap")
    return AdaptationSplit(update=update, calibration=calibration, test=test)
