"""Leakage-free chronological schedule for periodic trusted refreshes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils import ORIGINAL_RATE, WINDOW_SAMPLES


@dataclass(frozen=True)
class RollingPeriod:
    cycle: int
    evaluation_start_seconds: float
    evaluation_end_seconds: float
    refresh_update: pd.DataFrame
    refresh_calibration: pd.DataFrame


@dataclass(frozen=True)
class RollingSchedule:
    initial_update: pd.DataFrame
    initial_calibration: pd.DataFrame
    periods: tuple[RollingPeriod, ...]


def windows_inside(frame, start_seconds, end_seconds):
    """Select only windows whose complete raw samples fit in an interval."""
    ordered = frame.sort_values("start_sample")
    origin = int(ordered.start_sample.min())
    start = origin + round(start_seconds * ORIGINAL_RATE)
    end = origin + round(end_seconds * ORIGINAL_RATE)
    return ordered[
        (ordered.start_sample >= start)
        & (ordered.start_sample + WINDOW_SAMPLES <= end)
    ].copy()


def create_rolling_schedule(
    frame: pd.DataFrame,
    initial_update_seconds=30.0,
    initial_calibration_seconds=20.0,
    evaluation_seconds=50.0,
    refresh_seconds=20.0,
):
    """Create evaluation periods separated by trusted update/calibration data."""
    if frame.empty or min(initial_update_seconds, initial_calibration_seconds,
                          evaluation_seconds, refresh_seconds) <= 0:
        raise ValueError("A non-empty frame and positive durations are required")
    ordered = frame.sort_values("start_sample").reset_index(drop=True)
    duration = (
        ordered.start_sample.max() - ordered.start_sample.min() + WINDOW_SAMPLES
    ) / ORIGINAL_RATE
    initial_update = windows_inside(ordered, 0, initial_update_seconds)
    initial_calibration = windows_inside(
        ordered, initial_update_seconds,
        initial_update_seconds + initial_calibration_seconds,
    )
    cursor = initial_update_seconds + initial_calibration_seconds
    periods = []
    cycle = 0
    pending_update = ordered.iloc[0:0].copy()
    pending_calibration = ordered.iloc[0:0].copy()
    while cursor < duration:
        evaluation_end = min(cursor + evaluation_seconds, duration)
        if evaluation_end - cursor >= 2:
            periods.append(RollingPeriod(
                cycle=cycle,
                evaluation_start_seconds=cursor,
                evaluation_end_seconds=evaluation_end,
                refresh_update=pending_update,
                refresh_calibration=pending_calibration,
            ))
        cursor = evaluation_end
        if duration - cursor < refresh_seconds:
            break
        half = refresh_seconds / 2
        pending_update = windows_inside(ordered, cursor, cursor + half)
        pending_calibration = windows_inside(
            ordered, cursor + half, cursor + refresh_seconds
        )
        cursor += refresh_seconds
        cycle += 1
    if initial_update.empty or initial_calibration.empty or not periods:
        raise ValueError("Recording is too short for the rolling schedule")
    return RollingSchedule(initial_update, initial_calibration, tuple(periods))
