import numpy as np
import pandas as pd

from src.rolling_adaptation.protocol import create_rolling_schedule, windows_inside
from src.utils import WINDOW_SAMPLES


def frame(seconds=240):
    return pd.DataFrame({
        "start_sample": np.arange(seconds) * 128,
        "window_index": np.arange(seconds),
    })


def test_windows_must_fit_completely_inside_interval():
    selected = windows_inside(frame(), 10, 20)
    assert selected.start_sample.min() == 10 * 128
    assert selected.start_sample.max() + WINDOW_SAMPLES <= 20 * 128


def test_schedule_keeps_refresh_windows_out_of_evaluation_intervals():
    schedule = create_rolling_schedule(frame())
    assert len(schedule.initial_update) == 29
    assert len(schedule.initial_calibration) == 19
    for period in schedule.periods:
        evaluation = windows_inside(
            frame(), period.evaluation_start_seconds, period.evaluation_end_seconds
        )
        assert not set(evaluation.window_index) & set(period.refresh_update.window_index)
        assert not set(evaluation.window_index) & set(period.refresh_calibration.window_index)
