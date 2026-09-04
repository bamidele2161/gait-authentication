import pandas as pd
import pytest

from src.secure_adaptation.protocol import (
    split_normal_enrollment,
    split_trusted_session,
)
from src.utils import WINDOW_SAMPLES


def make_frame(count=80):
    return pd.DataFrame({
        "start_sample": [index * 128 for index in range(count)],
        "value": range(count),
    })


def test_trusted_session_is_chronological_and_has_raw_sample_guards():
    split = split_trusted_session(make_frame(), update_seconds=20, calibration_seconds=10)
    assert split.update.start_sample.max() + WINDOW_SAMPLES <= split.calibration.start_sample.min()
    assert split.calibration.start_sample.max() + WINDOW_SAMPLES <= split.test.start_sample.min()
    assert split.update.start_sample.max() < split.calibration.start_sample.min()
    assert split.calibration.start_sample.max() < split.test.start_sample.min()


def test_trusted_session_rejects_recordings_that_are_too_short():
    with pytest.raises(ValueError, match="too short"):
        split_trusted_session(make_frame(20), update_seconds=20, calibration_seconds=10)


def test_normal_enrollment_has_guard_windows():
    training, validation, testing = split_normal_enrollment(make_frame())
    assert training.start_sample.max() + WINDOW_SAMPLES <= validation.start_sample.min()
    assert validation.start_sample.max() + WINDOW_SAMPLES <= testing.start_sample.min()
