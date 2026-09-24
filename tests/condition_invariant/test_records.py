"""Checks for the gait-window data record."""

import numpy as np

from src.condition_invariant.records import GaitWindow


def assert_raises(expected_exception, expected_message, action) -> None:
    """Confirm that an action fails with the expected error."""

    try:
        action()
    except expected_exception as error:
        assert expected_message in str(error)
    else:
        raise AssertionError(f"Expected {expected_exception.__name__} to be raised")


def make_valid_window() -> GaitWindow:
    return GaitWindow(
        participant_id="sub_01",
        condition="st_control",
        window_index=3,
        start_sample=384,
        block_id=0,
        signal=np.zeros((256, 6), dtype=np.float32),
    )


def test_valid_window_preserves_signal_and_metadata() -> None:
    window = make_valid_window()

    assert window.participant_id == "sub_01"
    assert window.condition == "st_control"
    assert window.window_index == 3
    assert window.start_sample == 384
    assert window.block_id == 0
    assert window.signal.shape == (256, 6)


def test_window_rejects_wrong_signal_shape() -> None:
    assert_raises(
        ValueError,
        "Expected signal shape",
        lambda: GaitWindow(
            participant_id="sub_01",
            condition="st_control",
            window_index=0,
            start_sample=0,
            block_id=0,
            signal=np.zeros((128, 6), dtype=np.float32),
        ),
    )


def test_window_rejects_unknown_condition() -> None:
    assert_raises(
        ValueError,
        "Unknown condition",
        lambda: GaitWindow(
            participant_id="sub_01",
            condition="unknown",
            window_index=0,
            start_sample=0,
            block_id=0,
            signal=np.zeros((256, 6), dtype=np.float32),
        ),
    )


def test_window_rejects_non_finite_signal() -> None:
    signal = np.zeros((256, 6), dtype=np.float32)
    signal[0, 0] = np.nan

    assert_raises(
        ValueError,
        "NaN or infinite",
        lambda: GaitWindow(
            participant_id="sub_01",
            condition="st_control",
            window_index=0,
            start_sample=0,
            block_id=0,
            signal=signal,
        ),
    )
