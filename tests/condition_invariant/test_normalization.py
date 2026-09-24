"""Checks for development-only channel normalization."""

import numpy as np

from src.condition_invariant.normalization import fit_channel_normalizer
from src.condition_invariant.records import GaitWindow
from tests.condition_invariant.test_records import assert_raises


def make_constant_window(window_index: int, value: float) -> GaitWindow:
    return GaitWindow(
        participant_id="sub_01",
        condition="st_control",
        window_index=window_index,
        start_sample=window_index * 128,
        block_id=0,
        signal=np.full((256, 6), value, dtype=np.float32),
    )


def test_fit_normalizer_learns_six_channel_statistics() -> None:
    windows = (
        make_constant_window(0, 1.0),
        make_constant_window(1, 3.0),
    )

    normalizer = fit_channel_normalizer(windows)

    np.testing.assert_allclose(normalizer.mean, np.full(6, 2.0))
    np.testing.assert_allclose(normalizer.standard_deviation, np.full(6, 1.0))


def test_transform_centers_and_scales_learning_values() -> None:
    windows = (
        make_constant_window(0, 1.0),
        make_constant_window(1, 3.0),
    )
    normalizer = fit_channel_normalizer(windows)

    first = normalizer.transform(windows[0].signal)
    second = normalizer.transform(windows[1].signal)
    combined = np.concatenate([first, second], axis=0)

    np.testing.assert_allclose(combined.mean(axis=0), np.zeros(6), atol=1e-6)
    np.testing.assert_allclose(combined.std(axis=0), np.ones(6), atol=1e-6)
    assert first.dtype == np.float32


def test_transform_does_not_modify_original_signal() -> None:
    windows = (
        make_constant_window(0, 1.0),
        make_constant_window(1, 3.0),
    )
    original = windows[0].signal.copy()
    normalizer = fit_channel_normalizer(windows)

    normalizer.transform(windows[0].signal)

    np.testing.assert_array_equal(windows[0].signal, original)


def test_fit_normalizer_rejects_constant_channels() -> None:
    windows = (
        make_constant_window(0, 1.0),
        make_constant_window(1, 1.0),
    )

    assert_raises(
        ValueError,
        "standard deviation must be positive",
        lambda: fit_channel_normalizer(windows),
    )
