import numpy as np
import pandas as pd
import pytest

from src.robust_gait.data import LOCATIONS, WINDOWS_ROOT, _continuous, _raw, load_strides
from src.robust_gait.run_experiment import (
    ExperimentConfig, calibrate, fuse, participants, working_locations,
)
from src.robust_gait.verifier import VerifierConfig, augment_strides, stride_features

N_CHANNELS = 2 * len(LOCATIONS)


def test_raw_loader_reproduces_saved_sacrum_windows():
    windows = _continuous(WINDOWS_ROOT / "sacrum_time" / "st_control" / "sub_07_windows.npy")
    raw = _raw("sub_07", "st_control", "SA")  # includes the known recording trim
    np.testing.assert_allclose(raw[: len(windows)], windows)


def test_strides_are_phase_aligned_and_plausible():
    recording = load_strides("sub_01", "st_control")
    # 18 magnitude channels followed by 9 sensors x 6 raw axes.
    assert recording.strides.shape[1:] == (128, N_CHANNELS + 6 * len(LOCATIONS))
    sacrum_axes = recording.strides[:, :, N_CHANNELS:N_CHANNELS + 6]
    # Magnitudes are taken before stride resampling and interpolation does not
    # commute with the norm, so agreement is close but not exact.
    np.testing.assert_allclose(
        np.linalg.norm(sacrum_axes[:, :, 3:6], axis=2), recording.strides[:, :, 0], rtol=0.03
    )
    assert np.all((recording.duration >= 0.7) & (recording.duration <= 1.8))
    assert np.all(np.diff(recording.start) > 0)


def test_augmentation_keeps_gravity_baseline_and_shape():
    rng = np.random.default_rng(0)
    strides = rng.normal(1.0, 0.2, (20, 128, N_CHANNELS)).astype(np.float32)
    strides[:, :, 1::2] = np.abs(strides[:, :, 1::2]) * 50
    out = augment_strides(strides, 2, VerifierConfig(warp_range=0.0), rng)
    assert out.shape == (60, 128, N_CHANNELS)
    np.testing.assert_array_equal(out[:20], strides)
    np.testing.assert_allclose(
        out[20:40, :, 0::2].mean(axis=1), strides[:, :, 0::2].mean(axis=1), rtol=1e-5
    )


def test_mean_pooling_is_a_low_pass_filter():
    rng = np.random.default_rng(1)
    noise = rng.normal(size=(50, 128, N_CHANNELS))
    sliced = stride_features(noise, ("sacrum",), 4, "slice")
    pooled = stride_features(noise, ("sacrum",), 4, "mean")
    assert sliced.shape == pooled.shape == (50, 64)
    assert pooled.std() < 0.6 * sliced.std()


def test_fuse_is_a_causal_moving_average():
    np.testing.assert_allclose(fuse(np.arange(5.0), 2), [0.5, 1.5, 2.5, 3.5])


def test_faulty_enrolment_sensors_are_dropped_for_sub07_only():
    people = participants()
    data = {(p, c): load_strides(p, c) for p in people for c in ("st_control", "st_fatigue")}
    background = [p for p in people if p != "sub_07"]
    config = ExperimentConfig(VerifierConfig(locations=tuple(LOCATIONS)))
    kept = working_locations(data["sub_07", "st_control"].strides, background, data, config)
    assert set(LOCATIONS) - set(kept) == {"sacrum", "left_foot"}
    kept = working_locations(data["sub_01", "st_control"].strides, background, data, config)
    assert kept == tuple(LOCATIONS)


def test_calibration_never_touches_evaluation_users():
    people = participants()
    development, evaluation = list(people[:12]), set(people[12:])

    class Guard(dict):
        def __getitem__(self, key):
            if key[0] in evaluation:
                raise AssertionError(f"calibration accessed evaluation user {key}")
            return super().__getitem__(key)

    data = Guard({
        (p, c): load_strides(p, c) for p in development
        for c in ("st_control", "st_fatigue", "dt_control", "dt_fatigue")
    })
    config = ExperimentConfig(
        VerifierConfig(locations=("left_leg",)), calibration_probes="session2",
        operating_point="macro-eer", mirror_group_size=4,
    )
    threshold, frr, far = calibrate(development, data, config, np.random.default_rng(0))
    assert np.isfinite(threshold) and 0 <= frr <= 1 and 0 <= far <= 1
