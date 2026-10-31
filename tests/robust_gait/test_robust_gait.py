import numpy as np
import pandas as pd
import pytest

from src.robust_gait.data import LOCATIONS, WINDOWS_ROOT, _continuous, _raw, load_strides
from src.robust_gait.run_experiment import (
    DRIFT_CONDITIONS, ExperimentConfig, calibrate, drift_bank, fuse, participants,
    working_locations,
)
from src.robust_gait.verifier import (
    UserModel, VerifierConfig, augment_strides, drift_strides, stride_features,
)

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


def test_drift_copies_add_scaled_bank_vectors():
    rng = np.random.default_rng(2)
    strides = rng.normal(size=(10, 128, 4)).astype(np.float32)
    bank = np.ones((3, 128, 4), dtype=np.float32)
    config = VerifierConfig(drift_augmentation=2, drift_scale=(0.5, 1.0))
    copies = drift_strides(strides, bank, 2, config, rng)
    assert copies.shape == (20, 128, 4)
    shift = copies - np.concatenate([strides, strides])
    assert np.all((shift >= 0.5 - 1e-6) & (shift <= 1.0 + 1e-6))
    assert drift_strides(strides, None, 2, config, rng).shape[0] == 0


def test_drift_off_leaves_enrolment_randomness_unchanged():
    rng = np.random.default_rng(3)
    strides = rng.normal(1.0, 0.2, (30, 128, N_CHANNELS + 6 * len(LOCATIONS))).astype(np.float32)
    background = rng.normal(0.0, 1.0, strides.shape).astype(np.float32)
    config = VerifierConfig(locations=("sacrum",), enrolment_augmentation=1)
    with_bank = UserModel(config).fit(strides, background, 5, np.ones_like(strides[:2]))
    without = UserModel(config).fit(strides, background, 5)
    np.testing.assert_allclose(with_bank.score(strides), without.score(strides))


def test_drift_bank_uses_only_given_people_and_zeroes_dead_sensors():
    people = participants()
    data = {(p, c): load_strides(p, c) for p in people for c in
            ("st_control", "st_fatigue", "dt_control", "dt_fatigue")}
    config = ExperimentConfig(VerifierConfig(drift_augmentation=4))
    group = ["sub_07", "sub_01", "sub_02", "sub_03"]
    bank = drift_bank(group, data, config)
    assert bank.shape == (len(group) * len(DRIFT_CONDITIONS), 128, N_CHANNELS + 6 * len(LOCATIONS))
    names = list(LOCATIONS)
    sub07 = bank[: len(DRIFT_CONDITIONS)]  # sub_07's vectors come first
    for location in ("sacrum", "left_foot"):
        l = names.index(location)
        assert np.all(sub07[:, :, [2 * l, 2 * l + 1]] == 0)
        assert np.all(sub07[:, :, N_CHANNELS + 6 * l: N_CHANNELS + 6 * l + 6] == 0)
    assert drift_bank(group, data, ExperimentConfig()) is None
