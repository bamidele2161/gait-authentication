import numpy as np

from src.multisensor_problem_definition.run_experiment import (
    FEATURE_COUNT,
    fuse,
    statistical_features,
)


def test_feature_shape_dtype_and_finiteness():
    rng = np.random.default_rng(1)
    windows = rng.normal(size=(3, 256, 54)).astype(np.float32)
    features = statistical_features(windows)
    assert features.shape == (3, FEATURE_COUNT) == (3, 594)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_constant_signal_has_finite_shape_features():
    features = statistical_features(np.ones((2, 256, 54), dtype=np.float32))
    assert np.isfinite(features).all()
    # Standard deviation, range, IQR, MAD, skewness, and excess kurtosis.
    reshaped = features.reshape(2, 54, 11)
    np.testing.assert_array_equal(reshaped[:, :, 1], 0.0)
    np.testing.assert_array_equal(reshaped[:, :, 5:8], 0.0)
    np.testing.assert_array_equal(reshaped[:, :, 9:11], 0.0)


def test_fusion_is_causal_moving_average():
    np.testing.assert_allclose(fuse(np.arange(5.0), 2), [0.5, 1.5, 2.5, 3.5])
