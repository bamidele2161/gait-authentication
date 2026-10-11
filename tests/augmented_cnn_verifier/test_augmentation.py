import numpy as np
import pytest

from src.augmented_cnn_verifier.augmentation import augment_signal


@pytest.mark.parametrize("method", ("noise", "shift", "amplitude", "time_stretch"))
def test_each_augmentation_preserves_shape_and_finite_values(method):
    signal = np.column_stack((np.linspace(0, 1, 256), np.linspace(2, 4, 256))).astype(np.float32)
    result = augment_signal(signal, np.random.default_rng(7), 0.15, method)
    assert result.shape == (256, 2)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_amplitude_distortion_never_exceeds_fifteen_percent():
    signal = np.ones((256, 2), dtype=np.float32)
    result = augment_signal(signal, np.random.default_rng(3), 0.15, "amplitude")
    assert np.all(result >= 0.85)
    assert np.all(result <= 1.15)


def test_unknown_augmentation_is_rejected():
    with pytest.raises(ValueError, match="Unknown augmentation"):
        augment_signal(np.ones((256, 2)), np.random.default_rng(1), method="bad")


def test_time_stretch_actually_changes_a_non_linear_signal():
    signal = np.column_stack((np.sin(np.linspace(0, 8, 256)), np.cos(np.linspace(0, 5, 256))))
    result = augment_signal(signal, np.random.default_rng(3), 0.15, "time_stretch")
    assert not np.allclose(result, signal)
