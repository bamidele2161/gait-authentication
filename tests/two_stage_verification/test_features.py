import numpy as np
import pytest

from src.two_stage_verification.features import (
    SPECTRAL_COLS,
    extract_spectral_window,
    spectral_features,
)


def test_dominant_frequency_finds_known_periodic_signal():
    time = np.arange(256) / 128
    result = spectral_features(np.sin(2 * np.pi * 2 * time))
    assert result["dominant_frequency"] == pytest.approx(2.0)
    assert result["walking_band_ratio"] == pytest.approx(1.0)


def test_constant_signal_produces_finite_zero_features():
    result = spectral_features(np.ones(256))
    assert all(value == 0 for value in result.values())


def test_window_extractor_returns_declared_feature_set():
    result = extract_spectral_window(np.random.default_rng(2).normal(size=(256, 6)))
    assert set(result) == set(SPECTRAL_COLS)
    assert len(result) == 48
    assert np.isfinite(list(result.values())).all()
