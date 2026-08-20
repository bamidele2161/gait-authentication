"""Development-only channel normalization for raw sacrum windows."""

from dataclasses import dataclass

import numpy as np

from src.condition_invariant.config import (
    EXPECTED_WINDOW_SHAPE,
    NUMBER_OF_CHANNELS,
)
from src.condition_invariant.records import GaitWindow


@dataclass(frozen=True)
class ChannelNormalizer:
    """Frozen mean and standard deviation for each of the six IMU channels."""

    mean: np.ndarray
    standard_deviation: np.ndarray

    def __post_init__(self) -> None:
        """Validate and protect the learned normalization values."""

        mean = np.asarray(self.mean, dtype=np.float32).copy()
        standard_deviation = np.asarray(
            self.standard_deviation, dtype=np.float32
        ).copy()
        expected_shape = (NUMBER_OF_CHANNELS,)

        if mean.shape != expected_shape:
            raise ValueError(f"Expected mean shape {expected_shape}, received {mean.shape}")
        if standard_deviation.shape != expected_shape:
            raise ValueError(
                f"Expected standard-deviation shape {expected_shape}, "
                f"received {standard_deviation.shape}"
            )
        if not np.isfinite(mean).all() or not np.isfinite(standard_deviation).all():
            raise ValueError("Normalization values must be finite")
        if np.any(standard_deviation <= 0):
            raise ValueError("Every channel standard deviation must be positive")

        mean.setflags(write=False)
        standard_deviation.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", standard_deviation)

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """Apply the frozen channel rules to one raw gait window."""

        signal = np.asarray(signal)
        if signal.shape != EXPECTED_WINDOW_SHAPE:
            raise ValueError(
                f"Expected signal shape {EXPECTED_WINDOW_SHAPE}, "
                f"received {signal.shape}"
            )
        if not np.isfinite(signal).all():
            raise ValueError("Signal contains NaN or infinite values")

        normalized = (signal - self.mean) / self.standard_deviation
        return np.asarray(normalized, dtype=np.float32)


def fit_channel_normalizer(
    learning_windows: tuple[GaitWindow, ...],
) -> ChannelNormalizer:
    """Learn per-channel statistics from development-learning windows only."""

    if not learning_windows:
        raise ValueError("Cannot fit normalization on an empty window collection")

    channel_sum = np.zeros(NUMBER_OF_CHANNELS, dtype=np.float64)
    channel_squared_sum = np.zeros(NUMBER_OF_CHANNELS, dtype=np.float64)
    sample_count = 0

    for window in learning_windows:
        signal = np.asarray(window.signal, dtype=np.float64)
        channel_sum += signal.sum(axis=0)
        channel_squared_sum += np.square(signal).sum(axis=0)
        sample_count += signal.shape[0]

    mean = channel_sum / sample_count
    variance = channel_squared_sum / sample_count - np.square(mean)
    # Tiny negative values can occur from floating-point rounding only.
    variance = np.maximum(variance, 0.0)
    standard_deviation = np.sqrt(variance)

    return ChannelNormalizer(
        mean=mean,
        standard_deviation=standard_deviation,
    )
