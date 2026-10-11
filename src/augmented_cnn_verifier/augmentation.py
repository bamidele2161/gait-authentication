"""The four one-dimensional transformations described in Choi et al. (2025)."""

import numpy as np


AUGMENTATIONS = ("noise", "shift", "amplitude", "time_stretch")


def additive_noise(signal, rng, maximum_distortion=0.15):
    scale = rng.uniform(0.0, maximum_distortion)
    channel_sd = np.std(signal, axis=0, keepdims=True)
    noise = rng.normal(0.0, scale, size=signal.shape) * channel_sd
    return signal + noise


def temporal_shift(signal, rng, maximum_distortion=0.15):
    maximum = max(1, int(round(len(signal) * maximum_distortion)))
    amount = int(rng.integers(-maximum, maximum + 1))
    if amount == 0:
        return signal.copy()
    # Reflection avoids the physically false discontinuity created by roll().
    padded = np.pad(signal, ((maximum, maximum), (0, 0)), mode="reflect")
    start = maximum - amount
    return padded[start:start + len(signal)]


def amplitude_stretch(signal, rng, maximum_distortion=0.15):
    factor = rng.uniform(1.0 - maximum_distortion, 1.0 + maximum_distortion)
    return signal * factor


def time_stretch(signal, rng, maximum_distortion=0.15):
    factor = rng.uniform(1.0 - maximum_distortion, 1.0 + maximum_distortion)
    output_axis = np.arange(len(signal), dtype=np.float64)
    center = (len(signal) - 1) / 2.0
    # factor > 1 expands time around the centre; factor < 1 compresses it.
    source_axis = center + (output_axis - center) / factor
    source_axis = np.clip(source_axis, 0.0, len(signal) - 1.0)
    return np.stack([
        np.interp(source_axis, output_axis, signal[:, channel])
        for channel in range(signal.shape[1])
    ], axis=1)


def augment_signal(signal, rng, maximum_distortion=0.15, method=None):
    """Apply exactly one randomly selected paper transformation."""
    selected = method or AUGMENTATIONS[int(rng.integers(len(AUGMENTATIONS)))]
    functions = {
        "noise": additive_noise,
        "shift": temporal_shift,
        "amplitude": amplitude_stretch,
        "time_stretch": time_stretch,
    }
    if selected not in functions:
        raise ValueError(f"Unknown augmentation: {selected}")
    transformed = functions[selected](
        np.asarray(signal, dtype=np.float32), rng, maximum_distortion
    )
    return np.asarray(transformed, dtype=np.float32)
