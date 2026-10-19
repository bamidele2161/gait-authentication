"""Per-user discriminative stride verifier.

Why per-user instead of a global deep embedding: each fold has only 12
development identities, so a network trained to separate them memorizes those
12 people (development genuine distance 0.19 vs 0.95 for unseen users). Every
user, however, provides ~320 enrollment strides. A small regularized model per
user, trained against a background population of *other* people, uses that
data directly and needs no identity-level generalization from 12 people.

A user model is an L2-regularized logistic regression on phase-aligned stride
profiles: enrolled user (positive) versus background strides (negative), with
balanced class weights. Its decision value is the verification score for a
probe stride (higher = more likely the claimed user).

Sensor fusion:
- ``early``: one model on the concatenated sensor profiles.
- ``late``: one model per sensor; scores are averaged. A failed sensor then
  degrades only its own share of the score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.robust_gait.data import LOCATIONS

LOCATION_NAMES = tuple(LOCATIONS)


@dataclass(frozen=True)
class VerifierConfig:
    locations: tuple[str, ...] = ("sacrum", "left_leg", "right_leg")
    fusion: str = "early"              # early | late
    downsample: int = 4                # 128-sample stride -> 32 points/channel
    regularization: float = 0.1        # logistic-regression C
    background_strides_per_user: int = 120
    enrolment_augmentation: int = 0    # extra augmented copies per enrolment stride
    amplitude_range: float = 0.10      # dynamic-amplitude factor in 1 ± range
    warp_range: float = 0.05           # within-stride timing warp amplitude
    pooling: str = "slice"             # slice | mean (anti-aliased downsampling)
    combination: str = "mean"          # late fusion: mean | median across sensors
    channels: str = "magnitude"        # magnitude | axes | both
    derivative: bool = False           # append first differences of each profile
    drift_augmentation: int = 0        # extra copies shifted by learned drift vectors
    drift_scale: tuple[float, float] = (0.5, 1.0)  # drift vector scaled by U(lo, hi)


def augment_strides(
    strides: np.ndarray, copies: int, config: VerifierConfig, rng: np.random.Generator,
) -> np.ndarray:
    """Plausible within-person variation of enrolment strides.

    - Amplitude: each sensor's dynamic component is scaled by U(1 ± range).
      Acceleration magnitude is scaled around its stride mean so the ~1 g
      gravity baseline is unchanged; gyroscope magnitude is scaled directly.
    - Timing: one smooth monotonic warp t -> t + a*sin(pi*t)/pi, a ~ U(± range),
      shared by all sensors, shifts gait events within the stride.
    """

    if copies < 1:
        return strides
    samples = strides.shape[1]
    base = np.linspace(0.0, 1.0, samples)
    out = []
    for _ in range(copies):
        x = strides.copy()
        n, _, channels = x.shape
        locations = len(LOCATION_NAMES)
        factors = rng.uniform(1 - config.amplitude_range, 1 + config.amplitude_range,
                              (n, 1, locations))
        # One factor per sensor, shared by its magnitude and raw-axis channels.
        acc_cols = [2 * l for l in range(locations)]
        gyr_cols = [2 * l + 1 for l in range(locations)]
        acc_factor, gyr_factor = [factors] , [factors]
        if channels > 2 * locations:
            for l in range(locations):
                base_col = 2 * locations + 6 * l
                gyr_cols += [base_col + j for j in range(3)]
                acc_cols += [base_col + 3 + j for j in range(3)]
            per_axis = np.repeat(factors, 3, axis=2)
            acc_factor.append(per_axis)
            gyr_factor.append(per_axis)
        acc_f = np.concatenate(acc_factor, axis=2)
        gyr_f = np.concatenate(gyr_factor, axis=2)
        acc = x[:, :, acc_cols]
        acc_mean = acc.mean(axis=1, keepdims=True)
        x[:, :, acc_cols] = acc_mean + acc_f * (acc - acc_mean)
        x[:, :, gyr_cols] = gyr_f * x[:, :, gyr_cols]
        warp = rng.uniform(-config.warp_range, config.warp_range, n)
        for i in range(n):
            positions = base + warp[i] * np.sin(np.pi * base) / np.pi
            x[i] = np.column_stack([np.interp(positions, base, x[i, :, c]) for c in range(channels)])
        out.append(x)
    return np.concatenate([strides, *out]).astype(np.float32)


def drift_strides(
    strides: np.ndarray, bank: np.ndarray, copies: int, config: VerifierConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Copies of the enrolment strides shifted by measured condition drift.

    ``bank`` holds (k, samples, channels) drift vectors: how other people's
    mean stride changed from ST-control to a later condition.  Each copy adds
    one randomly chosen vector, scaled by U(drift_scale), to every stride.
    Returns only the new copies.
    """

    if copies < 1 or bank is None or not len(bank):
        return strides[:0]
    lo, hi = config.drift_scale
    out = []
    for _ in range(copies):
        pick = rng.integers(len(bank), size=len(strides))
        scale = rng.uniform(lo, hi, (len(strides), 1, 1))
        out.append(strides + scale * bank[pick])
    return np.concatenate(out).astype(np.float32)


def channel_columns(locations, channels: str = "magnitude") -> list[int]:
    """Column indices of the requested representation for these locations."""

    magnitude = [2 * LOCATION_NAMES.index(l) + k for l in locations for k in (0, 1)]
    axes = [
        2 * len(LOCATION_NAMES) + 6 * LOCATION_NAMES.index(l) + j
        for l in locations for j in range(6)
    ]
    return {"magnitude": magnitude, "axes": axes, "both": magnitude + axes}[channels]


def stride_features(
    strides: np.ndarray, locations, downsample: int,
    pooling: str = "slice", derivative: bool = False, channels: str = "magnitude",
) -> np.ndarray:
    """(n, 128, 2*all_locations) stride profiles -> (n, features).

    ``mean`` pooling averages each block of ``downsample`` samples, a box
    low-pass filter that suppresses high-frequency sensor noise before
    decimation instead of aliasing it.
    """

    selected = strides[:, :, channel_columns(locations, channels)]
    if pooling == "mean":
        n, length, channels = selected.shape
        profile = selected[:, : length - length % downsample].reshape(
            n, length // downsample, downsample, channels
        ).mean(axis=2)
    else:
        profile = selected[:, ::downsample]
    parts = [profile.reshape(len(strides), -1)]
    if derivative:
        parts.append(np.diff(profile, axis=1).reshape(len(strides), -1))
    return np.concatenate(parts, axis=1)


@dataclass
class UserModel:
    """One enrolled user's model(s) and the background scaling."""

    config: VerifierConfig
    models: list = field(default_factory=list)
    groups: list = field(default_factory=list)

    def fit(self, enrollment: np.ndarray, background: np.ndarray, seed: int = 0,
            drift_bank: np.ndarray | None = None):
        groups = (
            [self.config.locations] if self.config.fusion == "early"
            else [(location,) for location in self.config.locations]
        )
        rng = np.random.default_rng(seed)
        drifted = drift_strides(
            enrollment, drift_bank, self.config.drift_augmentation, self.config, rng,
        )
        enrollment = np.concatenate((
            augment_strides(enrollment, self.config.enrolment_augmentation, self.config, rng),
            drifted,
        ))
        labels = np.r_[np.ones(len(enrollment)), np.zeros(len(background))]
        for group in groups:
            x = np.vstack((
                stride_features(enrollment, group, self.config.downsample,
                                self.config.pooling, self.config.derivative,
                                self.config.channels),
                stride_features(background, group, self.config.downsample,
                                self.config.pooling, self.config.derivative,
                                self.config.channels),
            ))
            # Standardize with background statistics only, so the scale does
            # not depend on which user is enrolled.
            bg = x[len(enrollment):]
            mean, scale = bg.mean(0), bg.std(0) + 1e-9
            model = LogisticRegression(
                C=self.config.regularization, class_weight="balanced",
                max_iter=5000, random_state=seed,
            ).fit((x - mean) / scale, labels)
            self.models.append((model, mean, scale))
            self.groups.append(group)
        return self

    def score(self, strides: np.ndarray) -> np.ndarray:
        scores = [
            model.decision_function(
                (stride_features(strides, group, self.config.downsample,
                                 self.config.pooling, self.config.derivative,
                                self.config.channels)
                 - mean) / scale
            )
            for (model, mean, scale), group in zip(self.models, self.groups)
        ]
        if self.config.combination == "median":
            # Robust to one or two sensors whose signal shifted (e.g. a sensor
            # re-attached differently a week later).
            return np.median(scores, axis=0)
        return np.mean(scores, axis=0)
