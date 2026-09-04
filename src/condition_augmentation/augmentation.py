"""Learn and apply condition changes in standardized feature space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConditionTransform:
    """One development participant's location and scale change."""

    donor_participant: str
    condition: str
    donor_control_center: np.ndarray
    shift: np.ndarray
    scale_ratio: np.ndarray


def learn_condition_transforms(
    standardized_features: dict[str, dict[str, np.ndarray]],
) -> tuple[ConditionTransform, ...]:
    """Estimate each donor's ST-control-to-changed-condition transformation."""

    transforms = []
    for participant_id in sorted(standardized_features):
        conditions = standardized_features[participant_id]
        if "st_control" not in conditions:
            raise ValueError(f"Missing ST-control features for {participant_id}")
        control = np.asarray(conditions["st_control"], dtype=np.float64)
        control_center = np.median(control, axis=0)
        control_scale = np.std(control, axis=0)
        control_scale = np.maximum(control_scale, 1e-6)
        for condition in ("st_fatigue", "dt_control", "dt_fatigue"):
            if condition not in conditions:
                raise ValueError(f"Missing {condition} features for {participant_id}")
            changed = np.asarray(conditions[condition], dtype=np.float64)
            changed_center = np.median(changed, axis=0)
            changed_scale = np.maximum(np.std(changed, axis=0), 1e-6)
            transforms.append(
                ConditionTransform(
                    donor_participant=participant_id,
                    condition=condition,
                    donor_control_center=control_center.astype(np.float32),
                    shift=(changed_center - control_center).astype(np.float32),
                    scale_ratio=np.clip(
                        changed_scale / control_scale, 0.5, 2.0
                    ).astype(np.float32),
                )
            )
    return tuple(transforms)


def generate_condition_augmented_features(
    normal_features: np.ndarray,
    transforms: tuple[ConditionTransform, ...],
    seed: int,
    repetitions_per_condition: int = 2,
    lower_clip: np.ndarray | None = None,
    upper_clip: np.ndarray | None = None,
    compatible_donor_count: int = 4,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Generate condition-like positives while preserving the user's centre."""

    normal = np.asarray(normal_features, dtype=np.float32)
    if normal.ndim != 2 or len(normal) == 0:
        raise ValueError("normal_features must be a non-empty matrix")
    if repetitions_per_condition < 1:
        raise ValueError("repetitions_per_condition must be at least 1")
    if compatible_donor_count < 1:
        raise ValueError("compatible_donor_count must be at least 1")
    by_condition = {
        condition: tuple(item for item in transforms if item.condition == condition)
        for condition in ("st_fatigue", "dt_control", "dt_fatigue")
    }
    if any(not values for values in by_condition.values()):
        raise ValueError("Transforms must cover all three changed conditions")

    random_generator = np.random.default_rng(seed)
    user_center = np.median(normal, axis=0)
    generated = []
    labels = []
    source_rows = []
    for condition, all_donors in by_condition.items():
        donors = tuple(
            sorted(
                all_donors,
                key=lambda donor: np.linalg.norm(
                    user_center - donor.donor_control_center
                ),
            )[:compatible_donor_count]
        )
        for _ in range(repetitions_per_condition):
            donor_positions = random_generator.integers(len(donors), size=len(normal))
            for row_index, donor_position in enumerate(donor_positions):
                donor = donors[int(donor_position)]
                synthetic = (
                    user_center
                    + donor.shift
                    + (normal[row_index] - user_center) * donor.scale_ratio
                )
                if lower_clip is not None and upper_clip is not None:
                    synthetic = np.clip(synthetic, lower_clip, upper_clip)
                generated.append(synthetic)
                labels.append(condition)
                source_rows.append(row_index)
    return (
        np.asarray(generated, dtype=np.float32),
        tuple(labels),
        np.asarray(source_rows, dtype=np.int64),
    )
