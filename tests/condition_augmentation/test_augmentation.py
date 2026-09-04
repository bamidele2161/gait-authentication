"""Checks for development-derived statistical condition augmentation."""

import numpy as np

from src.condition_augmentation.augmentation import (
    generate_condition_augmented_features,
    learn_condition_transforms,
)


def test_learning_recovers_known_condition_shift() -> None:
    control = np.array([[0.0, 1.0], [2.0, 3.0]])
    data = {
        "sub_01": {
            "st_control": control,
            "st_fatigue": control + 2,
            "dt_control": control + 3,
            "dt_fatigue": control + 4,
        }
    }

    transforms = learn_condition_transforms(data)

    fatigue = next(item for item in transforms if item.condition == "st_fatigue")
    np.testing.assert_allclose(fatigue.shift, np.array([2.0, 2.0]))


def test_generation_preserves_count_and_source_identity() -> None:
    control = np.array([[0.0, 1.0], [2.0, 3.0]])
    data = {
        "sub_01": {
            "st_control": control,
            "st_fatigue": control + 1,
            "dt_control": control + 2,
            "dt_fatigue": control + 3,
        }
    }
    transforms = learn_condition_transforms(data)

    generated, conditions, source_rows = generate_condition_augmented_features(
        control, transforms, seed=3, repetitions_per_condition=2
    )

    assert generated.shape == (12, 2)
    assert set(conditions) == {"st_fatigue", "dt_control", "dt_fatigue"}
    assert set(source_rows) == {0, 1}
