import numpy as np

from src.two_stage_verification.decision import (
    and_decision,
    select_and_thresholds,
    select_group_robust_and_thresholds,
)


def test_and_decision_requires_both_stages():
    accepted = and_decision([0.9, 0.9, 0.1], [0.9, 0.1, 0.9], 0.5, 0.5)
    assert accepted.tolist() == [True, False, False]


def test_joint_threshold_can_reject_complementary_impostors():
    result = select_and_thresholds(
        positive_primary=[0.8, 0.9],
        positive_secondary=[0.8, 0.9],
        negative_primary=[0.85, 0.1],
        negative_secondary=[0.1, 0.85],
        target_far=0.0,
    )
    primary_threshold, secondary_threshold, frr, far = result
    assert frr == 0.0
    assert far == 0.0
    assert and_decision(
        np.array([0.85, 0.1]), np.array([0.1, 0.85]),
        primary_threshold, secondary_threshold,
    ).sum() == 0


def test_group_robust_threshold_controls_each_impostor_identity():
    result = select_group_robust_and_thresholds(
        positive_primary=[0.8, 0.9],
        positive_secondary=[0.8, 0.9],
        negative_primary=[0.85, 0.1, 0.2, 0.3],
        negative_secondary=[0.85, 0.1, 0.2, 0.3],
        negative_groups=["hard", "easy", "easy", "easy"],
        target_far=0.0,
    )
    assert result[2] <= 1.0
    assert result[3] == 0.0
    assert result[4] == 0.0
