"""Checks for similarity-threshold direction."""

from src.condition_augmentation.run_experiment import select_similarity_threshold


def test_similarity_threshold_accepts_high_scores_at_target_far():
    threshold, frr, far = select_similarity_threshold(
        positive_scores=[0.8, 0.9],
        negative_scores=[0.1, 0.2],
        target_far=0.0,
    )
    assert threshold == 0.8
    assert frr == 0.0
    assert far == 0.0
