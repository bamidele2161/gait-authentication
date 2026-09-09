import numpy as np

from src.adaptive_statistical.run_experiment import (
    adaptive_genuine_stream, cosine_scores, enrollment_parts, make_template,
    raw_channel_center, statistical_features, threshold_at_target_far,
)


def test_center_and_feature_dimensions():
    windows = np.random.default_rng(1).normal(size=(10, 256, 6))
    center = raw_channel_center(windows)
    features = statistical_features(windows, center)
    assert center.shape == (6,)
    assert features.shape == (10, 88)


def test_enrollment_partitions_are_disjoint():
    first, second, third = enrollment_parts(np.arange(100))
    assert len(first) == 60 and len(second) == 19 and len(third) == 19
    assert first[-1] + 1 < second[0]
    assert second[-1] + 1 < third[0]


def test_target_far_threshold():
    threshold, frr, far = threshold_at_target_far(
        np.asarray([0.8, 0.9]), np.asarray([0.1, 0.2]), 0.01
    )
    assert 0.2 < threshold <= 0.8
    assert frr == far == 0


def test_ema_updates_only_after_acceptance():
    initial = make_template(np.asarray([[1.0, 0.0]]))
    values = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    scores, decisions, final = adaptive_genuine_stream(values, initial, 0.8, 0.1, 1)
    assert decisions.tolist() == [True, False]
    assert cosine_scores(np.asarray([final]), initial)[0] == 1.0
