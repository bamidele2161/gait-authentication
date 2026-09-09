import numpy as np

from src.domain_adversarial_svm.projector import DomainAdversarialProjector
from src.domain_adversarial_svm.run_experiment import (
    causal_fusion, similarity_threshold_at_eer,
)


def test_projector_preserves_shape_and_removes_domain_direction():
    rng = np.random.default_rng(1)
    domains = np.repeat(np.arange(4), 30)
    values = rng.normal(size=(120, 12))
    values[:, 0] += domains * 5
    original = DomainAdversarialProjector(8, 0.0).fit(values, domains).transform(values)
    removed = DomainAdversarialProjector(8, 1.0).fit(values, domains).transform(values)
    assert original.shape == removed.shape == (120, 8)
    assert np.var(removed.mean(0)) <= np.var(original.mean(0)) + 1e-9


def test_similarity_threshold_and_fusion():
    threshold, frr, far = similarity_threshold_at_eer(
        np.asarray([0.8, 0.9]), np.asarray([0.1, 0.2])
    )
    assert 0.2 < threshold <= 0.8
    assert frr == far == 0
    np.testing.assert_allclose(causal_fusion([1, 3, 5], 2), [1, 2, 4])
