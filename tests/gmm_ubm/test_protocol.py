import numpy as np

from src.gmm_ubm.run_experiment import Config, GMMUBM, causal_fusion, target_far_threshold


def test_map_adaptation_and_llr_are_finite():
    rng = np.random.default_rng(3)
    population = rng.normal(size=(300, 12))
    enrollment = rng.normal(.5, 1, size=(30, 12))
    model = GMMUBM(Config(components=4, pca_components=8)).fit(population)
    means = model.adapt(enrollment)
    scores = model.score(enrollment, means)
    assert means.shape == (4, 8)
    assert scores.shape == (30,)
    assert np.isfinite(scores).all()


def test_threshold_and_fusion():
    threshold, frr, far = target_far_threshold(
        np.asarray([2., 3.]), np.asarray([-2., -1.]), .01
    )
    assert -1 < threshold <= 2
    assert frr == far == 0
    np.testing.assert_allclose(causal_fusion([1, 3, 5], 2), [1, 2, 4])
