import numpy as np

from src.constrained_drift_verification.scoring import (
    constrained_distance, fit_drift_model,
)


def test_constrained_template_cannot_exceed_learned_limit():
    probes = np.asarray([[0., 1.]])
    template = np.asarray([1., 0.])
    basis = np.asarray([[0.], [1.]])
    limited = constrained_distance(probes, template, basis, np.asarray([.1]))
    unlimited = constrained_distance(probes, template, basis, np.asarray([10.]))
    assert limited[0] > unlimited[0]


def test_drift_model_has_requested_rank_and_positive_limits():
    embeddings = {
        ("a", "st_control"): np.asarray([[1., 0., 0.]]),
        ("a", "st_fatigue"): np.asarray([[.9, .1, 0.]]),
        ("a", "dt_control"): np.asarray([[.9, 0., .1]]),
        ("a", "dt_fatigue"): np.asarray([[.8, .1, .1]]),
    }
    basis, limits = fit_drift_model(
        embeddings, ("a",),
        ("st_control", "st_fatigue", "dt_control", "dt_fatigue"),
        rank=2, quantile=.9, scale=1.,
    )
    assert basis.shape == (3, 2)
    assert np.all(limits > 0)
