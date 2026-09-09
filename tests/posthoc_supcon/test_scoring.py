import numpy as np

from src.posthoc_supcon.scoring import (
    augmented_templates, cohort_normalized_distance, cosine_distance,
    drift_aware_distance,
)


def test_cosine_distance_uses_closest_template():
    probes = np.asarray([[1., 0.]])
    templates = np.asarray([[0., 1.], [1., 0.]])
    np.testing.assert_allclose(cosine_distance(probes, templates), [0.])


def test_augmentation_keeps_unit_templates():
    templates = augmented_templates(
        np.asarray([1., 0.]), {"shift": np.asarray([0., 1.])}
    )
    np.testing.assert_allclose(np.linalg.norm(templates, axis=1), 1.)


def test_cohort_normalization_is_finite():
    values = cohort_normalized_distance(
        np.asarray([[1., 0.]]), np.asarray([1., 0.]),
        np.asarray([[0., 1.], [-1., 0.]]),
    )
    assert np.isfinite(values).all()


def test_zero_drift_weight_ignores_parallel_displacement():
    probes = np.asarray([[0., 1.]])
    template = np.asarray([0., 0.])
    basis = np.asarray([[0.], [1.]])
    np.testing.assert_allclose(
        drift_aware_distance(probes, template, basis, drift_weight=0.), [0.]
    )
