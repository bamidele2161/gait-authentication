"""Learn a drift basis and constrain probe-guided template movement within it."""

from __future__ import annotations

import numpy as np

from src.posthoc_supcon.scoring import centroid


def fit_drift_model(embeddings, participants, conditions, rank, quantile, scale):
    """Fit an SVD basis and robust per-axis movement limits from development users."""
    differences = []
    for person in participants:
        normal = centroid(embeddings[(person, "st_control")])
        for condition in conditions[1:]:
            differences.append(centroid(embeddings[(person, condition)]) - normal)
    differences = np.asarray(differences, dtype=np.float64)
    _, _, right = np.linalg.svd(differences, full_matrices=False)
    basis = right[:min(rank, len(right))].T
    observed_coefficients = differences @ basis
    limits = scale * np.quantile(
        np.abs(observed_coefficients), quantile, axis=0
    )
    return basis, np.maximum(limits, 1e-8)


def constrained_distance(probes, template, basis, limits):
    """Move a template toward each probe only along plausible drift directions."""
    probes = np.asarray(probes, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    requested = (probes - template) @ basis
    allowed = np.clip(requested, -limits, limits)
    plausible_templates = template + allowed @ basis.T
    plausible_templates /= np.maximum(
        np.linalg.norm(plausible_templates, axis=1, keepdims=True), 1e-12
    )
    return 1.0 - np.sum(probes * plausible_templates, axis=1)


def normalized_constrained_distance(
    probes, claimed_template, cohort_templates, basis, limits, top_k
):
    """T-normalize constrained claim distances against development-only templates."""
    claim = constrained_distance(probes, claimed_template, basis, limits)
    cohort = np.column_stack([
        constrained_distance(probes, template, basis, limits)
        for template in cohort_templates
    ])
    if top_k is not None and top_k < cohort.shape[1]:
        cohort = np.partition(cohort, top_k - 1, axis=1)[:, :top_k]
    return (claim - cohort.mean(axis=1)) / np.maximum(cohort.std(axis=1), 1e-6)
