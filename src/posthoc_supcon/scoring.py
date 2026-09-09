"""Isolated score transformations for frozen-embedding experiments."""

from __future__ import annotations

import numpy as np


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(np.linalg.norm(vector), 1e-12)


def centroid(embeddings):
    return unit(np.asarray(embeddings).mean(axis=0))


def cosine_distance(probes, templates):
    probes = np.asarray(probes, dtype=np.float64)
    templates = np.atleast_2d(np.asarray(templates, dtype=np.float64))
    return (1.0 - probes @ templates.T).min(axis=1)


def cohort_normalized_distance(probes, claimed_template, cohort_templates, top_k=None):
    """T-normalize a claim distance using the closest development cohort models."""
    claim = cosine_distance(probes, claimed_template)
    cohort = 1.0 - np.asarray(probes) @ np.asarray(cohort_templates).T
    if top_k is not None and top_k < cohort.shape[1]:
        cohort = np.partition(cohort, top_k - 1, axis=1)[:, :top_k]
    mean = cohort.mean(axis=1)
    std = cohort.std(axis=1)
    return (claim - mean) / np.maximum(std, 1e-6)


def population_shifts(embeddings, participants, conditions, scale):
    differences = {condition: [] for condition in conditions[1:]}
    for person in participants:
        normal = centroid(embeddings[(person, "st_control")])
        for condition in conditions[1:]:
            differences[condition].append(
                centroid(embeddings[(person, condition)]) - normal
            )
    return {
        condition: scale * np.mean(values, axis=0)
        for condition, values in differences.items()
    }


def augmented_templates(normal_template, shifts):
    return np.vstack([normal_template] + [
        unit(normal_template + shift) for shift in shifts.values()
    ])


def drift_basis(embeddings, participants, conditions, rank):
    differences = []
    for person in participants:
        normal = centroid(embeddings[(person, "st_control")])
        for condition in conditions[1:]:
            differences.append(centroid(embeddings[(person, condition)]) - normal)
    matrix = np.asarray(differences)
    matrix -= matrix.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(matrix, full_matrices=False)
    return right[:min(rank, len(right))].T


def drift_aware_distance(probes, template, basis, drift_weight):
    """Down-weight squared displacement lying inside the learned drift subspace."""
    delta = np.asarray(probes, dtype=np.float64) - np.asarray(template)
    parallel = (delta @ basis) @ basis.T
    perpendicular = delta - parallel
    # Division by two keeps ordinary squared Euclidean distance on the same
    # scale as cosine distance for unit vectors.
    return (
        np.square(perpendicular).sum(axis=1)
        + drift_weight * np.square(parallel).sum(axis=1)
    ) / 2.0
