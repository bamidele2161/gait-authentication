"""Joint threshold selection for two complementary gait verifiers."""

from __future__ import annotations

import numpy as np


def select_and_thresholds(
    positive_primary,
    positive_secondary,
    negative_primary,
    negative_secondary,
    target_far: float,
):
    """Minimize FRR subject to FAR when both stages must accept."""

    pp = np.asarray(positive_primary, dtype=float)
    ps = np.asarray(positive_secondary, dtype=float)
    np_ = np.asarray(negative_primary, dtype=float)
    ns = np.asarray(negative_secondary, dtype=float)
    if len(pp) == 0 or len(np_) == 0 or len(pp) != len(ps) or len(np_) != len(ns):
        raise ValueError("Paired non-empty positive and negative scores are required")
    candidates_primary = np.r_[np.unique(pp), np.inf]
    candidates_secondary = np.r_[np.unique(ps), np.inf]
    best = None
    for primary_threshold in candidates_primary:
        for secondary_threshold in candidates_secondary:
            positive_accept = (pp >= primary_threshold) & (ps >= secondary_threshold)
            negative_accept = (np_ >= primary_threshold) & (ns >= secondary_threshold)
            frr = float(1 - positive_accept.mean())
            far = float(negative_accept.mean())
            if far <= target_far + 1e-12:
                candidate = (frr, far, -primary_threshold, -secondary_threshold)
                if best is None or candidate < best[0]:
                    best = (candidate, primary_threshold, secondary_threshold)
    if best is None:
        raise ValueError("No joint threshold satisfies the requested FAR")
    (frr, far, _, _), primary_threshold, secondary_threshold = best
    return float(primary_threshold), float(secondary_threshold), frr, far


def select_group_robust_and_thresholds(
    positive_primary,
    positive_secondary,
    negative_primary,
    negative_secondary,
    negative_groups,
    target_far: float,
):
    """Require the FAR constraint for every development impostor identity."""

    pp = np.asarray(positive_primary, dtype=float)
    ps = np.asarray(positive_secondary, dtype=float)
    np_ = np.asarray(negative_primary, dtype=float)
    ns = np.asarray(negative_secondary, dtype=float)
    groups = np.asarray(negative_groups)
    if len(pp) == 0 or len(np_) == 0 or len(pp) != len(ps):
        raise ValueError("Paired non-empty positive and negative scores are required")
    if len(np_) != len(ns) or len(np_) != len(groups):
        raise ValueError("Negative scores and groups must be aligned")
    unique_groups = np.unique(groups)
    if len(unique_groups) == 0:
        raise ValueError("At least one negative group is required")

    best = None
    for primary_threshold in np.r_[np.unique(pp), np.inf]:
        for secondary_threshold in np.r_[np.unique(ps), np.inf]:
            positive_accept = (pp >= primary_threshold) & (ps >= secondary_threshold)
            negative_accept = (np_ >= primary_threshold) & (ns >= secondary_threshold)
            group_fars = [
                float(negative_accept[groups == group].mean())
                for group in unique_groups
            ]
            worst_far = max(group_fars)
            if worst_far <= target_far + 1e-12:
                frr = float(1 - positive_accept.mean())
                pooled_far = float(negative_accept.mean())
                candidate = (
                    frr, worst_far, pooled_far,
                    -primary_threshold, -secondary_threshold,
                )
                if best is None or candidate < best[0]:
                    best = (candidate, primary_threshold, secondary_threshold)
    if best is None:
        raise ValueError("No group-robust threshold satisfies the requested FAR")
    (frr, worst_far, pooled_far, _, _), primary_threshold, secondary_threshold = best
    return (
        float(primary_threshold), float(secondary_threshold),
        frr, pooled_far, worst_far,
    )


def and_decision(primary_scores, secondary_scores, primary_threshold, secondary_threshold):
    """Accept only comparisons approved by both independent stages."""

    primary = np.asarray(primary_scores)
    secondary = np.asarray(secondary_scores)
    if primary.shape != secondary.shape:
        raise ValueError("Primary and secondary scores must be aligned")
    return (primary >= primary_threshold) & (secondary >= secondary_threshold)
