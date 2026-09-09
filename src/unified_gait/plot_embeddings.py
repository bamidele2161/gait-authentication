"""Plot and quantify previously extracted frozen embeddings."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.unified_gait.config import CONDITIONS, RESULT_DIR


def between_group_variance_ratio(vectors, labels):
    overall = vectors.mean(axis=0)
    total = np.square(vectors - overall).sum()
    between = 0.0
    for label in np.unique(labels):
        group = vectors[labels == label]
        between += len(group) * np.square(group.mean(axis=0) - overall).sum()
    return float(between / max(total, 1e-12))


def run(fold=1):
    root = RESULT_DIR / f"fold_{fold}" / "diagnostics"
    rows = []
    for population in ("development", "evaluation"):
        vectors = np.load(root / f"embeddings_{population}.npy")
        labels = pd.read_csv(root / f"embedding_labels_{population}.csv")
        centered = vectors - vectors.mean(axis=0)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        coordinates = centered @ right[:2].T
        figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        for participant in sorted(labels.participant.unique()):
            selected = labels.participant.eq(participant).to_numpy()
            axes[0].scatter(*coordinates[selected].T, s=10, alpha=.65, label=participant)
        axes[0].set_title("Colour by participant")
        axes[0].legend(fontsize=7, ncol=2)
        for condition in CONDITIONS:
            selected = labels.condition.eq(condition).to_numpy()
            axes[1].scatter(*coordinates[selected].T, s=10, alpha=.65, label=condition)
        axes[1].set_title("Colour by condition")
        axes[1].legend(fontsize=8)
        for axis in axes:
            axis.set_xlabel("principal component 1")
            axis.set_ylabel("principal component 2")
        figure.suptitle(f"Frozen invariant encoder: {population} participants")
        figure.savefig(root / f"pca_{population}.png", dpi=220)
        plt.close(figure)
        rows.append({
            "population": population,
            "identity_between_variance_ratio": between_group_variance_ratio(
                vectors, labels.participant.to_numpy()
            ),
            "condition_between_variance_ratio": between_group_variance_ratio(
                vectors, labels.condition.to_numpy()
            ),
            "sample_count": len(labels),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "variance_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"PCA figures saved to {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    run(args.fold)


if __name__ == "__main__":
    main()
