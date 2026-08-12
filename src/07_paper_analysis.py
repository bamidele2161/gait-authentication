"""Create BIOSIG paper figures and reproducibility score files."""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (  # noqa: E402
    DT_CONTROL_FEATURES, DT_FATIGUE_FEATURES, EXCLUDED_PARTICIPANTS,
    FEATURE_COLS, FEATURES_DIR, MODELS_DIR, ST_FATIGUE_FEATURES,
)


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
FIGURE_DIR = PAPER_DIR / "figures"
SUPPLEMENT_DIR = PAPER_DIR / "supplement"
RESULTS_PATH = ROOT / "src" / "results" / "sacrum_time" / "evaluation_results.csv"

CONDITIONS = {
    "ST-control": FEATURES_DIR,
    "ST-fatigue": ST_FATIGUE_FEATURES,
    "DT-control": DT_CONTROL_FEATURES,
    "DT-fatigue": DT_FATIGUE_FEATURES,
}


def load_features(directory):
    frames = [pd.read_csv(path) for path in sorted(directory.glob("*_features.csv"))]
    data = pd.concat(frames, ignore_index=True)
    return data[~data["participant_id"].isin(EXCLUDED_PARTICIPANTS)].copy()


def export_system_scores():
    score_frames = []
    loaded = {name: load_features(path) for name, path in CONDITIONS.items()}
    for model_path in sorted(MODELS_DIR.glob("*_svm.pkl")):
        claimed_id = model_path.stem.replace("_svm", "")
        svm = joblib.load(model_path)
        scaler = joblib.load(MODELS_DIR / f"{claimed_id}_scaler.pkl")
        with open(MODELS_DIR / f"{claimed_id}_threshold.json") as handle:
            threshold = float(json.load(handle)["threshold"])
        for condition, data in loaded.items():
            scores = svm.decision_function(scaler.transform(data[FEATURE_COLS].to_numpy()))
            score_frames.append(pd.DataFrame({
                "claimed_id": claimed_id,
                "actual_id": data["participant_id"].to_numpy(),
                "condition": condition,
                "window_index": data["window_index"].to_numpy(),
                "start_sample": data["start_sample"].to_numpy(),
                "genuine": (data["participant_id"] == claimed_id).astype(int).to_numpy(),
                "score": scores,
                "threshold": threshold,
                "accepted": (scores >= threshold).astype(int),
            }))
    scores = pd.concat(score_frames, ignore_index=True)
    SUPPLEMENT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SUPPLEMENT_DIR / "system_scores.csv.gz", index=False, compression="gzip")
    return scores


def bootstrap_ci(values, repetitions=10000, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return np.quantile(samples, [0.025, 0.975])


def plot_macro_errors(results):
    import matplotlib.pyplot as plt

    labels = ["ST-control", "ST-fatigue", "DT-control", "DT-fatigue"]
    keys = ["baseline", "cross_session", "dt_control", "dt_fatigue"]
    frr = np.array([results[f"frr_{key}"].mean() for key in keys])
    far = np.array([results[f"far_{key}"].mean() for key in keys])
    frr_ci = np.array([bootstrap_ci(results[f"frr_{key}"], seed=42+i) for i, key in enumerate(keys)])
    far_ci = np.array([bootstrap_ci(results[f"far_{key}"], seed=52+i) for i, key in enumerate(keys)])

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.bar(x-width/2, frr*100, width, color="#2f6f9f", label="FNMR",
           yerr=np.vstack([(frr-frr_ci[:, 0])*100, (frr_ci[:, 1]-frr)*100]), capsize=3)
    ax.bar(x+width/2, far*100, width, color="#c75b39", label="FMR",
           yerr=np.vstack([(far-far_ci[:, 0])*100, (far_ci[:, 1]-far)*100]), capsize=3)
    ax.set_ylabel("Error rate (%)")
    ax.set_xticks(x, labels)
    # Leave headroom for the participant-bootstrap interval above the 51% mean
    # DT-fatigue FNMR in the final 16-participant experiment.
    ax.set_ylim(0, 70)
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "macro_errors.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "macro_errors.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_participant_errors(results):
    import matplotlib.pyplot as plt

    labels = ["ST-C", "ST-F", "DT-C", "DT-F"]
    keys = ["baseline", "cross_session", "dt_control", "dt_fatigue"]
    participants = results["participant_id"].str.replace("sub_", "S", regex=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), sharey=True)
    for ax, metric, title in zip(axes, ["frr", "far"], ["FNMR", "FMR"]):
        values = results[[f"{metric}_{key}" for key in keys]].to_numpy() * 100
        image = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
        ax.set_xticks(range(4), labels)
        ax.set_yticks(range(len(participants)), participants)
        ax.set_title(title)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                ax.text(column, row, f"{value:.0f}", ha="center", va="center",
                        fontsize=6, color="white" if value > 55 else "black")
    fig.colorbar(image, ax=axes, label="Error rate (%)", fraction=.03, pad=.03)
    fig.subplots_adjust(left=.10, right=.90, bottom=.12, top=.90, wspace=.12)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "participant_errors.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "participant_errors.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    scores = export_system_scores()
    print(f"Saved {len(scores):,} comparison scores to {SUPPLEMENT_DIR}")
    if "--scores-only" in sys.argv:
        return
    results = pd.read_csv(RESULTS_PATH)
    plot_macro_errors(results)
    plot_participant_errors(results)
    print(f"Saved paper figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
