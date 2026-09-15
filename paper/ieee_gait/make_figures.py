"""Generate the four IEEE-paper figures from saved experiment outputs.

Run from the repository root:
    venv/bin/python paper/ieee_gait/make_figures.py

Each figure is written as vector PDF for LaTeX and a 300-dpi PNG for inspection.
The fusion-sweep values are the recorded exploratory CLI results, not a CSV
produced by the current runner; keep them explicitly identified as such.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
SVM_CSV = ROOT / "src/results/sacrum_time/evaluation_results.csv"
EVAL_SCORES = (
    ROOT
    / "src/results/condition_invariant"
    / "session1_cnn_bilstm_hard/evaluation_scores.csv"
)

NAVY = "#17324D"
BLUE = "#32669A"
TEAL = "#177E86"
ORANGE = "#C77729"
RED = "#B44747"
PALE_BLUE = "#EAF1F8"
PALE_TEAL = "#E6F3F2"
PALE_ORANGE = "#F8EFE3"
GREY = "#626C75"
GRID = "#D8DEE3"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.close(fig)
    print(f"Saved {OUT / name}.pdf and .png")


def box(ax, xy, width, height, label, fc, ec=GRID, fontsize=9, weight="normal"):
    x, y = xy
    ax.add_patch(
        Rectangle((x, y), width, height, facecolor=fc, edgecolor=ec, linewidth=1)
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=NAVY,
        linespacing=1.25,
    )


def arrow(ax, start, end, color=GREY, mutation=11):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=1.15,
            color=color,
        )
    )


def protocol_timeline() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.15))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.text(0.1, 5.05, "Visit 1: single-task walking", weight="bold", color=NAVY)
    ax.text(6.45, 5.05, "Visit 2: dual-task walking", weight="bold", color=NAVY)
    box(ax, (0.12, 3.75), 2.35, 0.78, "ST-control\nbefore fatigue", PALE_BLUE)
    box(ax, (3.05, 3.75), 2.35, 0.78, "ST-fatigue\nafter fatigue", PALE_ORANGE)
    box(ax, (6.52, 3.75), 2.35, 0.78, "DT-control\nbefore fatigue", PALE_TEAL)
    box(ax, (9.45, 3.75), 2.35, 0.78, "DT-fatigue\nafter fatigue", PALE_ORANGE)
    arrow(ax, (2.52, 4.14), (3.0, 4.14))
    arrow(ax, (5.45, 4.14), (6.47, 4.14))
    arrow(ax, (8.92, 4.14), (9.40, 4.14))
    ax.text(5.96, 4.48, "≈ 7 days", ha="center", color=GREY, fontsize=8)

    ax.text(0.12, 2.95, "Experiment I · per-user SVM", weight="bold", color=NAVY)
    box(ax, (0.12, 2.05), 2.15, 0.63, "ST-C train\n60%", PALE_BLUE, fontsize=8)
    box(ax, (2.42, 2.05), 1.65, 0.63, "ST-C validate\n20%", PALE_BLUE, fontsize=8)
    box(ax, (4.22, 2.05), 1.6, 0.63, "ST-C test\n20%", PALE_BLUE, fontsize=8)
    box(ax, (6.32, 2.05), 5.48, 0.63, "Fixed threshold → ST-F / DT-C / DT-F", PALE_TEAL, fontsize=8)
    ax.text(
        0.12,
        1.78,
        "Chronological ST-C partitions have guards against shared raw samples.",
        fontsize=7.5,
        color=GREY,
    )

    ax.text(0.12, 1.23, "Experiment II · population encoder", weight="bold", color=NAVY)
    box(ax, (0.12, 0.35), 3.55, 0.65, "12 development people:\nST-C + ST-F only", PALE_BLUE, fontsize=8)
    box(ax, (4.0, 0.35), 3.55, 0.65, "4 unseen people:\nST-C enrollment only", PALE_BLUE, fontsize=8)
    box(ax, (7.9, 0.35), 3.9, 0.65, "Unseen-user test:\nDT-C + DT-F only", PALE_TEAL, fontsize=8)
    arrow(ax, (3.7, 0.67), (3.95, 0.67))
    arrow(ax, (7.6, 0.67), (7.85, 0.67))
    save(fig, "protocol_timeline")


def participant_errors() -> None:
    data = pd.read_csv(SVM_CSV).sort_values("participant_id")
    conditions = [
        ("ST-control", "baseline"),
        ("ST-fatigue", "cross_session"),
        ("DT-control", "dt_control"),
        ("DT-fatigue", "dt_fatigue"),
    ]
    frr = np.column_stack([data[f"frr_{key}"].to_numpy() for _, key in conditions]) * 100
    far = np.column_stack([data[f"far_{key}"].to_numpy() for _, key in conditions]) * 100
    assert len(data) == 16 and np.isfinite(frr).all() and np.isfinite(far).all()
    fig, axes = plt.subplots(
        1, 2, figsize=(7.1, 4.2), sharey=True, layout="constrained"
    )
    vmax = max(frr.max(), far.max())
    for ax, matrix, title in zip(axes, (frr, far), ("False rejection", "False acceptance")):
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_title(title, weight="bold", color=NAVY)
        ax.set_xticks(range(4), [label for label, _ in conditions], rotation=40, ha="right")
        ax.set_yticks(range(len(data)), data["participant_id"].tolist())
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(len(data)):
            for col in range(4):
                value = matrix[row, col]
                if value >= 15 or (value > 0 and value < 0.1):
                    ax.text(
                        col,
                        row,
                        f"{value:.0f}" if value >= 1 else "<1",
                        ha="center",
                        va="center",
                        fontsize=6.2,
                        color="white" if value >= vmax * 0.48 else NAVY,
                    )
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("Error rate (%)")
    save(fig, "svm_participant_errors")


def model_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.1)
    ax.axis("off")

    ax.text(0.1, 5.65, "Population development (12 people per fold)", weight="bold", color=NAVY)
    box(ax, (0.12, 4.48), 2.2, 0.72, "ST-C + ST-F\nraw IMU", PALE_BLUE, fontsize=8)
    box(ax, (2.72, 4.48), 2.0, 0.72, "Acc/gyr\nmagnitudes", PALE_BLUE, fontsize=8)
    box(ax, (5.12, 4.48), 3.05, 0.72, "Shared CNN → BiLSTM\n→ attention → 64-D", PALE_TEAL, fontsize=8)
    box(ax, (8.57, 4.48), 3.0, 0.72, "Batch-hard triplet\n+ identity loss", PALE_ORANGE, fontsize=8)
    for x0, x1 in ((2.34, 2.68), (4.74, 5.08), (8.19, 8.53)):
        arrow(ax, (x0, 4.84), (x1, 4.84))
    ax.text(0.1, 3.73, "New-user verification (4 unseen people per fold)", weight="bold", color=NAVY)
    box(ax, (0.12, 2.55), 2.45, 0.76, "ST-C only\nnew user", PALE_BLUE, fontsize=8)
    box(ax, (3.0, 2.55), 2.45, 0.76, "Frozen encoder\n+ embedding mean", PALE_TEAL, fontsize=7.6)
    box(ax, (5.88, 2.55), 2.10, 0.76, "Normalized\nuser template", PALE_TEAL, fontsize=8)
    arrow(ax, (2.59, 2.93), (2.96, 2.93))
    arrow(ax, (5.47, 2.93), (5.84, 2.93))

    box(ax, (0.12, 1.13), 2.45, 0.76, "Later DT-C or DT-F\nprobe window", PALE_ORANGE, fontsize=8)
    box(ax, (3.0, 1.13), 2.45, 0.76, "Frozen encoder\n+ 64-D probe", PALE_TEAL, fontsize=8)
    box(ax, (5.88, 1.13), 2.10, 0.76, "Distance to\ntemplate", PALE_TEAL, fontsize=8)
    box(ax, (8.45, 1.13), 3.1, 0.76, "10-score causal mean\n→ fold threshold", PALE_ORANGE, fontsize=8)
    arrow(ax, (2.59, 1.51), (2.96, 1.51))
    arrow(ax, (5.47, 1.51), (5.84, 1.51))
    arrow(ax, (8.00, 1.51), (8.41, 1.51))
    arrow(ax, (6.93, 2.52), (6.93, 1.94))
    ax.text(
        0.12, 0.32,
        "Threshold from held-out development ST-C/ST-F scores (unfused); DT test visits never train or calibrate.",
        color=GREY, fontsize=7.3,
    )
    save(fig, "model_pipeline")


def fusion_sweep() -> None:
    # Exploratory measurements provided in the recorded CLI results.
    sweep = pd.DataFrame(
        {
            "span": [2, 4, 6, 11, 16, 21, 31],
            "dtc_frr": [20.35, 18.11, 16.67, 15.84, 15.50, 15.30, 14.95],
            "dtc_far": [15.01, 13.29, 12.34, 11.42, 11.13, 10.97, 10.79],
            "dtf_frr": [44.97, 45.14, 45.72, 45.99, 46.81, 47.44, 47.95],
            "dtf_far": [15.29, 13.47, 12.86, 12.14, 11.71, 11.30, 10.88],
        }
    )
    # Verify the tabulated 11 s row against the saved per-window scores.
    ev = pd.read_csv(EVAL_SCORES)
    for condition, prefix in (("dt_control", "dtc"), ("dt_fatigue", "dtf")):
        frr, far = [], []
        for claimed, d in ev[ev.condition == condition].groupby("claimed_participant_id"):
            threshold = d.threshold.iloc[0]
            genuine, impostor = [], []
            for probe, rows in d.groupby("probe_participant_id"):
                v = rows.sort_values("start_sample").distance.values
                f = np.convolve(v, np.ones(10) / 10, "valid") if len(v) >= 10 else v
                (genuine if probe == claimed else impostor).append(f)
            genuine, impostor = np.concatenate(genuine), np.concatenate(impostor)
            frr.append((genuine > threshold).mean())
            far.append((impostor <= threshold).mean())
        for metric, observed in (("frr", np.mean(frr) * 100), ("far", np.mean(far) * 100)):
            claimed_value = sweep.loc[sweep["span"] == 11, f"{prefix}_{metric}"].item()
            assert abs(observed - claimed_value) < 0.02, (condition, metric, observed, claimed_value)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), layout="constrained")
    for ax, metric, title in zip(axes, ("frr", "far"), ("False rejection", "False acceptance")):
        ax.plot(
            sweep["span"], sweep[f"dtc_{metric}"], "o-", color=BLUE,
            linewidth=1.6, markersize=4, label="DT-control",
        )
        ax.plot(
            sweep["span"], sweep[f"dtf_{metric}"], "s--", color=ORANGE,
            linewidth=1.6, markersize=4, label="DT-fatigue",
        )
        ax.axvline(11, color=GREY, linewidth=0.9, linestyle=":")
        ax.set_title(title, weight="bold", color=NAVY)
        ax.set_xlabel("First-decision span (s)")
        ax.set_ylabel("Macro-average error rate (%)")
        ax.set_xticks(sweep["span"].tolist())
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.set_xlim(1, 32)
        yvals = np.r_[sweep[f"dtc_{metric}"], sweep[f"dtf_{metric}"]]
        ax.set_ylim(max(0, yvals.min() - 3), yvals.max() + 3)
    save(fig, "fusion_sweep")





def condition_errors() -> None:
    """Problem definition: how error grows outside the enrollment condition."""
    data = pd.read_csv(SVM_CSV)
    conds = [("ST-control\n(baseline)", "baseline"), ("ST-fatigue", "cross_session"),
             ("DT-control", "dt_control"), ("DT-fatigue", "dt_fatigue")]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), layout="constrained", sharey=True)
    rng = np.random.default_rng(0)
    for ax, metric, colour, title in zip(
            axes, ("frr", "far"), (BLUE, ORANGE),
            ("False rejection", "False acceptance")):
        vals = [data[f"{metric}_{k}"].to_numpy() * 100 for _, k in conds]
        means = [v.mean() for v in vals]
        ax.bar(range(4), means, color=colour, width=0.62, zorder=2)
        for x, v in enumerate(vals):                      # every participant
            ax.scatter(x + rng.uniform(-0.17, 0.17, len(v)), v, s=11,
                       color=NAVY, alpha=0.5, zorder=3, linewidths=0)
        for x, m in enumerate(means):
            ax.text(x, m + 3, f"{m:.1f}", ha="center", fontsize=8,
                    color=NAVY, fontweight="bold")
        ax.set_xticks(range(4), [c for c, _ in conds], fontsize=7.5)
        ax.set_title(title, weight="bold", color=NAVY)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    axes[0].set_ylabel("Error rate (%)")
    save(fig, "svm_condition_errors")


def _svm_dt_scores():
    """Recompute SVM decision scores on the two DT conditions."""
    import sys, joblib
    sys.path.insert(0, str(ROOT))
    from src.utils import (DT_CONTROL_FEATURES, DT_FATIGUE_FEATURES,
                           FEATURE_COLS, MODELS_DIR as SVM_MODELS)
    out = {}
    for cond, folder in (("dt_control", DT_CONTROL_FEATURES),
                         ("dt_fatigue", DT_FATIGUE_FEATURES)):
        frames = [pd.read_csv(p) for p in sorted(folder.glob("*_features.csv"))]
        d = pd.concat(frames, ignore_index=True)
        X = d[FEATURE_COLS].to_numpy()
        g, i = [], []
        for path in sorted(SVM_MODELS.glob("*_svm.pkl")):
            pid = path.stem.replace("_svm", "")
            svm = joblib.load(path)
            sc = joblib.load(SVM_MODELS / f"{pid}_scaler.pkl")
            s = svm.decision_function(sc.transform(X))
            own = (d["participant_id"] == pid).to_numpy()
            g.append(s[own]); i.append(s[~own])
        out[cond] = (np.concatenate(g), np.concatenate(i))
    return out


def _encoder_dt_scores(fuse=10):
    ev = pd.read_csv(ROOT / "src/results/condition_invariant"
                     / "session1_cnn_bilstm_hard/evaluation_scores.csv")
    out = {}
    for cond in ("dt_control", "dt_fatigue"):
        g, i = [], []
        for (_, cl), x in ev[ev.condition == cond].groupby(
                ["fold_index", "claimed_participant_id"]):
            for p, s in x.groupby("probe_participant_id"):
                v = s.sort_values("start_sample").distance.values
                f = np.convolve(v, np.ones(fuse) / fuse, "valid") if len(v) >= fuse else v
                (g if p == cl else i).append(-f)      # negate: higher = better match
        out[cond] = (np.concatenate(g), np.concatenate(i))
    return out


def _det(g, i):
    s = np.r_[g, i]; lab = np.r_[np.ones(len(g)), np.zeros(len(i))]
    o = np.argsort(-s); lab = lab[o]
    frr = 1 - np.cumsum(lab) / len(g)
    far = np.cumsum(1 - lab) / len(i)
    return far * 100, frr * 100


def det_curves() -> None:
    """Both systems on one error trade-off curve, per condition."""
    svm, enc = _svm_dt_scores(), _encoder_dt_scores()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), layout="constrained")
    for ax, cond, title in zip(axes, ("dt_control", "dt_fatigue"),
                               ("DT-control", "DT-fatigue")):
        for (g, i), colour, style, name in (
                (svm[cond], ORANGE, "--", "Statistical SVM"),
                (enc[cond], BLUE, "-", "Proposed encoder")):
            far, frr = _det(g, i)
            ax.plot(far, frr, style, color=colour, linewidth=1.8, label=name)
        ax.plot([0, 100], [0, 100], color=GREY, linewidth=0.7, linestyle=":")
        ax.set_title(title, weight="bold", color=NAVY)
        ax.set_xlabel("False acceptance rate (%)")
        ax.set_xlim(0, 60); ax.set_ylim(0, 100)
        ax.grid(color=GRID, linewidth=0.7)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("False rejection rate (%)")
    save(fig, "det_curves")


def score_distributions() -> None:
    """Why DT-fatigue is hard: the genuine and impostor scores overlap."""
    enc = _encoder_dt_scores()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), layout="constrained", sharey=True)
    for ax, cond, title in zip(axes, ("dt_control", "dt_fatigue"),
                               ("DT-control", "DT-fatigue")):
        g, i = (-enc[cond][0], -enc[cond][1])      # back to distances
        bins = np.linspace(0, 2, 70)
        ax.hist(i, bins=bins, color=ORANGE, alpha=0.75, density=True,
                label="Impostor", linewidth=0)
        ax.hist(g, bins=bins, color=BLUE, alpha=0.75, density=True,
                label="Genuine", linewidth=0)
        ax.set_title(title, weight="bold", color=NAVY)
        ax.set_xlabel("Distance to enrollment template")
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Density")
    save(fig, "score_distributions")


if __name__ == "__main__":
    protocol_timeline()
    participant_errors()
    condition_errors()
    model_pipeline()
    fusion_sweep()
    det_curves()
    score_distributions()
