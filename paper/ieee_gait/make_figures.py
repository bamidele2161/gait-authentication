"""Generate the four IEEE-paper figures from saved experiment outputs.

Run from the repository root:
    venv/bin/python paper/ieee_gait/make_figures.py

Each figure is written as a 300-dpi PNG.
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
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.close(fig)
    print(f"Saved {OUT / name}.png")


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


def fusion_sweep() -> None:
    # Exploratory measurements provided in the recorded CLI results.
    sweep = pd.DataFrame(
        {
            "span": [2, 4, 6, 11, 16, 21, 31],
            "dtc_frr": [20.95, 18.76, 17.27, 16.33, 15.98, 15.79, 15.42],
            "dtc_far": [14.53, 12.88, 12.02, 11.41, 11.14, 10.89, 10.79],
            "dtf_frr": [45.92, 46.37, 46.72, 47.55, 48.01, 48.48, 48.84],
            "dtf_far": [15.30, 13.83, 13.44, 13.06, 12.84, 12.65, 12.46],
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




# ---------------------------------------------------------------------------
# Schematic figures. Drawn as technical line art: hairline rules, white fills,
# serif type matching the body text, and a single accent for emphasis.
# ---------------------------------------------------------------------------

SERIF = ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]
INK = "#1A1A1A"
HAIR = "#4A4A4A"
FAINT = "#9A9A9A"
WASH = "#EFF3F7"


def _panel(ax, x, y, w, h, label, *, fill="white", lw=0.8, fs=7.6,
           weight="normal", edge=None, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                           edgecolor=edge or HAIR, linewidth=lw, linestyle=ls,
                           zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=INK, fontweight=weight, linespacing=1.35,
            zorder=3)


def _flow(ax, x0, y0, x1, y1, lw=0.8, ls="-", color=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=lw,
                                 linestyle=ls, color=color or HAIR,
                                 shrinkA=0, shrinkB=0, zorder=4))


def _band(ax, x, y, w, h, title):
    """A titled horizontal band that groups one stage."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none",
                           edgecolor=FAINT, linewidth=0.6, zorder=1))
    ax.text(x + 0.10, y + h - 0.16, title, ha="left", va="top",
            fontsize=7.4, color=INK, fontweight="bold", style="italic", zorder=3)


def protocol_timeline() -> None:
    with plt.rc_context({"font.family": "serif", "font.serif": SERIF}):
        fig, ax = plt.subplots(figsize=(7.1, 2.75))
        ax.set_xlim(0, 12); ax.set_ylim(0, 4.75); ax.axis("off")

        # ---- recording timeline -------------------------------------------
        ax.annotate("", xy=(11.95, 4.28), xytext=(0.05, 4.28),
                    arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=0.7))
        ax.text(11.95, 4.42, "time", ha="right", va="bottom",
                fontsize=7, color=FAINT, style="italic")

        for x0, title in ((0.05, "Single-task visit"),
                          (6.55, "Dual-task visit")):
            ax.text(x0, 3.98, title, fontsize=7.6, color=INK, fontweight="bold",
                    va="top")
        for x0, lab in ((0.05, "ST-control"), (2.75, "ST-fatigue"),
                        (6.55, "DT-control"), (9.25, "DT-fatigue")):
            _panel(ax, x0, 3.02, 2.45, 0.62, lab, fs=8)
        for x0 in (2.50, 9.00):
            _flow(ax, x0, 3.33, x0 + 0.25, 3.33)

        # seven-day separation, drawn as a break in the timeline
        for x in (5.35, 5.55):
            ax.plot([x, x + 0.16], [2.94, 3.72], color=FAINT, lw=0.7,
                    solid_capstyle="butt", zorder=3)
        ax.text(5.95, 3.33, "≈ 7 days", ha="center", va="center",
                fontsize=7.2, color=INK, style="italic",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

        # ---- experiment I --------------------------------------------------
        _band(ax, 0.05, 1.58, 11.90, 1.16, "Experiment I · per-user SVM")
        for x0, w, lab in ((0.35, 1.95, "train\n60\\%"),
                           (2.45, 1.45, "validate\n20\\%"),
                           (4.05, 1.45, "test\n20\\%")):
            _panel(ax, x0, 1.74, w, 0.60, lab.replace("\\%", "%"), fs=7.2)
        _flow(ax, 5.60, 2.04, 6.35, 2.04)
        _panel(ax, 6.45, 1.74, 5.30, 0.60,
               "threshold frozen, applied to\nST-fatigue, DT-control, DT-fatigue",
               fill=WASH, fs=7.2)

        # ---- experiment II -------------------------------------------------
        _band(ax, 0.05, 0.16, 11.90, 1.16, "Experiment II · population encoder")
        _panel(ax, 0.35, 0.32, 3.55, 0.60,
               "12 development identities\nST-control + ST-fatigue", fs=7.2)
        _panel(ax, 4.25, 0.32, 3.30, 0.60,
               "4 unseen identities\nenrol: ST-control only", fs=7.2)
        _panel(ax, 7.90, 0.32, 3.85, 0.60,
               "test: DT-control, DT-fatigue", fill=WASH, fs=7.2)
        _flow(ax, 3.95, 0.62, 4.20, 0.62)
        _flow(ax, 7.60, 0.62, 7.85, 0.62)
        save(fig, "protocol_timeline")


def model_pipeline() -> None:
    with plt.rc_context({"font.family": "serif", "font.serif": SERIF}):
        fig, ax = plt.subplots(figsize=(7.1, 3.15))
        ax.set_xlim(0, 12); ax.set_ylim(0, 5.45); ax.axis("off")

        # ---- stage 1: population development -------------------------------
        _band(ax, 0.05, 3.62, 11.90, 1.70,
              "Stage 1 · population development (12 identities per fold)")
        row = 3.86
        boxes = [(0.35, 2.05, "session-1 windows\n256 × 6 axes"),
                 (2.75, 1.85, "vector norms\n256 × 2"),
                 (4.95, 3.05, "shared encoder\nCNN → BiLSTM → attention"),
                 (8.35, 3.40, "batch-hard triplet\n+ auxiliary identity head")]
        for x0, w, lab in boxes:
            _panel(ax, x0, row, w, 0.78, lab, fs=7.2)
        for x0 in (2.40, 4.60, 8.00):
            _flow(ax, x0, row + 0.39, x0 + 0.35, row + 0.39)

        # ---- stage 2: unseen-user verification ------------------------------
        _band(ax, 0.05, 0.08, 11.90, 3.24,
              "Stage 2 · unseen-user verification (4 identities per fold)")
        enrol, probe = 2.28, 1.02
        _panel(ax, 0.35, enrol, 2.45, 0.66,
               "enrolment\nfull ST-control recording", fs=7.2)
        _panel(ax, 3.20, enrol, 2.45, 0.66, "frozen encoder\n(shared weights)", fs=7.0)
        _panel(ax, 6.05, enrol, 2.30, 0.66,
               "64-D embeddings\n\u2192 mean, $L_2$-normalised", fs=7.0)
        _panel(ax, 0.35, probe, 2.45, 0.66,
               "probe\nDT-control or DT-fatigue", fill=WASH, fs=7.2)
        _panel(ax, 3.20, probe, 2.45, 0.66, "frozen encoder\n(shared weights)", fs=7.0)
        _panel(ax, 6.05, probe, 2.30, 0.66,
               "64-D embedding\n(one per window)", fs=7.0)
        for y in (enrol, probe):
            for x0 in (2.80, 5.65):
                _flow(ax, x0, y + 0.33, x0 + 0.40, y + 0.33)
        # converge on the comparison
        _panel(ax, 8.85, 1.72, 2.90, 0.56,
               "Euclidean distance to template", fs=7.2)
        _panel(ax, 8.85, 0.94, 2.90, 0.62,
               "decision: one window (2 s)\nor 10-window mean (11 s)", fs=7.0)
        _flow(ax, 8.35, enrol + 0.33, 8.72, 2.14)
        _flow(ax, 8.35, probe + 0.33, 8.72, 1.90)
        _panel(ax, 8.85, 0.24, 2.90, 0.52, "accept / reject", fill=WASH,
               fs=7.2, weight="bold")
        _flow(ax, 10.30, 1.70, 10.30, 1.56)
        _flow(ax, 10.30, 0.90, 10.30, 0.78)
        ax.text(8.70, 0.50, "threshold frozen on\nsession-1 development scores",
                ha="right", va="center", fontsize=6.6, color=FAINT,
                style="italic", linespacing=1.3)
        save(fig, "model_pipeline")



def participant_errors() -> None:
    """One line per participant: how each person's error moves across conditions."""
    data = pd.read_csv(SVM_CSV).sort_values("participant_id")
    conds = [("ST-control", "baseline"), ("ST-fatigue", "cross_session"),
             ("DT-control", "dt_control"), ("DT-fatigue", "dt_fatigue")]
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), layout="constrained",
                             sharey=True)
    for ax, metric, colour, title in zip(
            axes, ("frr", "far"), (BLUE, ORANGE),
            ("False rejection", "False acceptance")):
        M = np.column_stack([data[f"{metric}_{k}"].to_numpy() for _, k in conds]) * 100
        for row, pid in zip(M, data["participant_id"]):
            ax.plot(x, row, color=colour, alpha=0.32, linewidth=0.9,
                    marker="o", markersize=2.4, zorder=2)
        ax.plot(x, M.mean(0), color=NAVY, linewidth=2.0, marker="o",
                markersize=4.5, zorder=4, label="macro-average")
        # name the participants that end highest, so the tail is identifiable
        order = np.argsort(M[:, 3])[-3:][::-1]        # highest first
        for rank, k in enumerate(order):
            ax.annotate(data["participant_id"].iloc[k].replace("sub_", "P"),
                        (3, M[k, 3]), xytext=(5, -rank * 8.5),
                        textcoords="offset points", fontsize=6.4,
                        color=NAVY, va="center")
        ax.set_xticks(x, [c for c, _ in conds], rotation=18, ha="right", fontsize=7.6)
        ax.set_title(title, weight="bold", color=NAVY)
        ax.set_xlim(-0.25, 3.55)
        ax.set_ylim(-3, 103)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("Error rate (%)")
    save(fig, "svm_participant_errors")

if __name__ == "__main__":
    protocol_timeline()
    participant_errors()
    condition_errors()
    model_pipeline()
    fusion_sweep()
    det_curves()
    score_distributions()
