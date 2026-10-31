"""Publication figures for the robust gait authentication paper.

One message per figure; consistent encoding across figures:
  blue = logistic-regression pipeline, orange = SVM pipeline, grey = problem-definition SVM baseline;
  marker = augmentation (circle hand-made, triangle learned drift, square hand + learned drift).

Run from the repository root:
  PYTHONPATH=. venv/bin/python docs/build/paper/paper_figs.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE / "fig"
OUT.mkdir(exist_ok=True)
RES = ROOT / "src" / "results"

# Palette (validated: blue/orange adjacent; aqua/violet all-pairs; never blue+violet together)
LR, SVM, BASE = "#2a78d6", "#eb6834", "#8d8f94"
AQUA, VIOLET, INK, MUTED, GRID = "#1baf7a", "#6250d6", "#1d1d1b", "#6b6a66", "#e4e3df"
COND = ["st_fatigue", "dt_control", "dt_fatigue"]
COND_LABEL = {"st_control": "ST-control\n(held-out)", "st_fatigue": "ST-fatigue", "dt_control": "DT-control",
              "dt_fatigue": "DT-fatigue"}
MARK = {"hand": "o", "drift": "^", "both": "s"}
AUG_LABEL = {"hand": "hand-made", "drift": "learned drift", "both": "hand-made + learned drift"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.edgecolor": "#b5b4ae", "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "legend.frameon": False,
})
FULL, HALF = 7.16, 3.5  # two-column paper widths (inches)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- data
def folders(pred):
    fs = sorted(glob.glob(str(RES / "robust_gait" / "*")))
    out = [f for f in fs if pred(Path(f).name) and (Path(f) / "participant_metrics.csv").exists()]
    if not out:
        raise FileNotFoundError("no result folders matched")
    return out


def load(fs):
    return pd.concat([pd.read_csv(Path(f) / "participant_metrics.csv").assign(seed=i) for i, f in enumerate(fs)])


THREE = lambda n: "probes-st_fatigue-dt_control-dt_fatigue" in n
CONFIGS = {  # (classifier, augmentation) -> folder predicate, three-condition protocol
    ("LR", "hand"): lambda n: THREE(n) and "svm" not in n and "bgcond" not in n and "drift" not in n,
    ("LR", "drift"): lambda n: THREE(n) and "svm" not in n and "_drift4" in n and "aug0a" in n,
    ("LR", "both"): lambda n: THREE(n) and "svm" not in n and "_drift4" in n and "aug4a" in n,
    ("SVM", "hand"): lambda n: THREE(n) and "svm" in n and "drift" not in n and "aug4a" in n,
    ("SVM", "drift"): lambda n: THREE(n) and "svm" in n and "_drift4" in n and "aug0a" in n,
    ("SVM", "both"): lambda n: THREE(n) and "svm" in n and "_drift4" in n and "aug4a" in n,
}
NEG_CTRL = lambda n: THREE(n) and "bgcond-st_control" in n
DATA = {k: load(folders(p)) for k, p in CONFIGS.items()}
DATA[("LR", "hand-ctrlneg")] = load(folders(NEG_CTRL))


def seed_stats(m, metric):
    """Per condition: mean over seeds of the macro (per-user) average, and seed min/max."""
    per_seed = m.groupby(["seed", "condition"])[metric].mean().unstack() * 100
    return per_seed.mean(), per_seed.min(), per_seed.max()


base_all = pd.read_csv(RES / "multisensor_problem_definition_rerun" / "raw_axes_9imu_targetfar0.01" / "participant_metrics.csv")
BASE11 = base_all[base_all.decision == "11s"].rename(columns={"participant_id": "claimed_participant_id"})


# ---------------------------------------------------------------- Fig 1
def fig1_problem_definition():
    """Problem definition: a threshold fixed on normal walking stops holding back impostors."""
    order = ["st_control", "st_fatigue", "dt_control", "dt_fatigue"]
    far = BASE11.pivot(index="claimed_participant_id", columns="condition", values="far")[order] * 100
    frr = BASE11.pivot(index="claimed_participant_id", columns="condition", values="frr")[order] * 100
    fig, ax = plt.subplots(figsize=(HALF, 2.55))
    x = np.arange(4)
    rng = np.random.default_rng(0)
    for i, c in enumerate(order):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(far)), far[c], s=9, color=BASE, alpha=0.55, lw=0, zorder=2)
    ax.plot(x, far.mean().values, "-o", color=INK, lw=1.6, ms=4.5, zorder=3)
    ax.plot(x, frr.mean().values, "-s", color=MUTED, lw=1.2, ms=3.8, zorder=3)
    ax.axhline(1, color=INK, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.15, 81, "dashed: 1 % FAR target, set on held-out ST-control", fontsize=6.5, color=MUTED)
    ax.text(3.08, far.mean().iloc[-1], f"FAR {far.mean().iloc[-1]:.0f} %", va="center", fontsize=7, color=INK)
    ax.text(3.08, frr.mean().iloc[-1] + 1.5, f"FRR {frr.mean().iloc[-1]:.0f} %", va="center", fontsize=7, color=MUTED)
    ax.text(0.15, 88, "dots: individual participants (FAR)", fontsize=6.5, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels([COND_LABEL[c] for c in order])
    ax.set_xlim(-0.35, 3.75); ax.set_ylim(-3, 103)
    ax.set_ylabel("error rate (%)")
    save(fig, "fig1_problem_definition")


# ---------------------------------------------------------------- Fig 2
def fig2_main_comparison():
    """Main result: same sensors, three systems, all three metrics."""
    systems = [("Problem-definition SVM", None, BASE, "D"), ("Our pipeline + SVM", ("SVM", "both"), SVM, "s"),
               ("Our pipeline + LR (final)", ("LR", "both"), LR, "s")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.35), gridspec_kw=dict(wspace=0.32))
    x = np.arange(3)
    for ax, metric, title in zip(axes, ["frr", "far", "eer"],
                                 ["FRR: genuine user rejected", "FAR: impostor accepted", "EER: per-user separability"]):
        for k, (name, key, col, mk) in enumerate(systems):
            off = (k - 1) * 0.22
            if key is None:
                v = BASE11.groupby("condition")[metric].mean()[COND] * 100
                ax.plot(x + off, v.values, mk, color=col, ms=5, label=name, zorder=3)
            else:
                mean, lo, hi = seed_stats(DATA[key], metric)
                mean, lo, hi = mean[COND], lo[COND], hi[COND]
                ax.errorbar(x + off, mean.values, yerr=[mean.values - lo.values, hi.values - mean.values], fmt=mk,
                            color=col, ms=5, elinewidth=1, capsize=0, label=name, zorder=3)
            for xi, vi in zip(x + off, (v if key is None else mean).values):
                ax.vlines(xi, 0, vi, color=col, lw=1.2, alpha=0.35, zorder=2)
        if metric in ("frr", "far"):
            ax.axhline(10, color=INK, lw=0.7, ls=(0, (3, 2)))
            ax.text(2.45, 10.8, "10 %", fontsize=6.5, color=MUTED, ha="right")
        ax.set_xticks(x); ax.set_xticklabels([COND_LABEL[c].replace("-", "-\n", 1) for c in COND])
        ax.set_xlim(-0.5, 2.5); ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("error rate (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    fig.text(0.5, -0.24, "Pipelines: mean of 3 seeds, bars = seed range. Baseline: single run. All systems trained on ST-control only.",
             ha="center", fontsize=6.5, color=MUTED)
    save(fig, "fig2_main_comparison")


# ---------------------------------------------------------------- Fig 3
def fig3_design_ablation():
    """Ablation: classifier x augmentation (and negatives) in the FRR-FAR plane, per condition."""
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, c in zip(axes, COND):
        ax.add_patch(plt.Rectangle((0, 0), 10, 10, color="#efeeea", zorder=0, lw=0))
        ax.text(0.4, 0.4, "both < 10 %", fontsize=6.2, color=MUTED, va="bottom")
        for (clf, aug), m in DATA.items():
            fr, fr_lo, fr_hi = (s[c] for s in seed_stats(m, "frr"))
            fa, fa_lo, fa_hi = (s[c] for s in seed_stats(m, "far"))
            col = LR if clf == "LR" else SVM
            neg = aug == "hand-ctrlneg"
            ax.errorbar(fr, fa, xerr=[[fr - fr_lo], [fr_hi - fr]], yerr=[[fa - fa_lo], [fa_hi - fa]],
                        fmt=MARK["hand" if neg else aug], ms=5.5, color=col, mfc="white" if neg else col,
                        mew=1.1, elinewidth=0.8, capsize=0, zorder=3)
            if (clf, aug) == ("LR", "both"):
                ax.annotate("LR, hand + drift", (fr, fa), (fr + 2.2, fa - 1.6), fontsize=6.5, color=INK,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
        ax.set_title(COND_LABEL[c], loc="left")
        ax.set_xlim(-0.8, 16); ax.set_ylim(0, 11)
        ax.set_xlabel("FRR (%)")
    axes[0].set_ylabel("FAR (%)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=LR, marker="o", ls="", ms=5, label="logistic regression"),
         Line2D([], [], color=SVM, marker="o", ls="", ms=5, label="SVM (RBF)"),
         Line2D([], [], color=MUTED, marker="o", ls="", ms=5, label="hand-made aug."),
         Line2D([], [], color=MUTED, marker="^", ls="", ms=5, label="learned drift"),
         Line2D([], [], color=MUTED, marker="s", ls="", ms=5, label="hand + learned drift"),
         Line2D([], [], color=LR, marker="o", mfc="white", ls="", ms=5, label="LR, negatives ST-control only")]
    fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1), columnspacing=1.0, handletextpad=0.3)
    save(fig, "fig3_design_ablation")


# ---------------------------------------------------------------- Fig 4
def fig4_separability_vs_threshold():
    """Per-user separability (EER) vs error at the deployed threshold: baseline vs final LR."""
    lr = DATA[("LR", "both")].groupby(["claimed_participant_id", "condition"])[["frr", "far", "eer"]].mean().reset_index()
    base = BASE11[BASE11.condition.isin(COND)]
    fig, axes = plt.subplots(1, 2, figsize=(HALF, 2.35), sharey=True, sharex=True, gridspec_kw=dict(wspace=0.1))
    for ax, d, col, title in ((axes[0], base, BASE, "Problem-definition SVM"), (axes[1], lr, LR, "Proposed (LR, final)")):
        worst = np.maximum(d.frr, d.far) * 100
        eer = d.eer * 100
        ax.scatter(eer, worst, s=14, color=col, alpha=0.75, lw=0.4, edgecolor="white", zorder=3)
        at_zero = int(((worst < 0.5) & (eer < 0.5)).sum())
        ax.text(0.97, 0.55, f"{at_zero}/{len(d)} pairs\nboth < 0.5 %", transform=ax.transAxes,
                ha="right", va="center", fontsize=6, color=MUTED)
        dw = d.assign(w=worst)
        r = dw[dw.eer < 0.01].sort_values("w", ascending=False).iloc[0]  # separable, yet wrong at the threshold
        ax.annotate(f"{r.claimed_participant_id}, {COND_LABEL[r.condition]}", (r.eer * 100, r.w),
                    (r.eer * 100 + 8, r.w - 14), fontsize=5.6, color=MUTED,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
        if title.startswith("Problem"):
            r2 = d.sort_values("eer", ascending=False).iloc[0]
            ax.annotate(f"{r2.claimed_participant_id}, {COND_LABEL[r2.condition]}", (r2.eer * 100, max(r2.frr, r2.far) * 100),
                        (r2.eer * 100 - 30, max(r2.frr, r2.far) * 100 - 28), fontsize=5.6, color=MUTED,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
        ax.set_title(title, loc="left", fontsize=7.5); ax.set_xlabel("per-user EER (%)", fontsize=7)
        ax.set_xlim(-2, 70); ax.set_ylim(-3, 103)
    axes[0].set_ylabel("max(FRR, FAR) at deployed\nthreshold (%)", fontsize=7)
    save(fig, "fig4_separability_vs_threshold")


# ---------------------------------------------------------------- Fig 5
def fig5_learned_drift_mechanism():
    """Why learned drift works: it spans where real change happens and covers the DT conditions."""
    sub = pd.read_csv(HERE / "subspace.csv")
    acc = pd.read_csv(HERE / "drift_accuracy.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, 2.45), gridspec_kw=dict(wspace=0.32, width_ratios=[1, 1.15]))
    rng = np.random.default_rng(1)
    for i, c in enumerate(COND):
        d = sub[sub.condition == c]
        for col, key, off in ((VIOLET, "captured_by_learned_span", -0.17), (BASE, "captured_by_random_span", 0.17)):
            v = d[key] * 100
            ax1.scatter(i + off + rng.uniform(-0.06, 0.06, len(v)), v, s=9, color=col, alpha=0.6, lw=0, zorder=2)
            ax1.hlines(v.mean(), i + off - 0.13, i + off + 0.13, color=col, lw=2, zorder=3)
        ax1.text(i - 0.33, d.captured_by_learned_span.mean() * 100, f"{d.captured_by_learned_span.mean()*100:.0f} %",
                 ha="right", va="center", fontsize=6.8, color=INK)
    ax1.text(2.17, 6, "chance", ha="center", fontsize=6.5, color=MUTED)
    ax1.text(1.83, 92, "36 learned directions", ha="center", fontsize=6.5, color=VIOLET)
    ax1.set_xticks(range(3)); ax1.set_xticklabels([COND_LABEL[c] for c in COND]); ax1.set_ylim(0, max(100, sub.captured_by_learned_span.max() * 100 + 5)); ax1.set_xlim(-0.75, 2.45)
    ax1.set_ylabel("test user's real change captured (%)")
    ax1.set_title("(a) Learned drift spans where change happens", loc="left")
    sets = [("none", "no augmentation", BASE, "o"), ("hand", "hand-made", AQUA, "o"),
            ("learned", "learned drift", VIOLET, "^"), ("hand+learned", "hand + learned", INK, "s")]
    y = np.arange(3)[::-1]
    for key, name, col, mk in sets:
        v = [acc[acc.condition == c][f"gen|{key}"].mean() for c in COND]
        ax2.plot(v, y, mk, color=col, ms=5.5, label=name, zorder=3)
    for yi, c in zip(y, COND):
        vs = [acc[acc.condition == c][f"gen|{k}"].mean() for k, *_ in sets]
        ax2.hlines(yi, min(vs), max(vs), color=GRID, lw=3, zorder=1)
    ax2.axvline(acc.floor.mean(), color=INK, lw=0.8, ls=(0, (3, 2)))
    ax2.text(acc.floor.mean() + 0.4, -0.3, "same-session floor", fontsize=6.5, color=MUTED, va="center")
    ax2.set_yticks(y); ax2.set_yticklabels([COND_LABEL[c] for c in COND]); ax2.set_ylim(-0.5, 3.0)
    ax2.set_xlabel("distance from real stride to nearest training stride")
    ax2.set_title("(b) Coverage of real later strides (lower = better)", loc="left")
    ax2.legend(loc="upper right", ncol=2, handletextpad=0.3, columnspacing=0.8, bbox_to_anchor=(1.0, 1.0))
    ax2.set_ylim(-0.5, 3.0)
    ax2.grid(axis="y", visible=False)
    save(fig, "fig5_learned_drift_mechanism")


# ---------------------------------------------------------------- Fig 6
def fig6_representation():
    """Representation ablation (DT-only protocol, LR, hand-made augmentation)."""
    old = lambda n: "probes" not in n
    reps = [("Sacrum only\n(6 raw axes)", lambda n: old(n) and n.startswith("sacrum_late"), "#86b6ef"),
            ("9 IMUs, magnitudes\n(2 per sensor)", lambda n: old(n) and n.startswith("sacrum+") and "_C0.3_mean_" in n, "#3987e5"),
            ("9 IMUs, raw axes\n(6 per sensor)", lambda n: old(n) and n.startswith("sacrum+") and "_axes_" in n, "#184f95")]
    data = [(name, load(folders(p)), col) for name, p, col in reps]
    fig, axes = plt.subplots(3, 1, figsize=(HALF, 4.3), sharex=True, gridspec_kw=dict(hspace=0.35))
    x = np.arange(2)
    for ax, metric, title in zip(axes, ["frr", "far", "eer"], ["FRR", "FAR", "EER"]):
        for k, (name, m, col) in enumerate(data):
            mean, lo, hi = seed_stats(m, metric)
            mean, lo, hi = (s[["dt_control", "dt_fatigue"]] for s in (mean, lo, hi))
            xs = x + (k - 1) * 0.26
            ax.bar(xs, mean.values, 0.24, color=col, label=name, zorder=2)
            ax.errorbar(xs, mean.values, yerr=[mean.values - lo.values, hi.values - mean.values], fmt="none",
                        ecolor=INK, elinewidth=0.7, capsize=0, zorder=3)
            for xi, vi, hv in zip(xs, mean.values, hi.values):
                ax.text(xi, hv + 0.25, f"{vi:.1f}", ha="center", va="bottom", fontsize=6, color=MUTED)
        ax.set_xticks(x); ax.set_xticklabels(["DT-control", "DT-fatigue"])
        ax.set_ylabel(f"{title} (%)"); ax.grid(axis="x", visible=False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend([x.replace("\n", " ") for x in l] and h, [x.replace("\n", " ") for x in l], loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.03), fontsize=6.5)
    save(fig, "fig6_representation_ablation")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["fig1_problem_definition", "fig2_main_comparison", "fig3_design_ablation",
                            "fig4_separability_vs_threshold", "fig5_learned_drift_mechanism", "fig6_representation"]
    for name in todo:
        globals()[name](); print("ok", name)
