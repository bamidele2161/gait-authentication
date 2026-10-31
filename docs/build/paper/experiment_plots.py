"""Per-experiment biometric plots (score distributions, FAR/FRR vs threshold, ROC/AUC,
per-user error map) and cross-experiment DET / AUC-EER comparisons.

Scores are threshold-aligned: margin = decision score − the threshold that system actually
used (fold threshold for our pipeline, personal threshold for the baseline), so
"accept" is always margin ≥ 0 and pooled curves describe the deployed system.

Run from the repository root:
  PYTHONPATH=. venv/bin/python docs/build/paper/experiment_plots.py [system ...]
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE / "fig" / "experiments"
RG = ROOT / "src" / "results" / "robust_gait"

INK, MUTED, GRID = "#1d1d1b", "#6b6a66", "#e4e3df"
LR, SVM, BASE = "#2a78d6", "#eb6834", "#8d8f94"
GEN, IMP = "#2a78d6", "#8d8f94"               # genuine / impostor in score plots
FRR_C, FAR_C = "#eb6834", "#1d1d1b"           # error-type curves
COND_C = {"st_control": "#8d8f94", "st_fatigue": "#1baf7a", "dt_control": "#2a78d6", "dt_fatigue": "#eb6834"}
LABEL = {"st_control": "ST-control (held-out)", "st_fatigue": "ST-fatigue", "dt_control": "DT-control",
         "dt_fatigue": "DT-fatigue"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.edgecolor": "#b5b4ae", "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "savefig.dpi": 300, "legend.frameon": False,
})

SUFFIX = "_probes-st_fatigue-dt_control-dt_fatigue"
SYSTEMS = {  # name -> (title, colour, scores source)
    "baseline_svm": ("Problem definition: statistical SVM (9 IMUs)", BASE, "baseline"),
    "lr_hand": ("Our pipeline + LR, hand-made augmentation", LR, "*_aug4a*_fuse10" + SUFFIX),
    "lr_drift": ("Our pipeline + LR, learned drift", LR, "*_aug0a*_fuse10_drift4" + SUFFIX),
    "lr_both": ("Our pipeline + LR, hand-made + learned drift (final)", LR, "*_aug4a*_fuse10_drift4" + SUFFIX),
    "svm_hand": ("Our pipeline + SVM, hand-made augmentation", SVM, "*_aug4a*_fuse10_svmC0.1g0.01" + SUFFIX),
    "svm_drift": ("Our pipeline + SVM, learned drift", SVM, "*_aug0a*_fuse10_drift4_svmC0.1g0.01" + SUFFIX),
    "svm_both": ("Our pipeline + SVM, hand-made + learned drift", SVM, "*_aug4a*_fuse10_drift4_svmC0.1g0.01" + SUFFIX),
}


def load_trials(system):
    """DataFrame of trials: claimed, owner, condition, margin (score - threshold)."""
    src = SYSTEMS[system][2]
    rows = []
    if src == "baseline":
        z = np.load(HERE / "baseline_scores.npz")
        for k in z.files:
            parts = k.split("|")
            if parts[0] != "11s":
                continue
            _, claimed, owner, cond = parts
            t = z[f"threshold|11s|{claimed}"][0]
            rows.append((claimed, owner, cond, z[k] - t))
    else:
        hits = [p for p in glob.glob(str(RG / src)) if (Path(p) / "scores.npz").exists()]
        if len(hits) != 1:
            raise FileNotFoundError(f"{system}: expected one scores.npz, found {len(hits)}")
        z = np.load(Path(hits[0]) / "scores.npz")
        for k in z.files:
            if k.startswith("threshold|"):
                continue
            fold, claimed, owner, cond = k.split("|")
            rows.append((claimed, owner, cond, z[k] - z[f"threshold|{fold}"][0]))
    return pd.DataFrame(rows, columns=["claimed", "owner", "condition", "margin"])


def pooled(trials, cond):
    d = trials[trials.condition == cond]
    g = np.concatenate(d[d.claimed == d.owner].margin.values)
    i = np.concatenate(d[d.claimed != d.owner].margin.values)
    return g, i


def eer_point(g, i):
    labels = np.r_[np.ones(len(g)), np.zeros(len(i))]
    fpr, tpr, thr = roc_curve(labels, np.r_[g, i])
    fnr = 1 - tpr
    k = int(np.argmin(np.abs(fpr - fnr)))
    return (fpr[k] + fnr[k]) / 2, thr[k]


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def conditions(trials):
    order = ["st_control", "st_fatigue", "dt_control", "dt_fatigue"]
    return [c for c in order if c in set(trials.condition)]


def plot_system(system):
    title, colour, _ = SYSTEMS[system]
    trials = load_trials(system)
    conds = conditions(trials)
    folder = OUT / system
    summary = []

    # 1. score distributions relative to the deployed threshold
    fig, axes = plt.subplots(1, len(conds), figsize=(7.16, 2.1), sharey=False, gridspec_kw=dict(wspace=0.18))
    for ax, c in zip(np.atleast_1d(axes), conds):
        g, i = pooled(trials, c)
        lo, hi = np.percentile(np.r_[g, i], [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax.hist(i, bins, density=True, color=IMP, alpha=0.55, label="impostor")
        ax.hist(g, bins, density=True, color=GEN, alpha=0.6, label="genuine")
        ax.axvline(0, color=INK, lw=1)
        ax.set_title(f"{LABEL[c]}\nFRR {np.mean(g < 0)*100:.1f} %  ·  FAR {np.mean(i >= 0)*100:.1f} %", loc="left", fontsize=7.5)
        ax.set_yticks([]); ax.set_xlabel("score − threshold")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    fig.suptitle(f"{title}: score distributions (accept if ≥ 0)", x=0.08, ha="left", y=1.08, fontsize=8.5, weight="bold")
    save(fig, folder, "1_score_distributions")

    # 2. FAR / FRR against threshold offset
    fig, axes = plt.subplots(1, len(conds), figsize=(7.16, 2.1), sharey=True, gridspec_kw=dict(wspace=0.1))
    for ax, c in zip(np.atleast_1d(axes), conds):
        g, i = pooled(trials, c)
        lo, hi = np.percentile(np.r_[g, i], [1, 99])
        t = np.linspace(lo, hi, 400)
        frr = np.array([np.mean(g < x) for x in t]) * 100
        far = np.array([np.mean(i >= x) for x in t]) * 100
        ax.plot(t, frr, color=FRR_C, lw=1.6, label="FRR")
        ax.plot(t, far, color=FAR_C, lw=1.6, label="FAR")
        e, et = eer_point(g, i)
        ax.plot([et], [e * 100], "o", color=INK, ms=4, mfc="white", mew=1)
        ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        ax.set_title(f"{LABEL[c]}\npooled EER {e*100:.1f} %", loc="left", fontsize=7.5); ax.set_xlabel("threshold offset")
    np.atleast_1d(axes)[0].set_ylabel("error rate (%)")
    np.atleast_1d(axes)[0].legend(loc="center left")
    fig.suptitle(f"{title}: FAR and FRR vs threshold (dashed = deployed, ○ = EER)", x=0.08, ha="left", y=1.04, fontsize=8.5, weight="bold")
    save(fig, folder, "2_far_frr_vs_threshold")

    # 3. ROC with AUC and deployed operating point
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    for c in conds:
        g, i = pooled(trials, c)
        labels = np.r_[np.ones(len(g)), np.zeros(len(i))]
        fpr, tpr, _ = roc_curve(labels, np.r_[g, i])
        auc = roc_auc_score(labels, np.r_[g, i])
        ax.plot(fpr * 100, tpr * 100, color=COND_C[c], lw=1.6, label=f"{LABEL[c]}  (AUC {auc:.4f})")
        ax.plot(np.mean(i >= 0) * 100, np.mean(g >= 0) * 100, "o", color=COND_C[c], ms=4.5, mec="white", mew=0.8)
        e, _ = eer_point(g, i)
        summary.append(dict(system=system, condition=c, auc=auc, pooled_eer=e * 100,
                            frr=np.mean(g < 0) * 100, far=np.mean(i >= 0) * 100))
    ax.plot([0, 100], [0, 100], color=GRID, lw=0.8)
    ax.set_xlim(-1, 60); ax.set_ylim(40, 101)
    ax.set_xlabel("FAR (%)"); ax.set_ylabel("genuine acceptance rate (%)")
    ax.set_title("ROC (dots = deployed threshold)", loc="left")
    ax.legend(loc="lower right")
    fig.suptitle(title, x=0.12, ha="left", y=1.02, fontsize=8, weight="bold")
    save(fig, folder, "3_roc_auc")

    # 4. per-user FRR / FAR map
    rows = []
    for (claimed, c), d in trials.groupby(["claimed", "condition"]):
        g = np.concatenate(d[d.owner == claimed].margin.values)
        i = np.concatenate(d[d.owner != claimed].margin.values)
        rows.append(dict(user=claimed, condition=c, FRR=np.mean(g < 0) * 100, FAR=np.mean(i >= 0) * 100))
    pu = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 3.6), gridspec_kw=dict(wspace=0.08))
    for ax, metric in zip(axes, ["FRR", "FAR"]):
        mat = pu.pivot(index="user", columns="condition", values=metric)[conds]
        im = ax.imshow(mat.values, cmap="Blues" if metric == "FRR" else "Greys", vmin=0, vmax=100, aspect="auto")
        for (r, cc), v in np.ndenumerate(mat.values):
            if v >= 0.5:
                ax.text(cc, r, f"{v:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 55 else INK)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([LABEL[c].split(" ")[0] for c in conds], rotation=30, ha="right")
        ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.index if metric == "FRR" else [])
        ax.set_title(f"{metric} per user (%)", loc="left"); ax.grid(False)
    fig.suptitle(f"{title}\nper-user errors at the deployed threshold (blank = 0 %)", x=0.12, ha="left", y=1.03,
                 fontsize=8, weight="bold")
    save(fig, folder, "4_per_user_errors")
    pu.to_csv(folder / "per_user_errors.csv", index=False)
    return pd.DataFrame(summary)


def comparison(summaries):
    """DET curves of all available systems per condition, and AUC / pooled-EER summary."""
    avail = [s for s in SYSTEMS if s in summaries]
    conds = ["st_fatigue", "dt_control", "dt_fatigue"]
    style = {"hand": "-", "drift": (0, (4, 2)), "both": "-"}
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7), sharey=True, gridspec_kw=dict(wspace=0.08))
    ticks = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 40, 60])
    for ax, c in zip(axes, conds):
        for s in avail:
            g, i = pooled(load_trials(s), c)
            labels = np.r_[np.ones(len(g)), np.zeros(len(i))]
            fpr, tpr, _ = roc_curve(labels, np.r_[g, i])
            fnr = 1 - tpr
            keep = (fpr > 5e-4) & (fnr > 5e-4) & (fpr < 0.7) & (fnr < 0.7)
            col = SYSTEMS[s][1]
            kind = s.split("_")[-1]
            lw = 2.0 if s in ("lr_both", "baseline_svm") else 1.0
            alpha = 1.0 if s in ("lr_both", "baseline_svm", "svm_both") else 0.55
            ax.plot(norm.ppf(fpr[keep]), norm.ppf(fnr[keep]), color=col, lw=lw, alpha=alpha,
                    ls=style.get(kind, "-") if s != "baseline_svm" else "-",
                    label=SYSTEMS[s][0].replace("Our pipeline + ", "").replace("Problem definition: ", ""))
            ax.plot(norm.ppf(np.clip(np.mean(i >= 0), 5e-4, 0.7)), norm.ppf(np.clip(np.mean(g < 0), 5e-4, 0.7)), "o",
                    color=col, ms=3.5 if lw < 2 else 5, alpha=alpha, mec="white", mew=0.6)
        ax.set_xticks(norm.ppf(ticks / 100)); ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_yticks(norm.ppf(ticks / 100)); ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.set_xlim(norm.ppf(0.0007), norm.ppf(0.65)); ax.set_ylim(norm.ppf(0.0007), norm.ppf(0.65))
        ax.plot([-4, 1], [-4, 1], color=GRID, lw=0.8)
        ax.set_title(LABEL[c], loc="left"); ax.set_xlabel("FAR (%)")
    axes[0].set_ylabel("FRR (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2), fontsize=6.5)
    save(fig, OUT, "compare_det_curves")

    table = pd.concat([summaries[s] for s in avail])
    table = table[table.condition.isin(conds)]
    table.round(4).to_csv(OUT / "summary_auc_eer.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw=dict(wspace=0.3))
    x = np.arange(3)
    for ax, metric, lab in ((axes[0], "auc", "pooled AUC"), (axes[1], "pooled_eer", "pooled EER (%)")):
        w = 0.8 / len(avail)
        for k, s in enumerate(avail):
            v = table[table.system == s].set_index("condition").loc[conds, metric].values
            ax.bar(x + (k - (len(avail) - 1) / 2) * w, v, w * 0.92, color=SYSTEMS[s][1],
                   alpha=1.0 if s.endswith("both") or s == "baseline_svm" else 0.5,
                   label=SYSTEMS[s][0].replace("Our pipeline + ", "").replace("Problem definition: ", ""))
        ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in conds]); ax.set_title(lab, loc="left")
        ax.grid(axis="x", visible=False)
        if metric == "auc":
            ax.set_ylim(max(0.8, table.auc.min() - 0.02), 1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.22), fontsize=6.5)
    fig.suptitle("Threshold-free comparison on threshold-aligned scores (faded = single augmentation)", x=0.08,
                 ha="left", y=1.03, fontsize=8.5, weight="bold")
    save(fig, OUT, "compare_auc_eer")


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(SYSTEMS)
    summaries = {}
    for s in SYSTEMS:
        try:
            summaries[s] = plot_system(s) if s in wanted else None
        except FileNotFoundError as e:
            print("skip", s, "-", e)
            continue
        if summaries[s] is None:
            del summaries[s]
        else:
            print("ok", s)
    if len(summaries) > 1:
        comparison(summaries)
        print("ok comparison", list(summaries))
