"""Figures: how closely each augmentation matches the real drift of each condition.

For every test user and changed condition, the measured drift (mean stride in the
condition minus mean ST-control stride; never used for training) is compared with
the variation that an augmentation adds to the same user's enrollment
(augmented copy minus original stride):

- top row: RMS of the measured drift of the 16 test users per IMU and gait phase;
- bottom row: RMS of the augmentation's variation, in the same units; for the
  CVAE and the drift bank only the vectors of that condition (30 generated and
  12 measured vectors), scaled as in training; hand-crafted augmentation has
  no condition, so its map is the same in every column;
- r: Pearson correlation between the two maps (where and when drift happens);
- coverage: median share of a user's measured drift lying in the first 30
  principal directions of the augmentation's variation as used in training
  (all copies of that user; 30 random directions cover about 2%).

Units: standard deviation of each axis over the development users' normal strides.
Drift sources and scaling use each fold's development users only.

    PYTHONPATH=. venv/bin/python docs/build/paper/augmentation_correctness.py OUTDIR
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from src.robust_gait import run_experiment as R
from src.robust_gait.data import CONDITIONS, LOCATIONS, load_strides
from src.robust_gait.protocol import create_outer_folds
from src.robust_gait.verifier import VerifierConfig, augment_strides, drift_strides

NAMES = list(LOCATIONS)
AXES = [2 * len(NAMES) + 6 * l + j for l in range(len(NAMES)) for j in range(6)]
LATER = R.DRIFT_CONDITIONS
TITLES = ["ST-fatigue", "DT-control", "DT-fatigue"]
SOURCES = {"hand": "Hand-crafted augmentation", "cvae": "CVAE drift", "bank": "Drift bank"}
ROWS = ["head", "sternum", "sacrum", "left_wrist", "right_wrist",
        "left_leg", "right_leg", "left_foot", "right_foot"]
LABELS = {"head": "Head", "sternum": "Sternum", "sacrum": "Sacrum",
          "left_wrist": "L wrist", "right_wrist": "R wrist", "left_leg": "L shank",
          "right_leg": "R shank", "left_foot": "L foot", "right_foot": "R foot"}
RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
        "#256abf", "#184f95", "#0d366b"]
INK, MUTED = "#0b0b0b", "#52514e"
COPIES, DIRECTIONS = 4, 30

plt.rcParams.update({
    "font.family": "serif", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
})

cfg = VerifierConfig(locations=tuple(NAMES), enrolment_augmentation=COPIES, amplitude_range=0.2,
                     warp_range=0.1, drift_augmentation=COPIES)
exp = R.ExperimentConfig(verifier=cfg, drift_model="cvae")
people = R.participants()
data = {(p, c): load_strides(p, c) for p in people for c in CONDITIONS}


def pool(v):  # (n, 128, 54) -> (n, 32, 54), as the classifier features
    return v.reshape(len(v), 32, 4, v.shape[2]).mean(axis=2)


def rms_map(v):  # (n, 32, 54) -> (9 IMUs in ROWS order, 32 phases)
    m = np.sqrt((v.reshape(len(v), 32, len(NAMES), 6) ** 2).mean(axis=(0, 3))).T
    return m[[NAMES.index(r) for r in ROWS]]


measured = {c: [] for c in LATER}
variation = {(s, c): [] for s in SOURCES for c in LATER}
coverage = {(s, c): [] for s in SOURCES for c in LATER}
for fold in create_outer_folds(people, 4):
    dev, test = list(fold.development_participants), list(fold.evaluation_participants)
    scale = pool(np.concatenate([data[p, "st_control"].strides[:, :, AXES] for p in dev])).std((0, 1))
    banks = {"cvae": R.drift_cvae(dev, data, exp), "bank": R.drift_bank(dev, data, exp)}
    drifts = R.measured_drifts(test + dev, data)
    for u in test:
        enrol = data[u, "st_control"].strides
        n = len(enrol)
        tiled = np.concatenate([enrol] * COPIES)
        diffs = {
            "hand": augment_strides(enrol, COPIES, cfg, np.random.default_rng(42))[n:] - tiled,
            "cvae": drift_strides(enrol, banks["cvae"], COPIES, cfg, np.random.default_rng(43)) - tiled,
            "bank": drift_strides(enrol, banks["bank"], COPIES, cfg, np.random.default_rng(43)) - tiled,
        }
        diffs = {k: pool(v[:, :, AXES]) / scale for k, v in diffs.items()}
        basis = {}
        for k, v in diffs.items():
            flat = v.reshape(len(v), -1).astype(np.float64)
            _, s, vt = np.linalg.svd(flat, full_matrices=False)
            basis[k] = vt[:min(DIRECTIONS, int((s > 1e-6 * s[0]).sum()))]
        for i, c in enumerate(LATER):
            d = pool(drifts[u, c][None][:, :, AXES]) / scale
            measured[c].append(d)
            flat = d.ravel().astype(np.float64)
            per_condition = {
                "hand": diffs["hand"],
                "cvae": banks["cvae"][i * 30:(i + 1) * 30],
                "bank": banks["bank"][i::3],
            }
            for k in SOURCES:
                if k == "hand":
                    variation[k, c].append(per_condition[k])
                else:
                    v = drift_strides(enrol, per_condition[k], COPIES, cfg,
                                      np.random.default_rng(44)) - tiled
                    variation[k, c].append(pool(v[:, :, AXES]) / scale)
                p = basis[k] @ flat
                coverage[k, c].append(p @ p / (flat @ flat))
    print("fold", fold.fold_index + 1, "done", flush=True)

maps = {c: rms_map(np.concatenate(measured[c])) for c in LATER}
smaps = {key: rms_map(np.concatenate(v)) for key, v in variation.items()}
vmax = np.percentile(np.concatenate([m.ravel() for m in list(maps.values()) + list(smaps.values())]), 99)
cmap = LinearSegmentedColormap.from_list("blue", RAMP)
out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)
rows = []
for name, title in SOURCES.items():
    fig = plt.figure(figsize=(7.16, 3.3))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.12, hspace=0.42,
                            left=0.15, right=0.93, bottom=0.12, top=0.86)
    for k, c in enumerate(LATER):
        r = np.corrcoef(maps[c].ravel(), smaps[name, c].ravel())[0, 1]
        cov = np.median(coverage[name, c])
        rows.append({"source": name, "condition": c, "r": r, "coverage": cov,
                     "measured_rms": maps[c].mean(), "augmentation_rms": smaps[name, c].mean()})
        for row, (m, label) in enumerate(((maps[c], "Measured drift\n(test users)"),
                                          (smaps[name, c], title.replace(" augmentation", "\naugmentation")
                                           .replace(" drift", "\ndrift").replace("Drift bank", "Drift\nbank")))):
            ax = fig.add_subplot(grid[row, k])
            im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                           extent=(0, 100, len(ROWS) - 0.5, -0.5), interpolation="nearest")
            ax.set_yticks(range(len(ROWS)))
            ax.set_yticklabels([LABELS[x] for x in ROWS] if k == 0 else [])
            ax.set_xticks([0, 50, 100])
            ax.set_xticklabels(["0", "50", "100"] if row == 1 else [])
            ax.tick_params(length=2, width=0.5)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(TITLES[k], color=INK, pad=3)
            else:
                ax.set_xlabel("Gait phase (%)")
                ax.set_title(f"r = {r:.2f}, coverage = {cov:.0%}", color=INK, fontsize=6.8, pad=3)
            if k == 0:
                ax.text(-0.36, 0.5, label, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", color=INK, fontsize=7)
    cax = fig.add_subplot(grid[:, 3])
    bar = fig.colorbar(im, cax=cax)
    bar.set_label("RMS (SD of normal walking)", labelpad=3)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2, width=0.5)
    fig.text(0.15, 0.965, f"{title} vs. measured drift", fontsize=8, color=INK, ha="left", va="top")
    fig.savefig(out / f"correctness_{name}.pdf")
    fig.savefig(out / f"correctness_{name}.png", dpi=200)
    plt.close(fig)

summary = pd.DataFrame(rows)
summary.to_csv(out / "augmentation_correctness.csv", index=False)
print(summary.round(3).to_string(index=False))

# Combined figure for the paper: per-IMU profile of the measured drift (bars)
# and of each augmentation (markers), one panel per condition.
SHORT = {"head": "Head", "sternum": "Stern.", "sacrum": "Sacr.", "left_wrist": "L wr.",
         "right_wrist": "R wr.", "left_leg": "L sh.", "right_leg": "R sh.",
         "left_foot": "L ft.", "right_foot": "R ft."}
SERIES = [("hand", "Hand-crafted", "#1baf7a", "s"), ("bank", "Drift bank", "#eb6834", "^"),
          ("cvae", "CVAE", "#2a78d6", "o")]
stats = summary.set_index(["source", "condition"])
x = np.arange(len(ROWS))
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3), sharey=True)
fig.subplots_adjust(left=0.07, right=0.995, bottom=0.2, top=0.8, wspace=0.06)
top = max(max(m.mean(1).max() for m in maps.values()),
          max(m.mean(1).max() for m in smaps.values()))
for ax, k, c in zip(axes, range(3), LATER):
    ax.bar(x, maps[c].mean(1), width=0.72, color="#d6d5d0", edgecolor="none", zorder=1)
    for offset, (name, label, color, marker) in zip((-0.2, 0.0, 0.2), SERIES):
        ax.plot(x + offset, smaps[name, c].mean(1), ls="", marker=marker, markersize=4.2,
                color=color, markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    lines = [f"{label:<12s} r = {stats.loc[(name, c)].r:.2f}, {stats.loc[(name, c)].coverage:.0%}"
             for name, label, _, _ in SERIES]
    ax.text(0.98, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=5.8, color=INK, family="monospace", linespacing=1.35)
    ax.set_title(TITLES[k], color=INK, pad=3)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[r] for r in ROWS], rotation=90, fontsize=6)
    ax.set_xlim(-0.6, len(ROWS) - 0.4)
    ax.set_ylim(0, top * 1.55)
    ax.tick_params(length=2, width=0.5)
    ax.grid(axis="y", color="#e4e3df", lw=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("RMS drift (SD of normal walking)")
from matplotlib.patches import Patch
handles = [Patch(color="#d6d5d0", label="Measured drift (test users)")] + [
    Line2D([], [], ls="", marker=m, color=col, markersize=4.5, label=lab)
    for _, lab, col, m in SERIES]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.53, 1.0), handletextpad=0.3, columnspacing=1.4, fontsize=6.5)
fig.savefig(out / "correctness_all.pdf")
fig.savefig(out / "correctness_all.png", dpi=200)

# Single-column figures in the style of the DET plot (one per condition):
# per-IMU drift profile of the measured drift and of each augmentation, with
# r and coverage in the legend.
STYLE = [("hand", "Hand-crafted only", "#1baf7a", "s"), ("bank", "Drift bank", "#eb6834", "^"),
         ("cvae", "CVAE drift", "#2a78d6", "o")]
FILE = {"st_fatigue": "stfatigue", "dt_control": "dtcontrol", "dt_fatigue": "dtfatigue"}
for c in LATER:
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.plot(x, maps[c].mean(1), color="#0b0b0b", lw=1.6, marker="o", ms=3,
            label="Measured drift (test users)")
    for name, label, color, marker in STYLE:
        st = stats.loc[(name, c)]
        ax.plot(x, smaps[name, c].mean(1), color=color, lw=1.2, marker=marker, ms=3.5,
                label=f"{label} (r = {st.r:.2f}, cov. {st.coverage:.0%})")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[r] for r in ROWS], rotation=90)
    ax.set_ylabel("RMS drift (SD of normal walking)")
    ax.set_ylim(0, 1.5 * maps[c].mean(1).max())
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.grid(alpha=.3, lw=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"correctness_{FILE[c]}.pdf")
    fig.savefig(out / f"correctness_{FILE[c]}.png", dpi=200)
    plt.close(fig)

# The three conditions in one image (DET style, shared y-axis).
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7), sharey=True)
ymax = 1.55 * max(maps[c].mean(1).max() for c in LATER)
for ax, k, c in zip(axes, range(3), LATER):
    ax.plot(x, maps[c].mean(1), color="#0b0b0b", lw=1.6, marker="o", ms=3, label="Measured drift")
    for name, label, color, marker in STYLE:
        st = stats.loc[(name, c)]
        ax.plot(x, smaps[name, c].mean(1), color=color, lw=1.2, marker=marker, ms=3.2,
                label=f"{label.replace(' only', '')} (r = {st.r:.2f}, {st.coverage:.0%})")
    ax.set_title(TITLES[k], fontsize=7.5, pad=3)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[r] for r in ROWS], rotation=90, fontsize=6)
    ax.set_ylim(0, ymax)
    ax.legend(fontsize=5.6, frameon=False, loc="upper right", handlelength=1.6, labelspacing=0.3)
    ax.grid(alpha=.3, lw=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("RMS drift (SD of normal walking)")
fig.tight_layout(w_pad=0.6)
fig.savefig(out / "correctness_conditions.png", dpi=300)
fig.savefig(out / "correctness_conditions.pdf")
plt.close(fig)
