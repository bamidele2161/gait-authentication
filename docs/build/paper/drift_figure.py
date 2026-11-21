"""Figure: what the drift CVAE learned (fold 1).

(a) Where each changed condition moves the stride: RMS of the CVAE-generated
    drift per IMU and gait phase, in units of the development users' normal
    stride-to-stride standard deviation of each axis.
(b) Generated drift vs. the measured drift of the fold's test users (never
    seen by the CVAE; one mean drift per user and condition), projected on the
    first two principal components of the generated vectors.

Strides are mean-pooled to 32 phase points, as for the classifiers.

Leakage-free: the CVAE, the PCA and the scaling use the fold's 12 development
users only; the test users' recordings appear only as the comparison target.

    PYTHONPATH=. venv/bin/python docs/build/paper/drift_figure.py OUT.pdf
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from src.robust_gait import run_experiment as R
from src.robust_gait.data import CONDITIONS, LOCATIONS, load_strides
from src.robust_gait.drift_cvae import SAMPLES_PER_CONDITION
from src.robust_gait.protocol import create_outer_folds
from src.robust_gait.verifier import VerifierConfig

FOLD = 0
NAMES = list(LOCATIONS)
AXES = [2 * len(NAMES) + 6 * l + j for l in range(len(NAMES)) for j in range(6)]
ROWS = ["head", "sternum", "sacrum", "left_wrist", "right_wrist",
        "left_leg", "right_leg", "left_foot", "right_foot"]  # top to bottom
LABELS = {"head": "Head", "sternum": "Sternum", "sacrum": "Sacrum",
          "left_wrist": "L wrist", "right_wrist": "R wrist", "left_leg": "L shank",
          "right_leg": "R shank", "left_foot": "L foot", "right_foot": "R foot"}
TITLES = ["ST-fatigue", "DT-control", "DT-fatigue"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
        "#256abf", "#184f95", "#0d366b"]    # one-hue sequential (blue)
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "serif", "font.size": 7, "axes.labelsize": 7,
    "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
})

people = R.participants()
data = {(p, c): load_strides(p, c) for p in people for c in CONDITIONS}
fold = create_outer_folds(people, 4)[FOLD]
dev, test = list(fold.development_participants), list(fold.evaluation_participants)
config = R.ExperimentConfig(verifier=VerifierConfig(drift_augmentation=4), drift_model="cvae")

def pool(v):  # (n, 128, 54) -> (n, 32, 54), as the classifier features
    return v.reshape(len(v), 32, 4, v.shape[2]).mean(axis=2)


generated = pool(R.drift_cvae(dev, data, config)[:, :, AXES])    # (90, 32, 54)
labels = np.repeat(np.arange(3), SAMPLES_PER_CONDITION)
# Scale: each axis's stride-to-stride SD in the development users' normal walking.
normal = pool(np.concatenate([data[p, "st_control"].strides[:, :, AXES] for p in dev]))
scale = normal.std(axis=(0, 1))

# (a) RMS drift per IMU and phase, averaged over the generated samples and axes.
z = generated / scale
heat = np.stack([
    np.sqrt((z[labels == k].reshape(-1, 32, len(NAMES), 6) ** 2).mean(axis=(0, 3))).T
    for k in range(3)
])                                                                 # (3, 9, 32)
heat = heat[:, [NAMES.index(r) for r in ROWS]]

# (b) Measured drift of each test user: mean stride per condition minus mean
# ST-control stride, with IMUs dead in either recording zeroed.
measured = R.measured_drifts(test + dev, data)
real = pool(np.stack([measured[u, c][:, AXES] for u in test for c in R.DRIFT_CONDITIONS]))
real_labels = np.tile(np.arange(3), len(test))


def flat(v):
    return (v / scale).reshape(len(v), -1)


g = flat(generated)
mean = g.mean(axis=0)
_, s, vt = np.linalg.svd(g - mean, full_matrices=False)
explained = s[:2] ** 2 / (s ** 2).sum()
pg, pr = (g - mean) @ vt[:2].T, (flat(real) - mean) @ vt[:2].T

fig = plt.figure(figsize=(7.16, 2.45))
grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.2,
                        left=0.075, right=0.575, bottom=0.17, top=0.84)
cmap = LinearSegmentedColormap.from_list("blue", RAMP)
vmax = np.percentile(heat, 99)
for k in range(3):
    ax = fig.add_subplot(grid[0, k])
    im = ax.imshow(heat[k], aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                   extent=(0, 100, len(ROWS) - 0.5, -0.5), interpolation="nearest")
    ax.set_title(TITLES[k], color=INK, pad=3)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Gait phase (%)")
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([LABELS[r] for r in ROWS] if k == 0 else [])
    ax.tick_params(length=2, width=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
cax = fig.add_subplot(grid[0, 3])
bar = fig.colorbar(im, cax=cax)
bar.set_label("RMS drift (SD of normal walking)", labelpad=3)
bar.outline.set_visible(False)
bar.ax.tick_params(length=2, width=0.5)
fig.text(0.075, 0.95, "(a) CVAE-generated drift per IMU and gait phase", color=INK,
         fontsize=7.5, ha="left")

ax = fig.add_axes([0.715, 0.17, 0.27, 0.67])
for k in range(3):
    ax.scatter(*pg[labels == k].T, s=10, color=COLORS[k], edgecolors="white",
               linewidths=0.4, alpha=0.85, zorder=2)
    ax.scatter(*pr[real_labels == k].T, s=34, facecolors="white",
               edgecolors=COLORS[k], linewidths=1.4, zorder=3)
both = np.vstack([pg, pr])
lo, hi = np.percentile(both, 1, axis=0), np.percentile(both, 99, axis=0)
pad = 0.12 * (hi - lo)
ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
ax.axhline(0, color="#e4e3df", lw=0.5, zorder=1)
ax.axvline(0, color="#e4e3df", lw=0.5, zorder=1)
ax.set_xlabel(f"PC 1 ({explained[0]:.0%} of generated variance)")
ax.set_ylabel(f"PC 2 ({explained[1]:.0%})", labelpad=1)
ax.tick_params(length=2, width=0.5)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
handles = [Line2D([], [], marker="o", ls="", color=COLORS[k], markersize=4, label=TITLES[k])
           for k in range(3)]
handles += [Line2D([], [], marker="o", ls="", color=MUTED, markersize=3.5, label="Generated (CVAE)"),
            Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=MUTED,
                   markeredgewidth=1.2, markersize=5, label="Measured, test user")]
ax.legend(handles=handles, loc="best", frameon=True, framealpha=0.9, edgecolor="none",
          handletextpad=0.2, borderaxespad=0.3, labelspacing=0.3, fontsize=6)
fig.text(0.66, 0.95, "(b) Generated vs. measured drift", color=INK, fontsize=7.5, ha="left")

out = sys.argv[1] if len(sys.argv) > 1 else "cvae_drift.pdf"
fig.savefig(out, dpi=300)
fig.savefig(out.rsplit(".", 1)[0] + ".png", dpi=200)
# Numbers quoted in the paper text.
order = [LABELS[r] for r in ROWS]
for k in range(3):
    top = np.argsort(heat[k].mean(1))[::-1][:3]
    print(TITLES[k], "largest IMUs:", [(order[i], round(heat[k, i].mean(), 2)) for i in top],
          "mean", round(heat[k].mean(), 2))
inside = []
for k in range(3):
    c = pg[labels == k]
    lo, hi = c.min(0), c.max(0)
    r = pr[real_labels == k]
    inside.append(np.mean(np.all((r >= lo) & (r <= hi), axis=1)))
print("PC variance", explained.round(3), "test-user drifts inside generated range per condition",
      np.round(inside, 2), "n", len(real))
for k in range(3):
    print(TITLES[k], "per-IMU mean:", {order[i]: round(heat[k, i].mean(), 2) for i in range(len(order))})
