"""Method overview figure for the paper (full column width, PNG)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "fig" / "fig0_method_overview.png"
INK, MUTED = "#1d1d1b", "#55544f"
ENR_F, ENR_E = "#eaf1fb", "#2a78d6"
VER_F, VER_E = "#fdf0e9", "#eb6834"
DEV_F, DEV_E = "#f1f0ec", "#8d8f94"
DEC_F, DEC_E = "#e8f6ef", "#1b8f62"
plt.rcParams.update({"font.family": "Arial"})


def box(ax, x, y, w, h, title, body, fill, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fill, ec=edge, lw=1.1))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=7.6, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.47, body, ha="center", va="top", fontsize=6.1, color=MUTED, linespacing=1.25)


def arrow(ax, a, b, col=INK, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=col, lw=1.0, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


fig = plt.figure(figsize=(7.16, 3.6)); ax = fig.add_axes([0.005, 0.005, 0.99, 0.99])
ax.set_xlim(0, 20); ax.set_ylim(0, 11.5); ax.axis("off")
W, H, gap = 4.62, 1.55, 0.43
xs = [0.05 + k * (W + gap) for k in range(4)]
ax.text(0.05, 11.45, "Enrolment (per user, ST-control walking only)", fontsize=7.6, weight="bold", color=ENR_E, va="top")
enr = [("Stride segmentation", "left-shank |ω|, 6 Hz zero-phase LP\nmid-swing peaks, strides 0.7–1.8 s"),
       ("Phase normalisation", "128 samples per stride, mean-pool ×4\n9 sensors × (6 axes × 32) features"),
       ("Enrolment augmentation", "hand-made: amplitude ±20 %, warp ±0.1\nlearned drift: x + s·δ_k (pool, below)"),
       ("9 per-sensor verifiers", "L2 logistic regression (or RBF-SVM)\nuser strides vs pool negatives")]
for x, (t, b) in zip(xs, enr):
    box(ax, x, 9.2, W, H, t, b, ENR_F, ENR_E)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, (a + W, 9.95), (b, 9.95))
ax.text(0.05, 7.6, "Verification (probes: ST-fatigue, DT-control, DT-fatigue)", fontsize=7.6, weight="bold", color=VER_E, va="top")
ver = [("Same front end", "segmentation, phase normalisation,\n9 × 192 features per stride"),
       ("Score and fuse sensors", "per-sensor decision value (log-odds),\nmedian of the 9 values"),
       ("Normalise and accumulate", "S-norm = ½(Z-norm + T-norm),\nmean of 10 strides (≈ 12 s)"),
       ("Decision", "accept if score ≥ τ;\nτ fixed per fold before testing")]
for i, (x, (t, b)) in enumerate(zip(xs, ver)):
    f, e = (DEC_F, DEC_E) if i == 3 else (VER_F, VER_E)
    box(ax, x, 5.3, W, H, t, b, f, e)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, (a + W, 6.05), (b, 6.05))
arrow(ax, (xs[3] + W / 2, 9.2), (xs[1] + W / 2, 6.85), col=ENR_E, ls="--", rad=-0.08)
ax.text(11.0, 8.15, "trained verifiers", fontsize=6.4, color=ENR_E, style="italic", ha="center")
ax.add_patch(FancyBboxPatch((0.05, 0.1), 19.85, 3.4, boxstyle="round,pad=0,rounding_size=0.08", fc=DEV_F, ec=DEV_E, lw=1.1))
ax.text(0.3, 3.3, "Development pool: the 12 non-test participants of the fold (test users never enter it)", fontsize=7.6,
        weight="bold", color=INK, va="top")
items = [("Negatives", "ST-control + ST-fatigue strides (240 each per person)"),
         ("Learned drift δ_k", "mean(condition) − mean(ST-control): 12 × 3 = 36 vectors"),
         ("Score-normalisation cohorts", "Z-norm (3 held-out rotations), T-norm (12 models)"),
         ("Calibration mirror", "6 pseudo-unseen pairs → minimax threshold τ")]
for k, (t, b) in enumerate(items):
    x = 0.3 + (k % 2) * 9.9; y = 2.55 - (k // 2) * 1.2
    ax.text(x, y, t, fontsize=6.5, weight="bold", color=INK, va="top")
    ax.text(x, y - 0.5, b, fontsize=6.3, color=MUTED, va="top")
for x in (xs[2] + W * 0.3, xs[3] + W * 0.3):   # cohorts -> normalisation, threshold -> decision
    arrow(ax, (x, 3.5), (x, 5.3), col=DEV_E, ls="--")
for x in (xs[2] + W * 0.88, xs[3] + W * 0.88):  # drift -> augmentation, negatives -> verifiers
    arrow(ax, (x, 3.5), (x, 9.2), col=DEV_E, ls="--")
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("ok", OUT)
