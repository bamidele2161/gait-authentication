"""System architecture figure (A-E layout) for the paper, PNG at 300 dpi."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "fig" / "fig0_architecture.png"
INK, MUTED, EDGE = "#111111", "#333333", "#222222"
plt.rcParams.update({"font.family": "Arial"})

fig = plt.figure(figsize=(7.16, 4.77))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(6, 72.6); ax.axis("off")


def panel(x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                fc="none", ec="#555555", lw=0.8, ls=(0, (4, 3))))
    ax.text(x + 1.2, y + h - 1.1, label, fontsize=9.2, weight="bold", color=INK, va="top")


def box(x, y, w, h, title, body, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fill, ec=EDGE, lw=1.0))
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=7.4, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 4.1, body, ha="center", va="top", fontsize=6.3, color=MUTED, linespacing=1.3)


def arrow(p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=INK, lw=0.9,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def elbow(pts):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=0.9, solid_capstyle="butt")
    arrow(pts[-2], pts[-1])


# ---------- A
panel(0.6, 55.4, 98.8, 16.6, "A.  Data and per-stride representation")
w, gap, y = 18.3, 1.6, 57.2
xs = [2.0 + k * (w + gap) for k in range(5)]
A = [("DUO-GAIT input", "16 subjects, 9 IMUs × 6 axes\nVisit 1: ST-control, ST-fatigue\nVisit 2 (7 days later):\nDT-control, DT-fatigue"),
     ("Stride segmentation", "Left-shank gyro magnitude\n6 Hz zero-phase Butterworth\nmid-swing peaks\nstrides of 0.7–1.8 s"),
     ("Phase normalisation", "Interpolate each stride\nto 128 samples\n(cadence removed)"),
     ("Per-sensor features", "Raw 6 axes, mean-pool ×4\n6 × 32 → 192 features\nper sensor and stride"),
     ("Feature scaling", "Z-score with negative\n(development) statistics")]
for x, (t, b) in zip(xs, A):
    box(x, y, w, 11.6, t, b)
for a, b in zip(xs[:-1], xs[1:]):
    arrow((a + w, y + 5.8), (b, y + 5.8))

# ---------- B
panel(0.6, 27.0, 50.4, 26.8, "B.  Per-user enrolment")
bw = 14.9
bx = [2.0, 18.0, 34.0]
B = [("Positives", "User's ST-control strides\n+ augmented copies"),
     ("Augmentation", "Hand-made: amplitude,\ntime warp; learned drift"),
     ("Negatives", "Development users'\nST-control + ST-fatigue")]
for x, (t, b) in zip(bx, B):
    box(x, 40.6, bw, 9.6, t, b)
box(2.0, 28.6, 16.0, 9.0, "Drift bank", "36 development-user\ndrift vectors")
box(22.5, 28.6, 26.4, 9.0, "9 per-sensor verifiers", "One logistic-regression (or SVM)\nverifier per sensor, trained separately")
for x in bx:
    ax.plot([x + bw / 2, x + bw / 2], [40.6, 39.0], color=INK, lw=0.9)
ax.plot([bx[0] + bw / 2, bx[2] + bw / 2], [39.0, 39.0], color=INK, lw=0.9)
arrow((35.7, 39.0), (35.7, 37.6))
arrow((18.0, 33.1), (22.5, 33.1))
ax.plot([25.45, 25.45], [40.6, 39.4], color="none")

# ---------- C
panel(52.4, 27.0, 47.0, 26.8, "C.  Scoring a probe")
cw = 13.7
cx = [53.8, 68.6, 83.4]
C = [("Sensor fusion", "Median of the 9\nper-sensor scores"),
     ("S-norm + temporal", "½(Z-norm + T-norm)\nmean of 10 strides\n(≈ 11–12 s)"),
     ("Decision", "Accept if\nscore ≥ τ")]
for x, (t, b) in zip(cx, C):
    box(x, 36.0, cw, 11.2, t, b)
for a, b in zip(cx[:-1], cx[1:]):
    arrow((a + cw, 41.6), (b, 41.6))
elbow([(48.9, 33.1), (51.0, 33.1), (51.0, 41.6), (cx[0], 41.6)])

# ---------- D
panel(0.6, 7.2, 50.4, 18.0, "D.  Threshold calibration (development users only)")
box(2.0, 8.8, 21.0, 11.6, "Pseudo-unseen mirror", "6 pairs of development\nusers act as unseen users;\nprobed with ST-fatigue,\nDT-control, DT-fatigue")
box(27.8, 8.8, 21.6, 11.6, "Minimax operating point", "One threshold τ per fold\nminimising the worst\nper-condition FRR / FAR")
arrow((23.0, 14.6), (27.8, 14.6))
elbow([(49.4, 14.6), (50.7, 14.6), (50.7, 26.1), (cx[2] + cw / 2, 26.1), (cx[2] + cw / 2, 36.0)])
ax.text(cx[2] + cw / 2 - 1.0, 31.0, "τ from D, fixed before\nany test user is scored", fontsize=6.2, color=MUTED, style="italic", ha="right", va="center")

# ---------- E
panel(52.4, 7.2, 47.0, 18.0, "E.  Evaluation protocol")
box(53.8, 8.8, 20.6, 11.6, "Outer folds", "4 participant-disjoint folds\n12 development / 4 test users\neach subject tested once")
box(77.8, 8.8, 20.2, 11.6, "Trials", "Genuine: own ST-fatigue,\nDT-control, DT-fatigue\nImpostors: the other\nunseen test users")
arrow((74.4, 14.6), (77.8, 14.6))
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.02)
print("ok", OUT)
