"""Generate the architecture diagram for the final robust-gait verifier."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper" / "figures" / "proposed_method_architecture.png"

BROWN = "#492F24"
GOLD = "#F5B335"
CREAM = "#FFF8E8"
PALE_BLUE = "#EAF2F8"
PALE_GREEN = "#EAF5EE"
PALE_RED = "#FBEDEE"
INK = "#202124"
GREY = "#5F6368"


def box(ax, x, y, w, h, title, detail="", fill="white", edge=BROWN,
        title_size=11, detail_size=9.2, linewidth=1.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fill, edgecolor=edge, linewidth=linewidth,
    )
    ax.add_patch(patch)
    if detail:
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=title_size, weight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center",
                fontsize=detail_size, color=GREY, linespacing=1.18)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, weight="bold", color=INK)
    return patch


def arrow(ax, start, end, color=BROWN, width=1.5, style="-"):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13,
        linewidth=width, color=color, linestyle=style,
        shrinkA=2, shrinkB=2, connectionstyle="arc3,rad=0",
    )
    ax.add_patch(patch)


def label(ax, x, y, text, color=BROWN):
    ax.text(x, y, text, ha="center", va="center", fontsize=9,
            weight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="none", alpha=0.95))


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 11), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")

    ax.text(9, 10.62, "Proposed Multi-Sensor Gait Verification Method",
            ha="center", va="center", fontsize=20, weight="bold", color=BROWN)
    ax.text(9, 10.28,
            "Participant-disjoint development, normal-only enrolment, and cross-session evaluation",
            ha="center", va="center", fontsize=11.5, color=GREY)

    # Shared signal-processing pipeline.
    box(ax, 0.35, 8.78, 2.55, 1.05, "DUO-GAIT recordings",
        "9 synchronized IMUs\n6 raw axes each, 128 Hz", CREAM)
    box(ax, 3.25, 8.78, 2.55, 1.05, "Quality control",
        "known trim + finite check\nshared recording length", "white")
    box(ax, 6.15, 8.78, 2.75, 1.05, "Stride detection",
        "6 Hz low-pass on left-shank gyro\nmid-swing peak boundaries", "white")
    box(ax, 9.25, 8.78, 2.65, 1.05, "Cycle normalization",
        "retain 0.7–1.8 s strides\nresample each stride to 128", "white")
    box(ax, 12.25, 8.78, 2.45, 1.05, "Raw-axis profiles",
        "54 channels\n9 sensors × 6 axes", PALE_BLUE)
    box(ax, 15.05, 8.78, 2.55, 1.05, "Mean pooling",
        "128 → 32 samples per axis\n192 features per sensor", PALE_BLUE)
    for a, b in [((2.9, 9.31), (3.25, 9.31)), ((5.8, 9.31), (6.15, 9.31)),
                 ((8.9, 9.31), (9.25, 9.31)), ((11.9, 9.31), (12.25, 9.31)),
                 ((14.7, 9.31), (15.05, 9.31))]:
        arrow(ax, a, b)

    # Fold split and development lane.
    box(ax, 0.35, 7.18, 2.55, 0.98, "Four outer folds",
        "12 development participants\n4 unseen evaluation participants", GOLD)
    arrow(ax, (1.62, 8.78), (1.62, 8.16))

    ax.add_patch(FancyBboxPatch(
        (3.15, 5.63), 14.45, 2.55,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor="#FCFCFC", edgecolor="#A77C2D", linewidth=1.25,
    ))
    ax.text(3.42, 7.92, "DEVELOPMENT AND CALIBRATION (12 participants)",
            fontsize=10.5, weight="bold", color=BROWN, va="center")
    box(ax, 3.45, 6.10, 2.62, 1.28, "Background population",
        "ST-control + ST-fatigue\nup to 240 strides per person\nand condition", CREAM)
    box(ax, 6.47, 6.10, 2.70, 1.28, "Pseudo-unseen users",
        "groups of 2; full ST-control\nused for enrolment", CREAM)
    box(ax, 9.57, 6.10, 2.66, 1.28, "Session-2 probes",
        "DT-control + DT-fatigue\nfrom development users only", PALE_BLUE)
    box(ax, 12.63, 6.10, 2.10, 1.28, "S-norm cohorts",
        "Z-normalization +\nT-normalization", PALE_GREEN)
    box(ax, 15.13, 6.10, 2.12, 1.28, "Minimax\ncalibration",
        "one threshold per fold\nlimits worst-case error", PALE_GREEN,
        title_size=10, detail_size=8.3)
    arrow(ax, (2.9, 7.67), (3.45, 7.15))
    arrow(ax, (6.07, 6.74), (6.47, 6.74))
    arrow(ax, (9.17, 6.74), (9.57, 6.74))
    arrow(ax, (12.23, 6.74), (12.63, 6.74))
    arrow(ax, (14.73, 6.74), (15.13, 6.74))

    # Evaluation/new-user lane.
    ax.add_patch(FancyBboxPatch(
        (0.35, 1.18), 17.25, 3.92,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor="#FCFCFC", edgecolor=BROWN, linewidth=1.4,
    ))
    ax.text(0.62, 4.82, "UNSEEN-USER ENROLMENT AND VERIFICATION (4 participants per fold)",
            fontsize=10.5, weight="bold", color=BROWN, va="center")

    box(ax, 0.65, 3.00, 2.35, 1.28, "Normal-only\nenrolment",
        "complete ST-control recording\nfor the claimed user", CREAM,
        title_size=10, detail_size=8.2)
    box(ax, 3.35, 3.00, 2.33, 1.28, "Sensor fault gate",
        "drop sensor below 25% of\ndevelopment gyro level", CREAM,
        title_size=10, detail_size=8.2)
    box(ax, 6.03, 3.00, 2.58, 1.28, "Positive\naugmentation",
        "4 copies: amplitude ±20%\n+ smooth timing warp ±10%", CREAM,
        title_size=10, detail_size=8.2)
    box(ax, 8.96, 3.00, 2.68, 1.28, "Per-sensor\nuser models",
        "9 balanced L2 logistic models\nuser vs development background", PALE_BLUE,
        title_size=10, detail_size=8.2)
    box(ax, 12.00, 3.00, 2.27, 1.28, "Session-2 probe",
        "DT-control or DT-fatigue\nfrom sacrum + 8 IMUs", CREAM)
    box(ax, 14.62, 3.00, 2.62, 1.28, "Sensor scores",
        "background-standardized input\none score from each model", PALE_BLUE,
        title_size=10, detail_size=8.2)

    for a, b in [((3.0, 3.64), (3.35, 3.64)), ((5.68, 3.64), (6.03, 3.64)),
                 ((8.61, 3.64), (8.96, 3.64)), ((14.27, 3.64), (14.62, 3.64))]:
        arrow(ax, a, b)
    arrow(ax, (11.64, 3.64), (12.0, 3.64), style="--")
    label(ax, 11.82, 3.86, "scores")
    arrow(ax, (2.9, 7.30), (1.82, 4.28), style="--")
    label(ax, 2.03, 5.34, "evaluation users")
    arrow(ax, (4.75, 6.10), (10.30, 4.28), color="#A77C2D", style="--")
    label(ax, 7.47, 5.20, "negative background")

    box(ax, 1.15, 1.50, 2.55, 0.98, "Robust late fusion",
        "median across working sensors", PALE_GREEN, detail_size=8.4)
    box(ax, 4.20, 1.50, 2.35, 0.98, "S-norm",
        "average Z- and T-normalized scores", PALE_GREEN, detail_size=8.0)
    box(ax, 7.05, 1.50, 2.45, 0.98, "Temporal fusion",
        "mean of 10 consecutive strides\n≈ 12-second decision", PALE_GREEN,
        detail_size=8.1)
    box(ax, 10.00, 1.50, 2.38, 0.98, "Frozen threshold",
        "minimax value learned only\nfrom development users", PALE_GREEN)
    box(ax, 13.00, 1.50, 1.80, 0.98, "ACCEPT",
        "score ≥ threshold", PALE_GREEN, edge="#39734D")
    box(ax, 15.20, 1.50, 1.80, 0.98, "REJECT",
        "score < threshold", PALE_RED, edge="#A23B42")
    arrow(ax, (15.93, 3.00), (2.42, 2.48))
    for a, b in [((3.70, 1.99), (4.20, 1.99)), ((6.55, 1.99), (7.05, 1.99)),
                 ((9.50, 1.99), (10.00, 1.99))]:
        arrow(ax, a, b)
    arrow(ax, (12.38, 1.99), (13.00, 1.99), color="#39734D")
    arrow(ax, (12.38, 1.82), (15.20, 1.82), color="#A23B42")
    arrow(ax, (16.19, 6.10), (11.18, 2.48), color="#39734D", style="--")
    label(ax, 14.05, 4.82, "frozen S-norm + threshold")

    ax.text(9, 0.67,
            "Strict separation: evaluation identities do not contribute to background training, score normalization, or threshold calibration.",
            ha="center", va="center", fontsize=10.5, weight="bold", color=BROWN)
    ax.text(9, 0.34,
            "A participant is genuine only when the claimed identity matches the probe owner; the other three evaluation users provide impostor attempts.",
            ha="center", va="center", fontsize=9.5, color=GREY)

    plt.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
