"""Generate the simple-English architecture diagram."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper" / "figures" / "proposed_method_simple_architecture.png"


def box(ax, x, y, w, h, title, detail, color="#FFF8E8", edge="#492F24",
        title_size=13, detail_size=10):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=color, edgecolor=edge, linewidth=1.6,
    ))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
            fontsize=title_size, weight="bold", color="#202124")
    ax.text(x + w / 2, y + h * 0.27, detail, ha="center", va="center",
            fontsize=detail_size, color="#5F6368", linespacing=1.25)


def arrow(ax, start, end, color="#492F24", dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.6,
        color=color, linestyle="--" if dashed else "-",
    ))


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 10), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(8, 9.62, "The Proposed System in Simple English", ha="center",
            fontsize=23, weight="bold", color="#492F24")
    ax.text(8, 9.25, "First the system learns its rules from other people; then a new person enrols normally.",
            ha="center", fontsize=12, color="#5F6368")

    steps = [
        (0.35, "Record movement", "9 body sensors\nmeasure motion"),
        (3.45, "Find each stride", "cut walking into\ncomplete gait cycles"),
        (6.55, "Line strides up", "make every stride\nthe same phase length"),
        (9.65, "Keep direction", "use X, Y, Z motion\nfrom every sensor"),
        (12.75, "Simplify gently", "average nearby points\nwithout changing to 32 Hz"),
    ]
    for x, title, detail in steps:
        box(ax, x, 7.45, 2.55, 1.15, title, detail,
            "#FFF8E8" if x < 6 else "#EAF2F8")
    for x in (2.90, 6.00, 9.10, 12.20):
        arrow(ax, (x, 8.02), (x + 0.55, 8.02))

    ax.add_patch(FancyBboxPatch((0.45, 4.52), 15.1, 2.05,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor="#FCFCFC", edgecolor="#B07C17", linewidth=1.5))
    ax.text(0.72, 6.28, "SYSTEM PREPARATION - uses other development participants",
            fontsize=12, weight="bold", color="#492F24")
    teacher = [
        (0.80, "Learn strangers", "normal + fatigued walking\nfrom other people"),
        (4.00, "Practice\nnew-user cases", "normal enrolment, then\nlater dual-task probes"),
        (7.20, "Learn score\nreferences", "what normal stranger\nscores look like"),
        (10.40, "Choose one\npass mark", "balance rejection and\nfalse acceptance"),
        (13.20, "Freeze the rules", "do not change them for\nthe evaluated users"),
    ]
    for x, title, detail in teacher:
        box(ax, x, 4.88, 2.25, 1.12, title, detail, "#EAF5EE", title_size=11.5, detail_size=9)
    for x1, x2 in ((3.05, 4.0), (6.25, 7.2), (9.45, 10.4), (12.65, 13.2)):
        arrow(ax, (x1, 5.44), (x2, 5.44), color="#7D5B16")

    ax.add_patch(FancyBboxPatch((0.45, 0.70), 15.1, 3.05,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor="#FCFCFC", edgecolor="#492F24", linewidth=1.5))
    ax.text(0.72, 3.47, "NEW USER - provides only normal ST-control walking for enrolment",
            fontsize=12, weight="bold", color="#492F24")
    user = [
        (0.72, "Build user models", "one small specialist\nfor each working sensor"),
        (3.75, "Score a later walk", "each specialist says\nhow well it matches"),
        (6.78, "Take the middle vote", "median reduces the effect\nof one bad sensor"),
        (9.81, "Correct the score", "compare with known\nstranger behavior"),
        (12.84, "Combine 10 strides", "stable decision after\nabout 12 seconds"),
    ]
    for x, title, detail in user:
        box(ax, x, 1.73, 2.40, 1.18, title, detail,
            "#EAF2F8" if x < 9 else "#EAF5EE", title_size=11.2, detail_size=8.9)
    for x1, x2 in ((3.12, 3.75), (6.15, 6.78), (9.18, 9.81), (12.21, 12.84)):
        arrow(ax, (x1, 2.32), (x2, 2.32))
    box(ax, 5.40, 0.82, 2.30, 0.72, "ACCEPT", "score reaches pass mark",
        "#EAF5EE", "#39734D", 12, 8.5)
    box(ax, 8.30, 0.82, 2.30, 0.72, "REJECT", "score is below pass mark",
        "#FBEDEE", "#A23B42", 12, 8.5)
    arrow(ax, (14.04, 1.73), (7.15, 1.54), color="#39734D")
    arrow(ax, (14.04, 1.73), (9.45, 1.54), color="#A23B42")
    arrow(ax, (14.32, 4.88), (11.0, 2.91), color="#39734D", dashed=True)
    ax.text(13.4, 3.85, "frozen pass mark", fontsize=9.5, weight="bold", color="#39734D")
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
