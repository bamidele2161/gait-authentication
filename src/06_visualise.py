import as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    RESULTS_DIR, FEATURE_COLS, FEATURE_DIR, MODELS_DIR,
    COLOR_BASELINE, COLOR_CROSS, COLOR_DELTA, FIGURES_DIR
)

def save_figure(fig, filename):
    path = FIGURES_DIR / filename

    fig.savefig(path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)

    print(f" Saved : {path}")


def plot_frr_comparison(results_df):
    participants = results_df['participant_id'].tolist()
    frr_baseline = results_df['frr_baseline'].tolist()
    frr_cross = results_df['frr_cross_session'].tolist()
    
    n = len(participants)
    x = np.arange(n)

    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    bars_baseline = ax.bar(
        x - bar_width / 2,
        frr_baseline,
        width = bar_width,
        color = COLOR_BASELINE,
        label = 'Baseline FRR (S1 -> S1)',
        alpha = 0.85
    )

    bars_cross = ax.bar(
        x + bar_width / 2,
        frr_cross,
        width = bar_width,
        color = COLOR_CROSS,
        label = 'Cross-Session FRR (S1 -> S2)',
        alpha = 0.85
    )

    ax.set_xlabel('Participant', fontsize=12)
    ax.set_ylabel('False Rejection Rate (FRR)', fontsize=12)
    ax.set_title(
        'Per-Participant FRR: Baseline vs Cross-Session\n'
        'Lower FRR = Better Authentication Performance',
        fontsize=13, fontweight='bold'
    )

    ax.set_xticks(x)

    ax.set_xticklabels(participants, rotation=45, ha='right', fontsize=9)

    ax.set_ylim(0, 1.05)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    ax.axhline(y=np.mean(frr_baseline), color=COLOR_BASELINE, 
    linestyle='--', linewidth=1.2, alpha=0.7, label=f"  Mean Baseline FRR: {np.mean(frr_baseline):.1%}")

    ax.axhline(y=np.mean(frr_cross), color=COLOR_CROSS, 
    linestyle='--', linewidth=1.2, alpha=0.7, label=f"  Mean Cross-Session FRR: {np.mean(frr_cross):.1%}")

    ax.legend(fontsize=10, loc='upper right')

    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()

    save_figure(fig, 'frr_comparison.png')