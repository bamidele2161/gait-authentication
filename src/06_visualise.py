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

    save_figure(fig, '01_frr_comparison.png')


def plot_frr_delta(results_df):
    df_sorted =results_df.sort_values('frr_delta', ascending=True)

    participants = df_sorted['partcipant_id'].tolist()
    deltas = df_sorted['frr_delta'].tolist()

    colors = [COLOR_DELTA if d > 0 else '#4C4F50' for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(participants, deltas, color=colors, alpha=0.85, edgecolor='white')

    for bar, delta in zip(bars, deltas):
        ax.text(
            x_pos + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f'{delta:+.3f}',
            va='center',
            ha='left',
            fontsize=8,
        )

    ax.axvline(x=0, color='black', linewidth=0.8)
 
    ax.set_xlabel('FRR Delta (Cross-Session − Baseline)', fontsize=12)
    ax.set_ylabel('Participant', fontsize=12)
    ax.set_title(
        'FRR Degradation per Participant\n'
        'Positive = FRR increased (worse authentication) across sessions',
        fontsize=13, fontweight='bold'
    )
 
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:+.0%}'))
 
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    n_degraded = sum(1 for d in deltas if d > 0)
    ax.text(0.98, 0.02,
            f'{n_degraded}/{len(deltas)} participants degraded',
            transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
 
    fig.tight_layout()
    save_figure(fig, '02_frr_delta.png')
    

def plot_boxplot(results_df):

    frr_baseline = results_df['frr_baseline'].tolist()
    frr_cross    = results_df['frr_cross_session'].tolist()
 
    fig, ax = plt.subplots(figsize=(8, 7))
 
    bp = ax.boxplot(
        [frr_baseline, frr_cross],
        labels    = ['Baseline\n(S1 → S1)', 'Cross-Session\n(S1 → S2)'],
        patch_artist = True,
        widths    = 0.5,
        medianprops = dict(color='black', linewidth=2),
    )


    bp['boxes'][0].set_facecolor(COLOUR_BASELINE)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(COLOUR_CROSS)
    bp['boxes'][1].set_alpha(0.7)
    
    for i, (data, colour) in enumerate(
        [(frr_baseline, COLOUR_BASELINE), (frr_cross, COLOUR_CROSS)], start=1
    ):
        x_jitter = np.random.normal(i, 0.04, size=len(data))
        ax.scatter(x_jitter, data, color=colour, alpha=0.6, s=40, zorder=3)
 
    ax.set_ylabel('False Rejection Rate (FRR)', fontsize=12)
    ax.set_title(
        'Distribution of FRR Across Participants\n'
        'Baseline vs Cross-Session',
        fontsize=13, fontweight='bold'
    )
 
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
 
    for i, (data, label) in enumerate(
        [(frr_baseline, 'Mean'), (frr_cross, 'Mean')], start=1
    ):
        mean_val = np.mean(data)
        ax.text(i, mean_val + 0.02, f'μ={mean_val:.1%}',
                ha='center', fontsize=9, color='black', fontweight='bold')
 
    ax.annotate(
        f'p = 0.0003\n(Wilcoxon)',
        xy         = (1.5, max(frr_cross) * 0.95),
        ha         = 'center',
        fontsize   = 10,
        color      = 'darkred',
        fontweight = 'bold',
        bbox       = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )
 
    fig.tight_layout()
    save_figure(fig, '03_boxplot.png')
 
 
 
def plot_confusion_matrices(results_df):
 
    fatigue_dir = FEATURES_DIR / "st_fatigue"
    dfs         = [pd.read_csv(f) for f in sorted(fatigue_dir.glob("*_features.csv"))]
    all_fatigue = pd.concat(dfs, ignore_index=True)
 
    participants = results_df['participant_id'].tolist()
    n            = len(participants)
 
    ncols = 4
    nrows = int(np.ceil(n / ncols))   
 
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    axes_flat  = axes.flatten()
 
    fig.suptitle(
        'Confusion Matrices — Cross-Session Evaluation (S1 → S2)',
        fontsize=14, fontweight='bold', y=1.01
    )
 
    for idx, participant_id in enumerate(participants):
 
        ax = axes_flat[idx]
 
        svm    = joblib.load(MODELS_DIR / f"{participant_id}_svm.pkl")
        scaler = joblib.load(MODELS_DIR / f"{participant_id}_scaler.pkl")
 
        X = all_fatigue[FEATURE_COLS].values
        y = (all_fatigue['participant_id'] == participant_id).astype(int).values
 
        X_scaled = scaler.transform(X)
        y_pred   = svm.predict(X_scaled)
 
        cm = confusion_matrix(y, y_pred, labels=[1, 0])
 
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
 
        for row in range(2):
            for col in range(2):
                value = cm[row, col]
                text_colour = 'white' if value > cm.max() / 2 else 'black'
                ax.text(col, row, str(value),
                        ha='center', va='center',
                        fontsize=11, fontweight='bold', color=text_colour)
 
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Accepted\n(Predicted +)', 'Rejected\n(Predicted −)'], fontsize=7)
        ax.set_yticklabels(['Legitimate\n(True +)', 'Impostor\n(True −)'], fontsize=7)
 
        tp = cm[0, 0]
        fn = cm[0, 1]
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0
        ax.set_title(f'{participant_id}\nFRR={frr:.1%}', fontsize=9, fontweight='bold')
 
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)
 
    fig.tight_layout()
    save_figure(fig, '04_confusion_matrices.png')
 
 

def main():
    print("=" * 60)
    print("DUO-GAIT  |  Visualisation")
    print("=" * 60)
 
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
 

    results_path = RESULTS_DIR / "evaluation_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}. "
            f"Run 05_evaluate.py first."
        )
 
    results_df = pd.read_csv(results_path)
    print(f"  Loaded results for {len(results_df)} participants\n")
 
    print("Generating Chart 1: Per-participant FRR comparison...")
    plot_frr_comparison(results_df)
 
    print("Generating Chart 2: FRR delta per participant...")
    plot_frr_delta(results_df)
 
    print("Generating Chart 3: Box plot distribution...")
    plot_boxplot(results_df)
 
    print("Generating Chart 4: Confusion matrix grid...")
    plot_confusion_matrices(results_df)
 
    print("\n" + "=" * 60)

 
 
if __name__ == "__main__":
    main()