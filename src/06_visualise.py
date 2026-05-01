import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from matplotlib.lines import Line2D
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    RESULTS_DIR, FEATURE_COLS, FEATURE_DIR, FEATURES_DIR, MODELS_DIR, ST_FATIGUE_FEATURES,
    COLOR_BASELINE, COLOR_CROSS, COLOR_DELTA, FIGURES_DIR
)

def save_figure(fig, filename):
    path = FIGURES_DIR / filename

    fig.savefig(path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)

    print(f" Saved : {path}")

def compute_eer(y_true, decision_scores):
    fpr, tpr, thresholds = roc_curve(y_true, decision_scores, pos_label=1)

    fnr = 1 - tpr

    abs_diff = np.abs(fnr -fpr)
    eer_idx = np.argmin(abs_diff)

    err = (fnr[eer_idx] + fpr[eer_idx]) / 2

    threshold = thresholds[eer_idx]

    return float(err), float(threshold)



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

    participants = df_sorted['participant_id'].tolist()
    deltas = df_sorted['frr_delta'].tolist()

    colors = [COLOR_DELTA if d > 0 else '#4C4F50' for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(participants, deltas, color=colors, alpha=0.85, edgecolor='white')

    for bar, delta in zip(bars, deltas):
        x_pos = bar.get_width()
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


    bp['boxes'][0].set_facecolor(COLOR_BASELINE)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(COLOR_CROSS)
    bp['boxes'][1].set_alpha(0.7)
    
    for i, (data, colour) in enumerate(
        [(frr_baseline, COLOR_BASELINE), (frr_cross, COLOR_CROSS)], start=1
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
 
    fatigue_dir = FEATURE_DIR / "st_fatigue"
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
        fp = cm[1, 0]
        tn = cm[1, 1]
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        ax.set_title(f'{participant_id}\nFRR={frr:.3%}\nFAR={far:.3%}', fontsize=9, fontweight='bold')
 
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)
 
    fig.tight_layout()
    save_figure(fig, '04_confusion_matrices.png')
 
def plot_far_comparison(results_df):
    participants = results_df['participant_id'].tolist()
    far_baseline = results_df['far_baseline'].tolist()
    far_cross    = results_df['far_cross_session'].tolist()

    n = len(participants)
    x = np.arange(n)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - bar_width/2, far_baseline, width=bar_width,
           color='#2196F3', alpha=0.85, label='Baseline FAR (S1 → S1)')
    ax.bar(x + bar_width/2, far_cross,    width=bar_width,
           color='#FF5722', alpha=0.85, label='Cross-Session FAR (S1 → S2)')

    ax.axhline(np.mean(far_baseline), color='#2196F3', linestyle='--',
               linewidth=1.2, alpha=0.7,
               label=f'Mean Baseline: {np.mean(far_baseline):.1%}')
    ax.axhline(np.mean(far_cross), color='#FF5722', linestyle='--',
               linewidth=1.2, alpha=0.7,
               label=f'Mean Cross-Session: {np.mean(far_cross):.1%}')

    ax.set_xlabel('Participant', fontsize=12)
    ax.set_ylabel('False Acceptance Rate (FAR)', fontsize=12)
    ax.set_title(
        'Per-Participant FAR: Baseline vs Cross-Session\n'
        'Lower FAR = Fewer Impostors Accepted',
        fontsize=13, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(participants, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    save_figure(fig, '05_far_comparison.png')


def plot_frr_far_summary(results_df):
    categories  = ['Baseline\nFRR', 'Baseline\nFAR',
                   'Cross-Session\nFRR', 'Cross-Session\nFAR']
    means = [
        np.mean(results_df['frr_baseline']),
        np.mean(results_df['far_baseline']),
        np.mean(results_df['frr_cross_session']),
        np.mean(results_df['far_cross_session']),
    ]
    stds = [
        np.std(results_df['frr_baseline']),
        np.std(results_df['far_baseline']),
        np.std(results_df['frr_cross_session']),
        np.std(results_df['far_cross_session']),
    ]
    colours = ['#2196F3', '#4CAF50', '#F44336', '#FF9800']

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(categories, means, color=colours, alpha=0.85,
                  yerr=stds, capsize=6, edgecolor='white', linewidth=0.8)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                mean + std + 0.015,
                f'{mean:.1%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title(
        'Authentication Error Rates — Full Summary\n'
        'FRR = user locked out  |  FAR = impostor let in',
        fontsize=13, fontweight='bold'
    )
    ax.set_ylim(0, max(means) + max(stds) + 0.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    ax.axvline(x=1.5, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.text(0.5, ax.get_ylim()[1]*0.95, 'Baseline', ha='center',
            fontsize=9, color='#555', style='italic')
    ax.text(2.5, ax.get_ylim()[1]*0.95, 'Cross-Session', ha='center',
            fontsize=9, color='#555', style='italic')
    fig.tight_layout()
    save_figure(fig, '06_frr_far_summary.png')
 
def plot_eer_comparison(eer_df):
    df_sorted    = eer_df.sort_values('eer_cross', ascending=True)
    participants = df_sorted['participant_id'].tolist()
    eer_base     = df_sorted['eer_baseline'].tolist()
    eer_cross    = df_sorted['eer_cross'].tolist()

    n = len(participants)
    y = np.arange(n)
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(y + bar_height/2, eer_base,  height=bar_height,
            color='#2196F3', alpha=0.85, label='Baseline EER')
    ax.barh(y - bar_height/2, eer_cross, height=bar_height,
            color='#F44336', alpha=0.85, label='Cross-Session EER')

    ax.set_yticks(y)
    ax.set_yticklabels(participants, fontsize=9)
    ax.set_xlabel('Equal Error Rate (EER)', fontsize=12)
    ax.set_title(
        'EER per Participant: Baseline vs Cross-Session\n'
        'EER = point where FRR equals FAR  |  Lower = Better',
        fontsize=13, fontweight='bold'
    )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.axvline(x=np.mean(eer_base),  color='#2196F3', linestyle='--',
               linewidth=1.2, alpha=0.7,
               label=f'Mean Baseline EER: {np.mean(eer_base):.1%}')
    ax.axvline(x=np.mean(eer_cross), color='#F44336', linestyle='--',
               linewidth=1.2, alpha=0.7,
               label=f'Mean Cross-Session EER: {np.mean(eer_cross):.1%}')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    fig.tight_layout()
    save_figure(fig, '07_eer_comparison.png')

def plot_det_curve(results_df):

    fatigue_dir = ST_FATIGUE_FEATURES
    dfs         = [pd.read_csv(f) for f in sorted(fatigue_dir.glob("*_features.csv"))]
    all_fatigue = pd.concat(dfs, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    eer_baseline_list = []
    eer_cross_list    = []

    for _, row in results_df.iterrows():
        pid    = row['participant_id']
        svm    = joblib.load(MODELS_DIR / f"{pid}_svm.pkl")
        scaler = joblib.load(MODELS_DIR / f"{pid}_scaler.pkl")


        holdout  = np.load(MODELS_DIR / f"{pid}_holdout.npz")
        X_h, y_h = holdout['X'], holdout['y']
        scores_h = svm.decision_function(X_h)

        fpr_b, tpr_b, _ = roc_curve(y_h, scores_h, pos_label=1)
        fnr_b = 1 - tpr_b   # FRR = 1 - TPR

        ax.plot(fpr_b, fnr_b, color='#2196F3', alpha=0.25, linewidth=0.9)

        eer_b, _ = compute_eer(y_h, scores_h)
        eer_baseline_list.append(eer_b)


        X_f      = all_fatigue[FEATURE_COLS].values
        y_f      = (all_fatigue['participant_id'] == pid).astype(int).values
        X_f_sc   = scaler.transform(X_f)
        scores_f = svm.decision_function(X_f_sc)

        fpr_f, tpr_f, _ = roc_curve(y_f, scores_f, pos_label=1)
        fnr_f = 1 - tpr_f

        ax.plot(fpr_f, fnr_f, color='#F44336', alpha=0.25, linewidth=0.9)

        eer_f, _ = compute_eer(y_f, scores_f)
        eer_cross_list.append(eer_f)


    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, alpha=0.5, label='EER line (FAR = FRR)')


    mean_eer_b = np.mean(eer_baseline_list)
    mean_eer_c = np.mean(eer_cross_list)
    ax.scatter(mean_eer_b, mean_eer_b, color='#2196F3', s=120, zorder=5,
               label=f'Mean Baseline EER: {mean_eer_b:.1%}')
    ax.scatter(mean_eer_c, mean_eer_c, color='#F44336', s=120, zorder=5,
               label=f'Mean Cross-Session EER: {mean_eer_c:.1%}')

    
    proxy = [
        Line2D([0],[0], color='#2196F3', lw=2, label='Baseline (S1→S1) — per participant'),
        Line2D([0],[0], color='#F44336', lw=2, label='Cross-Session (S1→S2) — per participant'),
        Line2D([0],[0], color='k',       lw=1, linestyle='--', label='EER diagonal'),
    ]

    ax.set_xlabel('False Acceptance Rate (FAR)', fontsize=12)
    ax.set_ylabel('False Rejection Rate (FRR)', fontsize=12)
    ax.set_title(
        'DET Curve — Detection Error Tradeoff\n'
        'Each line = one participant  |  Closer to origin = better system',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(handles=proxy + ax.get_legend_handles_labels()[0][-2:],
              fontsize=9, loc='upper right')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    fig.tight_layout()
    save_figure(fig, '08_det_curve.png')


def collect_eer_data(results_df):

    fatigue_dir = ST_FATIGUE_FEATURES
    dfs         = [pd.read_csv(f) for f in sorted(fatigue_dir.glob("*_features.csv"))]
    all_fatigue = pd.concat(dfs, ignore_index=True)


    control_dir = FEATURES_DIR
    ctrl_dfs    = [pd.read_csv(f) for f in sorted(control_dir.glob("*_features.csv"))]
    all_control = pd.concat(ctrl_dfs, ignore_index=True)

    eer_rows = []

    for _, row in results_df.iterrows():
        pid = row['participant_id']

        svm    = joblib.load(MODELS_DIR / f"{pid}_svm.pkl")
        scaler = joblib.load(MODELS_DIR / f"{pid}_scaler.pkl")

        holdout   = np.load(MODELS_DIR / f"{pid}_holdout.npz")
        X_h, y_h  = holdout['X'], holdout['y']
        scores_h  = svm.decision_function(X_h)
        
        eer_b, _  = compute_eer(y_h, scores_h)

       
        X_f       = all_fatigue[FEATURE_COLS].values
        y_f       = (all_fatigue['participant_id'] == pid).astype(int).values
        X_f_sc    = scaler.transform(X_f)
        scores_f  = svm.decision_function(X_f_sc)
        eer_f, _  = compute_eer(y_f, scores_f)

        eer_rows.append({
            'participant_id': pid,
            'eer_baseline'  : round(eer_b, 4),
            'eer_cross'     : round(eer_f, 4),
        })

        print(f"  {pid}  EER baseline={eer_b:.3f}  cross={eer_f:.3f}")

    return pd.DataFrame(eer_rows)

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

    print("Generating Chart 5: FAR comparison...")
    plot_far_comparison(results_df)
    
    print("Computing EER data for all participants...")
    eer_df = collect_eer_data(results_df)

    print("Generating Chart 6: FRR + FAR summary...")
    plot_frr_far_summary(results_df)

    print("Generating Chart 7: EER per participant...")
    plot_eer_comparison(eer_df)

    print("Generating Chart 8: DET curve...")
    plot_det_curve(results_df)

    print("\n" + "=" * 60)

 
 
if __name__ == "__main__":
    main()