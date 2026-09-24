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
    RESULTS_DIR, FEATURE_COLS, FEATURE_DIR, FEATURES_DIR, MODELS_DIR,
    ST_FATIGUE_FEATURES, DT_CONTROL_FEATURES, DT_FATIGUE_FEATURES,
    COLOR_BASELINE, COLOR_CROSS, COLOR_DELTA, FIGURES_DIR, COLOR_DT_CONTROL, COLOR_DT_FATIGUE
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
    frr_dt_control = results_df['frr_dt_control'].tolist()
    frr_dt_fatigue = results_df['frr_dt_fatigue'].tolist()
    
    n = len(participants)
    x = np.arange(n)

    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.bar(
        x - 1.5 * bar_width,
        frr_baseline,
        width = bar_width,
        color = COLOR_BASELINE,
        label = 'Baseline FRR (ST-CONTROL)',
        alpha = 0.85
    )

    ax.bar(
        x - 0.5 * bar_width,
        frr_cross,
        width = bar_width,
        color = COLOR_CROSS,
        label = 'Cross-Session FRR (ST-FATIGUE)',
        alpha = 0.85
    )

    ax.bar(
        x + 0.5 * bar_width,
        frr_dt_control,
        width = bar_width,
        color = COLOR_DT_CONTROL,
        label = 'DT Control FRR (+ Cognitive Task)',
        alpha = 0.85
    )

    ax.bar(
        x + 1.5 * bar_width,
        frr_dt_fatigue,
        width = bar_width,
        color = COLOR_DT_FATIGUE,
        label = 'DT Fatigue FRR (+ Fatigue + Cognitive Task)',
        alpha = 0.85
    )

    ax.axhline(np.mean(frr_baseline),   color=COLOR_BASELINE, linestyle='--',
               linewidth=1.0, alpha=0.6, label=f'Mean Baseline: {np.mean(frr_baseline):.1%}')
    ax.axhline(np.mean(frr_cross),      color=COLOR_CROSS,    linestyle='--',
               linewidth=1.0, alpha=0.6, label=f'Mean ST-Fatigue: {np.mean(frr_cross):.1%}')
    ax.axhline(np.mean(frr_dt_control), color='#FF9800',      linestyle='--',
               linewidth=1.0, alpha=0.6, label=f'Mean DT-Control: {np.mean(frr_dt_control):.1%}')
    ax.axhline(np.mean(frr_dt_fatigue), color='#8B0000',      linestyle='--',
               linewidth=1.0, alpha=0.6, label=f'Mean DT-Fatigue: {np.mean(frr_dt_fatigue):.1%}')

    ax.set_xlabel('Participant', fontsize=12)
    ax.set_ylabel('False Rejection Rate (FRR)', fontsize=12)
    ax.set_title(
        'Per-Participant FRR Across All Four Test Conditions\n'
        'Temporal Variation  |  Cognitive Load  |  Physical Fatigue',
        fontsize=13, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(participants, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    save_figure(fig, '01_frr_comparison.png')


def plot_frr_delta(results_df):
    participants = results_df['participant_id'].tolist()

    delta_st  = results_df['frr_cross_session'].tolist()  
    delta_dtc = (results_df['frr_dt_control'] - results_df['frr_baseline']).tolist()
    delta_dtf = (results_df['frr_dt_fatigue'] - results_df['frr_baseline']).tolist()

    n  = len(participants)
    y  = np.arange(n)
    bh = 0.25 

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.barh(y + bh,  delta_st,  height=bh, color=COLOR_CROSS, alpha=0.85,
            label='ST-Fatigue − Baseline (temporal variation)')
    ax.barh(y,       delta_dtc, height=bh, color='#FF9800',   alpha=0.85,
            label='DT-Control − Baseline (+ cognitive load)')
    ax.barh(y - bh,  delta_dtf, height=bh, color='#8B0000',   alpha=0.85,
            label='DT-Fatigue − Baseline (+ fatigue + cognitive)')

    ax.set_yticks(y)
    ax.set_yticklabels(participants, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('FRR Delta vs Baseline', fontsize=12)
    ax.set_ylabel('Participant', fontsize=12)
    ax.set_title(
        'FRR Degradation per Participant — All Conditions vs Baseline\n'
        'Shows compounding effect of temporal variation, cognitive load, and fatigue',
        fontsize=13, fontweight='bold'
    )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:+.0%}'))
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    fig.tight_layout()
    save_figure(fig, '02_frr_delta.png')


def plot_boxplot(results_df):
    frr_baseline   = results_df['frr_baseline'].tolist()
    frr_cross      = results_df['frr_cross_session'].tolist()
    frr_dt_control = results_df['frr_dt_control'].tolist()
    frr_dt_fatigue = results_df['frr_dt_fatigue'].tolist()

    data    = [frr_baseline, frr_cross, frr_dt_control, frr_dt_fatigue]
    labels  = ['Baseline\n(ST-Control)', 'Cross-Session\n(ST-Fatigue)',
               'DT-Control\n(+ cognitive)', 'DT-Fatigue\n(+ fatigue)']
    colours = [COLOR_BASELINE, COLOR_CROSS, '#FF9800', '#8B0000']

    fig, ax = plt.subplots(figsize=(11, 7))

    bp = ax.boxplot(
        data, labels=labels,
        patch_artist=True, widths=0.5,
        medianprops=dict(color='black', linewidth=2),
    )

    for patch, colour in zip(bp['boxes'], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)


    for i, (d, colour) in enumerate(zip(data, colours), start=1):
        x_jitter = np.random.normal(i, 0.04, size=len(d))
        ax.scatter(x_jitter, d, color=colour, alpha=0.6, s=40, zorder=3)


    for i, d in enumerate(data, start=1):
        mean_val = np.mean(d)
        ax.text(i, mean_val + 0.02, f'μ={mean_val:.1%}',
                ha='center', fontsize=9, color='black', fontweight='bold')

    ax.set_ylabel('False Rejection Rate (FRR)', fontsize=12)
    ax.set_title(
        'FRR Distribution Across All Four Conditions\n'
        'Each dot = one participant',
        fontsize=13, fontweight='bold'
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(0, 1.12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    save_figure(fig, '03_boxplot.png')


 
def plot_confusion_matrices(results_df):
    """Generate one confusion-matrix grid figure per test condition (4 total)."""

    def _load_features(feat_dir):
        dfs = [pd.read_csv(f) for f in sorted(feat_dir.glob("*_features.csv"))]
        return pd.concat(dfs, ignore_index=True)

    all_st_fatigue = _load_features(ST_FATIGUE_FEATURES)
    all_dt_control = _load_features(DT_CONTROL_FEATURES)
    all_dt_fatigue = _load_features(DT_FATIGUE_FEATURES)

    # (condition label, filename suffix, feature_df or None=use holdout)
    conditions = [
        ('Baseline (ST-Control — holdout)',         '04a_cm_baseline.png',   None),
        ('Cross-Session (ST-Fatigue)',               '04b_cm_st_fatigue.png', all_st_fatigue),
        ('DT-Control (+ Cognitive Task)',            '04c_cm_dt_control.png', all_dt_control),
        ('DT-Fatigue (+ Fatigue + Cognitive Task)',  '04d_cm_dt_fatigue.png', all_dt_fatigue),
    ]

    participants = results_df['participant_id'].tolist()
    n    = len(participants)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    def _draw_cm(ax, cm, pid):
        """Draw a single confusion-matrix cell onto ax."""
        ax.imshow(cm, interpolation='nearest', cmap='Blues')

        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                colour = 'white' if val > cm.max() / 2 else 'black'
                ax.text(c, r, str(val),
                        ha='center', va='center',
                        fontsize=11, fontweight='bold', color=colour)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Accepted\n(Pred +)', 'Rejected\n(Pred −)'], fontsize=7)
        ax.set_yticklabels(['Legitimate\n(True +)', 'Impostor\n(True −)'], fontsize=7)

        tp  = cm[0, 0]; fn = cm[0, 1]
        fp  = cm[1, 0]; tn = cm[1, 1]
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        ax.set_title(f'{pid}\nFRR={frr:.2%} | FAR={far:.2%}',
                     fontsize=8, fontweight='bold')

    for cond_label, filename, feat_df in conditions:

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.5, nrows * 3.2))
        axes_flat = axes.flatten()

        fig.suptitle(
            f'Confusion Matrices — {cond_label}\nEach cell = one participant',
            fontsize=13, fontweight='bold', y=1.01
        )

        for idx, pid in enumerate(participants):
            svm    = joblib.load(MODELS_DIR / f"{pid}_svm.pkl")
            scaler = joblib.load(MODELS_DIR / f"{pid}_scaler.pkl")

            if feat_df is None:
                # Baseline: held-out split saved during training
                holdout  = np.load(MODELS_DIR / f"{pid}_holdout.npz")
                X, y     = holdout['X'], holdout['y']
                y_pred   = svm.predict(X)
            else:
                X_raw  = feat_df[FEATURE_COLS].values
                y      = (feat_df['participant_id'] == pid).astype(int).values
                X_sc   = scaler.transform(X_raw)
                y_pred = svm.predict(X_sc)

            cm = confusion_matrix(y, y_pred, labels=[1, 0])
            _draw_cm(axes_flat[idx], cm, pid)

        for idx in range(n, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.tight_layout()
        save_figure(fig, filename)
 
def plot_far_comparison(results_df):
    participants = results_df['participant_id'].tolist()
    far_baseline = results_df['far_baseline'].tolist()
    far_cross    = results_df['far_cross_session'].tolist()

    n = len(participants)
    x = np.arange(n)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))

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
    conditions = ['Baseline\n(ST-Control)', 'ST-Fatigue\n(cross-session)',
                  'DT-Control\n(+ cognitive)', 'DT-Fatigue\n(+ fatigue)']

    frr_means = [
        np.mean(results_df['frr_baseline']),
        np.mean(results_df['frr_cross_session']),
        np.mean(results_df['frr_dt_control']),
        np.mean(results_df['frr_dt_fatigue']),
    ]
    far_means = [
        np.mean(results_df['far_baseline']),
        np.mean(results_df['far_cross_session']),
        np.mean(results_df['far_dt_control']),
        np.mean(results_df['far_dt_fatigue']),
    ]
    frr_stds = [
        np.std(results_df['frr_baseline']),
        np.std(results_df['frr_cross_session']),
        np.std(results_df['frr_dt_control']),
        np.std(results_df['frr_dt_fatigue']),
    ]
    far_stds = [
        np.std(results_df['far_baseline']),
        np.std(results_df['far_cross_session']),
        np.std(results_df['far_dt_control']),
        np.std(results_df['far_dt_fatigue']),
    ]

    x         = np.arange(len(conditions))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))

    bars_frr = ax.bar(x - bar_width/2, frr_means, bar_width,
                      yerr=frr_stds, capsize=5,
                      color=[COLOR_CROSS, COLOR_CROSS, COLOR_CROSS, COLOR_CROSS,],
                      alpha=0.85, label='FRR (user locked out)')
    bars_far = ax.bar(x + bar_width/2, far_means, bar_width,
                      yerr=far_stds, capsize=5,
                      color=[ '#8B0000', '#8B0000', '#8B0000', '#8B0000', '#8B0000'],
                      alpha=0.85, label='FAR (impostor accepted)',
                      edgecolor='grey', linewidth=0.5)

    for bar, mean, std in zip(bars_frr, frr_means, frr_stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                mean + std + 0.015,
                f'{mean:.1%}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title(
        'FRR and FAR Across All Four Test Conditions\n'
        'FRR = user locked out  |  FAR = impostor accepted',
        fontsize=13, fontweight='bold'
    )
    ax.set_ylim(0, max(frr_means) + max(frr_stds) + 0.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
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
    # Load all four test-condition feature sets
    def _load_features(feat_dir):
        dfs = [pd.read_csv(f) for f in sorted(feat_dir.glob("*_features.csv"))]
        return pd.concat(dfs, ignore_index=True)

    all_st_fatigue  = _load_features(ST_FATIGUE_FEATURES)
    all_dt_control  = _load_features(DT_CONTROL_FEATURES)
    all_dt_fatigue  = _load_features(DT_FATIGUE_FEATURES)

    # Condition config: (label, colour, alpha, data_df or None for holdout)
    conditions = [
        ('Baseline (ST-Control)',            COLOR_BASELINE,   None),
        ('Cross-Session (ST-Fatigue)',        COLOR_CROSS,      all_st_fatigue),
        ('DT-Control (+ cognitive)',          COLOR_DT_CONTROL, all_dt_control),
        ('DT-Fatigue (+ fatigue + cognitive)', COLOR_DT_FATIGUE, all_dt_fatigue),
    ]

    fig, ax = plt.subplots(figsize=(8, 8))

    eer_means = []

    for label, colour, feat_df in conditions:
        eer_list = []

        for _, row in results_df.iterrows():
            pid    = row['participant_id']
            svm    = joblib.load(MODELS_DIR / f"{pid}_svm.pkl")
            scaler = joblib.load(MODELS_DIR / f"{pid}_scaler.pkl")

            if feat_df is None:
                # Baseline: use the held-out training split
                holdout  = np.load(MODELS_DIR / f"{pid}_holdout.npz")
                X, y     = holdout['X'], holdout['y']
                scores   = svm.decision_function(X)
            else:
                X_raw  = feat_df[FEATURE_COLS].values
                y      = (feat_df['participant_id'] == pid).astype(int).values
                X      = scaler.transform(X_raw)
                scores = svm.decision_function(X)

            fpr, tpr, _ = roc_curve(y, scores, pos_label=1)
            fnr = 1 - tpr

            ax.plot(fpr, fnr, color=colour, alpha=0.20, linewidth=0.9)

            eer, _ = compute_eer(y, scores)
            eer_list.append(eer)

        mean_eer = np.mean(eer_list)
        eer_means.append((label, colour, mean_eer))

    # EER diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, alpha=0.5)

    # Mean EER scatter points
    for label, colour, mean_eer in eer_means:
        ax.scatter(mean_eer, mean_eer, color=colour, s=130, zorder=5,
                   label=f'Mean EER — {label}: {mean_eer:.1%}')

    # Legend proxies for the per-participant curves
    proxy_lines = [
        Line2D([0], [0], color=colour, lw=2, alpha=0.7, label=f'{label} — per participant')
        for label, colour, _ in conditions
    ]
    proxy_lines.append(
        Line2D([0], [0], color='k', lw=1, linestyle='--', label='EER diagonal')
    )

    ax.set_xlabel('False Acceptance Rate (FAR)', fontsize=12)
    ax.set_ylabel('False Rejection Rate (FRR)', fontsize=12)
    ax.set_title(
        'DET Curve — Detection Error Tradeoff (All 4 Conditions)\n'
        'Each line = one participant  |  Closer to origin = better',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Combine per-participant line proxies with mean-EER scatter handles
    scatter_handles = ax.get_legend_handles_labels()[0]
    ax.legend(handles=proxy_lines + scatter_handles,
              fontsize=8, loc='upper right', ncol=1)
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
 

    results_path = RESULTS_DIR / "right_wrist_evaluation_results.csv"
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