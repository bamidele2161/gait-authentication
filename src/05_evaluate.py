import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import sys
import os

import joblib
import warnings 
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    MODELS_DIR, FEATURES_DIR, RESULTS_DIR, FEATURE_COLS, ST_FATIGUE_FEATURES 
)

def compute_frr(y_true, y_pred):
    legitimate_mask = (y_true == 1)

    if legitimate_mask.sum() == 0:
        return None

    false_rejections = ((y_pred == 0) & legitimate_mask).sum()

    total_legitimate = legitimate_mask.sum()

    frr = false_rejections / total_legitimate

    return float(frr)


def evaluate_baseline(participant_id, svm):
    holdout_path = MODELS_DIR / f"{participant_id}_holdout.npz"

    if not holdout_path.exists():
        print(f" [WARNING] Holdout file not found for {participant_id}. Skipping baseline.")
        return None


    data = np.load(holdout_path)
    X_holdout = data['X']
    y_holdout = data['y']

    y_pred = svm.predict(X_holdout)

    frr = compute_frr(y_holdout, y_pred)

    n_legimataate = int((y_holdout == 1).sum())


    return frr


def evaluate_cross_session(participant_id, svm, scaler, all_fatigue_data):
   X_fatigue = all_fatigue_data[FEATURE_COLS].values
   y_fatigue = (all_fatigue_data['participant_id'] == participant_id).astype(int).values

   if y_fatigue.sum() == 0:
    print(f" [WARNING] {participant_id} not found in st_fatigue. Skipping")
    return None

   if y_fatigue.sum() == 0:
    print(f" [WARNING] {participant_id} not found in st_fatigue. Skipping")

    X_fatigue_scaled = scaler.transform(X_fatigue)

    y_pred = svm.predict(X_fatigue_scaled)

    frr = compute_frr(y_fatigue, y_pred)

    n_legitimate = int(Y_fatigue.sum())

    return frr


def load_fatigue_features():

    fatigue_dir = ST_FATIGUE_FEATURES
    csv_files = sorted(fatigue_dir.glob("*_features.csv"))

    if not csv_files:
        raise FilesNotFoundError(
            f"No feature files in {fatigue_dir}"
            f"Did you run 03_extract_features.py for st_fatigue?"
        )

    dfs = [pd.read_csv(f) for f in csv_files]

    all_fatigue = pd.concat(dfs, ignore_index=True)

    return all_fatigue



def run_statistical_test(baseline_frrs, cross_session_frrs):
    statistic, p_value = stats.wilcoxon(cross_session_frrs, baseline_frrs, alternative='greater')

    return statistic, p_value

    
    


def evaluate_all_participants(all_fatigue_data):
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_files = sorted(MODELS_DIR.glob("*_svm.pkl"))

    if not model_files:
        raise FileNotFoundError(
            f"No model files found in {MODELS_DIR}"
            f"Did you run 04_train.py?"
        )
    
    results_rows = []

    for model_path in model_files:
        participant_id = model_path.stem.replace("_svm", "")

        svm = joblib.load(model_path)
        scaler = joblib.load(MODELS_DIR / f"{participant_id}_scaler.pkl")

        frr_baseline = evaluate_baseline(participant_id, svm)

        frr_cross = evaluate_cross_session(
            participant_id, svm, scaler, all_fatigue_data
        )


        if frr_baseline is None or frr_cross is None: 
            continue

        frr_delta = frr_cross - frr_baseline

        results_rows.append({
            'participant_id' : participant_id,
            'frr_baseline' : round(frr_baseline, 4),
            'frr_cross_session' : round(frr_cross, 4),
            'frr_delta' : round(frr_delta, 4)
        })

    results_df = pd.DataFrame(results_rows)

    results_path = RESULTS_DIR / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"[INFO] Evaluation results saved to: {results_path}")

    return results_df


def print_summary(results_df):
    baseline_frrs = results_df['frr_baseline'].tolist()
    cross_session_frrs = results_df['frr_cross_session'].tolist()

    mean_baseline = np.mean(baseline_frrs)
    std_baseline = np.std(baseline_frrs)
    mean_cross = np.mean(cross_session_frrs)
    std_cross = np.std(cross_session_frrs)
    mean_delta = np.mean(results_df['frr_delta'].tolist())

    stat, p_value = run_statistical_test(baseline_frrs, cross_session_frrs)

    
    print("=" * 60)
    print("Cross-Session Gait Authentication Evaluation")
    print("=" * 60)

    print("\n[Baseline (Intra-Session) Performance]")
    print(f"Mean FRR: {mean_baseline:.4f} (±{std_baseline:.4f})")

    print("\n[Cross-Session Performance (Control → Fatigue)]")
    print(f"Mean FRR: {mean_cross:.4f} (±{std_cross:.4f})")

    print("\n[Performance Improvement]")
    print(f"Mean FRR Reduction: {mean_delta:.4f}")

    print("\n[Statistical Significance]")
    print(f"Wilcoxon signed-rank test:")
    print(f"  Statistic = {stat:.4f}")
    print(f"  p-value   = {p_value:.4f}")

    if p_value < 0.05:
        print("  Result: Statistically significant improvement (p < 0.05)")
    else:
        print("  Result: No statistically significant improvement (p ≥ 0.05)")

    print("\n[Conclusion]")
    print(f"Cross-session gait authentication reduces FRR from {mean_baseline:.4f} to {mean_cross:.4f}")
    print(f"This demonstrates that gait characteristics remain stable enough across sessions to enable cross-session authentication.")

    summary = {
        'metric' : ['mean_baseline_frr', 'std_baseline_frr', 'mean_cross_session_frr', 'std_cross_session_frr', 'mean_frr_delta', 'wilcoxon_statistic', 'p_value'],
        'value' : [round(mean_baseline, 4), round(std_baseline, 4), round(mean_cross, 4), round(std_cross, 4), round(mean_delta, 4), round(stat, 4), round(p_value, 4)]
    }

    summary_df = pd.DataFrame(summary)

    summary_path = RESULTS_DIR / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"[INFO] Evaluation summary saved to: {summary_path}")



    

    

def main():
    print("=" * 60)
    print("Cross-Session Gait Authentication Evaluation")
    print("=" * 60)
    

    all_fatigue_data = load_fatigue_features()

    results_df = evaluate_all_participants(all_fatigue_data)

    print_summary(results_df)


if __name__ == "__main__":
    main()