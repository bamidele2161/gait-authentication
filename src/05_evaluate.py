import numpy as np
import pandas as pd
from scipy import stats
import sys
import os

import joblib
import warnings 
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    MODELS_DIR, RESULTS_DIR, FEATURE_COLS, ST_FATIGUE_FEATURES, DT_CONTROL_FEATURES, DT_FATIGUE_FEATURES
)

def compute_frr(y_true, y_pred):
    legitimate_mask = (y_true == 1)

    if legitimate_mask.sum() == 0:
        return None

    false_rejections = ((y_pred == 0) & legitimate_mask).sum()

    total_legitimate = legitimate_mask.sum()

    frr = false_rejections / total_legitimate

    return float(frr)

def compute_far(y_true, y_pred):
    imposter_mask = (y_true == 0)

    if imposter_mask.sum() == 0:
        return None

    false_acceptances = ((y_pred == 1) & imposter_mask).sum()

    total_imposters = imposter_mask.sum()

    far = false_acceptances / total_imposters

    return float(far)


    
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

    far = compute_far(y_holdout, y_pred)

    n_legitimate = int((y_holdout == 1).sum())
    
    n_imposter = int((y_holdout == 0).sum())

    print(f"Baseline FRR for {participant_id}: {frr:.4f} ({n_legitimate} legitimate samples)")
    print(f"Baseline FAR for {participant_id}: {far:.4f} ({n_imposter} imposter samples)")

    return frr, far


def evaluate_cross_session(participant_id, svm, scaler, all_fatigue_data):
   X_fatigue = all_fatigue_data[FEATURE_COLS].values
   y_fatigue = (all_fatigue_data['participant_id'] == participant_id).astype(int).values
   print(y_fatigue)
   print(X_fatigue.sum())
   if y_fatigue.sum() == 0:
    print(f" [WARNING] {participant_id} not found in st_fatigue. Skipping")
    return None
    
   X_fatigue_scaled = scaler.transform(X_fatigue)

   y_pred = svm.predict(X_fatigue_scaled)

   frr = compute_frr(y_fatigue, y_pred)

   far = compute_far(y_fatigue, y_pred)

   n_legitimate = int(y_fatigue.sum())

   n_imposter = int((y_fatigue == 0).sum())

   print(f"CrossSession FRR for {participant_id}: {frr:.4f} ({n_legitimate} legitimate samples)")
   print(f"CrossSession FAR for {participant_id}: {far:.4f} ({n_imposter} imposter samples)")

   return frr, far


def evaluate_dt_control_cross_session(participant_id, svm, scaler, all_dt_control_data):
   X_dt_control = all_dt_control_data[FEATURE_COLS].values
   y_dt_control = (all_dt_control_data['participant_id'] == participant_id).astype(int).values
   print(y_dt_control)
   print(X_dt_control.sum())
   if y_dt_control.sum() == 0:
    print(f" [WARNING] {participant_id} not found in dt_control. Skipping")
    return None
    
   X_control_scaled = scaler.transform(X_dt_control)

   y_pred = svm.predict(X_control_scaled)

   frr = compute_frr(y_dt_control, y_pred)

   far = compute_far(y_dt_control, y_pred)

   n_legitimate = int(y_dt_control.sum())

   n_imposter = int((y_dt_control == 0).sum())

   print(f"Dual Task Control Cross Session FRR for {participant_id}: {frr:.4f} ({n_legitimate} legitimate samples)")
   print(f"Dual Task Control Cross FAR for {participant_id}: {far:.4f} ({n_imposter} imposter samples)")

   return frr, far

def evaluate_df_fatigure_cross_session(participant_id, svm, scaler, all_dt_fatigue_data):
   X_dt_fatigue = all_dt_fatigue_data[FEATURE_COLS].values
   y_dt_fatigue = (all_dt_fatigue_data['participant_id'] == participant_id).astype(int).values
   print(y_dt_fatigue)
   print(X_dt_fatigue.sum())
   if y_dt_fatigue.sum() == 0:
    print(f" [WARNING] {participant_id} not found in dt_fatigue. Skipping")
    return None
    
   X_fatigue_scaled = scaler.transform(X_dt_fatigue)

   y_pred = svm.predict(X_fatigue_scaled)

   frr = compute_frr(y_dt_fatigue, y_pred)

   far = compute_far(y_dt_fatigue, y_pred)

   n_legitimate = int(y_dt_fatigue.sum())

   n_imposter = int((y_dt_fatigue == 0).sum())

   print(f"Dual Task Fatigue Cross Session FRR for {participant_id}: {frr:.4f} ({n_legitimate} legitimate samples)")
   print(f"Dual Task Fatigue Cross Session FAR for {participant_id}: {far:.4f} ({n_imposter} imposter samples)")

   return frr, far


def load_fatigue_features():

    fatigue_dir = ST_FATIGUE_FEATURES
    csv_files = sorted(fatigue_dir.glob("*_features.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No feature files in {fatigue_dir}"
            f"Did you run 03_extract_features.py for st_fatigue?"
        )

    dfs = [pd.read_csv(f) for f in csv_files]

    all_fatigue = pd.concat(dfs, ignore_index=True)

    return all_fatigue

def load_dt_control_features():
    dt_control_dir = DT_CONTROL_FEATURES
    print(dt_control_dir, "Testing")
    csv_files = sorted(dt_control_dir.glob("*_features.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No features files in {dt_control_dir}"
            f"Did you run 03_extract_features.py for st_fatigue?"
        )
    
    dfs = [pd.read_csv(f) for f in csv_files]

    all_dt_control = pd.concat(dfs, ignore_index=True)
    return all_dt_control


def load_dt_fatigue_features():
    dt_fatigue_dir = DT_FATIGUE_FEATURES
    csv_files = sorted(dt_fatigue_dir.glob("*_features.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No features files in {dt_fatigue_dir}"
            f"Did you run 03_extract_features.py for st_fatigue?"
        )
    
    dfs = [pd.read_csv(f) for f in csv_files]

    all_dt_fatigue = pd.concat(dfs, ignore_index=True)
    return all_dt_fatigue


def run_statistical_test(baseline_frrs, cross_session_frrs):
    statistic, p_value = stats.wilcoxon(cross_session_frrs, baseline_frrs, alternative='greater')

    return statistic, p_value

    
    


def evaluate_all_participants(all_fatigue_data, all_dt_control_data, all_dt_fatigue_data):
    
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

        frr_baseline, far_baseline = evaluate_baseline(participant_id, svm)

        frr_cross, far_cross = evaluate_cross_session(
            participant_id, svm, scaler, all_fatigue_data
        )

        frr_dt_control, far_dt_control = evaluate_dt_control_cross_session(
            participant_id, svm, scaler, all_dt_control_data
        )

        frr_dt_fatigue, far_dt_fatigue = evaluate_df_fatigure_cross_session(
            participant_id, svm, scaler, all_dt_fatigue_data
        )


        if frr_baseline is None or frr_cross is None or frr_dt_control is None or frr_dt_fatigue is None: 
            continue
        
        if far_baseline is None or far_cross is None or far_dt_control is None or far_dt_fatigue is None: 
            continue

        frr_delta = frr_cross - frr_baseline

        far_delta = far_cross - far_baseline

        frr_dt_delta = frr_dt_fatigue - frr_dt_control

        far_dt_delta = far_dt_fatigue - far_dt_control

        results_rows.append({
            'participant_id' : participant_id,
            'frr_baseline' : round(frr_baseline, 4),
            'frr_cross_session' : round(frr_cross, 4),
            'frr_dt_control': round(frr_dt_control, 4),
            'frr_dt_fatigue': round(frr_dt_fatigue, 4),
            'far_baseline' : round(far_baseline, 4),
            'far_cross_session' : round(far_cross, 4),
            'far_dt_control': round(far_dt_control, 4),
            'far_dt_fatigue': round(far_dt_fatigue, 4),
            'frr_delta' : round(frr_delta, 4),
            'far_delta' : round(far_delta, 4),
            'frr_dt_delta' : round(frr_dt_delta, 4),
            'far_dt_delta' : round(far_dt_delta, 4)
        })

    results_df = pd.DataFrame(results_rows)

    results_path = RESULTS_DIR / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"[INFO] Evaluation results saved to: {results_path}")

    return results_df


def print_summary(results_df):
    baseline_frrs = results_df['frr_baseline'].tolist()
    cross_session_frrs = results_df['frr_cross_session'].tolist()
    baseline_fars = results_df['far_baseline'].tolist()
    cross_session_fars = results_df['far_cross_session'].tolist()

    mean_baseline = np.mean(baseline_frrs)
    std_baseline = np.std(baseline_frrs)
    mean_cross = np.mean(cross_session_frrs)
    std_cross = np.std(cross_session_frrs)
    mean_delta = np.mean(results_df['frr_delta'].tolist())

    mean_baseline_far = np.mean(baseline_fars)
    std_baseline_far = np.std(baseline_fars)
    mean_cross_far = np.mean(cross_session_fars)
    std_cross_far = np.std(cross_session_fars)
    mean_delta_far = np.mean(results_df['far_delta'].tolist())

    mean_dt_control_frr = np.mean(results_df['frr_dt_control'].tolist())
    std_dt_control_frr = np.std(results_df['frr_dt_control'].tolist())
    mean_dt_fatigue_frr = np.mean(results_df['frr_dt_fatigue'].tolist())
    std_dt_fatigue_frr = np.std(results_df['frr_dt_fatigue'].tolist())
    mean_dt_delta_frr = np.mean(results_df['frr_dt_delta'].tolist())

    mean_dt_control_far = np.mean(results_df['far_dt_control'].tolist())
    std_dt_control_far = np.std(results_df['far_dt_control'].tolist())
    mean_dt_fatigue_far = np.mean(results_df['far_dt_fatigue'].tolist())
    std_dt_fatigue_far = np.std(results_df['far_dt_fatigue'].tolist())
    mean_dt_delta_far = np.mean(results_df['far_dt_delta'].tolist())


    # stat_frr, p_value_frr = run_statistical_test(baseline_frrs, cross_session_frrs)

    # stat_far, p_value_far = run_statistical_test(baseline_fars, cross_session_fars)

    
    print("=" * 60)
    print("Cross-Session Gait Authentication Evaluation")
    print("=" * 60)

    print("\n[Baseline (Intra-Session) Performance]")
    print(f"Mean FRR: {mean_baseline:.4f} (±{std_baseline:.4f})")
    print(f"Mean FAR: {mean_baseline_far:.4f} (±{std_baseline_far:.4f})")

    print("\n[Cross-Session Performance (Control → Fatigue)]")
    print(f"Mean FRR: {mean_cross:.4f} (±{std_cross:.4f})")
    print(f"Mean FAR: {mean_cross_far:.4f} (±{std_cross_far:.4f})")

    print("\n[Performance Improvement]")
    print(f"Mean FRR Reduction: {mean_delta:.4f}")
    print(f"Mean FAR Reduction: {mean_delta_far:.4f}")

    # print("\n[Statistical Significance]")
    # print(f"Wilcoxon signed-rank test:")
    # print(f"  FRR Statistic = {stat_frr:.4f}")
    # print(f"  FRR p-value   = {p_value_frr:.4f}")
    # print(f"  FAR Statistic = {stat_far:.4f}")
    # print(f"  FAR p-value   = {p_value_far:.4f}")

    # if p_value_frr < 0.05:
    #     print("  Result: Statistically significant improvement (p < 0.05)")
    # else:
    #     print("  Result: No statistically significant improvement (p ≥ 0.05)")

    # if p_value_far < 0.05:
        # print("  Result: Statistically significant improvement (p < 0.05)")
    # else:
        # print("  Result: No statistically significant improvement (p ≥ 0.05)")

    print("\n[Conclusion]")
    print(f"Cross-session gait authentication reduces FRR from {mean_baseline:.4f} to {mean_cross:.4f}")
    print(f"Cross-session gait authentication reduces FAR from {mean_baseline_far:.4f} to {mean_cross_far:.4f}")

    # summary = {
    #     'metric': [
    #         'mean_baseline_frr', 'std_baseline_frr', 'mean_cross_session_frr', 
    #         'std_cross_session_frr', 'mean_frr_delta', # 'wilcoxon_statistic_frr', 'p_value_frr',
    #         'mean_baseline_far', 'std_baseline_far', 'mean_cross_session_far', 
    #         'std_cross_session_far', 'mean_far_delta', # 'wilcoxon_statistic_far', 'p_value_far'
    #     ],
    #     'value': [
    #         round(mean_baseline, 4), round(std_baseline, 4), round(mean_cross, 4), 
    #         round(std_cross, 4), round(mean_delta, 4), # round(stat_frr, 4), round(p_value_frr, 4),
    #         round(mean_baseline_far, 4), round(std_baseline_far, 4), round(mean_cross_far, 4), 
    #         round(std_cross_far, 4), round(mean_delta_far, 4), # round(stat_far, 4),  round(p_value_far, 4)
    #     ]
    # }
    summary = {
        'metric': [
            
            'mean_baseline_frr', 'std_baseline_frr', 
            'mean_cross_session_frr', 'std_cross_session_frr', 
            'mean_frr_delta',
            
            'mean_frr_dt_control', 'std_frr_dt_control',
            'mean_frr_dt_fatigue', 'std_frr_dt_fatigue',
            'mean_frr_dt_delta',
            
            'mean_baseline_far', 'std_baseline_far', 
            'mean_cross_session_far', 'std_cross_session_far', 
            'mean_far_delta',
            
            'mean_far_dt_control', 'std_frr_dt_control',
            'mean_far_dt_fatigue', 'std_far_dt_fatigue',
            'mean_far_dt_delta'
        ],
        'value': [
         
            round(mean_baseline, 4), round(std_baseline, 4), 
            round(mean_cross, 4), round(std_cross, 4), 
            round(mean_delta, 4),
            
            round(mean_dt_control_frr, 4), round(std_dt_control_frr, 4),
            round(mean_dt_fatigue_frr, 4), round(std_dt_fatigue_frr, 4),
            round(mean_dt_delta_frr, 4),

    
            round(mean_baseline_far, 4), round(std_baseline_far, 4), 
            round(mean_cross_far, 4), round(std_cross_far, 4), 
            round(mean_delta_far, 4),
            
            round(mean_dt_control_far, 4), round(std_dt_control_far, 4),
            round(mean_dt_fatigue_far, 4), round(std_dt_fatigue_far, 4),
            round(mean_dt_delta_far, 4)
        ]
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

    all_dt_control_data = load_dt_control_features()

    all_dt_fatigue_data = load_dt_fatigue_features()

    results_df = evaluate_all_participants(all_fatigue_data, all_dt_control_data, all_dt_fatigue_data)

    print_summary(results_df)


if __name__ == "__main__":
    main()