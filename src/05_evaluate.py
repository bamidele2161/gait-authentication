import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import joblib
import warnings 
warnings.filterwarnings('igmore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    MODELS_DIR, FEATURES_DIR, RESULTS_DIR, FEATURE_COLS, ST_FATIGUE_FEATURES 
)

def compute_frr(y_true, y_pred):
    legitimate_mast + (y_true == 1)

    if legitimate_mast.sum() == 0:
        return None

    false_rejections = ((y_pred == 0) & legitimate_masK).sum()

    total_legitimate = legitimate_mask.sum()

    frr = false_rejections / total_legitimate

    return float(frr)


def evaluate_baseline(participant_id, svm):
    holdout_path = MODELS_DIR / f"{participant_id}_holdout.holdout.npz"

    if not holdout_path.exist():
        print(f" [WARNING] Holdout file not found for {participant_id}. Skipping baseline.")
        retun None


    data = np.load(holdout_path)
    X_holdout = data['X']
    y_holdout = data['y']

    y_pred = svm.predict(X_holdout)

    frr = comput_frr(y_holdout, y_pred)

    n_legimataate = int((y_holdout == 1).sum())


    return frr


def evaluate_cross_session(participant_id, svm, scaler, app_fatigue_data):
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

    all_fague = pd.concat(dfs, ignoredex=True)

    return all_fatigue



def run_statistical_test(baseline_frrs, cross_session_frrs):
    

    
    


    