import numpy as np
import pandas as pd
from sklearn.svm import SVC
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    PROCESSED_DIR, SENSOR_COLS, SESSIONS, WINDOW_DIR, FEATURE_DIR,
    WINDOW_SECS, WINDOW_SAMPLES, STEP_SAMPLES, OVERLAP, MODELS_DIR, 
    FEATURES_DIR, FEATURE_COLS, CV_FOLDS, PARAM_GRID
)
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split

import joblib

import warnings
warnings.filterwarnings("ignore")



def load_all_features():
    csv_files = sorted(FEATURES_DIR.glob("*_features.csv"))
    if not csv_files:
        print(f"[Error]: No feature files found in {FEATURES_DIR}")
        return None

    dfs = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(all_data)} feature rows.")

    return all_data
    
def build_binary_labels(all_data, traget_participant): 
    X = all_data[FEATURE_COLS].values

    y = (all_data['participant_id'] == traget_participant).astype(int).values

    n_positive = int(y.sum())
    n_negative = int((y == 0).sum())

    return X, y, n_positive, n_negative

def scale_features(X_train, X_test=None):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.fit_transform(X_test)

        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler


def tune_and_train(X_train_scaled, y_train):
    base_svm = SVC(
        kernel = 'rbf',
        class_weight = 'balanced',
        probability = False,
        random_state = 42,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator = base_svm,
        param_grid = PARAM_GRID,
        cv = cv,
        scoring = 'f1',
        n_jobs = -1,
        verbose = 0
    )

    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    best_svm = grid_search.best_estimator_

    return best_svm, best_params, best_score


def train_all_participants(all_data):

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    participants = sorted(all_data['participant_id'].unique())

    summary_rows = []

    for participant in participants:

        X, y, n_pos, n_neg = build_binary_labels(all_data, participant)

        X_train, X_test_holdout, y_train, y_test_holdout = train_test_split(
            X, y,
            test_size = 0.3,
            stratify = y,
            random_state = 42
        )

        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test_holdout)

        svm, best_params, best_cv_score = tune_and_train(X_train_scaled, y_train)

        model_path = MODELS_DIR / f"{participant}_svm.pkl"
        scaler_path = MODELS_DIR / f"{participant}_scaler.pkl"


        joblib.dump(svm, model_path)
        joblib.dump(scaler, scaler_path)

        holdout_path = MODELS_DIR / f"{participant}_holdout.npz"
        np.savez(holdout_path, X=X_test_scaled, y=y_test_holdout)

        summary_rows.append({
            'participant_id' : participant,
            'n_positive' : int(y_train.sum()),
            'n_negative' : int((y_train == 0).sum()),
            'n_holdout_pos' : int(y_test_holdout.sum()),
            'n_holdout_neg' : int((y_test_holdout == 0).sum()),
            'best_C'   : best_params['C'],
            'best_gamma' : best_params['gamma'],
            'best_cv_f1' : round(best_cv_score, 4),
            
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = MODELS_DIR / "training_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nTraining summary saved to: {summary_path}")

    return summary_df


def main():
    print("=" * 60)
    
    all_data = load_all_features()

    summary = train_all_participants(all_data)

    print("\n" + "=" * 60)
    print("Training complete.")
    print("=" * 60)
    print(summary.to_string(index=False))

    

if __name__ == "__main__":
    main()

