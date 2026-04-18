import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    PROCESSED_DIR, SENSOR_COLS, SESSIONS, WINDOW_DIR, FEATURE_DIR,
    WINDOW_SECS, WINDOW_SAMPLES, STEP_SAMPLES, OVERLAP
)
def extract_features_window(window):
    features = {}

    for axis_index, axis_name in enumerate(SENSOR_COLS):
        signal = window[:, axis_index]

        features[f"{axis_name}_mean"] = np.mean(signal)
        features[f"{axis_name}_std"] = np.std(signal)
        features[f"{axis_name}_var"] = np.var(signal)
        features[f"{axis_name}_energy"] = np.sum(signal ** 2)
        features[f"{axis_name}_rms"] = np.sqrt(np.mean(signal ** 2))
        features[f"{axis_name}_min"] = np.min(signal)
        features[f"{axis_name}_max"] = np.max(signal)
        
    return features
        
def extract_features_for_participant(npy_path, csv_path):
    windows = np.load(npy_path)
    labels = pd.read_csv(csv_path)

    if windows.shape[0] != len(labels):
        print(f"[Warning] Mismatch between windows and labels for {npy_path.name}")
        return None
    n_windows = windows.shape[0]
    print(f" Windows loaded : {n_windows} shape={windows.shape}")

    all_feature_rows = []

    for i in range(n_windows):
        window = windows[i]

        features_row = extract_features_window(window)

        features_row['window_index'] = labels['window_index'].iloc[i]
        features_row['participant_id'] = labels['participant_id'].iloc[i]
        features_row['session_type'] = labels['session_type'].iloc[i]

        all_feature_rows.append(features_row)

    features_df = pd.DataFrame(all_feature_rows)

    return features_df
            

def process_session(session_name):

    windows_dir = WINDOW_DIR / session_name
    output_dir = FEATURE_DIR / session_name

    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(windows_dir.glob("*_windows.npy"))

    if not npy_files:
        print(f"[Warning]: No npy files found in {windows_dir}")
        return
    
    for npy_path in npy_files:

        participant_id = npy_path.stem.replace("_windows", "")

        csv_path = windows_dir / f"{participant_id}_labels.csv"

        if not csv_path.exists():
            print(f"[Warning]: No csv file found for {participant_id}")
            continue
        
        features_df = extract_features_for_participant(npy_path, csv_path)

        if features_df is None:
            continue

        output_path = output_dir / f"{participant_id}_features.csv"

        features_df.to_csv(output_path, index=False)

        print(f"Features saved to: {output_path}")


def main():
    n_features = len(SENSOR_COLS) * 7
    print("=" * 60)
    print("DUO-GAIT  |  Feature Extraction")
    print("=" * 60)
    print(f"  Axes      : {SENSOR_COLS}")
    print(f"  Total     : {n_features} features per window")
    print(f"  Input     : {WINDOW_DIR}") 
    print(f"  Output    : {FEATURE_DIR}")

    for session in SESSIONS:
        print(f"Checking session: {session}...") 
        process_session(session)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()