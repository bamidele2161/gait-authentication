import pandas as pd
import numpy as np
import os
import sys
import scipy.signal as resample_poly

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    DATA_DIR, PROCESSED_DIR, ensure_dirs, SENSOR_COLS, SAMPLE_RATE, ORIGINAL_RATE, STEP_SAMPLES, WINDOW_SAMPLES
)

def load_participant_csv(filepath):

    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Columns found: {list(df.columns)}")


    columns_to_drop = ['timestamp', 'Unnamed: 0']

    df = df.drop(columns=columns_to_drop, errors='ignore')
    for col in SENSOR_COLS:
        if col not in df.columns:
            print(f"Missing column: {col} in the dataset")
            return None

    return df



def resample_signal(df):

    UP = SAMPLE_RATE // np.gcd(SAMPLE_RATE, ORIGINAL_RATE)
    DOWN = ORIGINAL_RATE // np.gcd(SAMPLE_RATE, ORIGINAL_RATE)
    
    resampled_data = {}


    for col in SENSOR_COLS:
        signal = df[col].values

        resampled_signal = resample_poly(signal, UP, DOWN)
        resampled_data[col] = resampled_signal

    resampled_df = pd.DataFrame(resampled_data)

    resampled_df['participant_id'] = df['participant_id'].iloc[0]
    resampled_df['session_type'] = df['session_type'].iloc[0]

    return resampled_df
    

def create_windows(df, participant_id, session_type):

    signal = df[SENSOR_COLS].values
    n_samples = len(signal)

    windows=[]

    window_index = 0

    for start in range(0, n_samples - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        end = start + WINDOW_SAMPLES

        window_data = signal[start:end]

        if len(window_data) < WINDOW_SAMPLES:
            break

        window_dict= {}

        for i, col in enumerate(SENSOR_COLS):

            for t in range(WINDOW_SAMPLES):
                window_dict[f"{col}_{t}"] =window_data[t, i]

        window_dict['participant_id'] = participant_id
        window_dict['session_type'] = session_type
        window_dict['window_index'] = window_index

        windows.append(window_dict)
        window_index += 1

        return windows

def process_session(session_name):

    input_dir = DATA_DIR / session_name
    output_dir = PROCESSED_DIR / session_name

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"[Warning]: No CSV files found in {input_dir}")
        return 

    total_windows = 0

    for file_path in csv_files:
        participant_id = file_path.stem.replace("_SA", "")

        df = load_participant_csv(file_path)
        if df in None:
            continue

        original_samples = len(df)
        df_resampled = resample_signal(df)
        resampled_samples = len(df_resampled)

        expected = int(original_samples * SAMPLE_RATE / ORIGINAL_RATE)
        if abs(resampled_samples - expected) > 5:
            print(f"[Warning] unexpected resampled length: {resampled_samples}, expected: {expected}")
        
        windows = create_windows(df_resampled, participant_id=df['participant_id'].iloc[0], session_type=session_name)
        
        output_file = output_dir / f"{participant_id}_windows.csv"
        windows_df = pd.DataFrame(windows)

        windows_df.to_csv(output_file, index=False)

        total_windows += len(windows)

def main():
    ensure_dirs()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "gait_data.csv")

    if not os.path.exists(data_path):
        print(f"\n Error: {data_path} not found.")
        return

    try:
        load_participant_csv(data_path)
        print("\n Preprocessing complete!")
    except Exception as e:
        print(f" Error during preprocessing: {e}")

if __name__ == "__main__":
    main()
