import pandas as pd
import numpy as np
import os
import sys
from scipy.signal import resample_poly

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    DATA_DIR, PROCESSED_DIR, WINDOW_DIR, SENSOR_COLS, SAMPLE_RATE, ORIGINAL_RATE, STEP_SAMPLES, WINDOW_SAMPLES, WINDOW_SECS, OVERLAP, SESSIONS, 
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

    gcd = np.gcd(SAMPLE_RATE, ORIGINAL_RATE)
    UP = SAMPLE_RATE // gcd
    DOWN = ORIGINAL_RATE // gcd
    
    numeric_cols = ['GyrX', 'GyrY', 'GyrZ', 'AccX', 'AccY', 'AccZ']

    resampled_axes = []


    for col in numeric_cols:
        signal = df[col].values

        resampled_signal = resample_poly(signal, UP, DOWN)

        resampled_axes.append(resampled_signal)

    return np.column_stack(resampled_axes)

   
    

def create_windows(signal_array):

    n_samples = signal_array.shape[0]

    windows=[]

    

    for start in range(0, n_samples - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        end = start + WINDOW_SAMPLES

        window_data = signal_array[start:end, :]

        if window_data.shape[0] < WINDOW_SAMPLES:
            break

        windows.append(window_data)

    return np.stack(windows)



def save_windows(windows_array, participant_id, session_type, output_dir):

    n_windows = windows_array.shape[0]
    npy_path = output_dir / f"{participant_id}_windows.npy"

    np.save(npy_path, windows_array)


    labels_df = pd.DataFrame({
        'window_index' : np.arange(n_windows),

        'participant_id' : participant_id,

        'session_type' : session_type
    })

    csv_path = output_dir / f"{participant_id}_labels.csv"
    
    labels_df.to_csv(csv_path, index=False)
    

def process_session(session_name):

    input_dir = DATA_DIR / session_name
    output_dir = WINDOW_DIR / session_name

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    print(f"Files in {input_dir}: {csv_files}")
    if not csv_files:
        print(f"[Warning]: No CSV files found in {input_dir}")
        return 

    total_windows = 0

    for file_path in csv_files:
        print(f"Processing {file_path}")
        participant_id = file_path.stem.replace("_SA", "")

        df = load_participant_csv(file_path)
        if df is None:
            continue

        original_samples = len(df)
        df_resampled = resample_signal(df)

        resampled_len = df_resampled.shape[0]

        

        expected = int(original_samples * SAMPLE_RATE / ORIGINAL_RATE)
        if abs(resampled_len - expected) > 10:
            print(f"[Warning] unexpected resampled length: {resampled_len}, expected: {expected}")
        

        windows = create_windows(df_resampled)
        
        save_windows(
            windows, 
            participant_id = df['participant_id'].iloc[0],
            session_type = session_name,
            output_dir = output_dir
        )
    

        total_windows += windows.shape[0]

def main():
    
    print("DUO-GAIT Preprocessing Pipeline")
    print(f"  Original rate : {ORIGINAL_RATE} Hz")
    print(f"  Target rate   : {SAMPLE_RATE} Hz")
    print(f"  Window size   : {WINDOW_SECS}s ({WINDOW_SAMPLES} samples)")
    print(f"  Step size     : {STEP_SAMPLES} samples ({OVERLAP*100:.0f}% overlap)")
    print(f"  Axes          : {SENSOR_COLS}")

    for session in SESSIONS:
        process_session(session)

    print("\n\nAll sessions processed.")
    print(f"Output saved to: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()



