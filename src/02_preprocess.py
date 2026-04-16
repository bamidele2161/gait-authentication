import pandas as pd
import numpy as np
import os
import sys
import scipy.signal as resample_poly

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    PROCESSED_DIR, ensure_dirs, SENSOR_COLS, SAMPLE_RATE, ORIGINAL_RATE, SESSIONS
)

def load_and_preprocess_data(filepath):

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



def resmaple_signal(df):

    UP = TARGET_RATE // np.gcd(TARGET_RATE, ORIGINAL_RATE)
    DOWN = ORIGINAL_RATE // np.gcd(TARGET_RATE, ORIGINAL_RATE)
    
    resampled_data = {}


    for col in SENSOR_COLS:
        signal = df[col].values

        resmaple_signal=resample_poly(signal, UP, DOWN)
        resampled_data[col] = resampled_signal

    resampled_df = pd.DataFrame(resampled_data)

    resampled_df['participant_id'] = df['participant_id'].iloc[0]
    resampled_df['session_type'] = df['session_type'].iloc[0]

    return resampled_df
    
def create_windows(df, participant_id, session_type):
    signal = df[SENSOR_COLS].values
    
    n_samples = len(signal)

    windows = []
    

    window_index = 0

    for start in range(0, n_samples - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        
    
def process_session(session_name):


def main():

    for session in SESSIONS:
        process_session(session)

if __name__ == "__main__":
    main()
