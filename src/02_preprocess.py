import pandas as pd
import numpy as np
import os
import sys
import scipy.signal as resample_poly

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    PROCESSED_DIR, ensure_dirs, SENSOR_COLS, SAMPLE_RATE, ORIGINAL_RATE
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


    return resampled_df
    
def preprocess_all():
    ensure_dirs()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "gait_data.csv")

    if not os.path.exists(data_path):
        print(f"\n Error: {data_path} not found.")
        return

    try:
        load_and_preprocess_data(data_path)
        print("\n Preprocessing complete!")
    except Exception as e:
        print(f" Error during preprocessing: {e}")

if __name__ == "__main__":
    preprocess_all()
