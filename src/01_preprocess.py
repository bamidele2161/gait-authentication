import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    PROCESSED_DIR, ensure_dirs, SENSOR_COLS, SAMPLE_RATE
)

def load_and_preprocess_data(filepath):
    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Columns found: {list(df.columns)}")

    user_col = 'User'
    
    if not user_col:
        raise ValueError("Could not find a 'user' or 'User' column.")

    # Check for required sensor columns
    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected sensor columns: {missing}\nAvailable columns: {list(df.columns)}")

    users = df[user_col].unique()
    print(f"Found {len(users)} users.")

    for u in users:
        print(f"\nProcessing User: {u}")
        user_df = df[df[user_col] == u].copy()
        
        user_df = clean_signal(user_df)
        
        user_df['user'] = u
        if user_col != 'user':
            user_df = user_df.drop(columns=[user_col], errors='ignore')
        
        out_name = f"user_{u}_clean.csv"
        out_path = os.path.join(PROCESSED_DIR, out_name)
        
        out_cols = ['time', 'user'] + SENSOR_COLS
        user_df[out_cols].to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

def clean_signal(df):
    df = df.dropna(subset=SENSOR_COLS)

    df = df.reset_index(drop=True)
    df['time'] = np.arange(len(df)) / float(SAMPLE_RATE)

    df[SENSOR_COLS] = df[SENSOR_COLS].astype(float)

    print(f"  Duration: {df['time'].iloc[-1]:.1f} seconds")
    print(f"  Total samples: {len(df)}")

    return df

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
