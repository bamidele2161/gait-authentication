
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    list_raw_files, parse_participant_info,
    PROCESSED_DIR, ensure_dirs
)


def load_sensor_logger_csv(filepath):
  
    df = pd.read_csv(filepath)

  
    print(f"  Columns found: {list(df.columns)}")

   
    col_map = {}

    for col in df.columns:
        c = col.lower().strip()
        if c in ['seconds_elapsed', 'time(s)', 'time_s', 'elapsed']:
            col_map[col] = 'time'
        elif c in ['x', 'x (m/s^2)', 'accel_x', 'accelerometerx', 'x-axis']:
            col_map[col] = 'x'
        elif c in ['y', 'y (m/s^2)', 'accel_y', 'accelerometery', 'y-axis']:
            col_map[col] = 'y'
        elif c in ['z', 'z (m/s^2)', 'accel_z', 'accelerometerz', 'z-axis']:
            col_map[col] = 'z'

    df = df.rename(columns=col_map)


    required = ['time', 'x', 'y', 'z']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Could not find columns {missing} in {filepath}.\n"
            f"Available columns: {list(df.columns)}\n"
        )

    df = df[['time', 'x', 'y', 'z']].copy()
    return df


def clean_signal(df):
  
    df = df.dropna()
    df = df.sort_values('time').reset_index(drop=True)
    df['time'] = df['time'] - df['time'].iloc[0]   
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(float)

    print(f"  Duration: {df['time'].iloc[-1]:.1f} seconds")
    print(f"  Total samples: {len(df)}")

    return df


def preprocess_all():
    ensure_dirs()
    files = list_raw_files()

    if len(files) == 0:
        print("\n No CSV files found in data/raw/")
        print("   Drop your Sensor Logger CSV files there first.")
        print("   Name them like: participant_01_session1.csv")
        return

    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"\nProcessing: {filename}")

        try:
            participant_id, session = parse_participant_info(filepath)
            print(f"  Participant: {participant_id} | Session: {session}")

            df = load_sensor_logger_csv(filepath)
            df = clean_signal(df)

          
            df['participant_id'] = participant_id
            df['session'] = session

          
            out_name = f"participant_{participant_id}_session{session}_clean.csv"
            out_path = os.path.join(PROCESSED_DIR, out_name)
            df.to_csv(out_path, index=False)
            print(f" Saved: {out_path}")

        except Exception as e:
            print(f" Error with {filename}: {e}")

    print("\n Preprocessing complete!")


if __name__ == "__main__":
    preprocess_all()
