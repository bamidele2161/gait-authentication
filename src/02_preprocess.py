"""Create sacrum windows while retaining chronological grouping metadata."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (  # noqa: E402
    DATA_DIR, EXCLUDED_PARTICIPANTS, ORIGINAL_RATE, OVERLAP, SENSOR_COLS,
    SESSIONS, STEP_SAMPLES, WINDOW_DIR, WINDOW_SAMPLES, WINDOW_SECS,
)


def load_participant_csv(filepath):
    df = pd.read_csv(filepath)
    missing = [column for column in SENSOR_COLS if column not in df.columns]
    if missing:
        print(f"[WARNING] Skipping {filepath.name}; missing columns: {missing}")
        return None
    if "participant_id" not in df.columns:
        print(f"[WARNING] Skipping {filepath.name}; participant_id is missing")
        return None
    return df


def create_windows(signal_array):
    starts = np.arange(0, len(signal_array) - WINDOW_SAMPLES + 1, STEP_SAMPLES)
    if len(starts) == 0:
        return np.empty((0, WINDOW_SAMPLES, len(SENSOR_COLS))), starts
    windows = np.stack([signal_array[start:start + WINDOW_SAMPLES] for start in starts])
    return windows, starts


def save_windows(windows, starts, participant_id, session_type, output_dir):
    np.save(output_dir / f"{participant_id}_windows.npy", windows)
    # Twenty-second blocks are the grouping unit during cross-validation.
    block_samples = 20 * ORIGINAL_RATE
    labels = pd.DataFrame({
        "window_index": np.arange(len(windows)),
        "start_sample": starts,
        "block_id": starts // block_samples,
        "participant_id": participant_id,
        "session_type": session_type,
    })
    labels.to_csv(output_dir / f"{participant_id}_labels.csv", index=False)


def process_session(session_name):
    input_dir = DATA_DIR / session_name
    output_dir = WINDOW_DIR / session_name
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*_SA.csv"))
    if not csv_files:
        print(f"[WARNING] No sacrum CSV files in {input_dir}")
        return

    for filepath in csv_files:
        df = load_participant_csv(filepath)
        if df is None:
            continue
        participant_id = str(df["participant_id"].iloc[0])
        if participant_id in EXCLUDED_PARTICIPANTS:
            print(f"[EXCLUDED] {participant_id}: known invalid sacrum segment")
            continue

        sensor_data = df[SENSOR_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()
        finite_rows = np.isfinite(sensor_data).all(axis=1)
        if not finite_rows.all():
            print(f"[WARNING] {filepath.name}: dropping {(~finite_rows).sum()} non-finite rows")
            sensor_data = sensor_data[finite_rows]

        windows, starts = create_windows(sensor_data)
        if not len(windows):
            print(f"[WARNING] {filepath.name}: too short for one window")
            continue
        save_windows(windows, starts, participant_id, session_name, output_dir)
        print(f"{session_name:12s} {participant_id}: {len(windows)} windows")


def main():
    print("DUO-GAIT sacrum preprocessing")
    print(f"128 Hz, {WINDOW_SECS}s windows, {OVERLAP:.0%} overlap")
    for session in SESSIONS:
        process_session(session)
    print(f"Windows saved under {WINDOW_DIR}")


if __name__ == "__main__":
    main()
