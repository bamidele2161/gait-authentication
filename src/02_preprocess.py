"""Create sacrum windows while retaining chronological grouping metadata."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (  # noqa: E402
    DATA_DIR, EXCLUDED_PARTICIPANTS, KNOWN_TIME_TRIMS, ORIGINAL_RATE, OVERLAP,
    SENSOR_COLS, SESSIONS, STEP_SAMPLES, WINDOW_DIR, WINDOW_SAMPLES, WINDOW_SECS,
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


def apply_known_time_trim(df, participant_id, session_name, filepath):
    """Apply a pre-specified recording correction and validate its boundaries."""
    trim = KNOWN_TIME_TRIMS.get((session_name, participant_id))
    if trim is None:
        return df
    if "timestamp" not in df.columns:
        raise ValueError(f"{filepath.name}: timestamp is required for the known trim")

    timestamps = pd.to_numeric(df["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{filepath.name}: non-numeric timestamps prevent the known trim")

    selected = df.loc[timestamps.between(trim["start"], trim["end"], inclusive="both")].copy()
    if len(selected) != trim["expected_rows"]:
        raise ValueError(
            f"{filepath.name}: expected {trim['expected_rows']} rows after trimming "
            f"{trim['start']}--{trim['end']} s, found {len(selected)}"
        )

    selected_timestamps = pd.to_numeric(selected["timestamp"])
    if not (
        np.isclose(selected_timestamps.iloc[0], trim["start"], atol=1e-5)
        and np.isclose(selected_timestamps.iloc[-1], trim["end"], atol=1e-5)
    ):
        raise ValueError(f"{filepath.name}: trimmed timestamps do not match the expected boundaries")

    print(
        f"[TRIMMED] {participant_id} {session_name}: "
        f"{len(df)} -> {len(selected)} rows ({trim['start']}--{trim['end']} s)"
    )
    return selected


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

        df = apply_known_time_trim(df, participant_id, session_name, filepath)

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
