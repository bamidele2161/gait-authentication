"""Create IMU windows while retaining chronological grouping metadata.

Default (``GAIT_SENSOR_SET=sacrum``): sacrum CSVs organized by 01, as before.
``GAIT_SENSOR_SET=all9``: the nine synchronized IMU files of every recording
are read from ``raw_data/interim``, trimmed identically, and stacked side by
side into 9 x 6 = 54 channels before windowing.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (  # noqa: E402
    BLOCK_SAMPLES, DATA_DIR, EXCLUDED_PARTICIPANTS, KNOWN_TIME_TRIMS, OVERLAP,
    RAW_INTERIM_DIR, SENSOR_CODES, SENSOR_COLS, SENSOR_SET, SESSIONS,
    STEP_SAMPLES, WINDOW_DIR, WINDOW_SAMPLES, WINDOW_SECS,
)

# CSV timestamps are written with finite decimal precision.  A 0.0001-second
# tolerance is only 0.0128 of one sample at 128 Hz, so it accepts harmless
# rounding (observed maximum: 0.00001 s) while still detecting a real one-row
# misalignment (1 / 128 = 0.0078125 s).
TIMESTAMP_ALIGNMENT_TOLERANCE_SECONDS = 1e-4


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
        return np.empty((0, WINDOW_SAMPLES, signal_array.shape[1])), starts
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
    labels = pd.DataFrame({
        "window_index": np.arange(len(windows)),
        "start_sample": starts,
        "block_id": starts // BLOCK_SAMPLES,
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


def process_session_all_sensors(session_name):
    """Window the synchronized SENSOR_CODES recordings of every participant."""
    input_dir = RAW_INTERIM_DIR / f"OG_{session_name}"
    output_dir = WINDOW_DIR / session_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for participant_dir in sorted(input_dir.glob("sub_*")):
        participant_id = participant_dir.name
        if participant_id in EXCLUDED_PARTICIPANTS:
            print(f"[EXCLUDED] {participant_id}")
            continue
        blocks = []
        timestamps = []
        for code in SENSOR_CODES:
            filepath = participant_dir / f"{code}.csv"
            if not filepath.exists():
                raise FileNotFoundError(
                    f"{participant_id} {session_name}: missing sensor file {filepath.name}"
                )
            df = pd.read_csv(filepath)
            df = apply_known_time_trim(df, participant_id, session_name, filepath)
            missing = [column for column in ["timestamp", *SENSOR_COLS] if column not in df]
            if missing:
                raise ValueError(f"{filepath}: missing columns {missing}")
            time = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy()
            block = df[SENSOR_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()
            if not np.isfinite(time).all() or not np.isfinite(block).all():
                raise ValueError(
                    f"{participant_id} {session_name} {code}: non-finite timestamp/samples"
                )
            timestamps.append(time)
            blocks.append(block)
        # The sensors are synchronized; align on the shortest recording.
        length = min(len(block) for block in blocks)
        reference_time = timestamps[0][:length]
        for code, time in zip(SENSOR_CODES[1:], timestamps[1:]):
            if not np.allclose(
                time[:length], reference_time, rtol=0.0,
                atol=TIMESTAMP_ALIGNMENT_TOLERANCE_SECONDS,
            ):
                maximum_error = float(np.max(np.abs(time[:length] - reference_time)))
                raise ValueError(
                    f"{participant_id} {session_name}: {code} timestamps are not "
                    f"aligned with {SENSOR_CODES[0]} (max error {maximum_error:g} s)"
                )
        sensor_data = np.hstack([block[:length] for block in blocks])
        expected_channels = len(SENSOR_CODES) * len(SENSOR_COLS)
        if sensor_data.shape[1] != expected_channels:
            raise ValueError(
                f"{participant_id} {session_name}: expected {expected_channels} "
                f"channels, found {sensor_data.shape[1]}"
            )

        windows, starts = create_windows(sensor_data)
        if not len(windows):
            print(f"[WARNING] {participant_id}: too short for one window")
            continue
        save_windows(windows, starts, participant_id, session_name, output_dir)
        print(f"{session_name:12s} {participant_id}: {len(windows)} windows x {sensor_data.shape[1]} channels")


def main():
    if SENSOR_SET != "sacrum":
        print(f"DUO-GAIT {SENSOR_SET} preprocessing: {', '.join(SENSOR_CODES)}")
        print(f"128 Hz, {WINDOW_SECS}s windows, {OVERLAP:.0%} overlap")
        for session in SESSIONS:
            process_session_all_sensors(session)
        print(f"Windows saved under {WINDOW_DIR}")
        return
    print("DUO-GAIT sacrum preprocessing")
    print(f"128 Hz, {WINDOW_SECS}s windows, {OVERLAP:.0%} overlap")
    for session in SESSIONS:
        process_session(session)
    print(f"Windows saved under {WINDOW_DIR}")


if __name__ == "__main__":
    main()
