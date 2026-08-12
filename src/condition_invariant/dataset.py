"""Load processed sacrum windows for the condition-invariant experiment."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.condition_invariant.config import CONDITIONS, WINDOWS_DIR
from src.condition_invariant.records import GaitWindow


REQUIRED_LABEL_COLUMNS = (
    "window_index",
    "start_sample",
    "block_id",
    "participant_id",
    "session_type",
)


def load_participant_windows(
    participant_id: str,
    condition: str,
    windows_dir: Path = WINDOWS_DIR,
) -> list[GaitWindow]:
    """Load and validate one participant's windows from one condition."""

    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition!r}")

    condition_dir = windows_dir / condition
    signal_path = condition_dir / f"{participant_id}_windows.npy"
    labels_path = condition_dir / f"{participant_id}_labels.csv"

    if not signal_path.is_file():
        raise FileNotFoundError(f"Window file not found: {signal_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    signals = np.load(signal_path, allow_pickle=False)
    labels = pd.read_csv(labels_path)

    missing_columns = [
        column for column in REQUIRED_LABEL_COLUMNS if column not in labels.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{labels_path.name} is missing required columns: {missing_columns}"
        )

    if len(signals) != len(labels):
        raise ValueError(
            f"Signal/label count mismatch for {participant_id} {condition}: "
            f"{len(signals)} signals and {len(labels)} labels"
        )

    records = []
    for row_number, row in labels.reset_index(drop=True).iterrows():
        label_participant = str(row["participant_id"])
        label_condition = str(row["session_type"])

        if label_participant != participant_id:
            raise ValueError(
                f"Row {row_number} participant is {label_participant!r}; "
                f"expected {participant_id!r}"
            )
        if label_condition != condition:
            raise ValueError(
                f"Row {row_number} condition is {label_condition!r}; "
                f"expected {condition!r}"
            )

        records.append(
            GaitWindow(
                participant_id=label_participant,
                condition=label_condition,
                window_index=int(row["window_index"]),
                start_sample=int(row["start_sample"]),
                block_id=int(row["block_id"]),
                signal=np.asarray(signals[row_number], dtype=np.float32),
            )
        )

    return records
