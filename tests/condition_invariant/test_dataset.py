"""Checks for loading processed sacrum windows."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.condition_invariant.config import CONDITIONS
from src.condition_invariant.dataset import (
    discover_participants,
    load_participant_windows,
)
from tests.condition_invariant.test_records import assert_raises


def write_test_participant(
    root: Path,
    *,
    number_of_signals: int = 2,
    number_of_labels: int = 2,
    participant_id: str = "sub_01",
    label_participant_id: str = "sub_01",
    condition: str = "st_control",
    label_condition: str = "st_control",
) -> None:
    """Create a small processed-window fixture matching the real file format."""

    condition_dir = root / condition
    condition_dir.mkdir(parents=True)

    signals = np.zeros((number_of_signals, 256, 6), dtype=np.float64)
    signals[1:] = 1.0
    np.save(condition_dir / f"{participant_id}_windows.npy", signals)

    labels = pd.DataFrame({
        "window_index": list(range(number_of_labels)),
        "start_sample": [index * 128 for index in range(number_of_labels)],
        "block_id": [0] * number_of_labels,
        "participant_id": [label_participant_id] * number_of_labels,
        "session_type": [label_condition] * number_of_labels,
    })
    labels.to_csv(condition_dir / f"{participant_id}_labels.csv", index=False)


def write_participant_files_for_all_conditions(
    root: Path,
    participants: tuple[str, ...],
) -> None:
    """Create discoverable window filenames for every configured condition."""

    for condition in CONDITIONS:
        condition_dir = root / condition
        condition_dir.mkdir(parents=True)
        for participant_id in participants:
            np.save(
                condition_dir / f"{participant_id}_windows.npy",
                np.zeros((1, 256, 6), dtype=np.float32),
            )


def test_discover_participants_returns_sorted_shared_participants(tmp_path: Path) -> None:
    write_participant_files_for_all_conditions(tmp_path, ("sub_03", "sub_01"))

    participants = discover_participants(tmp_path)

    assert participants == ("sub_01", "sub_03")


def test_discover_participants_rejects_condition_mismatch(tmp_path: Path) -> None:
    write_participant_files_for_all_conditions(tmp_path, ("sub_01", "sub_02"))
    (tmp_path / "dt_fatigue" / "sub_02_windows.npy").unlink()

    assert_raises(
        ValueError,
        "Participant mismatch in dt_fatigue",
        lambda: discover_participants(tmp_path),
    )


def test_load_participant_windows_returns_valid_records(tmp_path: Path) -> None:
    write_test_participant(tmp_path)

    records = load_participant_windows("sub_01", "st_control", tmp_path)

    assert len(records) == 2
    assert records[0].participant_id == "sub_01"
    assert records[0].condition == "st_control"
    assert records[0].window_index == 0
    assert records[1].start_sample == 128
    assert records[0].signal.shape == (256, 6)
    assert records[0].signal.dtype == np.float32
    assert records[1].signal[0, 0] == 1.0


def test_load_participant_windows_rejects_count_mismatch(tmp_path: Path) -> None:
    write_test_participant(tmp_path, number_of_signals=2, number_of_labels=1)

    assert_raises(
        ValueError,
        "Signal/label count mismatch",
        lambda: load_participant_windows("sub_01", "st_control", tmp_path),
    )


def test_load_participant_windows_rejects_wrong_label_identity(tmp_path: Path) -> None:
    write_test_participant(tmp_path, label_participant_id="sub_02")

    assert_raises(
        ValueError,
        "expected 'sub_01'",
        lambda: load_participant_windows("sub_01", "st_control", tmp_path),
    )


def test_load_participant_windows_rejects_missing_file(tmp_path: Path) -> None:
    assert_raises(
        FileNotFoundError,
        "Window file not found",
        lambda: load_participant_windows("sub_01", "st_control", tmp_path),
    )
