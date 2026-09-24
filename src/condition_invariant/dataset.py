"""Load processed sacrum windows using accelerometer and gyroscope magnitude."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.condition_invariant.config import (
    CONDITIONS,
    STORED_SENSOR_CHANNELS,
    WINDOW_SAMPLES,
    WINDOWS_DIR,
)

from src.condition_invariant.records import (
    GaitWindow,
)


REQUIRED_LABEL_COLUMNS = (
    "window_index",
    "start_sample",
    "block_id",
    "participant_id",
    "session_type",
)


# Complete dataset organized as:
#
# dataset[participant_id][condition]
#     -> list of chronological GaitWindow objects
GaitDataset = dict[
    str,
    dict[
        str,
        list[GaitWindow],
    ],
]


def discover_participants(
    windows_dir: Path = WINDOWS_DIR,
) -> tuple[str, ...]:
    """Return participants present in every condition, rejecting mismatches."""

    participants_by_condition = {}

    for condition in CONDITIONS:

        condition_dir = (
            windows_dir
            / condition
        )

        if not condition_dir.is_dir():
            raise FileNotFoundError(
                f"Condition directory not found: "
                f"{condition_dir}"
            )

        participants = {
            path.name.removesuffix(
                "_windows.npy"
            )
            for path
            in condition_dir.glob(
                "*_windows.npy"
            )
        }

        if not participants:
            raise FileNotFoundError(
                f"No window files found in "
                f"{condition_dir}"
            )

        participants_by_condition[
            condition
        ] = participants

    reference_condition = (
        CONDITIONS[0]
    )

    reference_participants = (
        participants_by_condition[
            reference_condition
        ]
    )

    for condition in CONDITIONS[
        1:
    ]:

        condition_participants = (
            participants_by_condition[
                condition
            ]
        )

        if (
            condition_participants
            != reference_participants
        ):

            missing = sorted(
                reference_participants
                - condition_participants
            )

            unexpected = sorted(
                condition_participants
                - reference_participants
            )

            raise ValueError(
                f"Participant mismatch in "
                f"{condition}: "
                f"missing={missing}, "
                f"unexpected={unexpected}"
            )

    return tuple(
        sorted(
            reference_participants
        )
    )


def _compute_sensor_magnitudes(
    signals: np.ndarray,
    signal_path: Path,
) -> np.ndarray:
    """Convert six-channel IMU windows into AccMagnitude and GyrMagnitude."""

    if signals.ndim != 3:
        raise ValueError(
            f"{signal_path.name} should have "
            f"shape "
            f"(windows, samples, channels), "
            f"received {signals.shape}"
        )

    if (
        signals.shape[1]
        != WINDOW_SAMPLES
    ):
        raise ValueError(
            f"{signal_path.name} has "
            f"{signals.shape[1]} samples "
            f"per window; expected "
            f"{WINDOW_SAMPLES}"
        )

    expected_channels = len(
        STORED_SENSOR_CHANNELS
    )

    if (
        signals.shape[2]
        != expected_channels
    ):
        raise ValueError(
            f"{signal_path.name} has "
            f"{signals.shape[2]} channels; "
            f"expected {expected_channels}: "
            f"{STORED_SENSOR_CHANNELS}"
        )

    # Original saved channel order:
    #
    # 0 = GyrX
    # 1 = GyrY
    # 2 = GyrZ
    # 3 = AccX
    # 4 = AccY
    # 5 = AccZ

    gyro = np.asarray(
        signals[:, :, 0:3],
        dtype=np.float32,
    )

    accel = np.asarray(
        signals[:, :, 3:6],
        dtype=np.float32,
    )

    # Magnitude for every timestamp:
    #
    # GyrMagnitude =
    # sqrt(GyrX^2 + GyrY^2 + GyrZ^2)
    #
    # AccMagnitude =
    # sqrt(AccX^2 + AccY^2 + AccZ^2)

    gyro_magnitude = np.sqrt(
        np.sum(
            np.square(
                gyro,
                dtype=np.float32,
            ),
            axis=2,
        )
    )

    accel_magnitude = np.sqrt(
        np.sum(
            np.square(
                accel,
                dtype=np.float32,
            ),
            axis=2,
        )
    )

    # Stack as:
    #
    # channel 0 = AccMagnitude
    # channel 1 = GyrMagnitude
    #
    # Result:
    # (number_of_windows, 256, 2)

    magnitudes = np.stack(
        (
            accel_magnitude,
            gyro_magnitude,
        ),
        axis=2,
    )

    return np.asarray(
        magnitudes,
        dtype=np.float32,
    )


def load_participant_windows(
    participant_id: str,
    condition: str,
    windows_dir: Path = WINDOWS_DIR,
) -> list[GaitWindow]:
    """Load and validate one participant's magnitude windows."""

    if condition not in CONDITIONS:
        raise ValueError(
            f"Unknown condition: "
            f"{condition!r}"
        )

    condition_dir = (
        windows_dir
        / condition
    )

    signal_path = (
        condition_dir
        / f"{participant_id}_windows.npy"
    )

    labels_path = (
        condition_dir
        / f"{participant_id}_labels.csv"
    )

    if not signal_path.is_file():
        raise FileNotFoundError(
            f"Window file not found: "
            f"{signal_path}"
        )

    if not labels_path.is_file():
        raise FileNotFoundError(
            f"Label file not found: "
            f"{labels_path}"
        )

    # Load the original six-channel windows.
    stored_signals = np.load(
        signal_path,
        allow_pickle=False,
    )

    # Convert to:
    #
    # [AccMagnitude, GyrMagnitude]
    signals = _compute_sensor_magnitudes(
        stored_signals,
        signal_path,
    )

    labels = pd.read_csv(
        labels_path
    )

    missing_columns = [
        column
        for column
        in REQUIRED_LABEL_COLUMNS
        if column
        not in labels.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{labels_path.name} is "
            f"missing required columns: "
            f"{missing_columns}"
        )

    if (
        len(signals)
        != len(labels)
    ):
        raise ValueError(
            f"Signal/label count mismatch "
            f"for {participant_id} "
            f"{condition}: "
            f"{len(signals)} signals and "
            f"{len(labels)} labels"
        )

    records: list[
        GaitWindow
    ] = []

    for (
        row_number,
        row,
    ) in (
        labels
        .reset_index(
            drop=True
        )
        .iterrows()
    ):

        label_participant = str(
            row[
                "participant_id"
            ]
        )

        label_condition = str(
            row[
                "session_type"
            ]
        )

        if (
            label_participant
            != participant_id
        ):
            raise ValueError(
                f"Row {row_number} "
                f"participant is "
                f"{label_participant!r}; "
                f"expected "
                f"{participant_id!r}"
            )

        if (
            label_condition
            != condition
        ):
            raise ValueError(
                f"Row {row_number} "
                f"condition is "
                f"{label_condition!r}; "
                f"expected "
                f"{condition!r}"
            )

        records.append(
            GaitWindow(
                participant_id=(
                    label_participant
                ),
                condition=(
                    label_condition
                ),
                window_index=int(
                    row[
                        "window_index"
                    ]
                ),
                start_sample=int(
                    row[
                        "start_sample"
                    ]
                ),
                block_id=int(
                    row[
                        "block_id"
                    ]
                ),
                signal=np.asarray(
                    signals[
                        row_number
                    ],
                    dtype=np.float32,
                ),
            )
        )

    return records


def load_all_windows(
    windows_dir: Path = WINDOWS_DIR,
) -> GaitDataset:
    """Load every participant using AccMagnitude and GyrMagnitude."""

    participants = (
        discover_participants(
            windows_dir
        )
    )

    dataset: GaitDataset = {}

    for participant_id in (
        participants
    ):

        dataset[
            participant_id
        ] = {}

        for condition in CONDITIONS:

            records = (
                load_participant_windows(
                    participant_id=(
                        participant_id
                    ),
                    condition=(
                        condition
                    ),
                    windows_dir=(
                        windows_dir
                    ),
                )
            )

            if not records:
                raise ValueError(
                    f"No windows loaded for "
                    f"{participant_id} "
                    f"under {condition}"
                )

            dataset[
                participant_id
            ][
                condition
            ] = records

    return dataset
