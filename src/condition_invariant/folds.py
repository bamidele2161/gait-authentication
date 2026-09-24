"""Participant-disjoint splits for condition-invariant evaluation."""

from dataclasses import dataclass

from src.condition_invariant.config import TRAINING_CONDITIONS, WINDOW_SAMPLES
from src.condition_invariant.dataset import GaitDataset
from src.condition_invariant.records import GaitWindow


@dataclass(frozen=True)
class OuterFold:
    """One run's development population and completely unseen users."""

    fold_index: int
    development_participants: tuple[str, ...]
    evaluation_participants: tuple[str, ...]


@dataclass(frozen=True)
class DevelopmentRecordingSplit:
    """Chronological learning and validation windows from one recording."""

    learning_windows: tuple[GaitWindow, ...]
    validation_windows: tuple[GaitWindow, ...]


@dataclass(frozen=True)
class DevelopmentFoldData:
    """All learning and validation windows allowed in one outer fold."""

    learning_windows: tuple[GaitWindow, ...]
    validation_windows: tuple[GaitWindow, ...]



def create_outer_folds(
    participants: tuple[str, ...],
    number_of_folds: int = 2,
) -> tuple[OuterFold, ...]:
    """Create deterministic folds where every participant is unseen once."""

    unique_participants = tuple(sorted(set(participants)))
    if len(unique_participants) != len(participants):
        raise ValueError("Participant IDs must be unique")
    if number_of_folds < 2:
        raise ValueError("number_of_folds must be at least 2")
    if len(unique_participants) % number_of_folds != 0:
        raise ValueError(
            f"Cannot divide {len(unique_participants)} participants equally "
            f"across {number_of_folds} folds"
        )

    folds = []
    for fold_index in range(number_of_folds):
        evaluation_participants = unique_participants[
            fold_index::number_of_folds
        ]
        evaluation_set = set(evaluation_participants)
        development_participants = tuple(
            participant
            for participant in unique_participants
            if participant not in evaluation_set
        )

        if set(development_participants) & evaluation_set:
            raise AssertionError("Development and evaluation participants overlap")
        if set(development_participants) | evaluation_set != set(unique_participants):
            raise AssertionError("Fold does not contain every participant")

        folds.append(
            OuterFold(
                fold_index=fold_index,
                development_participants=development_participants,
                evaluation_participants=evaluation_participants,
            )
        )

    evaluated_once = [
        participant
        for fold in folds
        for participant in fold.evaluation_participants
    ]
    if sorted(evaluated_once) != list(unique_participants):
        raise AssertionError("Every participant must be evaluated exactly once")

    return tuple(folds)


def split_development_recording(
    windows: list[GaitWindow],
    learning_fraction: float = 0.80,
) -> DevelopmentRecordingSplit:
    """Split one recording chronologically using complete block IDs."""

    if not 0 < learning_fraction < 1:
        raise ValueError("learning_fraction must be between 0 and 1")
    if not windows:
        raise ValueError("Cannot split an empty recording")

    ordered_windows = sorted(windows, key=lambda window: window.start_sample)
    participant_ids = {window.participant_id for window in ordered_windows}
    conditions = {window.condition for window in ordered_windows}
    if len(participant_ids) != 1 or len(conditions) != 1:
        raise ValueError("All windows must come from one participant and condition")

    block_ids = sorted({window.block_id for window in ordered_windows})
    if len(block_ids) < 2:
        raise ValueError("At least two chronological blocks are required")

    learning_block_count = int(len(block_ids) * learning_fraction)
    learning_block_count = min(max(learning_block_count, 1), len(block_ids) - 1)
    learning_blocks = set(block_ids[:learning_block_count])
    validation_blocks = set(block_ids[learning_block_count:])

    learning_windows = [
        window for window in ordered_windows if window.block_id in learning_blocks
    ]
    validation_windows = [
        window for window in ordered_windows if window.block_id in validation_blocks
    ]

    # The last learning window can overlap the first validation window even
    # though their block IDs differ. Remove learning windows until no raw sample
    # can occur in both partitions.
    first_validation_start = validation_windows[0].start_sample
    learning_windows = [
        window
        for window in learning_windows
        if window.start_sample + WINDOW_SAMPLES <= first_validation_start
    ]

    if not learning_windows or not validation_windows:
        raise ValueError("Split produced an empty learning or validation partition")
    if learning_windows[-1].start_sample + WINDOW_SAMPLES > first_validation_start:
        raise AssertionError("Learning and validation windows share raw samples")

    return DevelopmentRecordingSplit(
        learning_windows=tuple(learning_windows),
        validation_windows=tuple(validation_windows),
    )


def prepare_development_fold_data(
    dataset: GaitDataset,
    fold: OuterFold,
    learning_fraction: float = 0.80,
) -> DevelopmentFoldData:
    """Collect chronological learning/validation windows for one outer fold."""

    dataset_participants = set(dataset)
    development_set = set(fold.development_participants)
    evaluation_set = set(fold.evaluation_participants)

    if development_set & evaluation_set:
        raise ValueError("Development and evaluation participants overlap")
    if not development_set <= dataset_participants:
        missing = sorted(development_set - dataset_participants)
        raise ValueError(f"Development participants missing from dataset: {missing}")

    learning_windows = []
    validation_windows = []
    for participant_id in fold.development_participants:
        participant_data = dataset[participant_id]
        for condition in TRAINING_CONDITIONS:
            if condition not in participant_data:
                raise ValueError(
                    f"Missing {condition} data for development participant "
                    f"{participant_id}"
                )
            recording_split = split_development_recording(
                participant_data[condition],
                learning_fraction=learning_fraction,
            )
            learning_windows.extend(recording_split.learning_windows)
            validation_windows.extend(recording_split.validation_windows)

    used_participants = {
        window.participant_id
        for window in (*learning_windows, *validation_windows)
    }
    if used_participants != development_set:
        raise AssertionError("Prepared data does not match development participants")
    if used_participants & evaluation_set:
        raise AssertionError("Evaluation participant leaked into development data")

    return DevelopmentFoldData(
        learning_windows=tuple(learning_windows),
        validation_windows=tuple(validation_windows),
    )
