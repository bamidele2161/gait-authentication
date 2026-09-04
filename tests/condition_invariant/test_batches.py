"""Checks for balanced identity-condition batch sampling."""

from collections import Counter

import numpy as np

from src.condition_invariant.batches import sample_identity_condition_batches
from src.condition_invariant.config import CONDITIONS
from src.condition_invariant.records import GaitWindow
from src.condition_invariant.triplets import build_triplet_index


def make_windows() -> tuple[GaitWindow, ...]:
    windows = []
    for participant_number in range(1, 5):
        participant_id = f"sub_{participant_number:02d}"
        for condition in CONDITIONS:
            for index in range(3):
                windows.append(
                    GaitWindow(
                        participant_id=participant_id,
                        condition=condition,
                        window_index=index,
                        start_sample=index * 128,
                        block_id=0,
                        signal=np.full((256, 6), participant_number + index),
                    )
                )
    return tuple(windows)


def test_batch_balances_participants_and_conditions() -> None:
    index = build_triplet_index(make_windows())
    labels = {participant_id: number for number, participant_id in enumerate(index)}

    batch = sample_identity_condition_batches(
        index,
        labels,
        seed=4,
        number_of_batches=1,
        participants_per_batch=3,
        windows_per_condition=2,
    )[0]

    assert len(batch.windows) == 3 * 4 * 2
    counts = Counter((window.participant_id, window.condition) for window in batch.windows)
    assert set(counts.values()) == {2}
    assert {
        label for label in batch.identity_labels
    } == {labels[participant] for participant in batch.participant_ids}


def test_balanced_batches_are_reproducible() -> None:
    index = build_triplet_index(make_windows())
    labels = {participant_id: number for number, participant_id in enumerate(index)}
    arguments = dict(
        index=index,
        identity_to_label=labels,
        seed=9,
        number_of_batches=2,
        participants_per_batch=2,
        windows_per_condition=1,
    )

    first = sample_identity_condition_batches(**arguments)
    second = sample_identity_condition_batches(**arguments)

    assert [batch.identity_labels for batch in first] == [
        batch.identity_labels for batch in second
    ]
    assert [
        [(window.participant_id, window.condition, window.window_index) for window in batch.windows]
        for batch in first
    ] == [
        [(window.participant_id, window.condition, window.window_index) for window in batch.windows]
        for batch in second
    ]
