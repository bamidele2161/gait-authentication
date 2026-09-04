"""Balanced identity-condition batches for metric learning."""

from dataclasses import dataclass

import numpy as np

from src.condition_invariant.config import CONDITIONS
from src.condition_invariant.records import GaitWindow
from src.condition_invariant.triplets import TripletIndex


@dataclass(frozen=True)
class IdentityConditionBatch:
    """Windows and integer identity labels for one balanced training batch."""

    windows: tuple[GaitWindow, ...]
    identity_labels: tuple[int, ...]
    participant_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("A balanced batch must not be empty")
        if len(self.windows) != len(self.identity_labels):
            raise ValueError("Every batch window needs one identity label")
        if len(set(self.participant_ids)) != len(self.participant_ids):
            raise ValueError("Batch participant IDs must be unique")


def sample_identity_condition_batches(
    index: TripletIndex,
    identity_to_label: dict[str, int],
    seed: int,
    number_of_batches: int,
    participants_per_batch: int,
    windows_per_condition: int,
) -> tuple[IdentityConditionBatch, ...]:
    """Sample equal windows from every condition for each selected identity."""

    if number_of_batches < 1:
        raise ValueError("number_of_batches must be at least 1")
    if participants_per_batch < 2:
        raise ValueError("participants_per_batch must be at least 2")
    if windows_per_condition < 1:
        raise ValueError("windows_per_condition must be at least 1")

    participants = tuple(sorted(index))
    if participants_per_batch > len(participants):
        raise ValueError("participants_per_batch exceeds available participants")
    if set(identity_to_label) != set(participants):
        raise ValueError("Identity-label mapping does not match indexed participants")

    random_generator = np.random.default_rng(seed)
    batches = []
    for _ in range(number_of_batches):
        selected = tuple(
            sorted(
                random_generator.choice(
                    participants,
                    size=participants_per_batch,
                    replace=False,
                ).tolist()
            )
        )
        windows = []
        labels = []
        for participant_id in selected:
            for condition in CONDITIONS:
                candidates = index[participant_id][condition]
                if len(candidates) < windows_per_condition:
                    raise ValueError(
                        f"{participant_id} {condition} has too few windows for a batch"
                    )
                chosen_positions = random_generator.choice(
                    len(candidates),
                    size=windows_per_condition,
                    replace=False,
                )
                for position in chosen_positions:
                    windows.append(candidates[int(position)])
                    labels.append(identity_to_label[participant_id])

        order = random_generator.permutation(len(windows))
        batches.append(
            IdentityConditionBatch(
                windows=tuple(windows[int(position)] for position in order),
                identity_labels=tuple(labels[int(position)] for position in order),
                participant_ids=selected,
            )
        )
    return tuple(batches)
