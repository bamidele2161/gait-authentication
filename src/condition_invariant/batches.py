"""Balanced session-1 batches for online hard-pair mining."""

from dataclasses import dataclass

import numpy as np

from src.condition_invariant.config import TRAINING_CONDITIONS
from src.condition_invariant.records import GaitWindow
from src.condition_invariant.triplets import TripletIndex


@dataclass(frozen=True)
class Session1Batch:
    windows: tuple[GaitWindow, ...]
    identity_labels: tuple[int, ...]
    condition_labels: tuple[int, ...]


def sample_session1_batch(
    index: TripletIndex,
    identity_to_label: dict[str, int],
    rng: np.random.Generator,
    participants_per_batch: int,
    windows_per_condition: int,
) -> Session1Batch:
    """Sample equal ST-control/ST-fatigue windows for selected identities."""
    people = tuple(index)
    if participants_per_batch > len(people):
        raise ValueError("participants_per_batch exceeds available identities")
    selected = rng.choice(people, participants_per_batch, replace=False)
    windows, identities, conditions = [], [], []
    for person in selected:
        for condition_label, condition in enumerate(TRAINING_CONDITIONS):
            candidates = index[str(person)][condition]
            positions = rng.choice(
                len(candidates), windows_per_condition,
                replace=len(candidates) < windows_per_condition,
            )
            for position in positions:
                windows.append(candidates[int(position)])
                identities.append(identity_to_label[str(person)])
                conditions.append(condition_label)
    order = rng.permutation(len(windows))
    return Session1Batch(
        tuple(windows[int(i)] for i in order),
        tuple(identities[int(i)] for i in order),
        tuple(conditions[int(i)] for i in order),
    )
