"""Adapt the paper's exhaustive action-combination mining to four conditions."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from src.paper_triplet.config import CONDITIONS


class PaperTripletDataset(Dataset):
    """Return 10 protocol triplets per selected DUO-GAIT anchor.

    With K conditions, the source protocol produces 3(K-1)+1 triplet types.
    The paper has K=6 (16 types); DUO-GAIT has K=4 (10 types).
    """

    def __init__(self, recordings, anchors_per_group=300, seed=42, negative_candidates=4):
        self.recordings = recordings
        self.participants = tuple(sorted(recordings))
        self.rng_seed = seed
        if negative_candidates < 1:
            raise ValueError("negative_candidates must be positive")
        self.negative_candidates = negative_candidates
        anchors = []
        rng = np.random.default_rng(seed)
        for participant in self.participants:
            for condition in CONDITIONS:
                count = len(recordings[participant][condition])
                take = min(anchors_per_group, count)
                positions = np.sort(rng.choice(count, take, replace=False))
                anchors.extend((participant, condition, int(position)) for position in positions)
        self.anchors = tuple(anchors)
        self.types_per_anchor = 3 * (len(CONDITIONS) - 1) + 1

    def __len__(self):
        return len(self.anchors) * self.types_per_anchor

    def __getitem__(self, index):
        anchor_index, protocol_index = divmod(index, self.types_per_anchor)
        participant, condition, position = self.anchors[anchor_index]
        other_conditions = tuple(value for value in CONDITIONS if value != condition)
        negative_participants = tuple(value for value in self.participants if value != participant)
        rng = np.random.default_rng(self.rng_seed + index * 104729)

        if protocol_index < len(other_conditions):
            # Same person/different condition; impostor in that different condition.
            positive_condition = other_conditions[protocol_index]
            negative_condition = positive_condition
        elif protocol_index < 2 * len(other_conditions):
            # Same person/same condition; impostor in a different condition.
            positive_condition = condition
            negative_condition = other_conditions[protocol_index - len(other_conditions)]
        elif protocol_index < 3 * len(other_conditions):
            # Same person/different condition; impostor in anchor condition.
            positive_condition = other_conditions[protocol_index - 2 * len(other_conditions)]
            negative_condition = condition
        else:
            # Same person/same condition; impostor in the same condition.
            positive_condition = condition
            negative_condition = condition

        positive_values = self.recordings[participant][positive_condition]
        positive_position = int(rng.integers(len(positive_values)))
        if positive_condition == condition and len(positive_values) > 1:
            while positive_position == position:
                positive_position = int(rng.integers(len(positive_values)))
        negatives = []
        for _ in range(self.negative_candidates):
            negative_participant = negative_participants[
                int(rng.integers(len(negative_participants)))
            ]
            negative_values = self.recordings[negative_participant][negative_condition]
            negative_position = int(rng.integers(len(negative_values)))
            negatives.append(negative_values[negative_position])
        return (
            self.recordings[participant][condition][position],
            positive_values[positive_position],
            np.stack(negatives),
        )
