"""Session-1 triplet construction for cross-session gait generalization."""

from dataclasses import dataclass

import numpy as np

from src.condition_invariant.config import TRAINING_CONDITIONS
from src.condition_invariant.records import GaitWindow


TripletIndex = dict[str, dict[str, tuple[GaitWindow, ...]]]


def window_identity(window: GaitWindow) -> tuple[str, str, int, int]:
    return (
        window.participant_id,
        window.condition,
        window.window_index,
        window.start_sample,
    )

@dataclass(frozen=True)
class GaitTriplet:
    anchor: GaitWindow
    positive: GaitWindow
    negative: GaitWindow

    def __post_init__(self) -> None:
        if self.anchor.condition != "st_control":
            raise ValueError("Every anchor must come from ST-control")
        if self.positive.condition not in TRAINING_CONDITIONS:
            raise ValueError("Positive must come from ST-control or ST-fatigue")
        if self.negative.condition not in TRAINING_CONDITIONS:
            raise ValueError("Negative must come from ST-control or ST-fatigue")
        if self.anchor.participant_id != self.positive.participant_id:
            raise ValueError("Anchor and positive must belong to the same participant")
        if window_identity(self.anchor) == window_identity(self.positive):
            raise ValueError("Anchor and positive must be different windows")
        if self.anchor.participant_id == self.negative.participant_id:
            raise ValueError("Negative must belong to a different participant")


def build_triplet_index(windows: tuple[GaitWindow, ...]) -> TripletIndex:
    """Index only the two session-1 conditions permitted for learning."""
    if not windows:
        raise ValueError("Cannot index an empty window collection")
    forbidden = sorted({w.condition for w in windows} - set(TRAINING_CONDITIONS))
    if forbidden:
        raise ValueError(f"Triplet learning received test-only conditions: {forbidden}")

    seen = set()
    grouped: dict[str, dict[str, list[GaitWindow]]] = {}
    for window in windows:
        identity = window_identity(window)
        if identity in seen:
            raise ValueError(f"Duplicate gait window found: {identity}")
        seen.add(identity)
        grouped.setdefault(window.participant_id, {}).setdefault(
            window.condition, []
        ).append(window)
    if len(grouped) < 2:
        raise ValueError("Triplet learning requires at least two participants")

    index = {}
    for participant in sorted(grouped):
        missing = [c for c in TRAINING_CONDITIONS if c not in grouped[participant]]
        if missing:
            raise ValueError(f"{participant} is missing training conditions: {missing}")
        index[participant] = {}
        for condition in TRAINING_CONDITIONS:
            ordered = tuple(sorted(
                grouped[participant][condition], key=lambda w: w.start_sample
            ))
            if len(ordered) < 2:
                raise ValueError(f"{participant} {condition} needs at least two windows")
            index[participant][condition] = ordered
    return index


def sample_triplet(
    index: TripletIndex,
    random_generator: np.random.Generator,
    anchor_participant: str,
    positive_condition: str,
    negative_condition: str,
) -> GaitTriplet:
    """Sample one ST-control anchor under the declared identity rules."""
    if anchor_participant not in index:
        raise ValueError(f"Unknown anchor participant: {anchor_participant}")
    if positive_condition not in TRAINING_CONDITIONS:
        raise ValueError(f"Unknown positive condition: {positive_condition}")
    if negative_condition not in TRAINING_CONDITIONS:
        raise ValueError(f"Unknown negative condition: {negative_condition}")

    anchors = index[anchor_participant]["st_control"]
    anchor = anchors[int(random_generator.integers(len(anchors)))]
    positives = tuple(
        w for w in index[anchor_participant][positive_condition]
        if window_identity(w) != window_identity(anchor)
    )
    positive = positives[int(random_generator.integers(len(positives)))]
    other_people = tuple(p for p in index if p != anchor_participant)
    negative_person = other_people[int(random_generator.integers(len(other_people)))]
    negatives = index[negative_person][negative_condition]
    negative = negatives[int(random_generator.integers(len(negatives)))]
    return GaitTriplet(anchor, positive, negative)


def sample_session1_triplets(
    index: TripletIndex,
    random_generator: np.random.Generator,
    count: int,
) -> tuple[GaitTriplet, ...]:
    """Balance positive and negative ST conditions while keeping ST-control anchors."""
    if count < 1:
        raise ValueError("count must be at least 1")
    people = tuple(index)
    roles = tuple(
        (positive, negative)
        for positive in TRAINING_CONDITIONS
        for negative in TRAINING_CONDITIONS
    )
    return tuple(
        sample_triplet(
            index,
            random_generator,
            people[position % len(people)],
            *roles[(position // len(people)) % len(roles)],
        )
        for position in random_generator.permutation(count)
    )
