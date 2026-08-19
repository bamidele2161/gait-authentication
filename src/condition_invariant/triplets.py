"""Triplet records and sampling for condition-invariant learning."""

from dataclasses import dataclass

import numpy as np

from src.condition_invariant.config import CONDITIONS
from src.condition_invariant.records import GaitWindow


TripletIndex = dict[str, dict[str, tuple[GaitWindow, ...]]]

# Four same-condition pairs plus every unordered cross-condition pair.
POSITIVE_CONDITION_PAIRS = tuple(
    [(condition, condition) for condition in CONDITIONS]
    + [
        (first_condition, second_condition)
        for first_index, first_condition in enumerate(CONDITIONS)
        for second_condition in CONDITIONS[first_index + 1:]
    ]
)


def window_identity(window: GaitWindow) -> tuple[str, str, int, int]:
    """Return the metadata fields that uniquely locate one gait window."""

    return (
        window.participant_id,
        window.condition,
        window.window_index,
        window.start_sample,
    )


@dataclass(frozen=True)
class GaitTriplet:
    """An anchor, same-person positive, and different-person negative."""

    anchor: GaitWindow
    positive: GaitWindow
    negative: GaitWindow

    def __post_init__(self) -> None:
        """Reject a triplet that violates identity-learning rules."""

        if self.anchor.participant_id != self.positive.participant_id:
            raise ValueError("Anchor and positive must belong to the same participant")
        if window_identity(self.anchor) == window_identity(self.positive):
            raise ValueError("Anchor and positive must be different windows")
        if self.anchor.participant_id == self.negative.participant_id:
            raise ValueError("Negative must belong to a different participant")


def build_triplet_index(windows: tuple[GaitWindow, ...]) -> TripletIndex:
    """Organize learning windows by participant and then by condition."""

    if not windows:
        raise ValueError("Cannot index an empty window collection")

    duplicate_check = set()
    grouped: dict[str, dict[str, list[GaitWindow]]] = {}
    for window in windows:
        identity = window_identity(window)
        if identity in duplicate_check:
            raise ValueError(f"Duplicate gait window found: {identity}")
        duplicate_check.add(identity)

        participant_conditions = grouped.setdefault(window.participant_id, {})
        participant_conditions.setdefault(window.condition, []).append(window)

    if len(grouped) < 2:
        raise ValueError("Triplet learning requires at least two participants")

    index: TripletIndex = {}
    for participant_id in sorted(grouped):
        condition_groups = grouped[participant_id]
        missing_conditions = [
            condition for condition in CONDITIONS if condition not in condition_groups
        ]
        if missing_conditions:
            raise ValueError(
                f"{participant_id} is missing triplet conditions: {missing_conditions}"
            )

        index[participant_id] = {}
        for condition in CONDITIONS:
            ordered = tuple(
                sorted(
                    condition_groups[condition],
                    key=lambda window: window.start_sample,
                )
            )
            if len(ordered) < 2:
                raise ValueError(
                    f"{participant_id} {condition} needs at least two windows"
                )
            index[participant_id][condition] = ordered

    return index


def sample_triplet(
    index: TripletIndex,
    random_generator: np.random.Generator,
    anchor_participant: str,
    anchor_condition: str,
    positive_condition: str,
    negative_condition: str,
) -> GaitTriplet:
    """Sample one reproducible triplet from explicit condition roles."""

    if anchor_participant not in index:
        raise ValueError(f"Unknown anchor participant: {anchor_participant}")
    for condition in (anchor_condition, positive_condition, negative_condition):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")

    anchor_candidates = index[anchor_participant][anchor_condition]
    positive_candidates = index[anchor_participant][positive_condition]
    anchor = anchor_candidates[
        int(random_generator.integers(len(anchor_candidates)))
    ]

    eligible_positives = tuple(
        window
        for window in positive_candidates
        if window_identity(window) != window_identity(anchor)
    )
    if not eligible_positives:
        raise ValueError("No positive window is available after excluding the anchor")
    positive = eligible_positives[
        int(random_generator.integers(len(eligible_positives)))
    ]

    negative_participants = tuple(
        participant_id
        for participant_id in index
        if participant_id != anchor_participant
    )
    if not negative_participants:
        raise ValueError("No different participant is available for the negative")
    negative_participant = negative_participants[
        int(random_generator.integers(len(negative_participants)))
    ]
    negative_candidates = index[negative_participant][negative_condition]
    negative = negative_candidates[
        int(random_generator.integers(len(negative_candidates)))
    ]

    return GaitTriplet(anchor=anchor, positive=positive, negative=negative)


def sample_balanced_triplets(
    index: TripletIndex,
    seed: int,
    repetitions: int = 1,
) -> tuple[GaitTriplet, ...]:
    """Sample balanced condition combinations for every participant."""

    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")

    random_generator = np.random.default_rng(seed)
    triplets = []
    for _ in range(repetitions):
        for anchor_participant in index:
            for first_condition, second_condition in POSITIVE_CONDITION_PAIRS:
                # Distance is symmetric, but swapping cross-condition roles
                # avoids always making ST-control the anchor.
                if (
                    first_condition != second_condition
                    and bool(random_generator.integers(2))
                ):
                    anchor_condition = second_condition
                    positive_condition = first_condition
                else:
                    anchor_condition = first_condition
                    positive_condition = second_condition

                for negative_condition in CONDITIONS:
                    triplets.append(
                        sample_triplet(
                            index=index,
                            random_generator=random_generator,
                            anchor_participant=anchor_participant,
                            anchor_condition=anchor_condition,
                            positive_condition=positive_condition,
                            negative_condition=negative_condition,
                        )
                    )

    order = random_generator.permutation(len(triplets))
    return tuple(triplets[int(position)] for position in order)
