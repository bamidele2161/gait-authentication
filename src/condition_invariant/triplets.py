"""Triplet records and sampling for condition-invariant learning."""

from dataclasses import dataclass

from src.condition_invariant.records import GaitWindow


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
