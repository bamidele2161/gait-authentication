"""Data records passed between condition-invariant experiment stages."""

from dataclasses import dataclass

import numpy as np

from src.condition_invariant.config import CONDITIONS, EXPECTED_WINDOW_SHAPE


@dataclass(frozen=True)
class GaitWindow:
    """One chronological sacrum window and the metadata that identifies it."""

    participant_id: str
    condition: str
    window_index: int
    start_sample: int
    block_id: int
    signal: np.ndarray

    def __post_init__(self) -> None:
        """Reject malformed records immediately after they are created."""

        if not self.participant_id.startswith("sub_"):
            raise ValueError(f"Invalid participant ID: {self.participant_id!r}")
        if self.condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {self.condition!r}")
        if self.window_index < 0:
            raise ValueError("window_index must be non-negative")
        if self.start_sample < 0:
            raise ValueError("start_sample must be non-negative")
        if self.block_id < 0:
            raise ValueError("block_id must be non-negative")
        if self.signal.shape != EXPECTED_WINDOW_SHAPE:
            raise ValueError(
                f"Expected signal shape {EXPECTED_WINDOW_SHAPE}, "
                f"received {self.signal.shape}"
            )
        if not np.issubdtype(self.signal.dtype, np.number):
            raise TypeError("signal must contain numeric values")
        if not np.isfinite(self.signal).all():
            raise ValueError("signal contains NaN or infinite values")
