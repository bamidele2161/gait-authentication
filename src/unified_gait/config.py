"""Single source of truth for the unified experiment."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WINDOWS_DIR = ROOT / "src/data/processed/windows/sacrum_time"
MODEL_DIR = ROOT / "src/models/unified_gait"
RESULT_DIR = ROOT / "src/results/unified_gait"
CONDITIONS = ("st_control", "st_fatigue", "dt_control", "dt_fatigue")


@dataclass(frozen=True)
class Config:
    seed: int = 42
    epochs: int = 40
    patience: int = 7
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    embedding_size: int = 64
    identities_per_batch: int = 8
    samples_per_condition: int = 2
    batches_per_epoch: int = 100
    margin: float = 0.2
    fusion_window: int = 30
    rotation_degrees: float = 12.0
