"""Fixed protocol for the paper-faithful triplet experiment."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WINDOWS_DIR = ROOT / "src" / "data" / "processed" / "windows" / "sacrum_time"
MODEL_DIR = ROOT / "src" / "models" / "paper_triplet"
RESULT_DIR = ROOT / "src" / "results" / "paper_triplet"

CONDITIONS = ("st_control", "st_fatigue", "dt_control", "dt_fatigue")
SENSOR_CHANNELS = ("GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ")
SAMPLE_RATE = 128
WINDOW_SAMPLES = 256
WINDOW_STEP = 128
EMBEDDING_SIZE = 48
MARGIN = 0.2


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    hidden_size: int = 64
    lstm_layers: int = 2
    anchors_per_participant_condition: int = 30
    batch_size: int = 128
    negative_candidates: int = 4
    patience: int = 6
    seed: int = 42
    fusion_window: int = 5
