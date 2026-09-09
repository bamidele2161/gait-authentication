from dataclasses import dataclass
from pathlib import Path

from src.utils import FEATURE_COLS, FEATURE_DIR, SESSIONS


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "src/models/fisher_metric"
RESULT_DIR = ROOT / "src/results/fisher_metric"
CONDITIONS = tuple(SESSIONS)
FEATURES = tuple(FEATURE_COLS)


@dataclass(frozen=True)
class Config:
    seed: int = 42
    outer_folds: int = 4
    inner_folds: int = 3
    learning_fraction: float = 0.8
    maximum_windows_per_group: int = 120
    pca_components: int = 30
    fisher_dimensions: int = 7
    candidate_feature_counts: tuple = (12, 20, 30, 40, 60, 88)
    fusion_window: int = 30
