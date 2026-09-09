from dataclasses import dataclass
from pathlib import Path

from src.utils import FEATURE_COLS, FEATURE_DIR, SESSIONS


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "src/results/domain_adversarial_svm"
MODEL_DIR = ROOT / "src/models/domain_adversarial_svm"
FEATURES = tuple(FEATURE_COLS)
CONDITIONS = tuple(SESSIONS)


@dataclass(frozen=True)
class Config:
    seed: int = 42
    pca_components: int = 30
    removal_strengths: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    svm_c: float = 10.0
    maximum_negative_windows_per_group: int = 40
    fusion_window: int = 30
    inner_folds: int = 3
