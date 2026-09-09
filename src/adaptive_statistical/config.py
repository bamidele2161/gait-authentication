from dataclasses import dataclass
from pathlib import Path

from src.utils import SENSOR_COLS, SESSIONS, WINDOW_DIR


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "src/results/adaptive_statistical"
MODEL_DIR = ROOT / "src/models/adaptive_statistical"
CONDITIONS = tuple(SESSIONS)
SENSORS = tuple(SENSOR_COLS)
STATISTICS = (
    "mean", "std", "median", "min", "max", "range", "iqr", "mad", "rms",
    "skewness", "kurtosis",
)
SIGNALS = (*SENSORS, "AccMag", "GyrMag")
FEATURES = tuple(f"{signal}_{stat}" for signal in SIGNALS for stat in STATISTICS)


@dataclass(frozen=True)
class Config:
    target_far: float = 0.01
    ema_alpha: float = 0.02
    fusion_window: int = 5
    seed: int = 42
    maximum_cohort_windows: int = 80
