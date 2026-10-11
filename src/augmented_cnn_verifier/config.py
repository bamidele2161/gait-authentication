"""Experiment paths and immutable defaults."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
WINDOWS_DIR = SRC_DIR / "data" / "processed" / "windows" / "sacrum_time"
MODELS_DIR = SRC_DIR / "models" / "augmented_cnn_verifier"
RESULTS_DIR = SRC_DIR / "results" / "augmented_cnn_verifier"

TRAINING_CONDITIONS = ("st_control", "st_fatigue")
EVALUATION_CONDITIONS = ("dt_control", "dt_fatigue")
WINDOW_SAMPLES = 256
CHANNELS = 2

