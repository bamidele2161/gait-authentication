"""Shared configuration for the sacrum, time-domain experiment."""

from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent

ORIGINAL_RATE = 128
WINDOW_SECS = 2
WINDOW_SAMPLES = ORIGINAL_RATE * WINDOW_SECS
OVERLAP = 0.5
STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP))

SENSOR_COLS = ["GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]
MAGNITUDE_COLS = ["AccMag", "GyrMag"]
FEATURE_SIGNALS = SENSOR_COLS + MAGNITUDE_COLS
TIME_FEATURE_STATS = [
    "mean", "std", "median", "min", "max", "range", "iqr", "mad",
    "rms", "skewness", "kurtosis",
]
FEATURE_COLS = [
    f"{signal}_{stat}"
    for signal in FEATURE_SIGNALS
    for stat in TIME_FEATURE_STATS
]

# sub_07's sacrum ST-control file has ~596k samples instead of ~46k and is not
# a valid six-minute segment. Exclude it consistently from every condition.
EXCLUDED_PARTICIPANTS = {"sub_07"}

SESSIONS = ["st_control", "st_fatigue", "dt_control", "dt_fatigue"]
DATA_DIR = SRC_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
WINDOW_DIR = PROCESSED_DIR / "windows" / "sacrum_time"
FEATURE_DIR = PROCESSED_DIR / "features" / "sacrum_time"
FEATURES_DIR = FEATURE_DIR / "st_control"
ST_FATIGUE_FEATURES = FEATURE_DIR / "st_fatigue"
DT_CONTROL_FEATURES = FEATURE_DIR / "dt_control"
DT_FATIGUE_FEATURES = FEATURE_DIR / "dt_fatigue"
MODELS_DIR = SRC_DIR / "models" / "sacrum_time"
RESULTS_DIR = SRC_DIR / "results" / "sacrum_time"
FIGURES_DIR = RESULTS_DIR / "figures"

PARAM_GRID = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.001, 0.01, 0.1],
}
CV_FOLDS = 5
TARGET_FAR = 0.01

COLOR_BASELINE = "#2196F3"
COLOR_CROSS = "#F44336"
COLOR_DELTA = "#FF9800"
COLOR_DT_CONTROL = "#8B4513"
COLOR_DT_FATIGUE = "#4CAF50"


def ensure_dirs():
    for directory in [WINDOW_DIR, FEATURE_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
