"""Shared configuration for the time-domain problem-definition experiment.

The sensor set is chosen with the ``GAIT_SENSOR_SET`` environment variable:

- ``sacrum`` (default): the original sacrum-only experiment, ``*_time`` paths
  under ``sacrum_time``, unchanged.
- ``all9``: all nine synchronized IMUs (sacrum, shanks, feet, wrists, head,
  sternum), six raw axes each, read directly from ``raw_data/interim``.
  Every output goes under ``all9_time`` so the sacrum results are untouched.
"""

import os
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
RAW_INTERIM_DIR = SRC_DIR.parent / "raw_data" / "interim"

SENSOR_SETS = {
    "sacrum": ("SA",),
    "all9": ("SA", "LL", "RL", "LF", "RF", "LW", "RW", "HE", "ST"),
}
SENSOR_SET = os.environ.get("GAIT_SENSOR_SET", "sacrum")
if SENSOR_SET not in SENSOR_SETS:
    raise ValueError(
        f"GAIT_SENSOR_SET must be one of {sorted(SENSOR_SETS)}, got {SENSOR_SET!r}"
    )
SENSOR_CODES = SENSOR_SETS[SENSOR_SET]
EXPERIMENT_NAME = f"{'sacrum' if SENSOR_SET == 'sacrum' else SENSOR_SET}_time"

ORIGINAL_RATE = 128
WINDOW_SECS = 2
WINDOW_SAMPLES = ORIGINAL_RATE * WINDOW_SECS
OVERLAP = 0.5
STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP))
BLOCK_SECONDS = 20
BLOCK_SAMPLES = BLOCK_SECONDS * ORIGINAL_RATE

SENSOR_COLS = ["GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]
MAGNITUDE_COLS = ["AccMag", "GyrMag"]
# Sacrum: six axes plus magnitudes (original baseline). Multi-sensor sets use
# the six raw axes of every sensor only.
FEATURE_SIGNALS = SENSOR_COLS + MAGNITUDE_COLS
TIME_FEATURE_STATS = [
    "mean", "std", "median", "min", "max", "range", "iqr", "mad",
    "rms", "skewness", "kurtosis",
]
# Sacrum keeps its original unprefixed names; multi-sensor sets prefix each
# feature with the sensor code (e.g. "LL_GyrX_mean").
SIGNAL_PREFIXES = ("",) if SENSOR_SET == "sacrum" else tuple(f"{c}_" for c in SENSOR_CODES)
if SENSOR_SET != "sacrum":
    FEATURE_SIGNALS = list(SENSOR_COLS)
FEATURE_COLS = [
    f"{prefix}{signal}_{stat}"
    for prefix in SIGNAL_PREFIXES
    for signal in FEATURE_SIGNALS
    for stat in TIME_FEATURE_STATS
]

# Known recording-level corrections, defined independently of authentication
# performance. sub_07's sacrum ST-control CSV contains the complete recording
# session. The synchronized sensors identify the actual ST-control walk below.
KNOWN_TIME_TRIMS = {
    ("st_control", "sub_07"): {
        "start": 769.53125,
        "end": 1146.875,
        "expected_rows": 48_301,
    },
}

# Retained as a shared hook for any future participant-level quality exclusions.
# sub_07 is included after applying the validated recording-level trim above.
EXCLUDED_PARTICIPANTS = set()

SESSIONS = ["st_control", "st_fatigue", "dt_control", "dt_fatigue"]
DATA_DIR = SRC_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
WINDOW_DIR = PROCESSED_DIR / "windows" / EXPERIMENT_NAME
FEATURE_DIR = PROCESSED_DIR / "features" / EXPERIMENT_NAME
FEATURES_DIR = FEATURE_DIR / "st_control"
ST_FATIGUE_FEATURES = FEATURE_DIR / "st_fatigue"
DT_CONTROL_FEATURES = FEATURE_DIR / "dt_control"
DT_FATIGUE_FEATURES = FEATURE_DIR / "dt_fatigue"
MODELS_DIR = SRC_DIR / "models" / EXPERIMENT_NAME
RESULTS_DIR = SRC_DIR / "results" / EXPERIMENT_NAME
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
