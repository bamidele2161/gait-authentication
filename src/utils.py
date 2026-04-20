import os
import numpy as np
import pandas as pd
from pathlib import Path

ORIGINAL_RATE = 128
SAMPLE_RATE     = 50        
WINDOW_SECS     = 2         
WINDOW_SAMPLES  = int(SAMPLE_RATE * WINDOW_SECS)  
OVERLAP         = 0.5       
STEP_SAMPLES    = int(WINDOW_SAMPLES * (1 - OVERLAP)) 

SENSOR_COLS     = [
    'GyrX', 'GyrY', 'GyrZ',
    'AccX', 'AccY', 'AccZ',
]
DATA_DIR = Path("data")
PROCESSED_DIR  = Path("data/processed")
MODELS_DIR     = Path("models")
RESULTS_DIR    = Path("results")
SESSIONS = ['st_control', 'st_fatigue']
WINDOW_DIR = PROCESSED_DIR / "windows"
FEATURE_DIR = PROCESSED_DIR / "features"
FEATURES_DIR = FEATURE_DIR / "st_control"
ST_FATIGUE_FEATURES = FEATURE_DIR / "st_fatigue"

FEATURE_COLS = [
    'AccX_mean', 'AccX_std', 'AccX_var', 'AccX_energy', 'AccX_rms', 'AccX_min', 'AccX_max',
    'AccY_mean', 'AccY_std', 'AccY_var', 'AccY_energy', 'AccY_rms', 'AccY_min', 'AccY_max',
    'AccZ_mean', 'AccZ_std', 'AccZ_var', 'AccZ_energy', 'AccZ_rms', 'AccZ_min', 'AccZ_max',
    'GyrX_mean', 'GyrX_std', 'GyrX_var', 'GyrX_energy', 'GyrX_rms', 'GyrX_min', 'GyrX_max',
    'GyrY_mean', 'GyrY_std', 'GyrY_var', 'GyrY_energy', 'GyrY_rms', 'GyrY_min', 'GyrY_max',
    'GyrZ_mean', 'GyrZ_std', 'GyrZ_var', 'GyrZ_energy', 'GyrZ_rms', 'GyrZ_min', 'GyrZ_max',
]

PARAM_GRID = {
    'C'     : [0.1, 1, 10, 100],
    'gamma' : ['scale', 'auto', 0.001, 0.01, 0.1],
}

CV_FOLDS = 5

COLOR_BASELINE = '#2196F3'
COLOR_CROSS = '#4F44336'
COLOR_DELTA = '#FF9800'

def ensure_dirs():
    for d in [PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
    print("All directories ready.")


