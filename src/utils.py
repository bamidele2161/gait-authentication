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
    'AccX', 'AccY', 'AccZ', 'session_type', 'participant_id', 'timestamp'
]

PROCESSED_DIR  = "data/processed"
MODELS_DIR     = "models"
RESULTS_DIR    = "results"
SESSIONS = ['st_control', 'st_fatigue']


def ensure_dirs():
    for d in [PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
    print("All directories ready.")
