import os
import numpy as np
import pandas as pd


SAMPLE_RATE     = 100        
WINDOW_SECS     = 2         
WINDOW_SAMPLES  = SAMPLE_RATE * WINDOW_SECS  
OVERLAP         = 0.5       
STEP_SAMPLES    = int(WINDOW_SAMPLES * (1 - OVERLAP)) 

SENSOR_COLS     = [
    'AG-X', 'AG-Y', 'AG-Z',
    'Acc-X', 'Acc-Y', 'Acc-Z',
    'Gravity-X', 'Gravity-Y', 'Gravity-Z',
    'RR-X', 'RR-Y', 'RR-Z',
    'RV-X', 'RV-Y', 'RV-Z'
]

PROCESSED_DIR  = "data/processed"
MODELS_DIR     = "models"
RESULTS_DIR    = "results"


def ensure_dirs():
    for d in [PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
    print("All directories ready.")
