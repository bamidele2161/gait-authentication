import os
import numpy as np
import pandas as pd


SAMPLE_RATE     = 100        
WINDOW_SECS     = 2         
WINDOW_SAMPLES  = SAMPLE_RATE * WINDOW_SECS  
OVERLAP         = 0.5       
STEP_SAMPLES    = int(WINDOW_SAMPLES * (1 - OVERLAP)) 

AXES            = ['x', 'y', 'z']

RAW_DIR        = "data/raw"
PROCESSED_DIR  = "data/processed"
MODELS_DIR     = "models"
RESULTS_DIR    = "results"


def ensure_dirs():
    for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
    print("All directories ready.")


def parse_participant_info(filename):
    name = os.path.basename(filename).replace(".csv", "")
    parts = name.split("_")
    participant_id = parts[1]
    session_number = int(parts[2].replace("session", ""))
    return participant_id, session_number


def list_raw_files():
    files = [
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.endswith(".csv")
    ]
    files.sort()
    print(f"Found {len(files)} raw file(s) in {RAW_DIR}/")
    return files
