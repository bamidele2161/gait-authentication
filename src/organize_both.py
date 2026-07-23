import pandas as pd
from pathlib import Path
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import LEGS_DATA_DIR, SESSIONS

def organise_wrists_data(parent_folder_path):
    parent = Path(parent_folder_path)
    output = LEGS_DATA_DIR

    for condition in SESSIONS:
        source_dir = parent / "interim" / f"OG_{condition}"
        target_dir = output / condition
        target_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            print(f"Skipping {source_dir}: Path not found.")
            continue

        for participant_folder in source_dir.iterdir():
            if not (participant_folder.is_dir() and
                    participant_folder.name.startswith("sub_")):
                continue

            participant_id = participant_folder.name.split('_')[1]
            target_filename = f"sub{participant_id}_RWLW.csv"
            target_path = target_dir / target_filename

            if target_path.exists():
                print(f"Skipping {target_filename}: already exists.")
                continue

            rl_path = participant_folder / "RW.csv"
            ll_path = participant_folder / "LW.csv"

            if not rl_path.exists():
                print(f"Missing RW.csv for sub_{participant_id} in {condition}")
                continue
            if not ll_path.exists():
                print(f"Missing LW.csv for sub_{participant_id} in {condition}")
                continue

            df_rl = pd.read_csv(rl_path)
            df_ll = pd.read_csv(ll_path)

            # Trim to same length if they differ slightly
            min_len = min(len(df_rl), len(df_ll))
            if len(df_rl) != len(df_ll):
                print(f"  [WARNING] sub_{participant_id} {condition}: "
                      f"RL={len(df_rl)}, LL={len(df_ll)} — trimming to {min_len}")
                df_rl = df_rl.iloc[:min_len].reset_index(drop=True)
                df_ll = df_ll.iloc[:min_len].reset_index(drop=True)

            # Rename columns to avoid clash
            df_rl = df_rl.rename(columns={
                'AccX': 'RW_AccX', 'AccY': 'RW_AccY', 'AccZ': 'RW_AccZ',
                'GyrX': 'RW_GyrX', 'GyrY': 'RW_GyrY', 'GyrZ': 'RW_GyrZ',
            })
            df_ll = df_ll.rename(columns={
                'AccX': 'LW_AccX', 'AccY': 'LW_AccY', 'AccZ': 'LW_AccZ',
                'GyrX': 'LW_GyrX', 'GyrY': 'LW_GyrY', 'GyrZ': 'LW_GyrZ',
            })

            # Combine 12 sensor columns
            combined = pd.concat([
                df_rl[['RW_AccX','RW_AccY','RW_AccZ',
                        'RW_GyrX','RW_GyrY','RW_GyrZ']],
                df_ll[['LW_AccX','LW_AccY','LW_AccZ',
                        'LW_GyrX','LW_GyrY','LW_GyrZ']],
            ], axis=1)

            # Add metadata
            combined['participant_id'] = f"sub_{participant_id}"
            combined['session_type']   = condition

            combined.to_csv(target_path, index=False)
            print(f"Processed: {target_filename} — {len(combined)} rows, "
                  f"12 sensor cols")


def main():
    parent_folder = Path(__file__).resolve().parent.parent / "raw_data"
    organise_wrists_data(parent_folder)


if __name__ == "__main__":
    main()