import pandas as pd
from pathlib import Path

def organize_duogait_data(parent_folder_path, output_path="data"):
    parent = Path(parent_folder_path)
    output = Path(output_path)
    
    conditions = ['st_control', 'st_fatigue', 'dt_control', 'dt_fatigue']
    
    for condition in conditions:
        
        source_dir = parent / "interim" / f"OG_{condition}"
        target_dir = output / condition
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if not source_dir.exists():
            print(f"Skipping {source_dir}: Path not found.")
            continue
        
        for participant_folder in source_dir.iterdir():
            if participant_folder.is_dir() and participant_folder.name.startswith("sub_"):
                participant_id = participant_folder.name.split('_')[1]

                source_file = participant_folder / "SA.csv"
                target_filename = f"sub{participant_id}_SA.csv"
                target_path = target_dir / target_filename

                if source_file.exists():

                    df = pd.read_csv(source_file)

                    df['participant_id'] = f"sub_{participant_id}"
                    df['session_type'] = condition
                    if target_path.exists():
                        print(f"Skipping {target_filename}: File already exists.")
                        continue
                    df.to_csv(target_path, index=False)
                    print(f"Processed: {target_filename}")
                else:
                    print(f"Skipping {source_file}: Path not found.")









def main():
    # parent_folder = r"C:\Users\bakinyem\Documents\gait-authentication\raw_data"
    parent_folder = Path(__file__).resolve().parent.parent / "raw_data"
    output_folder = "data"

    organize_duogait_data(parent_folder, output_folder)


    # file_path = Path(r"C:\Users\bakinyem\Documents\gait\src\data\st_control\sub01_SA.csv")


    # if file_path.exists():
    #     df = pd.read_csv(file_path)
    #     print(df.columns.tolist())
    # else:
    #     print("File not found. Check your path.")

if __name__ == "__main__":
    main()