import os
import pandas as pd

paths_to_check = [
    '../newfishdata/GameFish_Releases_Master.csv',
    'newfishdata/GameFish_Releases_Master.csv',
    './newfishdata/GameFish_Releases_Master.csv',
]

for p in paths_to_check:
    print(f"Checking path: {p}")
    print(f"  Exists: {os.path.exists(p)}")
    if os.path.exists(p):
        print(f"  Absolute path: {os.path.abspath(p)}")
        try:
            df = pd.read_csv(p, nrows=5)
            print(f"  Successfully read first 5 rows.")
            print(f"  Columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"  Error reading file: {e}")

print(f"\nCurrent working directory: {os.getcwd()}")
print(f"Contents of './newfishdata': {os.listdir('newfishdata') if os.path.exists('newfishdata') else 'Not found'}")
