import os
import pandas as pd

os.chdir("/Users/school/Library/CloudStorage/OneDrive-TheKing'sSchool/Classes/Heatmap")

INPUT_FILE = './newfishdata/GameFish_Releases_Master.csv'
DATE_COL = 'Release_Date'
LAT_COL = 'Latitude'
LON_COL = 'Longitude'

# Test CSV loading
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()[:5]}")
    print(f"Sample DATE_COL value: {df[DATE_COL].iloc[0]}")

    # Test date parsing
    sample_value = str(df[DATE_COL].iloc[0])
    print(f"Sample value as string: '{sample_value}', length: {len(sample_value)}, isdigit: {sample_value.isdigit()}")

    # Try parsing
    try:
        df['Date_Obj'] = pd.to_datetime(df[DATE_COL], errors='coerce', unit='ms')
        successful = (df['Date_Obj'].notna().sum() / len(df)) * 100
        print(f"Date parsing (milliseconds): {successful:.1f}% successful")
        if successful > 0:
            print(f"Sample parsed date: {df[df['Date_Obj'].notna()]['Date_Obj'].iloc[0]}")
        else:
            print("No dates were successfully parsed with unit='ms'. Testing without unit='ms'...")
            df['Date_Obj_Auto'] = pd.to_datetime(df[DATE_COL], errors='coerce')
            successful_auto = (df['Date_Obj_Auto'].notna().sum() / len(df)) * 100
            print(f"Date parsing (auto): {successful_auto:.1f}% successful")
            if successful_auto > 0:
                 print(f"Sample parsed date (auto): {df[df['Date_Obj_Auto'].notna()]['Date_Obj_Auto'].iloc[0]}")
    except Exception as e:
        print(f"Error parsing date: {e}")
except Exception as e:
    print(f"Error loading CSV: {e}")
