import pandas as pd
import os

INPUT_FILE = '../newfishdata/GameFish_Releases_Master.csv'
LAT_COL = 'Latitude'
LON_COL = 'Longitude'
SPECIES_COL = 'Species_Name'
DATE_COL = 'Release_Date'

def load_data():
    try:
        if not os.path.exists(INPUT_FILE):
             return None, f"FILE NOT FOUND: {os.path.abspath(INPUT_FILE)}"
             
        df = pd.read_csv(INPUT_FILE)
        
        found_cols = df.columns.tolist()
        missing = [c for c in [LAT_COL, LON_COL, SPECIES_COL, DATE_COL] if c not in found_cols]
        if missing:
            return None, f"MISSING COLUMNS: {missing}. Found: {found_cols}"

        original_count = len(df)
        df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors='coerce')
        df[LON_COL] = pd.to_numeric(df[LON_COL], errors='coerce')
        df = df.dropna(subset=[LAT_COL, LON_COL])
        after_latlon_drop = len(df)

        sample_value = str(df[DATE_COL].iloc[0]) if not df.empty else ''
        
        if sample_value.isdigit() and len(sample_value) >= 10:
            df['Date_Obj'] = pd.to_datetime(df[DATE_COL], errors='coerce', unit='ms')
            if df['Date_Obj'].isna().all():
                df['Date_Obj'] = pd.to_datetime(df[DATE_COL], errors='coerce', unit='s')
        else:
            df['Date_Obj'] = pd.to_datetime(df[DATE_COL], errors='coerce', format='%m/%d/%y')
            if df['Date_Obj'].isna().any():
                fallback_mask = df['Date_Obj'].isna()
                df.loc[fallback_mask, 'Date_Obj'] = pd.to_datetime(
                    df.loc[fallback_mask, DATE_COL], errors='coerce', dayfirst=True
                )
        
        before_date_drop = len(df)
        df = df.dropna(subset=['Date_Obj'])
        after_date_drop = len(df)

        df['Year'] = df['Date_Obj'].dt.year.astype(int)
        df['Month_Number'] = df['Date_Obj'].dt.month.astype(int)
        df['Month_Name'] = df['Date_Obj'].dt.month_name()
        
        before_species_drop = len(df)
        df = df.dropna(subset=[SPECIES_COL])
        after_species_drop = len(df)
        
        df[SPECIES_COL] = df[SPECIES_COL].astype(str).str.strip()
        
        stats = {
            "original_count": original_count,
            "after_latlon_drop": after_latlon_drop,
            "before_date_drop": before_date_drop,
            "after_date_drop": after_date_drop,
            "before_species_drop": before_species_drop,
            "after_species_drop": after_species_drop
        }
        
        return df, f"Success. Stats: {stats}"
    except Exception as e:
        return None, f"Critical Error: {str(e)}"

# Run the test
df, status_msg = load_data()
print(f"Status: {status_msg}")
if df is not None:
    print(f"Loaded {len(df)} rows.")
    print("Columns:", df.columns.tolist())
else:
    print("Failed to load data.")
