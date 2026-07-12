import pandas as pd
import os

# --- FILE PATHS ---
# Use the final cleaned dataset used by the app/model pipeline.
INPUT_FILE = 'newfishdata/Cleaned_Weather_GameFish_Releases_enriched.csv'
OUTPUT_FILE = 'NESA_AT3b_Spreadsheet_Weather_Redo.xlsx'

def generate_excel():
    print(f"Loading data from {INPUT_FILE}...")
    df_raw = pd.read_csv(INPUT_FILE)
    print(f"Starting with {len(df_raw)} total rows.")
    
    # Sea_Surface_Temp_C is intentionally NOT required here:
    # enrichment returns 100% null for SST in the protected datasets.
    # We still keep the column for provenance, but do not use it for row filtering.
    
    # 2. CLEAN FIRST: Drop rows that are completely blank (NaN) in these columns.
    # Note: dropna() automatically keeps 0 and 0.0!
    df_clean_all = df_raw.dropna(subset=['Species_Name', 'Weight', 'Latitude', 'Longitude']).copy()
    print(f"Rows with complete species, location, and weight data: {len(df_clean_all)}")

    if len(df_clean_all) == 0:
        print("ERROR: No rows have a recorded weight! Check your raw CSV.")
        return
        
    # 3. SAMPLE SECOND: Now grab our 30,000 rows from the pristine data
    df_sample = df_clean_all.head(18225).copy()
    print(f"Taking {len(df_sample)} rows for the final Excel file.")

    print("Generating Excel workbook...")
    # Use xlsxwriter engine to allow formula injection
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        
        # 1. Write the Cleaned Data (we don't need a separate "Original" sheet if it's identical)
        df_sample.to_excel(writer, sheet_name='Cleaned Data', index=False)
        
        # Access the workbook and the cleaned worksheet to add NESA requirements
        workbook = writer.book
        worksheet = writer.sheets['Cleaned Data']
        
        # 2. Add APPROPRIATE FORMULAE (NESA Requirement)
        header_format = workbook.add_format({'bold': True, 'border': 1})
        worksheet.write('P1', 'Weight_Classification', header_format)
        
        print("Injecting IF formulas into Column P...")
        # Since we know there are no blanks anymore, we can go back to the simple, clean IF statement!
        for row_num in range(2, len(df_sample) + 2):
            formula = f'=IF(H{row_num}>50, "Heavy Game Fish", "Standard")'
            worksheet.write_formula(f'P{row_num}', formula)

    print(f"SUCCESS! Open '{OUTPUT_FILE}' in Microsoft Excel to finish the final steps.")

if __name__ == "__main__":
    generate_excel()