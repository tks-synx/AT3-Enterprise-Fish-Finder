# Data Refresh Workflow (Shapefile Export -> Fish_Data_2.csv)

This project expects CSV data in this exact schema:

Tag_Number,Release_Da,Latitude,Longitude,Species_Na,Length,Weight,Latitude_F,Longitude_

## 1) Put your new export in a folder

Your website export should include matching files with the same base name, for example:
- GameFish_Tagging_Releases.shp
- GameFish_Tagging_Releases.dbf
- GameFish_Tagging_Releases.shx
- GameFish_Tagging_Releases.prj

Use a folder such as:
- incoming_data/GameFish_Tagging_Releases/

## 2) Install dependencies

From project root:

pip install -r requirements.txt

## 3) Convert + clean + replace

Run:

python scripts/import_shapefile_to_fish_csv.py \
  --input-dir incoming_data/GameFish_Tagging_Releases \
  --basename GameFish_Tagging_Releases \
  --output Fish_Data_2.csv \
  --backup

What this does:
- Reads the shapefile attributes
- Maps common source field names to app-required names
- Parses dates to month/day/2-digit-year
- Uppercases species names
- Builds numeric latitude/longitude from existing numeric fields, geometry points, or text values (like 33.52S)
- Drops rows missing required values (species/date/lat/lon)
- Replaces Fish_Data_2.csv and creates a timestamp backup

## 4) Verify quickly

Open Fish_Data_2.csv and confirm:
- Headers match exactly
- Latitude_F and Longitude_ are numeric
- Release_Da contains valid dates

## 5) Run app

python3 -m streamlit run BasicProduction/fish_app.py
