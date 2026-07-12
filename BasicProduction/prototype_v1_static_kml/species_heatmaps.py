import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import os
import re

# ================= CONFIGURATION =================
INPUT_FILE = 'new_raw_fish_data.csv' 

# Column Headers (Must match your CSV exactly)
LAT_COL = 'Latitude_F'
LON_COL = 'Longitude_'
SPECIES_COL = 'Species_Na'

# Settings
SIGMA = 5         # Blur amount
GRID_SIZE = 200   # Image resolution
# =================================================

def clean_filename(text):
    # Turns "Blue Fin Tuna" into "blue_fin_tuna" for safe filenames
    return re.sub(r'[^a-zA-Z0-9]', '_', str(text)).lower()

def generate_maps():
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("ERROR: File not found. Check the filename!")
        return

    # 1. Clean Data
    # Force Lat/Long to numbers and drop junk
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors='coerce')
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors='coerce')
    df = df.dropna(subset=[LAT_COL, LON_COL, SPECIES_COL])

    # Get list of all species
    unique_species = df[SPECIES_COL].unique()
    print(f"Found {len(unique_species)} species: {unique_species}")

    # Create an output folder so files don't clutter your desktop
    output_folder = "Species_Maps"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. Loop through every species found
    for species in unique_species:
        print(f"Processing: {species}...")
        
        # Filter data for JUST this fish
        fish_data = df[df[SPECIES_COL] == species].copy()
        
        if len(fish_data) < 2:
            print(f"  Skipping {species} (not enough data points)")
            continue

        # --- MATH SETUP ---
        # We calculate bounds based on THIS species' range, 
        # but you might want global bounds if comparing them side-by-side.
        # Here we use specific bounds so it zooms in on the fish.
        min_lat, max_lat = fish_data[LAT_COL].min(), fish_data[LAT_COL].max()
        min_lon, max_lon = fish_data[LON_COL].min(), fish_data[LON_COL].max()

        # Add buffer
        lat_buff = (max_lat - min_lat) * 0.1 or 0.01
        lon_buff = (max_lon - min_lon) * 0.1 or 0.01
        
        bounds = {
            'north': max_lat + lat_buff, 'south': min_lat - lat_buff,
            'east': max_lon + lon_buff, 'west': min_lon - lon_buff
        }

        # --- HEATMAP GENERATION ---
        x_edges = np.linspace(bounds['west'], bounds['east'], GRID_SIZE)
        y_edges = np.linspace(bounds['south'], bounds['north'], GRID_SIZE)
        
        # Note: We don't need a 'Count' column anymore. 
        # The script counts the rows automatically (weights=None).
        heatmap, _, _ = np.histogram2d(
            fish_data[LON_COL], fish_data[LAT_COL], 
            bins=(x_edges, y_edges)
        )
        
        heatmap = gaussian_filter(heatmap.T, sigma=SIGMA)

        # --- IMAGE SAVING ---
        safe_name = clean_filename(species)
        image_filename = f"{output_folder}/{safe_name}_overlay.png"
        
        fig = plt.figure(figsize=(10, 10), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Create Color Map (Transparent -> Blue -> Red)
        cmap = plt.get_cmap('jet')
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0, 1, cmap.N) 
        my_cmap[:10, -1] = 0 # Hide very low values
        my_cmap = mcolors.ListedColormap(my_cmap)

        ax.imshow(heatmap, aspect='auto', origin='lower', cmap=my_cmap)
        fig.savefig(image_filename, format='png', transparent=True)
        plt.close(fig)

        # --- KML SAVING ---
        kml_filename = f"{output_folder}/{safe_name}_map.kml"
        kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Folder>
    <name>{species} Heatmap</name>
    <GroundOverlay>
      <name>{species} Density</name>
      <Icon>
        <href>{os.path.basename(image_filename)}</href>
      </Icon>
      <LatLonBox>
        <north>{bounds['north']}</north>
        <south>{bounds['south']}</south>
        <east>{bounds['east']}</east>
        <west>{bounds['west']}</west>
      </LatLonBox>
    </GroundOverlay>
  </Folder>
</kml>
"""
        with open(kml_filename, "w") as f:
            f.write(kml_content)

    print("\nDone! Check the 'Species_Maps' folder.")

if __name__ == "__main__":
    generate_maps()