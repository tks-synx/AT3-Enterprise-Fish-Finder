import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

# ================= CONFIGURATION =================
# Replace with your actual filename
INPUT_FILE = 'heatmaporig.csv' 

# Column names in your CSV (Change if yours are different)
LAT_COL = 'Latitude_F'
LON_COL = 'Longitude_'
COUNT_COL = 'Count of Tag_Number'

# How "blurry/storm-like" you want the map (Higher = smoother blobs)
SIGMA = 5
# How detailed the grid should be (Higher = sharper image)
GRID_SIZE = 200
# =================================================

def create_overlay():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    # Force columns to be numbers. 
    # 'errors="coerce"' turns bad text (like "N/A" or "33deg") into NaN (Not a Number)
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors='coerce')
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors='coerce')
    df[COUNT_COL] = pd.to_numeric(df[COUNT_COL], errors='coerce')

    # Drop any rows that failed to convert (remove blank or bad data)
    initial_count = len(df)
    df = df.dropna(subset=[LAT_COL, LON_COL, COUNT_COL])
    print(f"Cleaned data: Kept {len(df)} valid rows out of {initial_count}")

    if len(df) == 0:
        print("ERROR: All data was removed! Check your CSV column headers and values.")
        return
    # 1. Define the Geographic Bounds
    min_lat = df[LAT_COL].min()
    max_lat = df[LAT_COL].max()
    min_lon = df[LON_COL].min()
    max_lon = df[LON_COL].max()

    # Add a tiny buffer so points on the edge aren't cut off
    lat_buffer = (max_lat - min_lat) * 0.1
    lon_buffer = (max_lon - min_lon) * 0.1
    
    bounds = {
        'north': max_lat + lat_buffer,
        'south': min_lat - lat_buffer,
        'east': max_lon + lon_buffer,
        'west': min_lon - lon_buffer
    }

    # 2. Convert Points to a Grid (Heatmap Matrix)
    # We use a 2D histogram to map the counts onto a grid
    x_edges = np.linspace(bounds['west'], bounds['east'], GRID_SIZE)
    y_edges = np.linspace(bounds['south'], bounds['north'], GRID_SIZE)
    
    # Create the weighted histogram
    heatmap, xedges, yedges = np.histogram2d(
        df[LON_COL], df[LAT_COL], 
        bins=(x_edges, y_edges), 
        weights=df[COUNT_COL]
    )

    # Transpose because numpy and images use different coord systems
    heatmap = heatmap.T

    # 3. Apply Gaussian Filter (The "Weather Map" Effect)
    # This blurs the grid squares into smooth clouds
    heatmap_smooth = gaussian_filter(heatmap, sigma=SIGMA)

    # 4. Generate the Image
    print("Generating image...")
    fig = plt.figure(figsize=(10, 10), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.]) # Make plot fill the whole canvas
    ax.set_axis_off()
    fig.add_axes(ax)

    # Create a custom colormap (Transparent -> Blue -> Red)
    # 'jet' is the classic weather map look, but we need the bottom to be transparent
    cmap = plt.get_cmap('jet')
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N) # Set Alpha (Transparency) gradient
    my_cmap[:20, -1] = 0 # Make the very lowest values completely invisible
    my_cmap = mcolors.ListedColormap(my_cmap)

    ax.imshow(heatmap_smooth, aspect='auto', origin='lower', cmap=my_cmap)
    
    image_filename = 'overlay_image.png'
    fig.savefig(image_filename, format='png', transparent=True)
    plt.close(fig)

    # 5. Create the KML File
    print("Creating KML...")
    kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Folder>
    <name>Fisheries Heatmap</name>
    <GroundOverlay>
      <name>Density Heatmap</name>
      <Icon>
        <href>{image_filename}</href>
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

    kml_filename = 'fisheries_heatmap.kml'
    with open(kml_filename, "w") as f:
        f.write(kml_content)

    print(f"Success! \n1. '{image_filename}' created.\n2. '{kml_filename}' created.")
    print("Double-click 'fisheries_heatmap.kml' to open in Google Earth.")

if __name__ == "__main__":
    create_overlay()