# Prototype v1: Static KML/PNG Heatmaps

This folder holds the **first working prototype** of the fisheries heatmap, built before the live
Streamlit + Folium app (`BasicProduction/fish_app.py`). It generates static heatmap overlays
(a transparent PNG + a `.kml` file) that can be opened directly in Google Earth, instead of an
interactive web map.

This is kept as evidence of iterative prototyping (NESA "Producing prototypes" criterion) showing
how the system evolved from static, per-species KML overlays into the live, filterable, in-browser
heatmap in `fish_app.py`.

## Files

- `main.py` — builds one combined heatmap overlay (`overlay_image.png`, `fisheries_heatmap.kml`)
  from `heatmaporig.csv`.
- `species_heatmaps.py` — builds one heatmap overlay per species (into `Species_Maps/`) from
  `new_raw_fish_data.csv`.
- `heatmaporig.csv`, `heatmapfixedlat.csv` — pre-aggregated catch-density CSVs used by `main.py`.
- `raw_fish_data.csv` — an early, small raw export (matches the original 9-column schema described
  in the Part A data dictionary, including the unfixed `Latitude_bad`/`Longitude_bad` columns).
- `new_raw_fish_data.csv` — a later, larger raw export used by `species_heatmaps.py`. This is the
  same export as the archived `archive/Fish_Data_2.csv` (kept alongside its script here so this
  prototype still runs standalone).
- `Species_Maps/` — generated output (one PNG + one KML per species).
- `overlay_image.png`, `fisheries_heatmap.kml` — generated output from `main.py`.

## Running

Both scripts use plain relative filenames (not `Cleaned_Weather_GameFish_Releases_enriched.csv` and
not the live app's data), so run them **from inside this folder**:

```bash
cd BasicProduction/prototype_v1_static_kml
python3 main.py
python3 species_heatmaps.py
```

No code changes were needed after moving these files here — the scripts and the CSVs they read
were moved together, so the existing relative paths still resolve correctly.

This prototype is not wired into the live app and is not part of the current predictive model.
