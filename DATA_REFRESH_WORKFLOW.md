# Data Refresh Workflow

This describes the **current, active** data pipeline used by the live app
(`BasicProduction/fish_app.py`) and the neural network trainer
(`BasicProduction/train_species_nn.py`). The older shapefile workflow that used to be
described in this file has been superseded and is kept only as archived history — see
[Archived workflow](#archived-workflow-shapefile--fish_data_2csv) at the bottom.

## Current pipeline (`newfishdata/`)

```
newfishdata/GameFish_Releases_Master.csv               (203,300 rows — raw export)
        |  scripts/enrich_fish_data.py
        v
newfishdata/GameFish_Releases_Master_enriched.csv       (203,300 rows — + weather/moon columns)
        |  filter: keep rows with usable Wind_Direction_10m / Rain_mm
        v
newfishdata/Weather_GameFish_Releases_enriched.csv      (49,893 rows)
        |  filter: keep rows with a recorded Length and Weight
        v
newfishdata/Cleaned_Weather_GameFish_Releases_enriched.csv   (41,440 rows)  <-- FINAL DATASET
```

### `newfishdata/Cleaned_Weather_GameFish_Releases_enriched.csv` is the final primary dataset

This is the single source of truth used by **both**:
- `BasicProduction/fish_app.py` (the Streamlit dashboard, map, and both intelligent systems), and
- `BasicProduction/train_species_nn.py` (the neural network trainer).

Using the same file for the app and the model guarantees the features shown to the user
and the features the model was trained on are consistent (see
[INTELLIGENT_SYSTEMS_IMPLEMENTATION.md](INTELLIGENT_SYSTEMS_IMPLEMENTATION.md) for the
bug this previously caused and how it was fixed).

Every row in this file has: a species, a valid latitude/longitude, a release date, a
recorded length and weight, and usable weather/moon data (`Moon_Phase`,
`Wind_Direction_10m`, `Rain_mm`, `Is_Raining`).

### Step 1: `scripts/enrich_fish_data.py`

```bash
python3 scripts/enrich_fish_data.py \
  --input newfishdata/GameFish_Releases_Master.csv \
  --output newfishdata/GameFish_Releases_Master_enriched.csv \
  --cache newfishdata/enrichment_cache.csv
```

For each record, this script looks up (at local midday, on the release date):
- **Moon phase** — calculated locally (0–1 fraction), always available.
- **Wind direction & rainfall** — from the Open-Meteo weather/archive API, batched by a
  ~10 nautical-mile grid (`--grid-size 0.1667`) and cached in `enrichment_cache.csv` to
  avoid repeat calls. Only succeeded for ~24.5% of rows (many older/remote records fall
  outside the archive API's coverage), which is why the next step filters down to rows
  with usable weather data.
- **Sea surface temperature** — from the Open-Meteo Marine API. **This came back 100%
  null for every single row**, in every enriched CSV in this repository. The marine API
  does not have coverage for the combination of dates/locations in this dataset. This
  was investigated and confirmed with `df['Sea_Surface_Temp_C'].isna().mean()` before
  deciding to drop the column from the model (see
  [INTELLIGENT_SYSTEMS_IMPLEMENTATION.md](INTELLIGENT_SYSTEMS_IMPLEMENTATION.md)).

### Step 2 & 3: filtering to the final dataset

The remaining two filtering steps (drop rows with no usable weather data, then drop rows
with no recorded `Length`/`Weight`) were applied with short, one-off pandas commands
rather than a saved script. In summary:

```python
weather_ok = df[df['Wind_Direction_10m'].notna() & df['Rain_mm'].notna()]
final = weather_ok.dropna(subset=['Length', 'Weight'])
```

If you need to reproduce `Cleaned_Weather_GameFish_Releases_enriched.csv` from a fresh
`GameFish_Releases_Master_enriched.csv`, running the two lines above (after step 1) will
recreate it.

### Protected CSVs (do not delete)

These files are intentionally kept in `newfishdata/` even though only the final cleaned
file is used by the app/model. They exist as **provenance/progress evidence** — showing
each stage of cleaning and enrichment for the assessment's data-cleaning evidence
requirements:

| File | Rows | Role |
|---|---|---|
| `GameFish_Releases_Master.csv` | 203,300 | Raw export, before any enrichment |
| `GameFish_Releases_Master_enriched.csv` | 203,300 | After weather/moon/SST enrichment, before filtering |
| `GameFish_Releases_Master_enriched_Cleaned.csv` | 203,300 | Byte-identical copy of the file above, kept as-is for provenance |
| `Weather_GameFish_Releases_enriched.csv` | 49,893 | After filtering to rows with usable weather data |
| `Cleaned_Weather_GameFish_Releases_enriched.csv` | 41,440 | **Final dataset** — also filtered to rows with recorded length/weight |
| `enrichment_cache.csv` | 18,224 | Cache of API responses keyed by grid cell + date, used by `enrich_fish_data.py` to avoid re-querying |

### Refreshing with new raw data

1. Add the new raw export to `newfishdata/GameFish_Releases_Master.csv` (matching the
   existing column schema).
2. Re-run `scripts/enrich_fish_data.py` (Step 1 above).
3. Re-apply the two filters shown in Step 2 & 3 to regenerate
   `Cleaned_Weather_GameFish_Releases_enriched.csv`.
4. Re-run `BasicProduction/train_species_nn.py` so the model is retrained on the
   refreshed data.
5. Run `python3 -m streamlit run BasicProduction/fish_app.py` to confirm the app still
   loads correctly.

---

## Archived workflow: shapefile → `Fish_Data_2.csv`

This project previously used a different pipeline based on shapefile exports, described
below for historical/process-diary reference only. **It is no longer used by the app or
training script.** The files it produced now live in `archive/` (see
`archive/README.md`), and the original raw shapefile-based prototype (which read
`Fish_Data_2.csv` directly) has been moved to
`BasicProduction/prototype_v1_static_kml/` (see
[PROTOTYPE_EVOLUTION.md](PROTOTYPE_EVOLUTION.md)).

Old schema: `Tag_Number,Release_Da,Latitude,Longitude,Species_Na,Length,Weight,Latitude_F,Longitude_`

1. Put a new shapefile export (`.shp`/`.dbf`/`.shx`/`.prj`) in `incoming_data/<name>/`.
2. Run:
   ```bash
   python scripts/import_shapefile_to_fish_csv.py \
     --input-dir incoming_data/GameFish_Tagging_Releases \
     --basename GameFish_Tagging_Releases \
     --output Fish_Data_2.csv \
     --backup
   ```
3. This mapped source field names, parsed dates, uppercased species names, built
   numeric lat/lon, dropped incomplete rows, and replaced `Fish_Data_2.csv` with a
   timestamped backup.

This workflow is kept only as prototype/iteration evidence — do not point the live app
or trainer at `Fish_Data_2.csv`.
