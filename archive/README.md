# Archive / Progress Files

These files are **not deleted** — they are kept for provenance and process-diary evidence, but are
no longer part of the active data pipeline or app.

## Legacy `Fish_Data_2` pipeline

- `Fish_Data_2.csv`, `Fish_Data_2.backup_20260513_135833.csv`, `Fish_Data_2.backup_20260513_140140.csv`,
  `Fish_Data_2.preclear_20260513_140120.csv`

These were produced by the older shapefile-import workflow described in the original
`DATA_REFRESH_WORKFLOW.md` (`scripts/import_shapefile_to_fish_csv.py` -> `Fish_Data_2.csv`). The
project has since moved to the `newfishdata/` pipeline
(`GameFish_Releases_Master.csv` -> `scripts/enrich_fish_data.py` ->
`newfishdata/Cleaned_Weather_GameFish_Releases_enriched.csv`), so nothing in the current app or
training script reads these files anymore. `Fish_Data_2.csv` and `Fish_Data_2.preclear_20260513_140120.csv`
are byte-identical copies of the same export.

## One-off debug scripts

- `debug_load_data.py`, `debug_path.py`, `test_data_load.py`

These were ad-hoc scripts used while diagnosing CSV loading/date-parsing issues (one even hardcodes
a personal `OneDrive` path). They are not part of the production pipeline, kept here as a record of
debugging work for the process diary/logbook.
