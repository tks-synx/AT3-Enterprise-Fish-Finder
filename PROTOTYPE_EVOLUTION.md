# Prototype Evolution

This document traces how the fisheries heatmap project evolved across its prototypes,
for the NESA "producing prototypes" / iterative development evidence.

## Stage 1 — Static KML/PNG heatmap (`BasicProduction/prototype_v1_static_kml/`)

The first working prototype. Given a raw CSV export of tagging releases, `main.py` and
`species_heatmaps.py` aggregate catch locations into a **static, pre-rendered heatmap
image** (`overlay_image.png`) plus a `.kml` overlay file, viewed by opening the KML in
Google Earth.

- Input: `raw_fish_data.csv` / `new_raw_fish_data.csv` (early raw exports, matching the
  original 9-column data-dictionary schema from Part A, including the unfixed
  `Latitude_bad`/`Longitude_bad` columns before location cleaning was added).
- Output: one combined overlay (`main.py`), or one overlay per species in
  `Species_Maps/` (`species_heatmaps.py`).
- Limitations that motivated the next stage: no interactivity (can't filter by
  species/date without regenerating the image), requires Google Earth to view, no way to
  layer in weather/moon context, and no predictive or expert-system reasoning at all —
  it is a pure visualisation, not an intelligent system.

This has been moved into its own folder (`BasicProduction/prototype_v1_static_kml/`,
with its own `README.md`) so it remains runnable in isolation as evidence, without being
confused with the current live app. It is not wired into `fish_app.py` and does not use
the current cleaned dataset or predictive model.

## Stage 2 — Interactive Streamlit + Folium dashboard (`BasicProduction/fish_app.py`)

The static image approach was replaced with a live, in-browser dashboard:
- **Folium** renders an interactive Leaflet map (pan/zoom/click) instead of a flat PNG.
- **Streamlit** adds filtering (species/year/month), manual and AI-assisted request
  modes, trip-distance estimation, GPX/KML route export, and a feedback/rating sidebar
  that the certainty-factor engine's "user preference match" rule later reused.
- **Ollama/TinyLlama** was added for local, offline natural-language summaries of the
  currently filtered map — kept local so the app still works without internet access,
  matching the Part A requirement for offline-first operation in the field.

This is a direct capability upgrade from Stage 1: the same underlying idea (show catch
density on a map) is now interactive, filterable, and forms the base the two intelligent
systems (below) are attached to, instead of a one-off static export.

## Stage 3 — Power BI reference dashboard (`Dashboard-FishFinder.pbix copy`)

A Power BI dashboard was also built as a second, business-intelligence-style
visualisation of the same underlying data, used as a design/analysis reference alongside
the Streamlit app. It is not modified as part of this cleanup/intelligent-systems work
and is out of scope for the Python codebase changes described in this document — it is
listed here only so the prototype history is complete for assessment purposes.

## Stage 4 — Cleaned dataset + predictive model pipeline (current)

The data pipeline itself went through several iterations before settling on
`newfishdata/Cleaned_Weather_GameFish_Releases_enriched.csv` as the final dataset (full
lineage in [DATA_REFRESH_WORKFLOW.md](DATA_REFRESH_WORKFLOW.md)):

1. Raw shapefile exports → `Fish_Data_2.csv` (archived, `archive/README.md`).
2. Raw CSV export → `GameFish_Releases_Master.csv` (203,300 rows, no weather context).
3. Weather/moon/SST enrichment via `scripts/enrich_fish_data.py` →
   `GameFish_Releases_Master_enriched.csv`.
4. Filtering to rows with usable weather data → `Weather_GameFish_Releases_enriched.csv`
   (49,893 rows).
5. Filtering to rows with a complete length/weight record →
   `Cleaned_Weather_GameFish_Releases_enriched.csv` (41,440 rows) — the current,
   final dataset used by both the app and the neural network.

Alongside this, the neural network (`train_species_nn.py` /
`fish_species_nn.joblib`) and the certainty-factor expert system
(`certainty_factor.py`) were added as two independent, complementary intelligent
systems on top of the same cleaned dataset — see
[INTELLIGENT_SYSTEMS_IMPLEMENTATION.md](INTELLIGENT_SYSTEMS_IMPLEMENTATION.md) for full
detail on both, including the feature-mismatch bug found and fixed during this stage.

## Summary of the evolution

| Stage | Visualisation | Data | Intelligence |
|---|---|---|---|
| 1. Static KML/PNG | Pre-rendered image, opened in Google Earth | Small raw CSV export | None |
| 2. Streamlit + Folium | Interactive, filterable, in-browser map | Same raw CSV, later swapped for the enriched pipeline | Local LLM summaries only |
| 3. Power BI | Business-intelligence dashboard (reference) | Same underlying export | None (BI visuals only) |
| 4. Current | Streamlit + Folium (unchanged) | Cleaned, weather-enriched, provenance-tracked pipeline | Neural network **and** certainty-factor expert system, side by side |

Each stage's files are kept in the repository (`prototype_v1_static_kml/`, `archive/`,
the protected `newfishdata/` CSVs) rather than deleted, so the full progression from a
static prototype to a data-driven + rule-based intelligent dashboard can be shown and
explained in the walkthrough video and hand-in evidence.
