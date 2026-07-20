# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is
The primary product is the **Fish Finder** Streamlit dashboard at `BasicProduction/fish_app.py`
(fisheries heatmaps, trip estimator, a scikit-learn neural network, and a certainty-factor
expert system). Standard run/setup commands live in `README.md` and `QUICK_START.md`.

### Services

- **Streamlit app (primary):** `python3 -m streamlit run BasicProduction/fish_app.py`.
  Serves on `http://localhost:8501`. Add `--server.headless true` when running non-interactively.
  Dependencies come from the root `requirements.txt` (the pinned `BasicProduction/requirements.txt`
  targets old package versions and is not used for the dev environment).
- **presentation_site (secondary/static):** a standalone Firebase Hosting scrollytelling site,
  unrelated to the Streamlit intelligent-systems app. Preview locally with
  `cd presentation_site/public && python3 -m http.server 8080`. Deploying needs Firebase CLI
  + auth (see `presentation_site/README.md`); not required to run the main app.

### Non-obvious notes

- **Data is committed.** The app reads `newfishdata/Cleaned_Weather_GameFish_Releases_enriched.csv`
  (~41k rows, 21 species, years 2000–2006) via a path relative to `fish_app.py`, so it works from
  any working directory. No data-prep step is needed to run the app.
- **Ollama/TinyLlama is optional.** The "AI Assisted" mapping mode and the "Ask Ollama to explain"
  buttons need a local Ollama server (`ollama serve`) with `ollama pull tinyllama`. The app detects
  this at runtime (`OLLAMA_AVAILABLE`) and degrades gracefully — "Manual Entry" heatmaps, the neural
  network, and the certainty-factor engine all work without Ollama. Ollama is NOT installed by the
  update script.
- **Pre-trained model is committed:** `BasicProduction/fish_species_nn.joblib`. `train_species_nn.py`
  regenerates it but is not needed to run the app.
- **App writes state:** `BasicProduction/user_profile.json` is read/written at runtime for saved
  preferences and map ratings.
- **No lint or test framework is configured** (no pytest/ruff/flake8/pyproject). Use
  `python3 -m py_compile <file.py>` as a syntax check. `archive/test_data_load.py` is an old script
  with a hardcoded local path and is not runnable here.
