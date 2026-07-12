# AT3 Enterprise Fish Finder

Streamlit app for exploring game-fish release data on interactive heatmaps, with an optional local TinyLlama assistant for asking questions about the filtered data.

## Features

- Interactive Folium heatmaps filtered by species, year, and month
- Local AI chat (Ollama + TinyLlama) for data insights
- Species prediction neural network utilities
- Data enrichment and import scripts under `scripts/`

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) (optional, for the AI assistant)

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

For the AI assistant:

```bash
ollama pull tinyllama
```

## Run

```bash
python3 -m streamlit run BasicProduction/fish_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Project layout

| Path | Purpose |
|------|---------|
| `BasicProduction/` | Main Streamlit app and models |
| `newfishdata/` | Fish release CSV data |
| `scripts/` | Data enrichment / import helpers |
| `QUICK_START.md` | Detailed AI + app walkthrough |

## Docs

- [QUICK_START.md](QUICK_START.md) — full setup and usage
- [SETUP_OLLAMA.md](SETUP_OLLAMA.md) — Ollama install notes
- [DATA_REFRESH_WORKFLOW.md](DATA_REFRESH_WORKFLOW.md) — refreshing fish data
