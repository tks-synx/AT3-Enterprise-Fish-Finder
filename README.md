# AT3 Enterprise Fish Finder

A Streamlit dashboard that turns game-fish release records into interactive heatmaps, trip estimates, and species insights for recreational fishing planning.

## What it does

Fish Finder loads a large dataset of tagged game-fish releases (location, species, date, and weather/moon context) and lets you explore **where fish have been caught over time**.

You start by choosing how to build a map:

- **Manual Entry** — pick species, years, and months yourself, then generate a heatmap of release density.
- **AI Assisted** — describe what you want in plain language (for example, “yellowfin tuna in late summer”), and the local TinyLlama model chooses the filters and builds the map.

Once a map is open, you can:

- Inspect catch density on an interactive Folium map (hotter colours mean more releases in that area)
- Measure routes on the map and estimate travel time and fuel from your boat profile (tank size, burn rate, cruise speed, vessel type)
- Rate maps and save preferences so later AI suggestions can lean toward what you found useful
- Explore a **neural network** that predicts likely species from location and season, including what-if scenarios
- Run a **certainty-factor expert system** that scores how strongly the data supports a species/month combination

## In short

It is a fisheries analysis tool: visualise historical catch patterns, plan a trip around those patterns, and use two intelligent systems (neural network + certainty-factor rules) plus optional local AI chat to interpret the data — all running from your machine against the enriched release dataset in `newfishdata/`.

## Run

```bash
pip install -r requirements.txt
python3 -m streamlit run BasicProduction/fish_app.py
```

For AI-assisted mapping, install [Ollama](https://ollama.ai) and run `ollama pull tinyllama`. See [QUICK_START.md](QUICK_START.md) for more detail.
