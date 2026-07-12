# Intelligent Systems Implementation

This project implements **two independent, complementary intelligent systems**, both
visible together in the "AI Assisted" page of `BasicProduction/fish_app.py`:

1. A **data-driven neural network** (`BasicProduction/train_species_nn.py` →
   `BasicProduction/fish_species_nn.joblib`) — predictive modelling.
2. A **rule-based certainty-factor expert system** (`BasicProduction/certainty_factor.py`)
   — explicit IF-THEN reasoning.

Together these satisfy the AT3b intelligent-systems criteria for predictive modelling,
what-if analysis, model evaluation, and expert-system/alternative-model reasoning,
without either system replacing the other.

---

## 1. Neural network (predictive modelling)

### Purpose

Predicts the most likely game-fish species for a given location, date, and set of
weather/lunar conditions, based on patterns in ~41,000 historical tagging records. Used
in the app for:
- A single random "example prediction" (actual vs predicted species).
- A "Predict from current map context" button that predicts the dominant species for
  whatever the user has currently filtered/mapped — this is the app's what-if hook
  (change the map filters → get a different prediction).
- Feature importance, training-loss, architecture, and confusion-matrix visualisations,
  for model evaluation.

### Input dataset

`newfishdata/Cleaned_Weather_GameFish_Releases_enriched.csv` (41,440 rows) — see
[DATA_REFRESH_WORKFLOW.md](DATA_REFRESH_WORKFLOW.md) for how this was produced. Both
`fish_app.py` and `train_species_nn.py` load this same file, so the features the model
was trained on always match the features the app builds at prediction time.

### Feature list (10 features)

| Feature | Description |
|---|---|
| `Latitude`, `Longitude` | Release location |
| `Year`, `Month_Number` | Release date, split out |
| `Moon_Phase_Sin`, `Moon_Phase_Cos` | Moon phase (0–1 fraction) encoded cyclically |
| `Wind_Direction_Sin`, `Wind_Direction_Cos` | Wind direction (0–360°) encoded cyclically |
| `Rain_mm` | Rainfall on the release date |
| `Is_Raining` | Whether it was raining (boolean) |

Moon phase and wind direction are encoded as sin/cos pairs rather than raw numbers so
the model understands that, e.g., 359° and 1° are almost the same direction (a plain
number would treat them as nearly opposite).

Target: `Species_Name`, grouped to the 12 most common species (each needing at least 75
samples) plus an `Other` bucket for everything else — 13 classes in total.

### Why `Sea_Surface_Temp_C` was dropped

The enrichment pipeline (`scripts/enrich_fish_data.py`) attempts to fetch sea-surface
temperature from the Open-Meteo Marine API. This was investigated and confirmed to be
**100% null** in every enriched CSV in the repository — the marine API simply has no
coverage for this dataset's combination of historical dates and locations. A feature
that is null for every single row carries no information and cannot be imputed
meaningfully, so it was removed from `FEATURE_COLUMNS` in `train_species_nn.py` rather
than left in as a permanently-missing input. This is documented in code with a comment
at the point it would otherwise have been added.

### Fixing the feature-mismatch bug

Earlier in this project, `fish_app.py` was hardcoding a 4-feature prediction input
(`Latitude`, `Longitude`, `Year`, `Month_Number`), while the model on disk had actually
been trained on 11 features (including the since-dropped `Sea_Surface_Temp_C`). This
caused `model.predict(...)` to raise `ValueError: The feature names should match those
that were passed during fit.` as soon as the dataset/feature alignment work above
started.

This was fixed in two parts:
1. `train_species_nn.py` now saves the exact list of features it trained on into the
   `.joblib` file (`model_info['feature_columns']`).
2. `fish_app.py` reads `feature_columns` back from the loaded model
   (`get_model_feature_columns`) and builds every prediction input dynamically from that
   list (`build_prediction_features_from_row`, `build_context_prediction_features`)
   instead of a hardcoded column list. If the feature set ever changes again, the app
   will automatically follow it.

### Retraining

`BasicProduction/fish_species_nn.joblib` was retrained after the above changes with:

```bash
python3 BasicProduction/train_species_nn.py
```

using `Cleaned_Weather_GameFish_Releases_enriched.csv` and the 10-feature list above
(explicitly excluding `Sea_Surface_Temp_C`).

### Model evaluation metrics (current trained model)

| Metric | Value |
|---|---|
| Accuracy | 62.2% |
| Macro F1 | 54.7% |
| Weighted F1 | 61.0% |
| Training rows | 33,152 |
| Test rows | 8,288 (80/20 stratified split) |
| Classes | 13 (12 named species + `Other`) |

These numbers are shown live in the app's "Neural Network Explorer" panel, along with a
confusion matrix, per-class precision/recall/F1, a training loss curve, and a
permutation-importance feature ranking — this is the model-evaluation evidence for the
assessment. Moderate accuracy is expected and explainable: fish species distribution
depends on many factors not captured here (bait, boat, angler, exact time of day), and
several species have visually/behaviourally overlapping ranges.

For consistency, the standalone comparison script `scripts/feature_impact.py` now also
defaults to `newfishdata/Cleaned_Weather_GameFish_Releases_enriched.csv` and uses the
same final 10-feature set (without `Sea_Surface_Temp_C`).

### Dedicated what-if analysis panel

Beyond the "Predict from current map context" button, a separate **"What-If Analysis
(Neural Network)"** panel (`render_what_if_analysis_panel()` in `fish_app.py`) sits
directly below the Neural Network Explorer. It lets the user adjust month, wind
direction, rain amount, is-raining, moon phase, and latitude/longitude away from the
current baseline context, rebuilds the model input using the same
`get_model_feature_columns()`/feature-pipeline used everywhere else in the app, and
shows a side-by-side comparison: baseline top prediction vs adjusted top prediction, a
bar chart of baseline vs adjusted probabilities, and a table with the percentage-point
change per species. This is the app's explicit what-if analysis evidence, in addition to
the context-prediction button above.

---

## 2. Certainty-factor expert system (rule-based reasoning)

### Purpose

Provides a second, independent recommendation using explicit IF-THEN rules and a
running certainty score, rather than a trained model. This is the "alternative
model"/expert-system comparison point against the neural network: it reasons in a way a
human can read rule-by-rule, whereas the neural network's reasoning is implicit in its
learned weights.

Implemented in `BasicProduction/certainty_factor.py` (no Streamlit dependency — it can
be read, tested, or run stand-alone: `python3 BasicProduction/certainty_factor.py`) and
surfaced in the app via `render_certainty_factor_panel()` in `fish_app.py`, directly
below the Neural Network Explorer, behind a "Show" toggle so it is clearly a separate
system rather than part of the neural network's output.

This directly implements the planning documents:
- `scripts/CertaintyFactors.md` — rule → point-value table
- `scripts/IF-THEN.md` — the IF-THEN rule logic and score thresholds
- `scripts/Flowchart.md` — the decision order the rules are evaluated in

### Rule list

| Rule | Points | What it checks |
|---|---|---|
| Species exists | +10 | Species appears anywhere in the dataset |
| Season match | +25 | Selected month is one of the species' top 4 historical months |
| High catch density | +35 | ≥20 historical catches of this species nearby (~10nm grid) under similar wind/rain/moon conditions |
| Medium catch density | +20 | 5–19 such catches |
| Low catch density | +5 | 1–4 such catches |
| Location match | +25 | Selected location is within the species' 5th–95th percentile lat/lon range |
| User preference match | +15 | The user has previously given positive feedback for this species or month (reuses the app's existing map-rating feedback data, `user_profile.json`) |

The catch-density rule is where wind direction, rainfall, "is it raining", and moon
phase feed into the engine: historical rows are only counted as "nearby" if they also
match the current weather/lunar context (±45° wind direction, ±0.1 moon-phase fraction,
same rain state), so the density score reflects catches made in similar conditions, not
just the same location.

Maximum possible score: 10 + 25 + 35 + 25 + 15 = **110**.

### Scoring / thresholds

Per `scripts/IF-THEN.md`:
- If the species doesn't exist in the dataset at all → score forced to `0`,
  **Insufficient data** (all other rules are skipped).
- `score ≥ 75` → **Strong recommendation**
- `45 ≤ score < 75` → **Moderate recommendation**
- `0 < score < 45` → **Weak recommendation** (advise changing species, season, or
  location)

Note: `scripts/CertaintyFactors.md`'s summary table rounds this to "75–100 Strong",
which is slightly misleading since the rules can sum above 100 (up to 110). The
`IF-THEN.md` operators (`>= 75`, `45–74`, `< 45`) are used directly in the code since
they are unambiguous at any score, including above 100.

### Example output

For ALBACORE in February, evaluated against the current map context (from a live test
run of the app):

> Moderate recommendation (50/110) for ALBACORE in February. 'ALBACORE' appears 1982
> time(s) in the historical dataset. February is one of this species' top 4 historical
> months. User has previously given positive feedback for month 'February'.

| Rule | Fired? | Points | Explanation |
|---|---|---|---|
| Species Exists | Yes | +10 | 'ALBACORE' appears 1,982 time(s) in the historical dataset. |
| Season Match | Yes | +25 | February is one of this species' top 4 historical months. |
| Catch Density | No | 0 | No historical catches found that match nearby/similar conditions. |
| Location Match | No | 0 | Location is outside this species' typical 5–95 percentile range. |
| User Preference Match | Yes | +15 | User has previously given positive feedback for month 'February'. |

The app shows this as a certainty score metric, a confidence-band metric, the plain-English
recommendation sentence, and a full rules table (including rules that did **not** fire,
for transparency).

---

## How both systems meet the assessment requirements

| Requirement | Neural network | Certainty-factor engine |
|---|---|---|
| Predictive modelling | ✅ Core purpose | — |
| What-if analysis | ✅ Dedicated what-if panel (adjust month/wind/rain/moon/location, compare baseline vs adjusted probabilities) plus "Predict from current map context" | ✅ Re-evaluate with a different species/month |
| Model evaluation | ✅ Accuracy/F1/confusion matrix/loss curve/feature importance | ✅ Rules-fired table shows exactly why a score was reached |
| Alternative models / rule-based comparison | (the "alternative" being compared against) | ✅ Explicit IF-THEN rules, independently explainable |
| Expert-system style reasoning | — | ✅ Certainty factors, thresholds, plain-English explanation |
| Explainability for hand-in/interview | Feature importance + confusion matrix | Every rule's pass/fail and points are shown, tied to named planning docs |

