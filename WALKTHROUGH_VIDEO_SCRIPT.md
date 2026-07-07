# AT3b Section 3.2.3 Walkthrough Video — Script & Screen Recording Cue Sheet

**Purpose:** This is the single document used to mark the practical task. It must show,
step by step, how each assessment section was achieved — not a generic pitch. Everything
below is written to match what actually exists in this repository right now.

**Target length:** ~13–15 minutes total (hard limit: 15 minutes for media attachments).
Section timings below add up to about 14:30 including a short intro — trim the spoken
lines (not the demos) if you run long.

**Order used (matches the assessment sections, not the folder structure):**

1. 2.1 Planning — Log book
2. 2.2 Planning — IF-THEN Flowchart
3. 2.3 Planning — Certainty-Factor Decision Tree
4. 3.2.1 Producing — Prototypes
5. 3.2.2.1 Implementing — Spreadsheet
6. 3.2.2.2 Implementing — Visualisation / Dashboard
7. 3.2.2.3 Implementing — Predictive Analysis Model
8. 3.2.2.4 Implementing — Presentation
9. 4.1 Evaluation — Testing and Evaluating

---

## 1. Pre-recording setup checklist

Do all of this **before** you press record. Recording in one continuous take (or a small
number of clean cuts between sections) is much easier if everything is already open.

### Apps and windows to open, in this order

1. **Logbook** — open your logbook document (OneDrive → `12ENC` folder → `AT3b` folder,
   referred to in Part A as `12ENC AT3 Logbook 2026.docx`). Scroll to a page that shows
   several dated entries so you're not stuck on a blank page.
2. **IF-THEN flowchart** — open your flowchart diagram file (the diagram you built from
   `scripts/Flowchart.md`). If it's a draw.io/Lucidchart file, open it in the editor at a
   zoom level where all symbols are readable.
3. **Certainty-factor decision tree** — two options:
   - Open `scripts/decision_tree.mmd` in a Mermaid renderer: paste its contents into
     <https://mermaid.live> (or draw.io → Arrange → Insert → Advanced → Mermaid), **or**
   - Open your already-exported decision tree diagram/image if you've saved one.
4. **Terminal**, ready in the repo root:
   `/Users/school/Documents/GitHub/AT3-Enterprise-Fish-Finder`
5. **Static prototype folder** — a Finder window open at
   `BasicProduction/prototype_v1_static_kml/` showing `overlay_image.png` /
   `fisheries_heatmap.kml`, plus the `Species_Maps/` folder.
6. **Power BI reference** — open the Power BI file/screenshot you used as a dashboard
   design reference (or have `presentation_site/public/assets/images/prototypes/powerbi-reference.png`
   ready as a fallback image).
7. **Spreadsheet** — open `NESA_AT3b_Spreadsheet_Weather.xlsx` (or
   `NESA_AT3b_Spreadsheet_Weather_Redo.xlsx`) in Excel, on the `Cleaned Data` sheet,
   scrolled so column P (`Weight_Classification`) is visible alongside a few data columns.
8. **Streamlit app** — not open yet; you'll launch it live on camera (see terminal
   commands below) so the audience sees it actually start up.
9. **Presentation website** — either the live Firebase URL
   `https://fishfinder-ad52d.web.app`, or run it locally (command below). Have it scrolled
   back to Section 1 before you need it.
10. **Documentation files** on hand for quick reference/paraphrasing while you talk (don't
    read them verbatim on screen): `DATA_REFRESH_WORKFLOW.md`,
    `INTELLIGENT_SYSTEMS_IMPLEMENTATION.md`, `scripts/CertaintyFactors.md`.

### Terminal commands to have ready (type/run live, don't pre-run and hide it)

```bash
cd /Users/school/Documents/GitHub/AT3-Enterprise-Fish-Finder
source env/bin/activate
python3 -m streamlit run BasicProduction/fish_app.py
```

Presentation site, if demoing locally instead of the live Firebase URL:

```bash
cd presentation_site/public
python3 -m http.server 8080
# then open http://localhost:8080 in a browser
```

Static prototype (only if you want to show it actually *running*, not just the saved
output files):

```bash
cd BasicProduction/prototype_v1_static_kml
python3 main.py
python3 species_heatmaps.py
```

### Screenshots to have ready as backups (open in Preview/a second tab, minimised)

- `presentation_site/public/assets/images/prototypes/kml-prototype.png` — static KML prototype
- `presentation_site/public/assets/images/prototypes/streamlit-dashboard.png` — Streamlit dashboard
- `presentation_site/public/assets/images/prototypes/powerbi-reference.png` — Power BI reference
- `presentation_site/public/assets/images/spreadsheet/spreadsheet-screenshot.png` — spreadsheet
- `presentation_site/public/assets/images/dashboard/dashboard-screenshot.png` — Power BI dashboard
- A saved screenshot of the Neural Network Explorer's metrics tab (accuracy/confusion matrix)
- A saved screenshot of a completed Certainty-Factor evaluation (score + rules table)

### Backup options if something doesn't load live

- **Streamlit app won't start in time** → cut to the saved
  `streamlit-dashboard.png` screenshot and narrate over it, then try again for the
  predictive-model section (it only needs to load once).
- **Ollama-independent features are all local, so this shouldn't apply** — but if the
  neural network model file fails to load, fall back to the metrics screenshot and the
  numbers quoted in this script (~62–63% accuracy) instead of a live prediction.
- **Power BI file unavailable on this machine** → use the dashboard screenshot; say
  clearly it's a design/reference screenshot, not a live demo.
- **Presentation website offline/Firebase down** → run it locally with the `http.server`
  command above, or fall back to scrolling the `presentation_site/public/index.html`
  file directly in a browser.
- **Logbook file won't open in time** → have 2–3 exported logbook page screenshots ready.

---

## 2. Timed walkthrough plan

| # | Section | Est. time | What to show | What to say (summary) | Criterion it proves |
|---|---|---|---|---|---|
| — | Intro | 0:20 | Face/title card, project name | One sentence: what the system is and who it's for | Sets context only |
| 1 | 2.1 Log book | 0:45 | Logbook entries, dated, several weeks | How you used it to track progress/decisions | 2.1 Planning — Logbook |
| 2 | 2.2 Flowchart | 1:15 | IF-THEN flowchart diagram | Walk the flow: species check → season → density → location → preference → score bands | 2.2 Planning — Flowchart |
| 3 | 2.3 Decision Tree | 1:15 | `decision_tree.mmd` rendered | Same rules as certainty factors, but as branching decisions with CF points at each branch | 2.3 Planning — Decision Tree |
| 4 | 3.2.1 Prototypes | 2:00 | Static KML/PNG → Streamlit dashboard → Power BI reference | Explain why each stage was replaced/kept, what feedback drove the change | 3.2.1 Producing — Prototypes |
| 5 | 3.2.2.1 Spreadsheet | 1:30 | Excel `Cleaned Data` sheet, column P formula | 18,225-row cleaned sample, `=IF(...)` weight classification formula, where the raw data came from | 3.2.2.1 Implementing — Spreadsheet |
| 6 | 3.2.2.2 Dashboard | 2:00 | Streamlit Manual Entry filters + heatmap | Filter by species/year/month, generate heatmap, explain density legend & measuring tool | 3.2.2.2 Implementing — Visualisation |
| 7 | 3.2.2.3 Predictive model | 2:45 | Neural Network Explorer tabs, What-If panel, Certainty-Factor panel | Model inputs/accuracy, adjust a variable and compare predictions, run one certainty-factor evaluation | 3.2.2.3 Implementing — Intelligent Systems |
| 8 | 3.2.2.4 Presentation | 1:30 | Firebase scrollytelling site | Scroll through a few sections, explain it's the client-facing summary | 3.2.2.4 Implementing — Presentation |
| 9 | 4.1 Testing/evaluating | 2:00 | Testing checklist, `DATA_REFRESH_WORKFLOW.md`, a bug you fixed | What you tested, how you found/fixed a real bug, what you'd still improve | 4.1 Evaluation — Testing and Evaluating |
| — | Outro | 0:15 | Face/title card | One-sentence wrap-up | — |

---

## 3. Full spoken script

Read this naturally — it's written the way I'd actually talk, not as a formal report.
Swap in your own filler words where it feels stiff. Anywhere it says "[read live number]"
or similar, say whatever the screen is actually showing at that moment, not a fixed value.

### Intro (0:20)

> Hi, I'm Eamon, and this is the walkthrough for my Enterprise Computing project — a fish
> catch heatmap system for fishing charter companies, built with Python, Streamlit and
> Folium. I'm going to go through this in the order of the assessment sections, showing
> what I actually built for each one.

### 1. 2.1 Planning — Log book (0:45)

**Show:** logbook document, scrolled to a page with multiple dated entries.

> This is my logbook. I used it to record what I worked on each session — so you can see
> entries like when I first got the shapefile data importing correctly, when I switched
> from the static KML prototype to the interactive Streamlit map, and when I hit and fixed
> the feature-mismatch bug in the neural network. Each entry has the date, what I did,
> what went wrong, and what I planned to do next. It's basically the running record of
> every decision I made across the project.

### 2. 2.2 Planning — IF-THEN Flowchart (1:15)

**Show:** the IF-THEN flowchart diagram.

> This flowchart represents the IF-THEN rules used by my expert system — the
> certainty-factor engine. It starts by checking whether the selected species actually
> exists in the historical dataset — if it doesn't, the system stops straight away and
> says there's insufficient data. If it does exist, the flow moves through four more
> checks: whether the selected month matches the species' usual season, whether the
> local area has a high historical catch density under similar wind, rain and moon
> conditions, whether the location falls inside the species' normal range, and whether
> I've previously rated similar searches positively. Each "yes" adds certainty points,
> and at the end those points are added up and compared against thresholds to decide if
> it's a strong, moderate, or weak recommendation. This flowchart is exactly what I later
> implemented in code in `certainty_factor.py`.

### 3. 2.3 Planning — Certainty-Factor Decision Tree (1:15)

**Show:** the rendered `decision_tree.mmd` decision tree.

> This decision tree represents the same rules as the flowchart, but laid out as a
> branching tree with the certainty-factor points shown at each decision point. You can
> see the species-exists check on the left, branching into season match worth 25 points,
> catch density worth up to 35, location match worth 25, and user preference worth 15 —
> so the maximum possible score is 110. On the right-hand side, the total score is
> compared against the bands: 75 or above is a strong recommendation, 45 to 74 is
> moderate, below 45 is weak, and if the species doesn't exist at all it's forced to zero
> and marked as insufficient data. This tree is what I coded rule-by-rule when I built the
> certainty-factor engine.

### 4. 3.2.1 Producing — Prototypes (2:00)

**Show:** static KML/PNG overlay → Streamlit dashboard screenshot/live app → Power BI reference.

> My first working prototype wasn't the Streamlit app at all — it was a script that
> generated static heatmap overlays as PNG images and KML files you could open directly
> in Google Earth. This let me prove the actual heatmap logic — turning raw catch
> coordinates into a density overlay — before I built any kind of interface around it.
> You can see it here: one combined overlay, and a separate overlay per species.
>
> The problem with that prototype was that it wasn't interactive — a fishing charter
> operator couldn't change filters without me re-running the script and generating new
> files. So the next stage was moving to Streamlit and Folium, which gave me an
> in-browser map that updates live when you change species, year, or month filters — this
> became the actual production app.
>
> Alongside that, I also used Power BI as a reference point while I was designing the
> dashboard layout — looking at how a proper business-intelligence tool lays out filters,
> charts and KPIs helped me decide what the Streamlit dashboard needed, even though the
> final interactive system runs in Streamlit rather than Power BI. So the evolution is:
> static per-species overlays, to a live filterable Folium heatmap, informed by a Power BI
> dashboard reference along the way.

### 5. 3.2.2.1 Implementing — Spreadsheet (1:30)

**Show:** `NESA_AT3b_Spreadsheet_Weather.xlsx`, `Cleaned Data` sheet, column P formula bar.

> This is the spreadsheet deliverable, built from the same final cleaned dataset the app
> uses — `Cleaned_Weather_GameFish_Releases_enriched.csv`. I dropped any row missing a
> species, weight, or location, and took a clean sample of just over eighteen thousand
> rows for the workbook. In column P I added a `Weight_Classification` formula — it's an
> IF formula that checks the weight column and labels anything over 50 kilograms as
> "Heavy Game Fish" and everything else as "Standard". That's a genuine formula running
> over every row, not a static label. The original raw export and each cleaning stage are
> also kept as separate CSVs in the `newfishdata` folder, so there's a clear trail from
> raw data to this cleaned spreadsheet.

### 6. 3.2.2.2 Implementing — Visualisation / Dashboard (2:00)

**Show:** launch the Streamlit app live, use Manual Entry, pick species/year/month, generate map.

> Now let's look at the actual dashboard. I'll start the app from the terminal...
> [wait for it to open] ...and here's the landing page. I'll go into Manual Entry, which
> gives me a grid of species, years and months to filter by. I'll pick a species —
> [name it] — a year range, and a couple of months, and hit Generate Map.
>
> That's a Folium heatmap built from the filtered rows, colour-coded from blue for low
> density up to red for high density, with the legend showing the actual maximum catch
> count for this selection. I can also use the measuring tool on the map to check distance
> between points, and there's a trip-estimator panel below that uses the boat profile I've
> set up to estimate fuel and time for a selected route. This is the core visualisation
> deliverable — species, year and month filtering feeding straight into an interactive
> geographic heatmap.

### 7. 3.2.2.3 Implementing — Predictive Analysis Model (2:45)

**Show:** Neural Network Explorer tabs → What-If Analysis panel → Certainty-Factor panel.

> Below the map is the intelligent-systems section, which actually has two separate
> systems working together. The first is a neural network that predicts the most likely
> species for a given location, date, and set of weather and moon conditions — it's
> trained on ten features: latitude, longitude, year, month, moon phase and wind
> direction encoded as sine/cosine pairs so the model understands they're cyclical, plus
> rainfall and whether it was raining. I dropped sea-surface temperature from the model
> entirely, because when I checked it, it came back null for a hundred percent of rows —
> the weather API just doesn't have coverage for this combination of dates and locations,
> so keeping a feature that's always empty would've been pointless.
>
> In the Neural Network Explorer you can see the training progress, the network
> architecture, feature importance, and the actual evaluation metrics — accuracy is
> around [read live number]%, across thirteen species classes, with a confusion matrix
> and per-class precision and recall so you can see exactly where it gets confused
> between similar species.
>
> Right below that is the What-If Analysis panel — this is where I can take the current
> prediction and change one variable, like the month or the wind direction, and see how
> the predicted species probabilities shift. [Adjust a slider/dropdown live.] You can see
> the baseline versus adjusted comparison chart update immediately.
>
> The second intelligent system is the certainty-factor expert system, which is separate
> from the neural network on purpose — it's rule-based rather than trained, so every score
> it gives can be explained rule by rule. I'll pick a species and month and run an
> evaluation... and here you can see the score out of 110, the confidence band, a
> plain-English explanation, and a full table showing exactly which rules fired and which
> didn't — this is the same logic from the flowchart and decision tree I showed earlier,
> now actually running.

### 8. 3.2.2.4 Implementing — Presentation (1:30)

**Show:** Firebase-hosted scrollytelling website, scroll through a few sections.

> For the presentation, I built a scroll-driven website hosted on Firebase, separate from
> the Streamlit app, so I could walk a client through the whole project without needing
> them to run any code. It's got twelve sections — starting with the client problem,
> moving through the data pipeline, the prototype evolution, the spreadsheet and dashboard
> evidence, both intelligent systems, testing, limitations, and finishing with what we
> learnt. As I scroll, the panel on the side updates with the matching screenshot or stat
> for whichever section is active. This is the version I'd actually send a fishing charter
> client to look at.

### 9. 4.1 Evaluation — Testing and Evaluating (2:00)

**Show:** testing checklist (presentation site Section 9, or your own notes), `DATA_REFRESH_WORKFLOW.md`.

> For testing, I checked each part of the system separately rather than just the whole
> app at once — app startup, the neural network prediction pipeline, the what-if panel,
> the certainty-factor rules, the spreadsheet generation script, and that the dataset path
> and feature columns stayed consistent between the app and the trainer.
>
> That last one is actually a real bug I found and fixed during development: early on, the
> app was hardcoding a four-feature prediction input, but the model on disk had actually
> been trained on eleven features including sea-surface temperature. That mismatch caused
> the app to crash with a feature-names error the moment I tried to run a prediction. I
> fixed it by having the training script save the exact feature list it used into the
> model file, and having the app read that list back and build every prediction from it
> dynamically — so if the feature set ever changes again, the app follows it automatically
> instead of crashing.
>
> For maintenance, I documented the full data-refresh pipeline — from the raw export,
> through weather enrichment, through the two filtering steps, down to the final cleaned
> dataset — so the process of refreshing the data and retraining the model is repeatable
> and doesn't rely on me remembering the steps.
>
> In terms of impact and limitations — the model's accuracy is useful but not perfect,
> which makes sense given how many real-world factors affect where fish actually are that
> aren't in this dataset, like bait, boat, or time of day. The dataset itself also leans
> towards recreational and sport-fishing catches, so it doesn't fully represent commercial
> catch patterns. If I kept developing this, I'd want to fix the sea-surface-temperature
> gap with a better data source, bring in more recent catch reports, and try comparing the
> neural network against an alternative model type to see if accuracy improves.

### Outro (0:15)

> That's the full system — from planning through to a working, tested predictive tool.
> Thanks for watching.

---

## 4. Screen recording cue sheet

| Cue | File/page/app | Action | Key point to say | Evidence shown |
|---|---|---|---|---|
| 1 | Logbook document | Scroll through 2–3 dated entries | "Running record of decisions and progress" | 2.1 Logbook |
| 2 | IF-THEN flowchart diagram | Point/highlight each decision box top to bottom | "Species check → season → density → location → preference → score bands" | 2.2 Flowchart |
| 3 | `scripts/decision_tree.mmd` rendered (mermaid.live or draw.io) | Trace left-to-right through columns | "Same rules, shown as branches with CF points, max 110" | 2.3 Decision Tree |
| 4a | `prototype_v1_static_kml/overlay_image.png` / `fisheries_heatmap.kml` | Open image, or open KML in Google Earth if available | "First working prototype — static per-species overlays" | 3.2.1 Prototype stage 1 |
| 4b | Streamlit app (live) or `streamlit-dashboard.png` | Show Manual Entry screen | "Became the live interactive map" | 3.2.1 Prototype stage 2 |
| 4c | Power BI file/`powerbi-reference.png` | Show dashboard layout | "Used as a design reference for the dashboard" | 3.2.1 Prototype stage 3 |
| 5 | `NESA_AT3b_Spreadsheet_Weather.xlsx` | Scroll `Cleaned Data` sheet, click cell P2 to show formula bar | "18,225 cleaned rows, IF formula for weight classification" | 3.2.2.1 Spreadsheet |
| 6a | Terminal | Run `source env/bin/activate` then `python3 -m streamlit run BasicProduction/fish_app.py` | "Launching the dashboard" | 3.2.2.2 setup |
| 6b | Streamlit landing page | Click "Manual Entry" | "Filter mode" | 3.2.2.2 |
| 6c | Manual Entry page | Tick a species, a year, a month → click "Generate Map" | "Species/year/month filtering into a heatmap" | 3.2.2.2 Visualisation |
| 6d | Generated map | Point at density legend, use measuring tool | "Colour-coded density + distance measuring" | 3.2.2.2 |
| 7a | Neural Network Explorer | Open tabs: Training Progress → Network Architecture → Feature Importance → Performance Details | "10 features, accuracy/confusion matrix" | 3.2.2.3 Predictive model |
| 7b | What-If Analysis panel | Change month or wind direction dropdown | "Baseline vs adjusted prediction comparison" | 3.2.2.3 What-if analysis |
| 7c | Certainty-Factor Expert System panel | Tick "Show", pick species + month, click "Evaluate certainty factor" | "Score out of 110, rules table, plain-English recommendation" | 3.2.2.3 Expert system |
| 8a | `https://fishfinder-ad52d.web.app` (or local `http.server`) | Scroll through Sections 1, 3, 6, 9, 12 | "Client-facing scrollytelling summary" | 3.2.2.4 Presentation |
| 9a | Testing checklist (site Section 9 or own notes) | Show pass/fail list | "Each part tested separately" | 4.1 Evaluating test data |
| 9b | `DATA_REFRESH_WORKFLOW.md` | Scroll the pipeline diagram at the top | "Repeatable maintenance/refresh process" | 4.1 Operation and maintenance documentation |
| 9c | Spoken only (no new screen) | — | "Real bug found and fixed: feature-mismatch crash" | 4.1 Modifying designs to improve functionality |
| 9d | Presentation site Section 10 ("Limitations & future work") | Show limitations/future-improvements card | "Honest evaluation + next steps" | 4.1 Reviewing impact / evaluating |

---

## 5. Final checklist

### What the video must show before you submit

- [ ] Every one of the 9 sections above appears in order, clearly labelled by what you're
      showing (say the section name out loud, e.g. "this is for section 2.2").
- [ ] The IF-THEN flowchart and the certainty-factor decision tree are both shown as
      actual diagrams, not just described verbally.
- [ ] At least one live, on-screen interaction with the Streamlit app (not only
      screenshots) — filtering the map and running one prediction/certainty-factor
      evaluation.
- [ ] The spreadsheet shows a real formula in the formula bar, not just a static number.
- [ ] The predictive model section shows an actual accuracy/metric number on screen.
- [ ] The testing section names at least one specific thing you tested and one specific
      problem you found and fixed (not "I tested everything and it worked").
- [ ] Total runtime is under 15 minutes.

### Common things to avoid

- Don't read this script word-for-word in a monotone — pause to actually interact with
  the app between sentences.
- Don't skip past a section too fast to show any evidence — every row in the cue sheet
  needs at least a few seconds on screen.
- Don't just say "the model is accurate" without showing the number.
- Don't present limitations/testing as an afterthought — Section 4.1 is worth marks on
  its own, treat it with the same care as the build sections.
- Don't leave dead air while the Streamlit app is starting up — talk through what's about
  to happen while it loads.

### Backup screenshots/evidence to have open in another window in case a live demo fails

- `streamlit-dashboard.png`, `kml-prototype.png`, `powerbi-reference.png`,
  `spreadsheet-screenshot.png`, `dashboard-screenshot.png` (all under
  `presentation_site/public/assets/images/`)
- A saved screenshot of the Neural Network Explorer metrics tab
- A saved screenshot of a completed certainty-factor evaluation
- Exported logbook page screenshots
