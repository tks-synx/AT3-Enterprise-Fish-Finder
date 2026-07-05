# Section 4.1 Overview Notes — Testing and Evaluating

Planning notes only — **not final paragraphs**. Each section below is a single ordered
list of points to talk about, in order — this already includes what evidence to show,
what problems to mention, what fixes to discuss, and how to judge success woven in at the
right point. Use these as a checklist, then write the actual 4.1 write-up (guide: 2 A4
pages) yourself.

Assessment requires you to verify and validate your system covering five things:
1. Evaluating test data
2. Trialling operation and maintenance documentation
3. Reviewing the impact of system implementation within relevant environments
4. Modifying designs to improve functionality
5. Testing, evaluating and maintaining the developed enterprise computing system

Your system has three evaluated components — **spreadsheet**, **Power BI dashboard**,
**intelligent systems/Streamlit** (NN + what-if + certainty-factor + dataset). Since the
guide is only ~2 pages, you don't need equal depth on all three for every point — pick
the strongest 1–2 examples per point rather than covering every component every time.

<span style="color:#1d4ed8">🔵 Added in this pass: lines starting with 🔵 are new — a
comparison of your AT3a requirements against what was actually built, for Section 6
below. You do NOT need to list every requirement you met (see Section 6's intro) — weave
a met requirement in briefly where natural, and spend most of your words on the gaps.</span>

---

## 1. Evaluating test data

1. State that this is about checking the *data* was fit for purpose (not testing
   software) — data quality/quantity before it's trusted for a spreadsheet, dashboard,
   or model.
2. Walk through the pipeline stages and row counts in order: raw export
   (`GameFish_Releases_Master.csv`, 203,300 rows) → weather/moon enrichment
   (still 203,300 rows, extra columns) → filter to rows with usable weather
   (49,893 rows) → filter to rows with complete length/weight (41,440 rows —
   the final dataset). A pipeline diagram/table of these counts is good evidence to show.
3. Raise the problem you found: weather API coverage was only ~24.5% of enriched rows,
   which is why the biggest row-count drop happens at the weather-filter step — a real
   external-data limitation, not something you could fix.
4. Raise the bigger problem: `Sea_Surface_Temp_C` came back 100% null after enrichment —
   a concrete data-quality finding you discovered, not assumed. A screenshot/snippet
   showing the null check (e.g. `isna().mean()` result) is strong evidence here.
5. Mention some records were also missing Length/Weight and had to be excluded.
6. Explain the fix: dropped SST from `FEATURE_COLUMNS` instead of leaving a
   permanently-null column, and filtered to rows with usable weather + complete
   length/weight rather than trying to impute values that would misrepresent the data.
7. Mention you kept every intermediate CSV as provenance evidence (not deleted) so the
   cleaning steps stay auditable — a screenshot of the spreadsheet showing cleaned vs
   original row counts supports this.
8. Mention the neural network's 80/20 stratified train/test split (33,152 / 8,288 rows)
   as evidence you evaluated whether there was enough data to train and fairly test a
   model.
9. State how you judged success: the final dataset is small enough to be usable but
   every row has complete species/location/weather data, and both the app and the model
   load and run against it without errors.

---

## 2. Trialling the operation and maintenance documentation

1. Define "trialling" for the marker: you didn't just write documentation, you actually
   followed/ran it yourself to check the instructions work.
2. Name the operation/maintenance documents you have: `DATA_REFRESH_WORKFLOW.md`
   (refreshing the dataset + retraining the model), `INTELLIGENT_SYSTEMS_IMPLEMENTATION.md`
   (how both intelligent systems work, for future maintenance),
   `presentation_site/README.md` (preview/deploy steps for the presentation site), and
   the prototype README (e.g. `BasicProduction/prototype_v1_static_kml/README.md`).
3. Give the exact run commands you trialled: `streamlit run BasicProduction/fish_app.py`,
   `python3 BasicProduction/train_species_nn.py`, `python3 build_nesa_spreadsheet.py`. A
   screenshot/snippet of one of these commands succeeding in a terminal is good evidence.
4. Raise a problem you found while trialling: docs and code had drifted apart early on
   (e.g. a stale reference in `TINYLLAMA_REFERENCE.md` to an old model name) — a real
   example of documentation going out of date.
5. Raise a second problem: the "refresh with new raw data" steps in
   `DATA_REFRESH_WORKFLOW.md` needed to be written explicitly, because the earlier
   shapefile-based pipeline was undocumented and hard to reproduce.
6. Explain the fixes: rewrote/updated the stale references once found, and added a
   numbered "Refreshing with new raw data" section describing exactly which scripts to
   re-run in order (enrich → filter → retrain → relaunch app).
7. Mention you kept the old/archived workflow documented separately (clearly marked as
   historical) rather than deleting it, so past decisions are still explainable.
8. State how you judged success: you (or someone else) can follow the documented steps
   from a clean checkout and get a working app/model/spreadsheet without needing to guess
   missing steps — any doc/code mismatch you found and fixed is evidence of this.

---

## 3. Reviewing the impact of system implementation within relevant environments

1. Define the "environments" you're reviewing: local desktop use, offline/field use
   (limited internet at sea), the Firebase presentation site (client-facing), and Power BI
   (business-reporting audience).
2. Explain the offline/field environment's impact on a real design decision: Ollama/
   TinyLlama runs locally instead of calling a cloud API, because internet access can't
   be assumed at sea.
3. Explain the presentation-site environment: it's a separate deployment (static Firebase
   Hosting site, previewed locally with `python3 -m http.server 8080`) from the local
   Streamlit environment, aimed at explaining evidence to a marker/client rather than
   running the live models. A screenshot of each environment running is good evidence.
4. Explain the Power BI environment: a different audience/tool (business intelligence
   dashboard) reviewing the same underlying data as Streamlit.
5. Raise the key problem: the feature-mismatch bug only appeared once the model and app
   were run together in the same environment — i.e. it wasn't caught during isolated
   training, only at integration.
6. Explain the fix: the app now reads `model_info['feature_columns']` dynamically and
   builds prediction inputs from that list, so the app and model can never drift apart
   again regardless of where/how it's run.
7. Raise a second, smaller problem: matplotlib/font cache permission issues appeared when
   running scripts from a new folder location in a sandboxed environment — an
   environment-specific issue unrelated to the code itself.
8. Mention a design change driven by environment review: the static KML prototype was
   moved into its own folder so it still runs standalone in its original environment
   without needing the current cleaned dataset.
9. Mention reviewing the **end-to-end** environment — running the full app (map → AI
   Assisted → neural network → what-if → certainty factor) together, not just testing
   scripts in isolation.
10. State how you judged success: the system behaves consistently whether run locally for
    development, demoed live, or explained via the presentation website, and
    environment-caused issues were identified and either fixed or clearly documented as
    constraints (e.g. offline-first design).

---

## 4. Modifying designs to improve functionality

1. State that this section is about changes made *because* testing/review found a
   problem — the "iteration" evidence, not just a list of features.
2. Pick 2–3 of the strongest examples below and, for each, cover: what was wrong → how
   you found it (error message, wrong output, manual check) → what you changed → why the
   new design is better. This is the richest of the five areas, so don't try to cover all
   of them in equal depth.
3. Example 1 (recommended — pick this one): the app hardcoded 4 prediction features while
   the model was trained on 11 (later 10), causing a shape mismatch. Fix: the app now
   reads `model_info['feature_columns']` dynamically and builds prediction inputs from
   that list. A before/after code snippet of this is strong evidence.
4. Example 2 (recommended — pick this one): `Sea_Surface_Temp_C` was 100% null. Fix:
   removed from `FEATURE_COLUMNS` in the trainer and from the default input path in
   `scripts/feature_impact.py`.
5. Example 3: the spreadsheet script filtered on the now-removed SST column, producing 0
   usable rows. Fix: removed the SST filter so the script uses the cleaned CSV directly,
   producing 18,225 rows.
6. Example 4: the what-if panel's baseline prediction didn't match the "no changes" case
   due to a sin/cos round-trip issue. Fix: the baseline is now reconstructed through the
   same rounded feature path as the adjusted prediction, so unchanged controls give
   identical results. A screenshot of baseline == adjusted with no inputs changed is good
   evidence.
7. Example 5 (optional, lower priority): the static KML prototype had no interactivity,
   so it was replaced with the live Streamlit + Folium dashboard for filtering without
   regenerating files.
8. Make sure each example you pick explains the *reasoning* — not just "I changed X to
   Y" but why the new design is more correct or robust.
9. State how you judged success: re-running the affected feature after the fix produces
   correct output — e.g. `model.predict()` no longer raises a `ValueError`, the
   spreadsheet no longer returns 0 rows, baseline equals adjusted when no what-if inputs
   change.

---

## 5. Testing, evaluating and maintaining the developed enterprise computing system

1. Frame this as the "tie it all together" section — ongoing/repeated testing across all
   three components, plus how the system would be maintained going forward.
2. Spreadsheet: re-ran `build_nesa_spreadsheet.py` end-to-end after the SST fix and
   confirmed the row count (18,225) and the `Weight_Classification` `IF` formula still
   work correctly.
3. Power BI: checked the dashboard as a separate visualisation of the same underlying
   data, confirming both visualisation tools tell a consistent story.
4. Intelligent systems: tested app startup, neural network prediction (valid
   probabilities summing to 100%, no shape errors), the what-if panel, and
   certainty-factor rules separately — each rule tested individually with known inputs to
   confirm correct points and that the total never exceeds 110. A pass/fail checklist
   (already styled in the presentation site, section 9) and a screenshot of the
   certainty-factor rules-fired table are good evidence here.
5. Raise an honest limitation as an evaluation finding: model accuracy (62.2%) is
   moderate, not perfect — explain briefly why (many real-world factors aren't in the
   dataset: bait, boat, angler skill, time of day). A screenshot of the Neural Network
   Explorer's confusion matrix/metrics supports this as evidence you evaluated it
   properly, not just built it.
6. Raise a second limitation: historical recreational catch data may be biased toward
   popular fishing spots and active/reporting anglers — a data-quality limitation
   relevant to ongoing maintenance.
7. Reference `DATA_REFRESH_WORKFLOW.md`'s "refreshing with new raw data" steps as the
   documented maintenance plan for keeping the system up to date if new tagging data
   arrives.
8. List forward-looking maintenance/fixes: repair SST enrichment if a better data source
   becomes available, add more recent/live catch reports over time, trial/compare
   alternative models (this also strengthens your 3.2.2.3 intelligent systems marks if
   you can show it), and add more automated tests instead of relying on manual checks.
9. State how you judged success: each component (spreadsheet, dashboard, intelligent
   systems) was tested on its own before being judged as part of the whole system, and
   you can point to a specific metric or check for each — row counts, formula output,
   model accuracy/F1, rule totals — rather than just saying "it works."

---

<span style="color:#1d4ed8">🔵 ## 6. Reconciling AT3a requirements vs the final build (new)</span>

<span style="color:#1d4ed8">🔵 This section doesn't map to one of the 5 numbered rubric areas on its own — fold these
points into "reviewing the impact of system implementation" and "modifying designs to
improve functionality" above, wherever they fit best. This is where you explicitly own
the differences between what you planned in AT3a and what you actually built, which is
exactly what "verify and validate" is checking. You do not need to also list every
requirement you *did* meet — that's already evident elsewhere in AT3b (spreadsheet,
dashboard, model sections). Pick 2–4 of the strongest gaps below rather than all 8, to
stay within the page guide.</span>

<span style="color:#1d4ed8">🔵 1. Automatic weekly GitHub data sync (planned in AT3a §2.2 "Ease of Maintenance" —
   pulling new fish catch data from GitHub every Monday). Not built — no scheduler/cron
   exists anywhere in the repo. `DATA_REFRESH_WORKFLOW.md` documents a manual refresh
   process instead. State this as a deliberate scope decision and give a reason: no
   reliable free way to run an always-on scheduled job within the project timeframe, and
   manual control reduces the risk of an untested automated job corrupting the live
   dataset.</span>

<span style="color:#1d4ed8">🔵 2. "No external API calls... needed for constant functioning" (AT3a §3.1.4). In
   reality, `scripts/enrich_fish_data.py` calls the Open-Meteo weather API, and the
   presentation site runs on Firebase Hosting. Clarify (this is good news, not a
   contradiction): the weather API is only called once, offline, during dataset
   preparation — never at runtime in the field — so the deployed Streamlit app still has
   zero live API dependency, matching the original offline requirement. Firebase is only
   used for the separate marker-facing presentation site, not the enterprise system
   itself.</span>

<span style="color:#1d4ed8">🔵 3. Direct boat GPS receiver hardware integration (AT3a §3.1.2, Technical Feasibility).
   Not implemented — no serial/NMEA/GPS code anywhere in the codebase. GPX/KML export for
   Garmin devices was built instead. Reason to give: real GPS hardware integration would
   need physical boat hardware to test against, which wasn't feasible for a school
   project timeline; GPX/KML export is a practical substitute that still gets coordinates
   into Garmin devices.</span>

<span style="color:#1d4ed8">🔵 4. "Instant" coordinate copy-to-clipboard (AT3a §1.1/§3.1.1, and your own Context
   Diagram/Level 1 DFD, which literally labels an output "Generated GPS Coordinates
   (Copied to Clipboard)"). Checked `fish_app.py` — there is no clipboard functionality;
   coordinates come from the Folium map interaction (`st_folium`) but aren't
   auto-copied. Own this gap explicitly rather than letting a marker discover it: state
   it wasn't implemented, and why (Streamlit's browser sandboxing makes native
   clipboard-write awkward without extra custom JS components), and what's used instead.</span>

<span style="color:#1d4ed8">🔵 5. High-contrast dark/light mode + oversized buttons for sun-glare and boat motion
   (AT3a §1.1, §2.2, §3.1.1, §3.1.4 — repeated four times as a major non-functional
   requirement). Checked `fish_app.py` — only `st.set_page_config(layout="wide", ...)`
   and two `st.markdown` calls for headings; no custom contrast/theme CSS or oversized
   button styling. This is your biggest gap since it was emphasised the most in AT3a — say
   so directly rather than leaving it silent. Reason to give: deprioritised in favour of
   the intelligent-systems features (NN, what-if, certainty-factor) given the time
   budget; note it as a limitation/future improvement.</span>

<span style="color:#1d4ed8">🔵 6. Bias in data collection — recreational vs commercial fisher bias and the
   "unconscious bias / overfishing feedback loop" you identified yourself in AT3a §3.1.4.
   Your current "Evaluating test data" paragraph in 4.1 doesn't mention it at all. Since
   you flagged this risk yourself in Part A, close the loop here: state whether the final
   dataset shows evidence of it and how it's acknowledged (already covered as a
   limitation on the presentation site and in Section 5 above — just add 1 sentence
   referencing it back to your original AT3a risk).</span>

<span style="color:#1d4ed8">🔵 7. "Read-only CSV access enforced in code" (AT3a §3.1.4, Risks — Data Corruption).
   Not literally implemented as an enforced file permission anywhere. You achieved the
   same underlying goal a different way — keeping copies of the CSV at each pipeline
   stage (already in your Section 1 "Evaluating test data" paragraph). Reframe this
   explicitly: the planned "read-only access" mechanism was replaced with a
   versioned-copies workflow, which is simpler to implement reliably than OS-level
   permission enforcement and preserves full provenance rather than just blocking writes.</span>

<span style="color:#1d4ed8">🔵 8. Pilot implementation with real fishing charter customers (AT3a §3.1.3/§3.1.4,
   "preferred system implementation method"). For a school assessment this didn't happen
   with real paying customers — your customer research was AI-generated personas, not a
   genuine pilot rollout. One line is enough: acknowledge real-world pilot deployment is a
   next step beyond this assessment's scope, distinguishing simulated/persona-based
   validation from the genuine pilot method described in Part A.</span>

---

## Quick self-check before you write

- Have you got at least one **specific example** (with a before/after or an error
  message) for each of the 5 numbered areas above?
- Have you mentioned **all three components** (spreadsheet, dashboard, intelligent
  systems) somewhere across the section, even if not in every single point?
- Have you stated **how you know** each fix worked (a number, a test, a re-run), not
  just that you made a change?
- Are you within the ~2 A4 page guide? If long, cut the weaker/duplicate examples first
  (e.g. keep the feature-mismatch fix and the SST fix, drop less detailed ones).
- <span style="color:#1d4ed8">🔵 Have you addressed at least 2–3 of the AT3a-vs-final-build gaps in Section 6 (not
  all 8 — pick the strongest), rather than only listing requirements you met?</span>
