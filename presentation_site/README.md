# Fish Finder — Presentation Site (Firebase Hosting)

Static scrollytelling presentation for the AT3b Fish Finder project.

**This site is separate from the intelligent systems submission.** The neural network,
what-if analysis, and certainty-factor engine run in the local Streamlit app
(`BasicProduction/fish_app.py`). This website explains the project with scroll-driven
text and static visuals only.

## Folder structure

```
presentation_site/
├── firebase.json          # Firebase Hosting config (no .firebaserc yet)
├── README.md
└── public/
    ├── index.html         # Single scrollytelling page
    ├── css/style.css
    ├── js/scrolly.js      # IntersectionObserver scroll triggers
    └── assets/images/     # Screenshots go here in later stages
```

## Preview locally (recommended)

Use a static server so CSS/JS paths resolve correctly:

```bash
cd presentation_site/public
python3 -m http.server 8080
# open http://localhost:8080
```

Scroll through all 11 sections — the sticky visual panel (left on desktop, top on mobile)
should update as each text step enters the viewport. Nav dots highlight the active section
on desktop; a **Section X of 11** label appears in the header on mobile.

You can also open `presentation_site/public/index.html` directly in a browser, but a
local server is preferred for consistent asset paths.

### Quick QA checklist (Stage E)

- [ ] Progress bar fills smoothly as you scroll
- [ ] Nav dots (desktop) match the visible section
- [ ] Mobile header shows **Section X of 11**
- [ ] Sticky panel swaps on substeps (e.g. §2a→§2d, §6a→§6b)
- [ ] Placeholders show asset filename paths under `assets/images/`
- [ ] No broken internal links (only `css/style.css` and `js/scrolly.js`)
- [ ] `prefers-reduced-motion`: instant transitions, no smooth scroll on nav click

## Firebase project

| Setting | Value |
|---|---|
| Project ID | `fishfinder-ad52d` |
| Hosting URL | https://fishfinder-ad52d.web.app |
| Alt URL | https://fishfinder-ad52d.firebaseapp.com |

Config lives in `public/js/firebase-init.js` (loaded as an ES module from the Firebase CDN).
Analytics initialises only when supported (may not run on `localhost`).

`.firebaserc` is included and points at `fishfinder-ad52d`.

## Firebase Hosting — one-time setup

```bash
npm install -g firebase-tools   # or: npx firebase-tools
firebase login
cd presentation_site
```

If you have not linked Hosting yet:

```bash
firebase init hosting
# Use an existing project → fishfinder-ad52d
# Public directory: public
# Single-page app: No
# GitHub auto-deploy: optional
```

## Preview with emulator

```bash
cd presentation_site
firebase emulators:start --only hosting
```

## Deploy

```bash
cd presentation_site
firebase deploy --only hosting
```

After deploy, the site is live at **https://fishfinder-ad52d.web.app**.

## Assets still needed

Capture these as PNG/JPEG into `public/assets/images/` — **never copy CSV, .joblib, or .pbix files**.
Replace each placeholder `<div>` with an `<img src="assets/images/...">` when ready.

| Path | Content |
|---|---|
| `hero/hero.png` | Problem/client need hero image |
| `pipeline/pipeline-diagram.png` | Data pipeline diagram |
| `prototypes/kml-prototype.png` | Static KML/PNG prototype |
| `prototypes/streamlit-dashboard.png` | Streamlit map dashboard |
| `prototypes/powerbi-reference.png` | Power BI reference (prototype stage) |
| `spreadsheet/spreadsheet-screenshot.png` | Excel workbook screenshot |
| `dashboard/dashboard-screenshot.png` | Power BI dashboard evidence |
| `neural-network/metrics.png` | NN metrics from Streamlit |
| `what-if/baseline-vs-adjusted.png` | What-if probability comparison |
| `certainty-factor/rules-table.png` | Certainty-factor output panel |
| `testing/checklist.png` | Testing checklist visual |
| `walkthrough/video-thumbnail.png` | Walkthrough video thumbnail (+ external link in §11) |

## Current stage

**Stage E (complete):** visual polish and QA pass — all 11 sections presentation-ready
with labelled screenshot placeholders, responsive layout, nav/progress improvements.

**Next:** finish remaining screenshot placeholders, then `firebase deploy --only hosting`.

## Content sources for later stages

- `DATA_REFRESH_WORKFLOW.md`
- `INTELLIGENT_SYSTEMS_IMPLEMENTATION.md`
- `PROTOTYPE_EVOLUTION.md`
- `AssessmentTask.md`
- Part A documentation (`12ENC AT3a Enterprise Project 2026 copy.docx`)
