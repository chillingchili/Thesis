# Pickleball Serve Analysis Thesis

Research code and an Android proof of concept for the Pascua–Leones thesis on pickleball serve classification and feedback. The repository contains the thesis source, model experiments, evaluation artifacts, mobile and desktop apps, and a static research showcase.

## Start here

- **[Partner handoff — current state and next steps](docs/PARTNER_HANDOFF.md)** — read this first; it supersedes old pending-status notes.
- **[Coach C evaluation folder](evaluation/coach_c/README.md)** — responses, frozen reports, revised feedback and all 30 clips.
- **[Current Session 2 form script](scripts/google_apps_script/README.md)** — setup and updating an existing Google form.
- [Thesis source](docs/Pascua-Leones_Thesis.md) — thesis proposal and methodology.
- [Historical results summary](docs/RESULTS_SUMMARY.md) — September experiments; read alongside [the report index](analysis/README.md).
- [Chapter 5 and 6 draft](analysis/THESIS_CHAPTERS_5_6_DRAFT.md) — draft findings and conclusions; review against the final thesis requirements before submission.
- [Research showcase](research-site/README.md) — run the local findings website.
- [Android app](android/README.md) — mobile proof of concept and build notes.
- [Desktop app](desktop/README.md) — recorded-clip analysis workflow.
- [Coaching development](docs/COACHING_DEVELOPMENT.md) — opt-in measurement and feedback safeguards, plus a calibration audit.

## Directory guide

```text
.
├── README.md                      Project overview and this directory guide
├── .gitignore                     Shared ignore rules
├── .gitattributes                 Preserve evaluation evidence bytes and hashes
├── docs/                          Thesis, results summary, and product description
├── launchers/                     Windows shortcuts for the tracker, site, and Serve Lab
├── analysis/                      Audits, diagnostics, and chapter 5–6 drafts
├── android/                       Android Studio app, TFLite assets, and tests
├── archive/                       Superseded experiments and legacy analytics project
├── data/                          Training, testing, and serve-study data
├── datasets/                      Local detector datasets (excluded from GitHub)
├── desktop/                       Desktop analysis app and tests
├── evaluation/coach_c/            Tracked Coach C evidence, responses and 30 videos
├── features/                      Saved timing and biomechanical feature matrices
├── label_tool/                    Paddle annotation app, frames, and labels.json
├── logs/                          Local run logs and error output (ignored)
├── models/                        GRU, classical, refinement, and TFLite assets
├── outputs/                       Generated reports and review material (ignored)
├── research-site/                 Static thesis findings website and bundled assets
├── rule-based/                    Rule mapping and annotation audit files
├── scripts/                       Training, extraction, export, evaluation, and utilities
│   └── google_apps_script/        Current Session 2 form builder and local checks
├── weights/                       Shared pretrained YOLO model weights
└── runs/                          Local training outputs (ignored)
```

The root contains the project guide and Git configuration. Documents, launchers, labels, and weights are grouped in their respective folders. Original frozen paths remain intact; the organized `evaluation/coach_c/` handoff contains verified copies with a provenance manifest. See [scripts/README.md](scripts/README.md) for commands and [analysis/README.md](analysis/README.md) for study reports. `archive/` holds superseded experiments.

## Current evaluation

Session 1 is received: 27 Bad Form and 3 Good Form judgments. Independent serve-type labels were intentionally removed from the actual instrument. Session 2 ratings are pending. The current Google form shows the frozen predicted subtype and approved revised feedback, with Valid / Not valid / No coaching feedback to assess choices. The current hybrid produced 16 corrective, one positive, 12 low-confidence and one partial output. These are coverage counts, not accuracy or appropriateness results.

## Dataset availability

The `datasets/` directory is present in the local workspace but is ignored by Git. GitHub rejected its upload because the repository owner's Git LFS budget is exhausted. The other tracked project files, including the thesis, code, model assets, results, and website, are published in this repository. Do not interpret the absence of `datasets/` from a fresh clone as an empty dataset; it must be transferred separately or uploaded after Git LFS access is restored.

## Running the research site

The 30 selected Coach C videos and compact evidence are tracked under `evaluation/coach_c/`. Repeated ZIPs, run caches, detector training runs and generated build files remain local. A clone does not contain every ignored artifact needed to retrain or reproduce all historical runs.

Open `research-site/index.html` directly, or run `launchers/Research Website.bat` from Windows. The site is static and documents the sources and evaluation limits alongside the findings. See [its README](research-site/README.md) for details.

## Reproducibility notes

Run scripts from the repository root unless a script says otherwise. Model scores retain the evaluation protocol under which they were produced; cross-validation and participant-held-out results are different measures. The results summary and individual audit reports describe those limits. The research site is a showcase, not a replacement for the underlying evaluation artifacts.
