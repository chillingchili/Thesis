# Pickleball Serve Analysis Thesis

Research code and an Android proof of concept for the Pascua–Leones thesis on pickleball serve classification and feedback. The repository contains the thesis source, model experiments, evaluation artifacts, mobile and desktop apps, and a static research showcase.

## Start here

- [Thesis source](Pascua-Leones_Thesis.md) — thesis proposal and methodology.
- [Results summary](RESULTS_SUMMARY.md) — reported classification results, holdout findings, and model comparisons.
- [Chapter 5 and 6 draft](analysis/THESIS_CHAPTERS_5_6_DRAFT.md) — draft findings and conclusions; review against the final thesis requirements before submission.
- [Research showcase](research-site/README.md) — run the local findings website.
- [Android app](android/README.md) — mobile proof of concept and build notes.
- [Desktop app](desktop/README.md) — recorded-clip analysis workflow.

## Directory guide

```text
.
├── README.md                         Project overview and this directory guide
├── Pascua-Leones_Thesis.md           Thesis source
├── RESULTS_SUMMARY.md                Consolidated experimental findings
├── PRODUCT.md                        Android prototype scope and behavior
├── Play with Tracker.bat              Windows launcher for the tracker
├── Research Website.bat               Windows launcher for the research site
├── Serve Lab.bat                      Windows launcher for Serve Lab
├── analysis/                          Audits, diagnostics, and thesis chapter drafts
├── android/                           Android Studio app, TFLite assets, and tests
├── archive/
│   ├── experimental_runs/             Superseded GRU and classifier checkpoints
│   └── legacy_pickleball_analytics/    Separate, earlier match-analysis project
├── data/
│   ├── training/                      Training clips and extracted representations
│   ├── testing/                       Beginner holdout clips and extracted features
│   ├── serve_v2/                      Serve refinement experiment inputs/artifacts
│   └── serve_study_v3/                Serve study experiment inputs/artifacts
├── datasets/                           Local detector datasets; excluded from GitHub
├── desktop/                            Desktop analysis app and tests
├── features/                           Saved timing and biomechanical feature matrices
├── label_tool/                         Browser-based paddle annotation tool and frames
├── models/                              GRU, classical, refinement, and TFLite assets
├── outputs/                             Local generated reports and review material (ignored)
├── research-site/                       Static thesis findings website and bundled assets
├── rule-based/                          Rule mapping and annotation audit files
├── scripts/                             Training, extraction, export, evaluation, and utilities
├── weights/                             Shared pretrained model weights
└── runs/                                Local training outputs (ignored)
```

The main serve-subtype pipeline uses `scripts/`, `data/`, `features/`, and `models/`. The Android and desktop apps consume exported assets and share parts of that pipeline. `analysis/` contains supporting audits and drafts; `archive/` holds older, non-current experiments and the earlier standalone project.

## Dataset availability

The `datasets/` directory is present in the local workspace but is ignored by Git. GitHub rejected its upload because the repository owner's Git LFS budget is exhausted. The other tracked project files, including the thesis, code, model assets, results, and website, are published in this repository. Do not interpret the absence of `datasets/` from a fresh clone as an empty dataset; it must be transferred separately or uploaded after Git LFS access is restored.

## Running the research site

Open `research-site/index.html` directly, or run `Research Website.bat` from Windows. The site is static and documents the sources and evaluation limits alongside the findings. See [its README](research-site/README.md) for details.

## Reproducibility notes

Run scripts from the repository root unless a script says otherwise. Model scores retain the evaluation protocol under which they were produced; cross-validation and participant-held-out results are different measures. The results summary and individual audit reports describe those limits. The research site is a showcase, not a replacement for the underlying evaluation artifacts.
