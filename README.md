# Pickleball Serve Analysis Thesis

Research code and an Android proof of concept for the Pascua–Leones thesis on pickleball serve classification and feedback. The repository contains the thesis source, model experiments, evaluation artifacts, mobile and desktop apps, and a static research showcase.

## Start here

- [Thesis source](docs/Pascua-Leones_Thesis.md) — thesis proposal and methodology.
- [Results summary](docs/RESULTS_SUMMARY.md) — reported classification results, holdout findings, and model comparisons.
- [Chapter 5 and 6 draft](analysis/THESIS_CHAPTERS_5_6_DRAFT.md) — draft findings and conclusions; review against the final thesis requirements before submission.
- [Research showcase](research-site/README.md) — run the local findings website.
- [Android app](android/README.md) — mobile proof of concept and build notes.
- [Desktop app](desktop/README.md) — recorded-clip analysis workflow.

## Directory guide

```text
.
├── README.md                      Project overview and this directory guide
├── .gitignore                     Shared ignore rules
├── docs/                          Thesis, results summary, and product description
├── launchers/                     Windows shortcuts for the tracker, site, and Serve Lab
├── analysis/                      Audits, diagnostics, and chapter 5–6 drafts
├── android/                       Android Studio app, TFLite assets, and tests
├── archive/                       Superseded experiments and legacy analytics project
├── data/                          Training, testing, and serve-study data
├── datasets/                      Local detector datasets (excluded from GitHub)
├── desktop/                       Desktop analysis app and tests
├── features/                      Saved timing and biomechanical feature matrices
├── label_tool/                    Paddle annotation app, frames, and labels.json
├── logs/                          Local run logs and error output (ignored)
├── models/                        GRU, classical, refinement, and TFLite assets
├── outputs/                       Generated reports and review material (ignored)
├── research-site/                 Static thesis findings website and bundled assets
├── rule-based/                    Rule mapping and annotation audit files
├── scripts/                       Training, extraction, export, evaluation, and utilities
├── weights/                       Shared pretrained YOLO model weights
└── runs/                          Local training outputs (ignored)
```

The root contains only the project guide and Git ignore rules; project documents, launchers, labels, and pretrained weights are grouped in their respective folders. The main serve-subtype pipeline uses `scripts/`, `data/`, `features/`, and `models/`. The Android and desktop apps consume exported assets and share parts of that pipeline. `analysis/` contains supporting audits and drafts; `archive/` holds older, non-current experiments and the earlier standalone project.

## Dataset availability

The `datasets/` directory is present in the local workspace but is ignored by Git. GitHub rejected its upload because the repository owner's Git LFS budget is exhausted. The other tracked project files, including the thesis, code, model assets, results, and website, are published in this repository. Do not interpret the absence of `datasets/` from a fresh clone as an empty dataset; it must be transferred separately or uploaded after Git LFS access is restored.

## Running the research site

Open `research-site/index.html` directly, or run `launchers/Research Website.bat` from Windows. The site is static and documents the sources and evaluation limits alongside the findings. See [its README](research-site/README.md) for details.

## Reproducibility notes

Run scripts from the repository root unless a script says otherwise. Model scores retain the evaluation protocol under which they were produced; cross-validation and participant-held-out results are different measures. The results summary and individual audit reports describe those limits. The research site is a showcase, not a replacement for the underlying evaluation artifacts.
