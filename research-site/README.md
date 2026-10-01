# The Serve Study

A static, local research showcase for the Pascua–Leones thesis. Open `index.html`
directly, or launch `launchers/Research Website.bat` from the project root and visit
http://127.0.0.1:7863.

The site includes the original reported experiments, bundled GRU/kNN/hybrid,
v2 retraining, v3 augmentation/input comparisons, v4 calibrated fusion,
detector training logs, actual training-clip output samples, thesis objectives,
and pending expert/phone evaluation. It makes no network requests. Archivo is
self-hosted under the included Open Font License.

Regenerate the saved dataset and media copies after source results change:

```powershell
python scripts/build_research_site.py
```

This exports existing artifacts; it does not retrain or run inference. It needs
the project's OpenCV installation and ffmpeg on PATH. Website files are independent
of the Android app and Serve Lab.

Scores retain their original evaluation contexts. Historical September figures
are reported values, often rounded, not fresh re-evaluations. Detector values
come from each run's final logged validation epoch, not a best-checkpoint test.
The original paper remains a proposal/methodology source; completed experiments
and pending steps are explicitly identified on the page.

The gallery deliberately includes three correct training examples (Drive, Lob,
Topspin) and one error example. This curation is not a performance estimate.
The Drive result is from the retrained v2 ensemble; the others use the original
GRU ensemble. Lob and Topspin were run through the unchanged full desktop pipeline
to save their complete reports; no weights, thresholds or aggregate scores changed.
Only already inspected training examples are included as video samples. Coach C
staging videos and the researcher key are excluded. The selected beginner review
clips have prior diagnostic exposure, so that review is retrospective rather
than an untouched holdout. No Coach C judgments are invented.

Deploy the entire `research-site` directory to any static host when ready to
publish. Research recordings are bundled in the local site; decide which media
to share before public publication.
