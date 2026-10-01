"""Document the completed training-only retrain and frozen diagnostic evaluation."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models/serve_v2_fixed"


def pct(value):
    return f"{100*value:.1f}%"


def main():
    cv = json.loads((OUT / "cv_results.json").read_text())
    pilot = json.loads((ROOT / "models/serve_v2/cv_results.json").read_text())
    config = json.loads((OUT / "training_config.json").read_text())
    evaluated = json.loads((OUT / "evaluation.json").read_text())
    frozen = json.loads((OUT / "frozen_models.json").read_text())
    quality = json.loads((ROOT / "data/serve_v2/quality.json").read_text())
    predictions = list(csv.DictReader((OUT / "predictions.csv").open()))
    trained = len(config["training_ids"])
    test = evaluated["results"]["diagnostic_test"]
    score = test["v2_ensemble"]["all"]
    old = test["legacy_desktop_ensemble"]["all"]
    lines = ["# GRU retraining result — 2026-10-01", "",
             f"Retrained the GRU using **{trained} Coach B annotated training clips only**. "
             "No Beginner 3/4 testing clip was passed to model fitting, early stopping, "
             "normalization fitting, confidence tuning, or checkpoint selection. BlazePose "
             "weights were not trained or modified.", "",
             f"The new five-fold ensemble scores **{pct(score['accuracy_all_clips_quality_failures_wrong'])} "
             f"on the existing 175 diagnostic test clips**, versus {pct(old['accuracy'])} for the old "
             "desktop ensemble on the same freshly extracted Lite landmarks. This is the "
             "relevant desktop baseline; the earlier 38.3% audit used cached Heavy poses and "
             "the old training preprocessing, so it is a different comparison. **The retrain did "
             "not improve unseen-player accuracy; the old desktop model remains the default.**", "",
             "## Dataset and separation", "",
             "- Located all 210 annotated raw training videos, including the five absent from old feature caches.",
             f"- {trained} passed the predeclared tracking quality gate (at least 32 source frames, "
             "at least 80% valid source and resampled frames; required landmark visibility ≥0.50).",
             "- Excluded from fitting: " + ", ".join(config["excluded_training_ids"]) + ". Its annotation "
             "and quality statistics remain in the frozen inventory; no label was changed.",
             f"- Training class counts in drive/lob/topspin order: {config['class_counts']}.",
             "- Froze video SHA-256 hashes, Coach B workbook hash, labels, player IDs and split "
             "membership in data/serve_v2/dataset.json. Extracted all clips with separate tracker instances.",
             "- The existing 175 Beginner 3/4 clips are diagnostic testing data only. They were already "
             "inspected during earlier project experiments. Model weights were frozen before this "
             "run's test-label evaluation. No fitting code opens diagnostic_test.npz.",
             "- The user confirmed that Coach C's independently annotated 30-clip set does not exist "
             "yet. Its evaluation remains pending; neither this model nor these 175 labels represent "
             "that final thesis evaluation. The actual 209 eligible training count also differs from "
             "the planned 210 usable clips and must be documented in the thesis.", "",
             "## Pipeline corrections", "",
             "1. Shared scripts/serve_sequence.py between training and desktop: the same Tasks VIDEO "
             "BlazePose Lite asset, settings, and fresh tracker per selected recording.",
             "2. Resample the complete selected serve into 128 steps over elapsed time. Preserve "
             "both ends and normalize interpolated sin/cos pairs. Do not take only the first 128 "
             "frames or estimate contact to crop the classifier input. Clips must contain one serve "
             "from setup through follow-through, at most 12 seconds.",
             "3. Retain visibility and time metadata. Mask unreliable required joints; do not interpolate "
             "through a missing source frame. Apply the same quality rejection in training and desktop.",
             "4. Remove inference-only moment matching for the candidate. The model sees the same "
             "feature representation during training and inference.",
             "5. Fix SequenceAugment so padding and dropped frames stay zero after sin/cos encoding "
             "and noise. Previously cos(atan2(0,0)) populated missing frames with artificial poses.",
             "6. Reset the original legacy Heavy extractor before each new video too. Existing cached "
             "Heavy files and original model checkpoints were preserved.", "",
             "## Training and validation", "",
             "An initial training-only pilot used inner validation to select an epoch and refit each "
             "outer training fold. Its stopping point varied from 12 to 94 epochs; the early-stopped "
             "fold was undertrained. The final candidate uses a fixed 100 epochs, selected from this "
             "training-only development evidence. The two runs and their histories are preserved. "
             "Therefore the final stratified score is a **development CV score**, not a pristine "
             "independent estimate after hyperparameter selection. The diagnostic test was not used "
             "to choose between these schedules.", "",
             "The final GRU has 32 units, dropout 0.15, L2 0.0001, batch size 16, and conservative "
             "training-only angle/noise/time/frame-drop augmentation. No horizontal flip is used. "
             "The current implementation augments extracted pose features; it does not implement "
             "the paper's brightness/contrast/crop augmentation of raw videos, and this remains "
             "a methodological difference to reconcile. Seeds, fold membership, training histories, "
             "and model hashes are saved. Each outer model is fitted on its training indices only; "
             "outer validation is never passed to fit in the final run.", "",
             "| Evaluation | Initial nested pilot | Fixed 100-epoch candidate |",
             "|---|---:|---:|",
             f"| Stratified five-fold training CV | {pct(pilot['stratified']['metric']['accuracy'])} | {pct(cv['stratified']['metric']['accuracy'])} |",
             f"| Leave-one-training-player-out | {pct(pilot['participant']['metric']['accuracy'])} | {pct(cv['participant']['metric']['accuracy'])} |", "",
             "Final stratified fold accuracies: " + ", ".join(pct(r["accuracy"]) for r in cv["stratified"]["folds"]) + ".", "",
             "The leave-one-player-out check trains three additional models. Every held-out player's "
             "clips are excluded from that model's fitting. These diagnostic models are not included "
             "in the deployment ensemble. A separate single GRU is fitted on all eligible training "
             "clips for 100 epochs; the primary ensemble was chosen before testing.", "",
             "## Frozen-model diagnostic results", "",
             "Scores below count quality rejection as an incorrect/unavailable result, rather than "
             "silently dropping difficult test clips.", "",
             "| Model | Beginner 3 (85) | Beginner 4 (90) | All testing (175) |",
             "|---|---:|---:|---:|"]
    for mode, title in [("legacy_desktop_ensemble", "Old desktop ensemble, fresh Lite poses"),
                        ("v2_ensemble", "Retrained five-fold ensemble"), ("v2_single", "Retrained single GRU")]:
        lines.append(f"| {title} | " + " | ".join(pct(test[mode][g]["accuracy_all_clips_quality_failures_wrong"]) for g in ["Beginner3", "Beginner4", "all"]) + " |")
    lines += ["", f"The retrained ensemble accepts tracking quality on {score['quality_accepted']}/175 clips. "
              f"At confidence ≥0.60 its coverage is {pct(score['pipeline_coverage'])}, confident-only accuracy "
              f"is {pct(score['pipeline_confident_accuracy']) if score['pipeline_confident_accuracy'] is not None else 'N/A'}, "
              f"and {score['pipeline_confident_wrong']} accepted predictions are wrong. Confidence must not "
              "be interpreted as validated correctness.", "",
              "### Previously reported examples", "",
              "These are training clips, so this table is a regression illustration, not evidence of "
              "generalization to new players.", "",
              "| Clip | Coach B label | Old desktop ensemble | Retrained ensemble |",
              "|---|---|---|---|"]
    for cid in ["Beginner2_Drive_001", "Beginner2_Drive_002"]:
        cells = []
        for mode in ["legacy_desktop_ensemble", "v2_ensemble"]:
            row = next(r for r in predictions if r["clip_id"] == cid and r["model"] == mode)
            cells.append(row["predicted"] + (f" ({pct(float(row['confidence']))})" if row["confidence"] else ""))
        lines.append(f"| {cid} | drive | " + " | ".join(cells) + " |")
    lines += ["", "## Desktop use and deployment status", "",
              "Analysis settings → Serve classifier → **Retrained GRU · research candidate** "
              "uses models/serve_v2_fixed/tflite. The old model options remain available for comparisons. "
              "Refresh/restart Serve Lab if it was already running. The candidate forces moment "
              "matching off, validates the preprocessing contract, and rejects poor tracking explicitly.", "",
              ("The candidate meets the 85% training development-CV threshold, but independent Coach C "
               "validation and physical Android evaluation remain pending." if frozen["cv_target_met"] else
               "**The candidate does not meet the thesis's 85% training CV threshold.** It is available "
               "for inspecting predictions, but its coaching output is withheld. It was not promoted "
               "over the existing baseline or installed as the Android model."), "",
              "Android still has its original model assets. Its asynchronous live stream and motion "
              "gate have not been validated against this complete-recording contract; copying the "
              "new files into Android without implementing and validating that path would recreate "
              "the mismatch. No Android latency or live accuracy improvement is claimed.", "",
              "The retrain changes serve classification, not the repertoire of biomechanical diagnoses. "
              "Wrist-lag and early-wiper coaching rules still need validated measurements; this GRU "
              "does not learn those diagnoses from deviation comments.", "",
              "The next evidence needed is additional independently annotated **training** players "
              "and an assessment of whether these five 2D angles preserve the differences between "
              "serve subtypes. Keep the current testing partition out of fitting. These results "
              "do not justify tuning on it or claiming that another epoch increase will solve the "
              "remaining player/domain gap.", "",
              "## Verification and artifacts", "",
              f"- Exported all five fold models plus the single model. TFLite/Keras parity checked "
              f"on {sum(r['windows'] for r in evaluated['parity'])} real training windows: zero label disagreements, "
              f"maximum probability difference {max(r['max_probability_diff'] for r in evaluated['parity']):.9g}.",
              "- Sequence tests cover late motion, endpoints, frame-rate invariance, angle wraparound, "
              "missing joints, invalid time axes, and preservation of padding/frame-drop masks.",
              "- Desktop tests cover candidate preprocessing parity, isolation from cached legacy models, "
              "contract mismatch rejection, below-target coaching suppression, and existing API/stream behavior.",
              "- A real end-to-end desktop run on Beginner2_Drive_001 completed all core streams: "
              "the candidate classified Drive at 82.5%, resampled all 142 frames without truncation, "
              "and produced the analysis report, playback and contact image. The running desktop "
              "health endpoint confirms that candidate files are available.",
              "- Original checkpoints and cached feature arrays remain intact. New data are under "
              "data/serve_v2; pilot models under models/serve_v2; final candidate under models/serve_v2_fixed.", "",
              "[Final training configuration](../models/serve_v2_fixed/training_config.json) · "
              "[CV results](../models/serve_v2_fixed/cv_results.json) · "
              "[Frozen model hashes](../models/serve_v2_fixed/frozen_models.json) · "
              "[Diagnostic evaluation](../models/serve_v2_fixed/evaluation.json) · "
              "[Per-clip predictions](../models/serve_v2_fixed/predictions.csv)", "",
              "```powershell", "python scripts/prepare_serve_v2.py --workers 3",
              "python scripts/retrain_serve_v2.py",
              "python scripts/retrain_serve_v2.py --out-dir models/serve_v2_fixed --schedule fixed100",
              "python scripts/evaluate_serve_v2.py --src models/serve_v2_fixed",
              "python scripts/report_serve_v2.py", "python scripts/test_serve_sequence.py",
              "python -m unittest desktop.test_desktop", "```", "",
              "The dataset/contract is frozen: source or configuration changes require a new version. "
              "Resume checks reject stale pose provenance or a different training configuration.", "",
              "![Retraining comparison](../outputs/serve_v2_results.png)", "",
              "Implementation references: [TensorFlow determinism](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism), "
              "[MediaPipe PoseLandmarker options](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarkerOptions).", ""]
    path = ROOT / "analysis/GRU_RETRAINING_V2.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2, figsize=(11,4.8))
    cv_values = [cv["stratified"]["metric"]["accuracy"],cv["participant"]["metric"]["accuracy"]]
    test_values = [old["accuracy"],score["accuracy_all_clips_quality_failures_wrong"]]
    for ax, values, labels, title in [(axes[0],cv_values,["Stratified\ndevelopment CV","Held-out\ntraining player"],f"Training partition only · {trained} clips"),
                                     (axes[1],test_values,["Old desktop\nensemble","Retrained\nensemble"],"Frozen evaluation · 175 testing clips")]:
        bars=ax.bar(labels,100*np.array(values),color=["#667a8a","#127f78"],width=.6)
        ax.set(ylim=(0,105),ylabel="Accuracy (%)",title=title)
        ax.bar_label(bars,labels=[pct(x) for x in values],padding=4)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].axhline(85,color="#ad5348",linestyle="--",label="Thesis CV target")
    axes[0].legend(loc="upper left",frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/serve_v2_results.png",dpi=180)
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
