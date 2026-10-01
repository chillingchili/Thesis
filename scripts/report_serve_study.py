"""Render completed study artifacts into a reproducible research report."""
import json
from pathlib import Path
import numpy as np
from serve_study import ROOT, DATA, OUT, PROTOCOL, write_json

NAMES = {"angles_control":"Five angles, no augmentation", "angles_thesis_aug":"Five angles, thesis video augmentation",
         "skeleton":"33-joint coordinates + visibility", "skeleton_motion":"33-joint coordinates + motion + visibility"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cv={c:json.loads((OUT/c/"cv_results.json").read_text()) for c in PROTOCOL["conditions"]}
    frozen=json.loads((OUT/"frozen_selection.json").read_text())
    evaluation=json.loads((OUT/"diagnostic_evaluation.json").read_text())
    extraction=json.loads((DATA/"extraction_complete.json").read_text())
    hybrid=json.loads((ROOT/"outputs/serve_study_v3/hybrid/results.json").read_text())
    coach=json.loads((ROOT/"outputs/serve_study_v3/coach_review/status.json").read_text())
    effects=json.loads((ROOT/"outputs/serve_study_v3/augmentation_effects.json").read_text())
    out=ROOT/"outputs/serve_study_v3"
    out.mkdir(exist_ok=True,parents=True)
    rows=[]
    for c in PROTOCOL["conditions"]:
        rows.append(dict(condition=c,development_cv=cv[c]["stratified"]["gru"]["accuracy"],
                         held_out_player=cv[c]["participant"]["gru"]["accuracy"],
                         mean_subject_accuracy=cv[c]["participant"]["gru"]["mean_subject_accuracy"],
                         diagnostic=evaluation["conditions"][c]["gru"]["accuracy"]))
    fig,axes=plt.subplots(1,3,figsize=(13,4.7),sharey=True)
    colors=["#67758b","#bf6b21","#387c7d","#85639c"]
    for ax,key,title in zip(axes,["development_cv","held_out_player","diagnostic"],
                           ["Same-player development CV\n209 clips / 5 folds","Held-out-player development CV\n209 clips / 3 players","Frozen-model diagnostic test\n175 clips / 2 other players"]):
        values=[r[key]*100 for r in rows]
        bars=ax.bar(np.arange(4),values,color=colors,width=.66)
        ax.bar_label(bars,labels=[f"{v:.1f}%" for v in values],padding=4,fontsize=10)
        ax.set_title(title,fontsize=11,pad=12)
        ax.set_xticks(np.arange(4),["Angles","Video aug.","Skeleton","+ Motion"],rotation=20,ha="right")
        ax.set_ylim(0,105);ax.set_axisbelow(True);ax.grid(axis="y",alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Serve classification: matched training budget, different inputs",fontsize=15,y=1.01)
    fig.text(.01,-.035,"One seed; only three training players. Diagnostic clips were previously inspected. Quality failures count as unavailable/incorrect.",fontsize=9)
    fig.tight_layout()
    for suffix in ("png","svg"):
        fig.savefig(out/f"comparison.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    pct=lambda v:f"{100*v:.1f}%"
    table="\n".join(f"| {NAMES[r['condition']]} | {pct(r['development_cv'])} | {pct(r['held_out_player'])} | {pct(r['diagnostic'])} |" for r in rows)
    participants="\n".join(f"| {NAMES[c]} | "+" | ".join(pct(cv[c]["participant"]["gru"]["per_subject"][s]["accuracy"]) for s in ["Beginner1","Beginner2","CoachA"])+" |" for c in PROTOCOL["conditions"])
    hybridoof="\n".join(f"| {NAMES[c]} | {pct(cv[c]['participant']['gru']['accuracy'])} | {pct(cv[c]['participant']['knn']['accuracy'])} | {pct(cv[c]['participant']['hybrid']['accuracy'])} |" for c in PROTOCOL["conditions"])
    deployed="\n".join(f"| {w} | {r['hybrid']['available']}/175 | {pct(r['gru']['accuracy'])} | {pct(r['knn']['accuracy'])} | {pct(r['hybrid']['accuracy'])} |" for w,r in hybrid.items())
    diagnostic_hybrid="\n".join(f"| {NAMES[c]} | {pct(evaluation['conditions'][c]['gru']['accuracy'])} | {pct(evaluation['knn']['accuracy'])} | {pct(evaluation['conditions'][c]['hybrid']['accuracy'])} |" for c in PROTOCOL["conditions"])
    parameters={c:json.loads((OUT/c/"participant/fold1/result.json").read_text())["parameters"]
                for c in PROTOCOL["conditions"]}
    # Paired counts are descriptive; clips within a player are not independent.
    paired={}
    for protocol in ["participant","stratified"]:
        a=np.load(OUT/"angles_control"/f"{protocol}_oof.npz")
        b=np.load(OUT/"angles_thesis_aug"/f"{protocol}_oof.npz")
        assert a["clip_ids"].tolist()==b["clip_ids"].tolist()
        ac=a["probabilities"].argmax(1)==a["y"];bc=b["probabilities"].argmax(1)==b["y"]
        paired[protocol]=dict(both_correct=int((ac&bc).sum()),augmentation_only_correct=int((~ac&bc).sum()),
                              control_only_correct=int((ac&~bc).sum()),both_wrong=int((~ac&~bc).sum()))
    write_json(out/"summary.json",dict(rows=rows,selected_condition=frozen["selected_condition"],paired_augmentation=paired,
               coach_review_pending=True,android_assets_replaced=False))
    text=f"""# Four follow-up investigations — 2026-10-01

Completed model experiments and a local Android-classifier benchmark. The coach
review packet is prepared; human judgments remain pending. No source label,
original model, or Android model asset was replaced.

Thesis video augmentation improved held-out-player GRU accuracy from
**{pct(cv['angles_control']['participant']['gru']['accuracy'])} to {pct(cv['angles_thesis_aug']['participant']['gru']['accuracy'])}**
({100*(cv['angles_thesis_aug']['participant']['gru']['accuracy']-cv['angles_control']['participant']['gru']['accuracy']):+.1f} percentage points)
against the matched control. Diagnostic accuracy changed from
**{pct(evaluation['conditions']['angles_control']['gru']['accuracy'])} to {pct(evaluation['conditions']['angles_thesis_aug']['gru']['accuracy'])}**
({100*(evaluation['conditions']['angles_thesis_aug']['gru']['accuracy']-evaluation['conditions']['angles_control']['gru']['accuracy']):+.1f} points).

Skeleton and skeleton-motion inputs reached
**{pct(evaluation['conditions']['skeleton']['gru']['accuracy'])} and {pct(evaluation['conditions']['skeleton_motion']['gru']['accuracy'])}**
on the diagnostic clips, while their held-out-player development scores were
lower than the angle control. Model ranking therefore varies across these
players and evaluation procedures. The fixed research hybrids scored lower
than their corresponding GRUs on the diagnostic set. These one-seed results
show improvements in specific comparisons, without establishing a consistently
reliable model or a single cause for the remaining errors.

## 1. Thesis video augmentation

The pipeline now applies brightness/contrast, independent per-pixel Gaussian
noise (3–5% of the 0–255 intensity range), slight isotropic crop/translation, and
2–10% temporal subsampling **before** fresh BlazePose Lite VIDEO extraction.
No flips, shears, or nonuniform stretching are used. Parameters are fixed within
each clip; Gaussian noise varies by frame. Original raw videos are untouched.

Two deterministic composite variants were extracted for each of the 209
eligible original training clips: **{extraction['variants']} variants**, of which
**{extraction['accepted']} passed** the existing tracking gate. Rejected variants
fall back to their original clip during sampling. This produces additional
training views, not additional independent participants. The original 210-clip
inventory retains the one pre-existing tracking rejection.

Visual spot checks were rendered for each training player. Across all variants,
the median of per-clip mean angular changes was
{effects['median_mean_angle_change_degrees']:.2f} degrees on jointly valid frames;
{effects['identical_to_original']} variants were identical to their original
feature window. These changes include tracker sensitivity and temporal
resampling, and are not new independent serve mechanics.
[Example original/augmented frames and skeletons](../outputs/serve_study_v3/previews/Beginner2_Drive_001.jpg).

Each condition sees exactly one example per original training clip per epoch,
for 100 epochs, batch size 16, GRU 32, dropout 0.15, L2 0.0001 and Adam 0.001.
The augmented condition samples original/variant 1/variant 2 uniformly; other
conditions see the original. Batch order and model seed match where dimensions
permit. This controls optimizer-update count. No early stopping or validation
feedback is used during these fits. No feature-level augmentation is mixed into
the new ablation, isolating the new video-augmentation treatment.

Original IDs are split first. All variants inherit their original's membership.
Validation sees unaugmented originals only. The primary development comparison
holds out one entire training player; the secondary five-fold split uses the
same original memberships as v2. Every fold saves IDs, hashes and histories.

## 2. Richer input comparison

| Input condition | Stratified development CV | Held-out-player CV | Diagnostic test |
|---|---:|---:|---:|
{table}

The skeleton representation uses 33 image-space joints, with x corrected for
image aspect ratio, per-frame hip centering, median visible torso-length scaling,
and a visibility mask per joint (99 channels). The motion condition appends 66
adjacent-step displacements (165 channels). It preserves all 128 full-serve
timeline samples. Missing joints are zeroed; neither interpolation nor motion
differences bridge missing observations. The model does not receive player IDs.
These displacements describe movement per normalized serve step, not measured
physical velocity. Richer inputs also increase parameter count, so this compares
input systems rather than proving an information-only causal effect. Aspect-ratio
correction also differs from the legacy normalized-image angle representation.
The five-angle GRUs have {parameters['angles_control']:,} parameters; the skeleton
and skeleton-motion GRUs have {parameters['skeleton']:,} and
{parameters['skeleton_motion']:,}, respectively.

| Input condition | Beginner 1 held out | Beginner 2 held out | Coach A held out |
|---|---:|---:|---:|
{participants}

Training-only selection chose **{NAMES[frozen['selected_condition']]}**, using the
predeclared mean accuracy across held-out players. The choice and all 32 model
hashes were frozen before opening the diagnostic test arrays. Each diagnostic
GRU result averages that condition's five stratified-fold models, all trained
on training clips only. The existing 175 Beginner 3/4 clips were previously
inspected and are not an untouched final evaluation. One diagnostic quality
failure remains in the denominator as unavailable/incorrect.

![Accuracy comparisons](../outputs/serve_study_v3/comparison.png)

## 3. GRU / kNN / hybrid

The deployed bank contains 438 older training examples. Its contents were
verified against the saved training-feature source and checked for test-ID
overlap. The local benchmark invokes the actual five Android TFLite files and
bundled kNN bank. GRU gets moment-matched features; kNN gets unmatched timing
features; fusion remains the app's fixed 50/50 average.

| Bundled-model input window | Available hybrid results | GRU | kNN5 | Hybrid |
|---|---:|---:|---:|---:|
{deployed}

The primary window is the first 128 valid frames of each recorded clip, matching
the desktop comparison. The last-128 window probes the phone's rolling buffer.
Gate replay ports the current cold-start MainActivity gate, at source-video
cadence with every available pose, and selects the last completed event. It does
not force an event to finish at EOF. Short/no completed events count as
unavailable. Gate replay is **not physical Android accuracy**: LIVE_STREAM frame
drops, callback timing, continuous tracker state and live warm-up are absent.
Its result demonstrates sensitivity to segmentation assumptions, not a measured
failure rate for the connected phone. No device was connected.

The existing Kotlin timing/kNN fixture tests, Android build and lint checks pass.
The desktop first-window benchmark is independently reproduced with TFLite.
The earlier 56.1% kNN figure used a different cached-feature evaluation and
per-player averaging; it is not a measured score for the phone's fixed hybrid.

For a fair new-training comparison, kNN was refitted inside each same training
fold using the current full-serve 32 timing features, with no scaling or weight
tuning. It never uses the held-out player's clips. These are new research hybrids,
not the existing 438-example Android bank:

| Input condition | Held-out-player GRU | Fold-fitted kNN | Fixed hybrid |
|---|---:|---:|---:|
{hybridoof}

| Input condition | Diagnostic GRU | 209-example kNN | Fixed hybrid |
|---|---:|---:|---:|
{diagnostic_hybrid}

Hybrid weights and confidence threshold (0.60 for reporting) were not selected
using diagnostic labels. Confidence coverage, wrong confident predictions,
macro-F1, per-player results and confusion matrices are saved with every result.

## 4. Coach and pose audit

Audited all {coach['training_clips']} training annotations against the frozen
inventory: **{coach['label_file_disagreements']} label-file disagreements** and
**{coach['quality_rejected']} tracking-quality rejection**. Agreement between two
files does not establish that the visible executed serve supports the label.

Prepared **{coach['review_cases']} blinded review cases**, balancing player/class
cells with two confident participant-held-out errors and one seeded comparison
per cell, then adding the quality-rejected clip and named regression examples.
Only training clips are included. The coach first judges the renamed raw videos,
then reviews six raw/skeleton frame pairs per clip. The response sheet allows
ambiguous/unjudgeable labels and specific wrist/hand, occlusion and identity
errors. The researcher key, original labels, model predictions and previous
coach comments are outside the shared ZIP.

**Human review remains pending; no label was corrected automatically.** Blank
responses produce 30 pending judgments, not a fabricated agreement score.
The scoring script validates IDs, reviewer/date and response values and writes
agreement/disagreement findings separately from the original dataset. Because
the packet deliberately samples difficult cases, its agreement rate cannot be
reported as dataset-wide or final-evaluation accuracy.

[Coach packet](../outputs/serve_study_v3/coach_review/coach_review_packet.zip) ·
[Response sheet](../outputs/serve_study_v3/coach_review/packet/responses.csv) ·
[Researcher inventory](../outputs/serve_study_v3/coach_review/training_audit.csv)

## Interpretation limits

- These are development experiments with one seed and only three training
  participants. They do not establish a hard accuracy ceiling or prove that data
  quantity, GRU architecture, or a specific feature is the sole bottleneck.
- Player and recording conditions are partly confounded: the thesis records
  Coach A at a different venue from the beginners. A held-out-player gap cannot
  be attributed exclusively to personal serving style.
- The matched control is a new fit with a controlled sampler, not a rerun of the
  older 438-example GRU. Compare augmentation against this matched control;
  compare deployed assets separately.
- Uniform full-clip resampling can remove much of a global playback-rate change.
  Temporal subsampling here primarily changes which frames reach the tracker
  and interpolation. Its benefit is not isolated from the other augmentations.
- No paddle/ball classifier branch was trained: the richer-input experiment
  measures skeletal information already available with reliable provenance.
- Coach C's independent evaluation and a physical-phone replay/live check remain
  unavailable. No candidate was promoted into the desktop default or Android.

## Reproduce and inspect

```powershell
python scripts/test_serve_study.py
python scripts/test_serve_study_integrity.py
python scripts/prepare_serve_study.py --workers 6
python scripts/train_serve_study.py
python scripts/evaluate_serve_study.py
python scripts/benchmark_serve_hybrid.py
python scripts/prepare_coach_review.py
python scripts/score_coach_review.py outputs/serve_study_v3/coach_review/packet/responses.csv
python scripts/report_serve_study.py
```

[Frozen protocol](../data/serve_study_v3/protocol.json) ·
[Frozen selection and model hashes](../models/serve_study_v3/frozen_selection.json) ·
[Diagnostic evaluation](../models/serve_study_v3/diagnostic_evaluation.json) ·
[Android asset benchmark](../outputs/serve_study_v3/hybrid/results.json) ·
[Per-clip diagnostic predictions](../models/serve_study_v3/diagnostic_predictions.csv)

Implementation references: [MediaPipe VIDEO pose API](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
and [scikit-learn split/preprocessing isolation](https://scikit-learn.org/stable/modules/cross_validation.html).
"""
    (ROOT/"analysis/FOUR_FOLLOWUP_STUDY.md").write_text(text,encoding="utf-8")
    print(table)
    print("Report written: analysis/FOUR_FOLLOWUP_STUDY.md")


if __name__=="__main__":
    main()
