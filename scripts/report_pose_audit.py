"""Build the audit report and figures from saved, reproducible evaluation outputs."""
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from audit_pose_classifier import OUT, ROOT, CLASSES, inventory, metrics
from coach_annotations import read_annotations


def read_csv(name):
    with (OUT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def percent(x):
    return f"{100*x:.1f}%"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cached = json.loads((OUT / "cached_metrics.json").read_text())
    fresh = json.loads((OUT / "fresh_metrics.json").read_text())
    comparisons = json.loads((OUT / "fresh_comparisons.json").read_text())
    inv = json.loads((OUT / "inventory.json").read_text())
    order = json.loads((OUT / "order_check.json").read_text())
    quality = read_csv("cached_pose_quality.csv")
    fresh_quality = read_csv("fresh_pose_quality.csv")
    predictions = read_csv("cached_predictions.csv")
    fresh_predictions = read_csv("fresh_predictions.csv")
    annotations = read_annotations()
    available, _ = inventory()
    ids = {r["clip_id"] for r in available}
    annotated = {Path(r["Filename"]).stem: r for r in annotations}
    annot_missing = sorted(set(annotated) - ids)
    annotated_usable = [r for r in predictions if r["mode"] == "saved_training_features" and r["clip_id"] in annotated]
    label_conflicts = [r["clip_id"] for r in annotated_usable if r["truth"] != annotated[r["clip_id"]]["Serve Type"].lower()]
    original = cached["saved_training_features"]
    lines = ["# BlazePose and serve-classifier audit — 2026-10-01", "",
             "The inconsistency is confirmed. The saved Heavy-pose/GRU pipeline scores "
             f"{percent(original['train']['accuracy'])} on its training clips, "
             f"{percent(cached['reconstructed_oof']['train']['accuracy'])} in reconstructed stratified cross-validation, "
             f"and {percent(original['test_combined']['accuracy'])} on the 175 existing test clips. "
             "These are **serve classification** scores, not BlazePose joint-location accuracy. "
             "BlazePose is the pretrained landmark extractor; the GRU is the classifier trained "
             "on this project's sequences.", "",
             "No model was trained, no checkpoint or threshold was chosen using these results, "
             "and the desktop/Android model settings were left unchanged. Only evaluation reporting "
             "was corrected. All results below use the existing five-model ensemble unless marked OOF.", "",
             "## Scope and protocol", "",
             "- Re-evaluated all 438 available training sequences and 175 test sequences (Beginner 3: 85; Beginner 4: 90).",
             "- Re-extracted both legacy BlazePose Heavy and Tasks Lite on 33 videos: two randomly selected clips "
             "per player × class using seed 42 (30 total), plus Beginner2_Drive_001, _002 and _007. "
             "Fresh sample results are diagnostic; they are not full-dataset desktop accuracy estimates.",
             "- Used manifest serve-type labels. The 205 available Coach B annotated clips have "
             f"{len(label_conflicts)} label conflicts with their manifest labels. There is no manual "
             "landmark ground truth or identified Coach C 30-clip annotation set in this audit.",
             "- OOF is reconstructed from the current manifest order, StratifiedKFold(5, shuffle=True, "
             "random_state=42), and saved fold checkpoints. Original fold IDs were not saved; the "
             "reconstruction reproduces the previously reported fold scores but is conditional on that order/seed. "
             "Validation also guided early stopping. Players occur on both sides of stratified folds, "
             "so this does not measure generalization to a new player.",
             "- Four of five models have seen each training clip. Ensemble training accuracy is a "
             "resubstitution diagnostic, not an independent evaluation.",
             "- The existing docs/RESULTS_SUMMARY.md and analysis scripts already examined Beginners 3/4 "
             "and compared adaptation methods on them. Treat these as previously inspected test sets, "
             "not a pristine final holdout.", "",
             "## Full-dataset results", "",
             "| Input pipeline using cached Heavy poses | Training (438) | Beginner 3 (85) | Beginner 4 (90) | Combined test (175) |",
             "|---|---:|---:|---:|---:|"]
    mode_names = {"saved_training_features": "Original: contact + 10 trim, normalized angles, no matching",
                  "untrimmed_no_matching": "Untrimmed first 128, no matching",
                  "untrimmed_matching": "Desktop preprocessing: untrimmed first 128 + moment matching",
                  "trimmed_matching": "Trimmed first 128 + moment matching"}
    for mode, name in mode_names.items():
        lines.append("| " + name + " | " + " | ".join(percent(cached[mode][g]["accuracy"]) for g in ["train", "test_beg3", "test_beg4", "test_combined"]) + " |")
    lines += ["", "The desktop-preprocessing rows still use **cached Heavy landmarks** to isolate preprocessing. "
              "The actual desktop uses Lite; its fresh-sample results appear below. Moment matching "
              "rescales each channel to training mean/std at inference, although the GRU was trained "
              "on unmodified sin/cos features. It changes the signal the model sees. Its mixed effects "
              "do not establish it as a validated fix.", "",
              "OOF fold accuracies: " + ", ".join(percent(m["accuracy"]) for m in cached["oof_folds"]) + ".", "",
              "### Original test confusion matrix", "",
              "Rows are manifest labels; columns are predicted classes.", "",
              "| Actual / predicted | Drive | Lob | Topspin |", "|---|---:|---:|---:|"]
    for name, row in zip(CLASSES, original["test_combined"]["confusion"]):
        lines.append(f"| {name} | " + " | ".join(map(str, row)) + " |")
    test = original["test_combined"]
    lines += ["", f"At the existing 0.60 confidence threshold, {test['accepted']}/175 predictions are accepted "
              f"({percent(test['coverage'])} coverage). **{test['accepted_wrong']} of those accepted predictions are wrong**; "
              f"accepted-only accuracy is {percent(test['accepted_accuracy'])}. Accuracy with abstentions counted "
              f"as wrong is {percent(test['accuracy_abstentions_wrong'])}. Mean softmax confidence is "
              f"{percent(test['mean_confidence'])}, despite only {percent(test['accuracy'])} accuracy. "
              "A high confidence number is therefore not a calibrated probability of correctness.", "",
              "## Pose quality and temporal coverage", "",
              "| Cached poses | Frames | Frames with pose | Estimated contact outside first 128 |",
              "|---|---:|---:|---:|"]
    for split in ["train", "test_beg3", "test_beg4"]:
        group = [r for r in quality if r["split"] == split]
        frames = sum(int(r["frames"]) for r in group)
        detected = sum(int(r["detected_frames"]) for r in group)
        outside = sum(r["contact_outside_window"] == "True" for r in group)
        lines.append(f"| {split} | {frames:,} | {detected:,} ({100*detected/frames:.3f}%) | {outside}/{len(group)} |")
    lines += ["", "All saved trims exactly reproduce the current contact heuristic, and every saved angle array "
              f"exactly reproduces angles regenerated from normalized trimmed landmarks (max difference {inv['angle_maxdiff']}). "
              f"No duplicate raw-pose file hashes were found among the {inv['usable']} available clips. "
              "This is a cache-consistency check, not proof of independence between adjacent recorded serves.", "",
              "The contact estimate is a peak wrist-speed heuristic, not an annotated paddle/ball impact. "
              "For 159 training clips the estimated impact is entirely outside the classifier's first "
              "128 frames. Merely trimming after impact does not bring a late impact into that window.", "",
              "Very high detection coverage does not validate joint locations, wrist tracking, player "
              "identity, or biomechanics. The original extractor stores x/y only and discards visibility. "
              "The fresh audit also records visibility, but model visibility is not human ground truth.", "",
              "## Fresh Heavy vs Lite comparison", "",
              "| Pose extractor / preprocessing | Training sample (21) | Test sample (12) | All sampled (33) |",
              "|---|---:|---:|---:|"]
    for variant in ["cached_heavy", "heavy", "lite"]:
        for mode in ["training_preprocessing", "desktop_default"]:
            group = fresh[f"{variant}/{mode}"]
            lines.append(f"| {variant} / {mode} | " + " | ".join(f"{group[k]['correct']}/{group[k]['n']} ({percent(group[k]['accuracy'])})" for k in ["train", "test_combined", "all"]) + " |")
    lines += ["", "Fresh Heavy uses the training script's model/thresholds but resets the tracker between clips. "
              "Lite uses the desktop's model/thresholds. Differences therefore include the model "
              "variant, graph/API, and tracking settings; this is not a controlled benchmark of network "
              "weights alone. Each variant gets the same decoded full video and GRU.", ""]
    paired = {(r["clip_id"], r["variant"], r["mode"]): r for r in fresh_predictions}
    cids = sorted({r["clip_id"] for r in fresh_predictions})
    disagreement_counts = {}
    for mode in ["training_preprocessing", "desktop_default"]:
        flips = sum(paired[(cid, "heavy", mode)]["predicted"] != paired[(cid, "lite", mode)]["predicted"] for cid in cids)
        disagreement_counts[mode] = flips
        lines.append(f"- Heavy/Lite predicted-label disagreements with {mode}: **{flips}/{len(cids)}**.")
    angular = [r["median_angle_disagreement_deg"] for r in comparisons if r.get("pair") == "heavy_vs_lite"]
    medians = np.median(angular, axis=0)
    lines += ["", "Median across clips of per-clip median angle disagreement (degrees): " +
              ", ".join(f"{name} {value:.2f}" for name, value in zip(["elbow", "shoulder", "wrist", "torso", "twist"], medians)) +
              ". Disagreement does not identify which extractor is more accurate.", "",
              "### Repeated-run and clip-order checks", ""]
    for r in comparisons:
        if "max_coordinate_diff" in r:
            lines.append(f"- {r['clip_id']} / {r['variant']}: maximum repeated-run coordinate change "
                         f"{r['max_coordinate_diff']:.9g}; identical missing-frame masks: {r['same_missing_mask']}.")
    lines += ["", "The original extractor creates one Heavy tracker outside its video loop and does not reset "
              "it across videos. In the controlled Drive 2 check, processing Drive 1 first changes "
              f"Drive 2's coordinates by up to {order['max_coordinate_diff_carried']:.6f} normalized units. "
              f"Resetting the tracker brings that maximum difference to {order['max_coordinate_diff_after_reset']:.9g}. "
              "This establishes an input-order dependency; cached arrays can depend on extraction order "
              "and which earlier files were skipped. The predicted labels did not change in this "
              "controlled order check. It does not establish that this alone causes all errors.", "",
              "## Beginner 2 Drive 1 and 2", "",
              "Both are labeled Drive by Coach B and by the manifest. The table below uses freshly "
              "extracted landmarks; percentages are model confidence, not correctness.", "",
              "| Clip | Heavy + training preprocessing | Lite + training preprocessing | Lite + desktop defaults |",
              "|---|---|---|---|"]
    for cid in ["Beginner2_Drive_001", "Beginner2_Drive_002"]:
        cells = []
        for variant, mode in [("heavy", "training_preprocessing"), ("lite", "training_preprocessing"), ("lite", "desktop_default")]:
            r = paired[(cid, variant, mode)]
            cells.append(f"{r['predicted']} ({percent(float(r['confidence']))})")
        lines.append(f"| {cid} | " + " | ".join(cells) + " |")
    lines += ["", "The current serve classifier does not predict Coach B's wrist-lag or early-wiper "
              "annotations. Those detailed feedback disagreements cannot be resolved just by raising "
              "the serve-classification confidence threshold.", "",
              "## Conversion, dataset, and reporting checks", "",
              f"- TFLite vs Keras: 132 real windows × 5 folds = 660 comparisons; zero label disagreements, "
              f"maximum probability difference {max(r['max_probability_diff'] for r in cached['tflite_parity']):.9g}. "
              "Deployed model assets are byte-identical to exported files. These checks were on this "
              "Windows CPU runtime, not a physical Android device.",
              "- Nine training-manifest rows have no usable cached arrays; their IDs are in inventory.json.",
              f"- Coach B workbook has 210 rows, but only {len(annotated_usable)} have available model features. "
              f"Unavailable IDs: {', '.join(annot_missing)}. The existing model was evaluated/trained "
              "on 438 available training clips, which does not match the thesis's 210 annotated training "
              "and independent 30-clip evaluation protocol. A subset score cannot retroactively make "
              "this a model trained under that protocol.",
              "- Fixed scripts/testrun.py: its report previously said abstentions were counted wrong while "
              "actually reporting unconditional argmax accuracy. It now prints both explicitly, "
              "reports coverage and confident errors, handles missing classes in the confident subset, "
              "and defaults to its documented 0.60 threshold instead of 0.40. The old behavior can "
              "still be requested explicitly with --threshold 0.40. Two regression checks pass.", "",
              "## What should change next", "",
              "1. Freeze a versioned split manifest that matches the thesis, including Coach B labels "
              "and the independently annotated Coach C evaluation set. Recover or document missing "
              "clips; use participant-held-out development checks as well as the required stratified CV. "
              "Obtain a new untouched final set if inspected test results guide improvements.",
              "2. Use one explicit extraction and preprocessing contract for training, desktop, and "
              "Android: same pose variant, tracker reset per recording, visibility handling, frame "
              "sampling rate, and serve window. Retrain and validate under that contract. Do not "
              "assume swapping Lite for Heavy fixes the classifier without validation.",
              "3. Validate contact detection and place the whole serve into a consistent time window. "
              "The phone additionally uses asynchronous live frames and a rolling buffer of valid "
              "poses, unlike the offline full-frame sequences. Android live accuracy remains unmeasured.",
              "4. Annotate representative shoulder/elbow/wrist/index/hip locations, especially near "
              "contact and during occlusion, to measure dataset-specific landmark error/PCK. Review "
              "the contact overlays saved by this audit before attributing all error to BlazePose.",
              "5. Validate confidence/abstention on development data and gate specific coaching rules "
              "on validated measurements. Accurate serve type alone cannot validate wrist-lag or "
              "swing-path diagnoses.", "",
              "## Reproduce and inspect", "",
              "```powershell", "python scripts/audit_pose_classifier.py cached",
              "python scripts/audit_pose_classifier.py fresh --per-stratum 2",
              "python scripts/audit_pose_classifier.py order", "python scripts/report_pose_audit.py",
              "python scripts/test_testrun_reporting.py", "```", "",
              "Fresh extraction resumes files under outputs/pose_audit/fresh. Remove or use a separate "
              "output directory before evaluating changed weights, videos, or extraction settings; "
              "the saved audit is a snapshot of the current artifacts.", "",
              "Machine-readable evidence: [cached metrics](../outputs/pose_audit/cached_metrics.json), "
              "[per-clip predictions](../outputs/pose_audit/cached_predictions.csv), "
              "[fresh metrics](../outputs/pose_audit/fresh_metrics.json), "
              "[fresh predictions](../outputs/pose_audit/fresh_predictions.csv), "
              "[clip-order check](../outputs/pose_audit/order_check.json), "
              "[provenance](../outputs/pose_audit/provenance.json).", "",
              "![Accuracy and confusion matrices](../outputs/pose_audit/accuracy.png)", "",
              "![Pose overlays at cached estimated contact](../outputs/pose_audit/drive_examples.png)", "",
              "MediaPipe reference: [Pose solution and model complexity](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md), "
              "[Pose Landmarker overview](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker). "
              "Published benchmark accuracy is not measured accuracy on these pickleball videos.", ""]
    (ROOT / "analysis/POSE_CLASSIFIER_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, group, title in zip(axes, [original["train"], cached["reconstructed_oof"]["train"], original["test_combined"]],
                                 ["Training ensemble (seen clips)", "Reconstructed stratified OOF", "Existing test: new players"]):
        cm = np.array(group["confusion"])
        ax.imshow(cm / cm.sum(1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="white" if cm[i,j]/cm[i].sum() > .55 else "black", fontsize=15)
        ax.set(xticks=range(3), xticklabels=CLASSES, yticks=range(3), yticklabels=CLASSES,
               xlabel="Predicted", ylabel="Actual", title=f"{title}\n{percent(group['accuracy'])} · n={group['n']}")
    fig.suptitle("Same saved Heavy-pose features and GRU ensemble; OOF uses one fold per clip", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy.png", dpi=180)
    plt.close(fig)
    montage()
    tracked = [ROOT / "scripts/audit_pose_classifier.py", ROOT / "scripts/extract_keypoints.py", ROOT / "desktop/pipeline.py",
               ROOT / "android/app/src/main/assets/manifest.json", ROOT / "android/app/src/main/assets/pose_landmarker_lite.task"]
    tracked += list((ROOT / "models/gru_runs_angles_strat5").glob("*.keras"))
    tracked += [ROOT / base / angles / "manifest.csv" for _, base, _, angles, _ in __import__("audit_pose_classifier").SPECS]
    import cv2
    packages = {p: version(p) for p in ["tensorflow", "mediapipe", "numpy", "scikit-learn"]}
    packages["opencv"] = cv2.__version__
    provenance = dict(date="2026-10-01", packages=packages,
                      sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
                      heavy_lite_label_disagreements=disagreement_counts, annotation_label_conflicts=label_conflicts)
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Wrote analysis/POSE_CLASSIFIER_AUDIT.md and outputs/pose_audit figures/provenance")


def montage():
    import cv2
    from trim_after_contact import estimate_contact_frame
    videos = {p.stem: p for p in (ROOT / "data/training/raw").rglob("*.mp4")}
    strips = []
    for cid in ["Beginner2_Drive_001", "Beginner2_Drive_002"]:
        cached = np.load(ROOT / "data/training/keypoints" / (cid + ".npy"))
        contact = estimate_contact_frame(cached)
        cap = cv2.VideoCapture(str(videos[cid]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, contact)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot inspect contact frame for {cid}")
        images = []
        for name, points in [("Cached Heavy", cached), ("Fresh Heavy", np.load(OUT / "fresh/heavy" / (cid + ".npz"))["points"]),
                             ("Fresh Lite", np.load(OUT / "fresh/lite" / (cid + ".npz"))["points"])]:
            image = cv2.resize(frame, (640, 360))
            coords = points[contact] * [640, 360]
            for a, b in [(11,12), (11,23), (12,24), (23,24), (12,14), (14,16), (16,20), (11,13), (13,15), (23,25), (25,27), (24,26), (26,28)]:
                if np.isfinite(coords[[a,b]]).all():
                    cv2.line(image, tuple(coords[a].astype(int)), tuple(coords[b].astype(int)), (0,255,255), 2)
            for index in [12, 14, 16, 20]:
                if np.isfinite(coords[index]).all():
                    cv2.circle(image, tuple(coords[index].astype(int)), 4, (20,30,255), -1)
            cv2.rectangle(image, (0,0), (640,45), (20,20,20), -1)
            cv2.putText(image, f"{cid} | {name} | frame {contact}", (8,28), cv2.FONT_HERSHEY_SIMPLEX, .52, (255,255,255), 1, cv2.LINE_AA)
            images.append(image)
        strips.append(np.concatenate(images, axis=1))
    cv2.imwrite(str(OUT / "drive_examples.png"), np.concatenate(strips))


if __name__ == "__main__":
    main()
