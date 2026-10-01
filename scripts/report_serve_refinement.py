"""Publish complete v4 outputs with per-class results and pipeline limitations."""
import json
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from serve_study import ROOT, write_json
from serve_refinement import DEST, OUTPUT, CONDITIONS

NAMES=dict(angles_thesis_aug='Angles + augmentation',lean_control='Lean arm/torso, control',
    lean_thesis_aug='Lean arm/torso + augmentation',skeleton_thesis_aug='Full skeleton + augmentation',
    skeleton_motion_thesis_aug='Full skeleton/motion + augmentation')


def pc(value): return '—' if value is None else f'{value*100:.1f}%'


def main():
    frozen=json.loads((DEST/'frozen_selection.json').read_text())
    diagnostic=json.loads((OUTPUT/'diagnostic_results.json').read_text())
    cv={c:json.loads((DEST/c/'cv_results.json').read_text()) for c in CONDITIONS}
    selected=frozen['selected_condition'];selected_policy=frozen['policies'][selected]
    lines=['# Capture, augmented geometry and calibrated fusion — v4','',
        'Completed automated work requested in items 1, 3 and 5. Coach review and new-player collection remain deferred. '
        'All studies use existing recordings. No coach judgments or new independent evaluation are claimed.','',
        f'Training-only selection chose **{NAMES[selected]}**, based on mean macro-F1 across held-out players for the nested tuned hybrid. '
        'The selected research model is available in Android recorded replay; the existing live GRU/kNN models remain the default.','',
        '## 1. Match capture and training','',
        'Android now includes **Replay Serve → Choose serve video → Save comparison**. It processes one trimmed complete serve '
        '(maximum 12 seconds, Android 9+) with a fresh MediaPipe Lite VIDEO tracker. It decodes indexed video frames and pairs '
        'them with sorted presentation timestamps from the video track. Unsupported duplicate timestamps or disagreement '
        'between track sample count and decoded frame count are rejected rather than guessed. Missing detections retain their '
        'timeline slots. Required-joint visibility is checked at 0.5, with at least 32 usable source frames and 80% usable '
        'source/output frames. The complete timeline is resampled to 128 steps; angle pairs are renormalized after interpolation.','',
        'The saved JSON includes source-video SHA-256, frame timestamps, points, visibility, angle window, quality and predictions. '
        'The old first-128 and last-128 paths and the separate complete-serve research path are shown on the same recording. '
        'Research GRU inputs receive no moment matching; its kNN uses timing features from the resampled angles. '
        'The new policy is associated only with its new model/bank, never applied to the legacy assets.','',
        '**No physical Android is connected.** The app compiles and the numerical feature contract is verified with real '
        'training-clip and missing-pose fixtures. Android/Python pose detection, decoder rotation, native inference performance '
        'and real LIVE_STREAM callback behavior still require the phone trace. Recorded VIDEO replay deliberately bypasses '
        'the live motion gate; it does not establish that automatic live serve boundaries are fixed.','',
        'Offline evaluation of the unchanged bundled classifier illustrates window sensitivity:','',
        '| Existing assets / window | GRU | Fixed hybrid | Clips losing poses | Discarded poses |',
        '|---|---:|---:|---:|---:|']
    for mode,r in diagnostic['capture'].items():
        lines.append(f"| {mode} | {pc(r['gru']['accuracy'])} | {pc(r['hybrid']['accuracy'])} | {r['clips_losing_frames']} | {r['discarded_pose_frames']} |")
    lines+=['','The old models were trained under their legacy input rules. Resampling alone is not assumed to improve their '
        'accuracy. The new full-serve pipeline is paired with models trained under that pipeline. Source-pose counts exclude '
        'failed detections in the legacy windows, while complete-timeline quality counts missing poses.','',
        '## 3. Augmentation with richer features','',
        'Reused all 418 frozen training-only video augmentation variants, including their newly extracted landmarks. '
        'No raw recording is changed. Every variant follows its original clip into a training fold; validation uses originals only. '
        'Each model receives one sampled original/variant per original per epoch, for 100 epochs with the same GRU32 training '
        'settings as v3. The angle baseline reuses its three verified v3 participant-held-out models.','',
        'The lean representation contains the existing ten angle channels plus aspect-corrected, hip-centered coordinates, '
        'visibility and adjacent displacements for shoulders, right elbow/wrist/index, and hips (45 channels). Full skeleton '
        'and full skeleton/motion use 99 and 165 channels. Lean control versus lean augmented isolates augmentation at fixed '
        'input size; cross-representation comparisons also change model capacity.','',
        '| Condition | Held-out-player GRU | Fixed hybrid | Nested tuned hybrid | Mean-player tuned macro-F1 |',
        '|---|---:|---:|---:|---:|']
    for c,r in cv.items():
        lines.append(f"| {NAMES[c]} | {pc(r['gru']['accuracy'])} | {pc(r['fixed_hybrid']['accuracy'])} | {pc(r['nested_tuned']['accuracy'])} | {r['mean_player_macro_f1']:.3f} |")
    lines+=['','## 5. Fusion weights and confidence','',
        'For each outer held-out player, two inner folds hold out one remaining player and train on the other. '
        'Their group-held-out predictions fit the fusion policy. The outer player contributes no original/augmented training '
        'clip, label, calibration example or weight-selection example. The fixed grid tests GRU weights 0, 0.25, 0.5, 0.75 and 1. '
        'It selects mean-player macro-F1, with probability loss and proximity to 0.5 as declared tie breakers. '
        'A single temperature on the log fused probabilities minimizes inner player-balanced log loss. '
        'Temperature changes probability sharpness while preserving the fused class ranking.','',
        'Acceptance thresholds are tested only on those inner predictions: at least 80% observed accuracy, at least 20 '
        'accepted cases and at least five per inner player. If none qualifies, confidence acceptance is disabled. '
        'These small-sample thresholds are development criteria, not a guarantee of correctness.','',
        '| Condition | Fixed hybrid log loss | Nested calibrated log loss | Fixed ECE | Nested ECE | Nested acceptance coverage | Accepted accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c,r in cv.items():
        a=r['fixed_hybrid'];b=r['nested_tuned']
        lines.append(f"| {NAMES[c]} | {a['log_loss_available']:.3f} | {b['log_loss_available']:.3f} | {a['ece_available']:.3f} | {b['ece_available']:.3f} | {pc(r['nested_selected_coverage'])} | {pc(r['nested_selected_accuracy'])} |")
    lines+=['','The final research policy is fit on all participant-held-out training predictions after nested evaluation. '
        'Inner-fold acceptance thresholds did not transfer successfully: outer accepted-case accuracy was only '
        '25.6–35.0%, despite meeting 80% in the inner fitting data. These thresholds are unsuitable as validated '
        'coaching-confidence filters. All final policies disable confidence acceptance. '
        'It accompanies a single model refit on all 209 eligible clips. The shift from one/two-player base fits to three-player '
        'final fits can affect calibration. Policies trained here cannot be transplanted onto the old five-model ensemble.','',
        f"Selected policy: GRU weight **{selected_policy['gru_weight']:.2f}**, kNN weight **{1-selected_policy['gru_weight']:.2f}**, "
        f"temperature **{selected_policy['temperature']:.3f}**, acceptance threshold **{selected_policy['confidence_threshold']}** "
        '(None means disabled).','',
        '## Diagnostic evaluation after selection','',
        'All candidates, policies and checkpoint hashes were frozen before this run opened diagnostic labels. '
        'These 175 clips were inspected in earlier work; they remain diagnostic, not an untouched final test. '
        'This table evaluates single full-training fits, whereas v3 diagnostic results used five-fold ensembles. '
        'One tracking-quality rejection remains incorrect/unavailable in the denominator.','',
        '| Condition / mode | Accuracy | Macro-F1 | Drive recall | Lob recall | Topspin recall |',
        '|---|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        for mode in ['gru','fixed_hybrid','tuned_calibrated']:
            r=diagnostic['conditions'][c][mode]
            lines.append(f"| {NAMES[c]} / {mode} | {pc(r['accuracy'])} | {r['macro_f1']:.3f} | {' | '.join(pc(x) for x in r['class_recall'])} |")
    lines+=['','## Interpretation and remaining work','',
        '- Three training players remain the entire independent training population. Nested inner models train on one player; '
        'their predictions can be substantially worse than models trained on two or three. Reported metrics are small-sample development evidence.',
        '- Each condition has properly separated outer evaluation of its fusion policy. Selecting the best condition on these '
        'outer scores can inflate the winner’s apparent performance. A separate new-player final test remains unavailable.',
        '- Pose geometry cannot directly measure ball spin. Per-class recall and macro-F1 accompany accuracy so a collapsed '
        'serve class is visible. No source label was changed without coach review.',
        '- A better calibration score need not imply better recognition. Log loss/Brier combine discrimination and probability '
        'quality; reliability bins and ECE are also saved. Temperature alone preserves class predictions.',
        '- Real phone replay and live motion-gate validation remain pending. No connected-device accuracy or speed is claimed.','',
        '## Outputs and reproduction','',
        'Verification: **20 Python checks and 35 Kotlin unit tests pass**. Android debug build, lint and the '
        'phone-test APK build pass. The selected TFLite model matches Keras across all 175 diagnostic inputs '
        'within the declared numerical tolerance. Research model/bank hashes inside the APK were checked. '
        'Physical-phone instrumentation tests remain pending.','',
        '- [Frozen protocol](../models/serve_refinement_v4/protocol.json)',
        '- [Frozen selection and policies](../models/serve_refinement_v4/frozen_selection.json)',
        '- [Diagnostic results](../outputs/serve_refinement_v4/diagnostic_results.json)',
        '- [Per-clip predictions](../outputs/serve_refinement_v4/diagnostic_predictions.csv)',
        '- [Research mobile bundle](../models/serve_refinement_v4/android_bundle/manifest.json)',
        '- [Complete automated-study package](../outputs/serve_refinement_v4/complete_outputs.zip)',
        '- [Android debug APK](../android/app/build/outputs/apk/debug/app-debug.apk)','',
        '```powershell',
        'python scripts/train_serve_refinement.py',
        'python scripts/evaluate_serve_refinement.py',
        'python scripts/test_serve_refinement.py',
        'python scripts/test_serve_refinement_integrity.py',
        'python scripts/report_serve_refinement.py',
        '# After Replay Serve → Save comparison on Android:',
        'python scripts/compare_serve_replay.py path/to/serve-comparison.json --video path/to/the-same.mp4',
        '```','',
        'A connected-phone instrumentation test, `com.thesis.pickleballserve.ServeResearchModelTest`, checks the '
        'actual Android TFLite runtime, kNN bank and calibrated probabilities against three training-only '
        'numerical fixtures. It has not been run without a device. The replay trace comparison separately '
        'checks decoded-video/pose behavior.','',
        'References: [independent calibration data](https://scikit-learn.org/stable/modules/calibration.html), '
        '[MediaPipe Android modes](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/android), '
        '[indexed video frames](https://developer.android.com/reference/android/media/MediaMetadataRetriever), '
        '[video timestamps](https://developer.android.com/reference/android/media/MediaExtractor).','']
    report=ROOT/'analysis/SERVE_REFINEMENT_V4.md';report.write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    y=np.arange(len(CONDITIONS));labels=[NAMES[c].replace(' + ','\n+ ') for c in CONDITIONS]
    for mode,color,offset in [('gru','#42638f',-.2),('fixed_hybrid','#d09c3f',0),('nested_tuned','#427965',.2)]:
        axes[0].barh(y+offset,[100*cv[c][mode]['accuracy'] for c in CONDITIONS],height=.19,label=mode.replace('_',' '),color=color)
    axes[0].set(yticks=y,yticklabels=labels,xlim=(0,100),xlabel='Accuracy (%)',title='Held-out training players')
    axes[0].invert_yaxis();axes[0].legend(fontsize=8)
    for mode,color,offset in [('gru','#42638f',-.14),('tuned_calibrated','#427965',.14)]:
        axes[1].barh(y+offset,[100*diagnostic['conditions'][c][mode]['accuracy'] for c in CONDITIONS],height=.26,label=mode.replace('_',' '),color=color)
    axes[1].set(yticks=y,yticklabels=[],xlim=(0,100),xlabel='Accuracy (%)',title='Previously inspected diagnostic clips')
    axes[1].invert_yaxis();axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUTPUT/'comparison.png',dpi=180);fig.savefig(OUTPUT/'comparison.svg');plt.close(fig)
    # Store every v4 checkpoint, membership, history, probability output, policy and plot.
    package=OUTPUT/'complete_outputs.zip'
    files=[p for p in DEST.rglob('*') if p.is_file()]+[p for p in OUTPUT.rglob('*') if p.is_file() and p!=package]+[report]
    files+=[ROOT/'scripts'/name for name in ['serve_refinement.py','train_serve_refinement.py','evaluate_serve_refinement.py',
        'compare_serve_replay.py','prepare_serve_replay_fixtures.py','test_serve_refinement.py','test_serve_refinement_integrity.py','report_serve_refinement.py','finish_serve_refinement.py']]
    files+=[ROOT/'android/app/src/main/java/com/thesis/pickleballserve'/name for name in
            ['ServeSequence.kt','ProbabilityPolicy.kt','ServeReplayActivity.kt']]
    files+=[ROOT/'android/app/src/test/java/com/thesis/pickleballserve/ServeSequenceTest.kt',
            ROOT/'android/app/src/test/resources/fixtures/serve_sequence.json',
            ROOT/'android/app/src/androidTest/java/com/thesis/pickleballserve/ServeResearchModelTest.kt',
            ROOT/'android/app/src/androidTest/assets/serve_refinement.json']
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(files)):z.write(path,path.relative_to(ROOT).as_posix())
    write_json(OUTPUT/'complete.json',dict(automated_outputs_complete=True,selected_condition=selected,
        new_model_fits=47,reused_outer_fits=3,physical_phone_pending=True,coach_review_deferred=True,
        new_players_deferred=True,package=str(package.relative_to(ROOT)),report=str(report.relative_to(ROOT))))
    print('REPORT AND PACKAGE COMPLETE',selected,flush=True)


if __name__=='__main__':main()
