Pickleball serve-subtype thesis — repository map
================================================

Run all scripts from the project root (paths below assume that).

RESULTS
-------
RESULTS_SUMMARY.md   Full results: NFR2 86.7% strat5, group-CV, holdouts,
                     quick wins, real training changes, recommendations.

CORE PIPELINE (scripts/)
------------------------
bp.py                      BlazePose smoke test
build_manifest.py          Build manifest.csv from video tree
extract_keypoints.py       Extract BlazePose keypoints per clip
normalize_keypoints.py     Normalize keypoints (center/scale)
trim_after_contact.py      Trim frames after paddle contact
extract_joint_angles.py    Keypoints -> joint-angle sequences
extract_timing_features.py Keypoints -> 32 timing features
extract_biomech_features.py Keypoints -> 42 biomechanical features
train_gru.py               Train GRU (group or stratified CV; aug options)
testrun.py                 Evaluate GRU checkpoint(s) on sealed holdout
train_timing.py            Train sklearn timing-feature models
final_holdout_table.py     Fit baselines on all train; eval Beg3/Beg4
export_tflite.py           Export GRU folds -> TFLite (+ moment-matching stats)
export_knn_assets.py       Bundle kNN5 train matrix for the Android app
oof_subject_eval.py        Per-subject OOF/ensemble eval (e.g. coach lob check)
derive_shift_threshold.py  Coach p10 body-weight-shift threshold -> app assets

EXPERIMENTAL TRAINERS (scripts/)
--------------------------------
train_gru_metric.py        Metric / center-loss GRU (real change 1)
train_gru_adv.py           Adversarial subject-invariance GRU (real change 2)

OFFICIAL MODEL (models/)
------------------------
gru_runs_angles_strat5/    gru_fold1..5.keras  (NFR2: 86.7%)

FEATURE / MODEL ARTIFACTS
-------------------------
features/timing_train.npz           Train timing features
features/timing_test.npz            Beg3 holdout timing
features/timing_beg4.npz            Beg4 holdout timing
features/timing_enriched_*.npz      Timing + biomech merged
features/biomech_*.npz              Raw biomech features
models/timing_{svm,lr,rf,gb,knn5,knn7s}.pkl   Sklearn baselines

ANALYSIS (analysis/)
--------------------
quick_wins.py / quick_wins2.py   TTA, ensembles, CORAL
check_class_ood_distance.py      OOD distance diagnostic
compare_serve_groups.py          Serve-group comparison
eval_timing_threshold.py         Confidence / abstention analysis
flag_jumpy_keypoints.py          Jumpy keypoint finder
margin_model_compare.py          SVM vs kNN margins
scan_npy.py / npyvis.py / pose_visualizer.py   Viz helpers
*.png / *.csv / intermediate .npy|npz dumps

ARCHIVED EXPERIMENTAL RUNS (archive/experimental_runs/)
-------------------------------------------------------
Old GRU / metric / adversarial checkpoints not used by the thesis gate.
Official model stays at models/gru_runs_angles_strat5/.

ANDROID APP (android/)
----------------------
Thesis proof-of-concept: camera -> BlazePose -> joint angles -> GRU / kNN5
TFLite on-device. Modes SINGLE / ENSEMBLE / HYBRID; motion-gated serve window;
deterministic body-weight-shift verdict. Build in Android Studio; the pose
model must be downloaded manually (see android/app/src/main/assets/README.md).

DATA (data/)
------------
data/training/   438 train clips + keypoint stages (raw, keypoints,
                 keypoints_norm, keypoints_trimmed, keypoints_angles, ...)
data/testing/    Beg3 / Beg4 sealed holdouts + keypoint stages

NOTES (from earlier extraction)
-------------------------------
Jumpy / unreliable raw keypoints (15 clips) — see git history or re-flag:
  CoachA_Drive_003, _018, _020, _023, _033, _040, _043
  CoachA_Lob_039, _044
  CoachA_Topspin_004, _007, _009, _011, _015, _023

.gitignore
----------
Ignores __pycache__, .kilo/ (agent tooling), venvs, OS junk.