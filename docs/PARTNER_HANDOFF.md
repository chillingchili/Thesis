> Latest mobile follow-up (12 October): corrective output coverage increased from 2/30 to 18/30 on reused live-path replays. Eleven cues use guarded general-mechanics rules despite uncertain classification. These outputs were not rated by Coach C. Three captures failed; one output was 3.126 seconds after file end. A subsequent eight-clip longer-hold pilot added one positive output but did not recover missing captures or remove the late case, so default 350 ms hold remains. Current manuscript and website now distinguish this development result from the frozen expert study. See `evaluation/coach_c/mobile_gpu_rate_20261012/REPORT.md` and `docs/thesis/Pascua-Leones_Thesis_Current.md`.

# Partner handoff — current original-GRU evaluation
## Session 2 received — 12 October 2026

[Scored response export](../evaluation/coach_c/session2_gru/coach_c_scores_20261012/REPORT.md): 27 Valid, one Not valid (C001), two No coaching feedback to assess (C021, C024), zero missing. Primary Valid/30 = 90.0%; secondary Valid/(Valid + Not valid) = 96.4%.

[Interpretation review](../evaluation/coach_c/session2_gru/coach_c_scores_20261012/INTERPRETATION_REVIEW.md): eight Valid answers concern frozen notice-only outputs, while two corrective outputs were marked no coaching. The present deployed form was checked: all 30 output texts match the frozen reports. The remaining limitation is interpretation of the selected categories. Answers remain unchanged. Chapters 5–6 and the website now report these results and limitations.


The selected Session 2 classifier is the **original five-fold GRU ensemble**,
without kNN weighting. Android and desktop defaults now select it. Hip and paddle
measurements and the 0.60 gate remain; the apps now use the approved feedback wording.

The change was authorized before Session 2 was sent, but after researchers had
read Session 1. The manuscript discloses that timing. Selection uses the prior
matched diagnostic comparison: original GRU 51.4%, hybrid 49.7%, retrained GRU 46.9%.
That difference does not prove general superiority. No retraining or tuning to
Coach C comments occurred.

## Use these files

- [Active GRU outputs, preview and amendment](../evaluation/coach_c/session2_gru/README.md).
- [New Session 2 form instructions](../scripts/google_apps_script/README.md).
  The original-GRU form has been completed. Retain it and its responses unchanged; the current displayed outputs have been verified.
- [Current manuscript](thesis/README.md), aligned with the GRU selection and actual results.
- [Scoring plan](../evaluation/coach_c/session2/SCORING_PLAN.md).
  Run python scripts/score_coach_c_session2.py with the response export; its
  default reports are now session2_gru/reports.

## Results and remaining work

Session 1 contains 27 Bad Form and three Good Form judgments (C008, C019, C022).
Independent subtype labels were intentionally omitted, so independent Coach C
classification accuracy is unavailable. The original 30 clips remain unchanged.

The GRU version produces 19 corrective, two positive, eight low-confidence and
one partial output: 21/30 substantive coaching (70.0%). The archived hybrid had
17/30 (56.7%). Coverage is distinct from the submitted validity ratings. All 30 remain in the primary denominator; missing and no-coaching responses remain separate.

Table 12 and the conclusions now include the completed ratings. Adviser review, final typesetting and any additional live-timing evidence remain. The [Xiaomi recorded-video benchmark](../evaluation/phone_latency/note12_20261010/REPORT.md)
is complete: nine successful measured runs, 68.6 s median and 104.7 s slowest.
Paddle processing is the main bottleneck. Live capture-to-display latency still
needs measurement; these results do not verify the two-second target or desktop/phone parity.

The manuscript retains the prior diagnostic exposure of the beginner clips and
the separate legacy versus annotated-training populations. Original GRU's 86.7%
historical CV and retrained GRU's 90.9% CV do not imply those accuracies on new players.

## Historical evidence

evaluation/coach_c/baseline and evaluation/coach_c/session2 remain unchanged
hybrid snapshots. Their old readiness statuses are superseded by
evaluation/coach_c/active_session2.json. Original run artifacts remain under
outputs/. Do not rerun the historical handoff exporter to overwrite these records.
Session 2 responses have now been received; see the scored results above. No coach message has been sent here.


Paddle preprocessing optimization is installed and verified on the Xiaomi: grayscale crop preparation fell from 5.20 to 1.38 ms with identical tested pixels and angles. This is a component improvement; the subsequent full recorded-clip benchmark found effectively unchanged latency; live latency remains unmeasured. See [the optimization evidence](../evaluation/phone_latency/note12_optimization_20261010/REPORT.md).


[Full optimized Android rerun](../evaluation/phone_latency/note12_optimized_full_20261010/REPORT.md): 9/9 measured trials passed. Median 68.576 s versus 68.619 s before (0.06% lower), effectively unchanged. This is recorded-video processing, not live-camera latency.
