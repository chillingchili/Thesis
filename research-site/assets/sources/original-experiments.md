# Thesis Results Summary — Pickleball Serve Subtype Classification

Date: 2026-09-24  
Dataset: 438 train clips (Beginner1, Beginner2, CoachA) | 32 timing features  
Holdouts: Beginner3 (85), Beginner4 (90) — sealed, never trained on

---

## 1. NFR2 — 85% stratified 5-fold CV (thesis metric, Section 4.3.2)

Plain `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on training clips.

| Model | CV accuracy (mean) | Meets 85%? |
|---|---|---|
| Gradient Boosting | 92.9% | YES |
| Random Forest | 91.8% | YES |
| SVM (RBF, scaled) | 89.9% | YES |
| kNN-7 + Std + manhattan | 87.9% | YES |
| **GRU (thesis model)** | **86.7%** (folds 90.9, 92.0, 85.2, 89.7, 75.9) | **YES** |
| kNN-7 + Std (euclidean) | 85.4% | YES |
| kNN-5 + Std | 84.0% | NO (just under) |
| Logistic Regression | 83.1% | NO |
| kNN-5 raw (no scaling) | 78.3% | NO |

**Conclusion:** GRU (named throughout the thesis) passes NFR2 as written at 86.7%. No document amendment required. Feature scaling was the key fix for classical kNN.

---

## 2. Honest subject-held-out generalization (secondary; not the thesis gate)

### 2a. StratifiedGroupKFold(3) on train (leave-one-participant-out style)

| Model | Group-CV acc |
|---|---|
| Random Forest | 57.8% |
| Gradient Boosting | 57.1% |
| SVM | 56.0% |
| Logistic Regression | 53.7% |
| kNN-5 / kNN-7 family | ~51–52% |
| GRU (older group run) | 56.8% |
| Strong-aug GRU | 55.3% |
| Adversarial GRU (λ=1) | ~56–67% (unstable across folds) |

### 2b. Sealed holdout (fit on all 438 train clips, then evaluate Beg3 / Beg4)

| Model | Beg3 acc | Beg3 macro-F1 | Beg4 acc | Beg4 macro-F1 | Mean |
|---|---|---|---|---|---|
| kNN-5 (distance, raw) | 58.8% | 0.549 | 53.3% | 0.478 | **56.1%** |
| kNN-7 + Std + manhattan | 54.1% | 0.501 | 43.3% | 0.351 | 48.7% |
| SVM | 50.6% | 0.425 | 45.6% | 0.448 | 48.1% |
| Logistic Regression | 45.9% | 0.370 | 51.1% | 0.432 | 48.5% |
| Random Forest | 40.0% | 0.313 | 57.8% | 0.520 | 48.9% |
| Gradient Boosting | 37.6% | 0.283 | 52.2% | 0.474 | 44.9% |
| GRU strat5 ensemble @0.40 | 34% | 0.27 | 42% | 0.45 | 38% |
| Strong-aug GRU ensemble | 39% | 0.31 | 44% | 0.47 | 41.5% |
| Adversarial GRU ensemble (λ=1) | 40–48% | — | 42–47% | — | ~44% (seed-unstable) |
| Metric/center-loss GRU (prototype) | 52.9% | — | 51.1% | — | **52.0%** |

### 2c. Quick wins (post-hoc; no retraining of sealed holdout)

| Method | Beg3 | Beg4 | Mean |
|---|---|---|---|
| **bestTTA_mix (moment-match TTA + ensemble)** | — | — | **58.3%** |
| SVM + moment TTA | — | — | 57.2% |
| knn7s + biomech features (Beg3) | 62.4% | — | 55.1% |
| CORAL (knn5+svm+rf) Beg3 | 64.7% | — | 55.7% |
| Weighted knn5+knn7s+rf | — | — | 56.6% |

All quick wins peak at **58.3% mean** — well below an informal 70% escalation bar, confirming the subject-shift ceiling.

### 2d. Real training changes (retrained on train only; holdout re-evaluated)

| Change | Best holdout mean | vs baseline GRU (38%) | vs quick-win best (58.3%) |
|---|---|---|---|
| 1. Metric / center-loss + cosine head (prototype) | **52.0%** | +14 pts | −6 pts |
| 2. Adversarial subject-invariance (DANN GRL) | ~44–48% (unstable) | +6–10 pts | −10–14 pts |
| 3. Strong subject-style augmentation | 41.5% | +3.5 pts | −17 pts |
| Combo: adversarial + strong aug | 36–37.5% | −0.5 pts | −21 pts |

**Conclusion:** None of the three planned real training changes, nor their combination, rescue unseen-subject holdout above 70%. Metric learning was the strongest single real change (52.0%) but still trails the best classical quick-win (58.3%). The bottleneck is structural: only 3 training participants, sealed Beg3/Beg4 distribution shift.

### Per-class pattern (consistent across models)

- **Beg3:** lob collapses (recall near 0–18%); drive/topspin partially recoverable; errors often confidently wrong.
- **Beg4:** topspin collapses (recall near 0–10%); drive/lob partially OK; RF recovers lob best (73%).
- GRU strat5 on Beg3: nearly everything → topspin (lob recall 0). On Beg4: topspin → drive (18/30), drive → topspin (22/30).

This is a **subject-style distribution shift**, not a fixed class-pair failure (margin analysis: Beg3 pulls lob, Beg4 pulls topspin). Consistent with having only 3 training participants.

---

## 3. Recommendation for thesis writeup

1. Report **GRU 86.7% stratified 5-fold** as the NFR2 result — matches the named model and the stated CV procedure exactly.
2. Report **group-CV (~56–58%)** and **Beg3/Beg4 holdout** as the honest generalization / limitations discussion (N=3 train-subject ceiling).
3. Include classical baselines (GB / RF / SVM / kNN-7) in a comparison table — all pass the same CV gate; none rescue holdout.
4. Optional: briefly note that metric-learning, adversarial, and strong-augmentation variants of the GRU were tried; none surpassed the classical quick-win ceiling (~58% mean holdout).
5. Do **not** train on Beg3/Beg4; do **not** claim 85%+ generalizes to unseen players.

---

## 4. Body-weight-shift detector (deterministic output)

Separate from serve-subtype classification: a rule-based form check for whether the
server transfers body weight laterally, derived **only from CoachA's good-form clips**.

- **Metric** (identical offline & on-device): hip-center image x (landmarks 23/24)
  → 5-frame moving average → `(p95 − p5) / median body height` (nose→ankle), so the
  value is invariant to camera distance/framing.
- **Threshold** = 10th percentile of CoachA's 142 clips = **0.1438** (`coach_p10`).
- **Validation (all 613 clips):**

  | subject | n | detected | rate |
  |---|---|---|---|
  | CoachA | 142 | 127 | **0.89** |
  | Beginner1 | 146 | 43 | 0.29 |
  | Beginner2 | 150 | 0 | 0.00 |
  | Beginner3 | 85 | 2 | 0.02 |
  | Beg4 | 90 | 2 | 0.02 |

  Beginners 2–4 essentially never pass; Beginner1 is the overlap case (some beginner
  clips do contain notable lateral movement). Reproduce: `scripts/derive_shift_threshold.py`.
- **On-device**: `ShiftDetector.kt` computes it over the gated serve window and shows
  `WEIGHT SHIFT: ✓/✗` with the serve classification; config in
  `android/app/src/main/assets/shift_config.json`.

---

## 5. Artifacts

**Official thesis model**

- `models/gru_runs_angles_strat5/gru_fold{1..5}.keras` — official thesis GRU (stratified 5-fold, 86.7%)
- `scripts/train_gru.py --cv stratified --n_folds 5` — reproduces the 86.7% run
- `scripts/train_gru.py --style_offset/--style_scale` — strong subject-style augmentation

**Real training changes (scripts)**

- `scripts/train_gru_metric.py` — center-loss / cosine-head metric GRU (real change 1)
- `scripts/train_gru_adv.py` — DANN gradient-reversal subject-adversarial GRU (real change 2)
- Experimental checkpoints archived under `archive/experimental_runs/`:
  - `gru_runs_strongaug/` — strong-aug group-CV (real change 3)
  - `gru_adv_runs/`, `gru_adv_strongaug/`, `gru_adv_lambda3/` — adversarial ensembles
  - `gru_metric_runs/`, `gru_metric_runs_group/` — metric-learning models
  - `gru_runs/`, `gru_runs_angles/`, `gru_runs_angles_aug2/`, `gru_runs_angles_strong/` — earlier GRU runs

**Classical baselines & data**

- `models/timing_{svm,lr,rf,gb,knn5,knn7s}.pkl` — sklearn baselines fit on all train
- `scripts/final_holdout_table.py` — regenerates holdout table
- `features/timing_train.npz`, `features/timing_test.npz` (Beg3), `features/timing_beg4.npz`
- `features/timing_enriched_{train,beg3,beg4}.npz` — +42 biomechanical features
- `features/biomech_{train,beg3,beg4}.npz` — raw biomechanical features

**Analysis / one-offs** (run from project root)

- `analysis/quick_wins.py`, `analysis/quick_wins2.py` — TTA / ensemble / CORAL
- `analysis/` — diagnostics, PCA plot, GRU probs, intermediate feature dumps

**Docs**

- `docs/RESULTS_SUMMARY.md` — this file
- `README.md` — repository guide and directory map
