# RSNA project status

This document is the canonical high-level execution state for the repository. Detailed module behavior lives in the dedicated docs; this file answers **what is ready, what is blocked, and what should be worked on next**.

## Mission

Build a reproducible and competitive pipeline for the RSNA Knee Abnormality Detection challenge, moving from verified data preparation to real OOF evidence, controlled model search, ensemble selection and final Kaggle inference.

## Current canonical state

```text
FOUNDATION                     READY
DATA PIPELINE                  READY
SYNTHETIC E2E                  READY
TRAINING ENGINE                READY
OOF EVALUATION                 READY
CNN-224 CODE                   READY
REAL RSNA DATA VALIDATION      PENDING
REAL CNN-224 5-FOLD OOF        PENDING
COMPETITIVE ABLATIONS          PENDING
MODEL DIVERSITY                PENDING
ENSEMBLE                       PENDING
KAGGLE INFERENCE               PENDING
ENDGAME                        PENDING
```

## Current critical path

The project should now optimize for **real competitive evidence**, not more foundation work.

1. Establish operational access to the real RSNA dataset in the intended training runtime.
2. Run a bounded real-data preflight: metadata/index/series mapping/labels/folds/leakage.
3. Inventory DICOM transfer syntaxes and prove deterministic pixel decoding for the representative sample.
4. Confirm the competition/rules constraints relevant to external weights, data, offline runtime and packaging.
5. Run a small GPU dry run for the existing CNN-224 baseline and choose only the batch-size/accumulation values required to fit memory.
6. Freeze the baseline configuration before fold 0.
7. Train folds 0-4 with the same preprocessing/model/fold/target contracts.
8. Build the full OOF artifact and report all 12 target AUCs plus macro AUC.
9. Only after the real baseline exists, begin controlled ablations and new model arms.

## Issue groups

### Foundation — complete

The merged repository already contains the data preparation, synthetic smoke, training and OOF infrastructure needed for the first real baseline. Do not reopen this layer without a concrete regression.

Relevant completed work includes the Data Pipeline Integration, Synthetic End-to-End Data Smoke, Training Engine and OOF Evaluation Engine.

### Immediate milestone

**Issue #17 — CNN-224 Baseline**

The code path is implemented and documented in [`CNN224_BASELINE.md`](CNN224_BASELINE.md). The issue remains materially incomplete until a real five-fold run produces complete OOF predictions and reproducible metrics.

The required real-run evidence is:

- folds 0-4 complete;
- checkpoint per fold;
- per-fold `PredictionArtifact`s;
- exactly-once OOF study coverage;
- 12 target AUCs;
- macro AUC;
- recorded config/seed/fold/model/preprocessing identities;
- training/inference timing, throughput and peak VRAM when available.

### Real-data gates

The real-data path is covered by existing issues rather than new infrastructure work:

- **#33 — Real Dataset Validation Campaign**: bounded real-data preflight and diagnostics.
- **#34 — DICOM Codec Compatibility**: transfer-syntax/decoder compatibility.
- **#41 — Competition Rules Compliance**: current rule/provenance/offline-runtime gate.

These should be implemented only to the depth needed to make the real baseline trustworthy and runnable.

### Controlled ablations after the baseline

Recommended order:

1. **#18 — CNN-288 Resolution Ablation**
2. **#19 — Slice Selection Arena**
3. **#25 — Augmentation & Regularization Arena**
4. **#26 — Hard vs Soft Label Arena**

Change one material factor per comparison and keep folds/seeds/model factors paired.

### Model-diversity phase

Recommended order:

1. **#20 — 2.5D Triplet Model**
2. **#21 — MIL / Attention Pooling**
3. **#22 — CoAtNet Model Arm**
4. **#23 — DINOv2 Model Arm**
5. **#24 — RadImageNet Model Arm**, only if provenance/license/rules allow it.

Synthetic smokes prove wiring only. Real promotion decisions require matched OOF evidence.

### Ensemble and tournament phase

After multiple real OOF candidates exist:

- **#27 — Ensemble Diversity Engine**
- **#28 — Target-Specific Blending**
- **#29 — TTA Arena**
- **#31 — Continuous Tournament**

The tournament should consume real candidate evidence; do not build a large mutable orchestration layer before there are candidates worth comparing.

### Kaggle and endgame

- **#30 — Kaggle Inference Notebook**
- **#38 — Runtime Budget Gate**
- **#39 — Artifact Integrity Audit**
- **#40 — Failure Recovery**
- **#41 — Competition Rules Compliance**
- **#32 — Endgame Search**

The final lock should be reproducible from immutable candidate/config/artifact identities and must fit runtime/rules constraints.

## Deferred infrastructure

Useful but not on the critical path before the first real OOF result:

- **#35 — Cloud Provider Benchmark**
- **#36 — Experiment Registry**
- **#37 — Seed Registry**

Do not let these delay the first real baseline.

## Baseline freeze policy

For the first real CNN-224 run, do not tune the architecture during folds. Freeze before fold 0:

- 224x224 input resolution;
- 24 selected slices;
- explicit Study -> Series mapping;
- current CNN-224 architecture/pooling;
- official 12-target order;
- canonical five-fold plan;
- preprocessing/intensity policy;
- optimizer/scheduler/loss settings from the versioned config;
- seed/config identity.

If memory is insufficient, adjust batch size and/or gradient accumulation only, record the change, and keep the effective training policy consistent across all folds.

## Promotion policy

A synthetic result can establish **code readiness** but cannot establish **competitive superiority**.

Use these labels consistently:

```text
CODE_READY        implementation + deterministic synthetic/CPU coverage
REAL_READY        real-data preflight passed
OOF_ESTABLISHED   complete real five-fold OOF + metrics + lineage
PROMOTED          matched real evidence beats the current comparison gate
```

## Definition of the next milestone

The next milestone is complete only when the repository can truthfully state:

```text
CNN-224 REAL BASELINE ESTABLISHED
```

with complete five-fold OOF evidence. Until then, the current status is:

```text
CNN-224 BASELINE CODE READY
REAL BASELINE NOT YET ESTABLISHED
```
