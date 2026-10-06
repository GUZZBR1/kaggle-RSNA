# CNN-224 Real Five-Fold OOF Evaluation

## Verdict

`OOF_EVALUATION_BLOCKED`

The Mini-MVP 3 prerequisite is not satisfied. Its dedicated audit concluded
`FIVE_FOLDS_BLOCKED`, with no training started and no checkpoints, TrainingResults,
or validation PredictionArtifacts produced. The user confirmed that Mini-MVP 2 still
needs to be completed. This evaluation therefore did not aggregate predictions or
calculate metrics.

This is an unavailable-input verdict, not an OOF integrity failure. No real OOF or
competitive-performance claim is made.

## Execution identity

| Field | Value |
|---|---|
| `BASE_SHA` | `5e39a11aa1efb2203610b6f45b59bea6bbcbf5ad` (`origin/main` after fetch) |
| `EXECUTION_SHA` | `N/A` — no OOF evaluator execution; audit-only checkout at the base SHA above |
| Branch | `codex/rsna-real-oof-evaluation` |
| Frozen `CONFIG_HASH` | Not available; Mini-MVP 2 freeze is not established |
| CNN-224 template SHA-256 | `90333b0346e643a80b5b1a6483e3c6f2c3064c1f812a17699c3f0d5539b1b6f3` (template only; not the frozen config) |
| `MODEL_CANDIDATE_ID` | Not available |
| `DATASET_ID` | Not available |
| `FOLD_PLAN_ID` | Not available |
| `TARGET_REGISTRY_ID` | Defined by the canonical source registry; no run artifact binds it to these folds |
| Mini-MVP 3 evidence | `FIVE_FOLDS_BLOCKED`; report commit `7b10537694fc6b99da6432b77724cdd7b1773793` on `codex/rsna-cnn224-real-five-fold` |

## Fold artifact inventory

| Fold | Rows | Targets | Config hash | Candidate ID | PredictionArtifact ID | Status |
|---:|---:|---:|---|---|---|---|
| 0 | N/A | N/A | N/A | N/A | N/A | `MISSING` |
| 1 | N/A | N/A | N/A | N/A | N/A | `MISSING` |
| 2 | N/A | N/A | N/A | N/A | N/A | `MISSING` |
| 3 | N/A | N/A | N/A | N/A | N/A | `MISSING` |
| 4 | N/A | N/A | N/A | N/A | N/A | `MISSING` |

The inspected dry-run checkout at the base SHA had no real fold artifacts or run
report. Its `prepared-dataset.synthetic*` fixtures are explicitly synthetic and are
not inputs to this evaluation. The Mini-MVP 3 report records all five folds as pending.

## Coverage and alignment

| Measure | Result |
|---|---:|
| `EXPECTED_ROWS` | N/A — real FoldPlan unavailable |
| `OBSERVED_ROWS` | 0 real OOF rows available in the inspected artifact locations |
| `COVERAGE_PERCENT` | Not computable |
| `DUPLICATE_ROWS` | Not computable |
| `MISSING_ROWS` | Not computable |
| `FOREIGN_ROWS` | Not computable |

No coverage gate was run. Reporting 0% coverage would be misleading because the
expected-study denominator is unavailable.

## Target order, labels, and metrics

The canonical source registry defines this order:

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's
11. Contusion
12. Fracture

No fold artifact exists to check against the registry. Real LabelRecords and observed
masks are also unavailable, so class counts and missing-label handling cannot be
audited. No AUC was calculated and no aggregate Evaluation artifact was emitted.

| Target | Observed | Positive | Negative | AUC | Status |
|---|---:|---:|---:|---:|---|
| ACL | N/A | N/A | N/A | N/A | `UNAVAILABLE` — real predictions and labels absent |
| MCL | N/A | N/A | N/A | N/A | `UNAVAILABLE` — real predictions and labels absent |
| Medial Meniscus | N/A | N/A | N/A | N/A | `UNAVAILABLE` — real predictions and labels absent |
| Lateral Meniscus | N/A | N/A | N/A | N/A | `UNAVAILABLE` — real predictions and labels absent |
| Medial OA | N/A | N/A | N/A | N/A | `UNAVAILABLE` — real predictions and labels absent |
| Lateral OA | N/A | N/A | N/A | N/A | `UNAVAILABLE` — real predictions and labels absent |
| PF OA | N/A | N/A | N/A | N/A | `UNAVAILABLE` — real predictions and labels absent |
| Effusion | N/A | N/A | N/A | N/A | `UNAVAILABLE` — real predictions and labels absent |
| Synovitis | N/A | N/A | N/A | N/A | `UNAVAILABLE` — real predictions and labels absent |
| Baker's | N/A | N/A | N/A | N/A | `UNAVAILABLE` — real predictions and labels absent |
| Contusion | N/A | N/A | N/A | N/A | `UNAVAILABLE` — real predictions and labels absent |
| Fracture | N/A | N/A | N/A | N/A | `UNAVAILABLE` — real predictions and labels absent |

`MISSING_LABEL_COUNTS`: unavailable. `INVALID_PROBABILITIES`: unavailable. The
evaluator's macro-AUC policy (macro only over defined target AUCs, currently requiring
all 12) is documented in `docs/OOF_EVALUATION.md`; no targets were included or
excluded because no evaluation ran. `MACRO_AUC`: unavailable.

## Fold diagnostics, sanity, and reproducibility

No fold diagnostics, probability-distribution checks, or repeated evaluation were
possible. Row order, AUCs, macro AUC, report hash comparison, constant predictions,
and per-fold divergence are all `NOT EVALUATED` because there are no real predictions.

## Provenance and adversarial review

The required lineage cannot be established:

```text
DatasetVersion         MISSING
  -> FoldPlan           MISSING
  -> ModelCandidate     MISSING
  -> TrainingJobs 0–4   MISSING (Mini-MVP 3 did not train)
  -> Checkpoints        MISSING
  -> PredictionArtifacts MISSING
  -> OOF aggregate      NOT CREATED
  -> Evaluation         NOT CREATED
```

| Adversarial question | Finding |
|---|---|
| Is a fold missing? | Yes; all five fold PredictionArtifacts are unavailable. |
| Could a fold be partial? | Cannot determine without fold artifacts. |
| Duplicate or cross-fold StudyInstanceUIDs? | Not auditable without rows and FoldPlan. |
| Target order changed in an artifact? | Canonical registry order is known; no artifact to inspect. |
| Were missing labels masked? | Not auditable without real LabelRecords and predictions. |
| Did all folds use one frozen config/candidate? | Not established; Mini-MVP 2 freeze is missing. |
| Did any result come from synthetic data? | The discovered examples are synthetic and excluded; no real results were found. |
| Does OOF contain only held-out validation predictions? | Not auditable because no OOF predictions exist. |
| Was a study used to train the model that predicts it? | Not auditable without jobs, plan, and artifact provenance. |
| Did macro AUC exclude targets without justification? | No macro was calculated. |
| Was AUC calculated outside the canonical evaluator? | No AUC was calculated. |

The requested specialist agent launches were attempted but the available agent models
were rejected by this Codex account/runtime. Their roles were therefore handled
sequentially by the coordinator from the accessible repository and Mini-MVP 3 evidence;
the findings above delimit checks that cannot be performed without inputs.

## Outputs and next gate

| Required output | Status |
|---|---|
| OOF aggregate PredictionArtifact | Not created — no fold inputs |
| Evaluation artifact | Not created — evaluation blocked |
| Target AUC table | This blocked-status table only; no numeric AUCs |
| Fold diagnostics | Not created |
| Comparison/report artifact | This audit report only |
| Provenance manifest | Not created — upstream lineage absent |
| OOF artifact ID / Evaluation ID | N/A |
| PR | [#68](https://github.com/GUZZBR1/kaggle-RSNA/pull/68) — open; CI succeeded |

Next gate: complete Mini-MVP 2 and Mini-MVP 3, provide the frozen config and its hash,
real DatasetVersion, labels/masks, canonical FoldPlan, and five consistent validation
PredictionArtifacts with checkpoint/training provenance. Then rerun this audit using
the canonical OOF Evaluation Engine. Do not claim `REAL_OOF_ESTABLISHED` until every
coverage, metric, reproducibility, and provenance gate passes.
