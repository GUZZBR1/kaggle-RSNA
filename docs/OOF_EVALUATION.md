# Out-of-fold evaluation

`rsna evaluate oof --job JOB.json` validates externally produced per-fold predictions against an `ExperimentSpec`, a canonical `FoldPlanManifest`, and complete `LabelRecord` coverage. It requires one OOF `PredictionArtifact` per declared model candidate and fold. Each payload is a content-addressed JSON artifact with this shape:

```json
{
  "schema_version": 1,
  "class_names": ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"],
  "rows": [
    {"StudyInstanceUID": "1.2.840.113619.2.55.3.604688433.123.1599472500.467", "scores": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3]}
  ]
}
```

Rows are keyed only by `StudyInstanceUID`; content `study_id` values are not accepted as row identifiers. Scores follow the exact `TARGETS` order and are probabilities in `[0, 1]`. `schemas/oof-predictions.schema.json` describes this payload. Each row must belong to the artifact's held-out fold, and every study in the plan must have exactly one prediction for each candidate. Inference artifacts, unknown studies, duplicate rows, missing studies, invalid target order, dataset/fold mismatches, leakage, and artifact hash mismatches fail with diagnostics.

The job JSON contains an `experiment` object serialized from `ExperimentSpec`, a `fold_plan` path, a `labels` path containing a list of serialized `LabelRecord`s (or `{"labels": [...]}`), and `prediction_artifacts`, either a list of serialized `PredictionArtifact`s or a JSON path containing that list. Artifact URIs in references must be readable from the command's working directory. An optional `output_dir` in the job or `--output-dir` stores outputs; the CLI flag takes precedence.

The evaluator computes ROC AUC by the deterministic average-rank definition (`rank_auc_v1`), including half-credit for tied scores. A target with no observed labels has AUC `null` with `NO_SUPERVISION`; a target with no positives has `NO_POSITIVES`; a target with no negatives has `NO_NEGATIVES`. Unknown labels are masked and never treated as negatives. Macro AUC is left `null` unless all 12 target AUCs are defined, matching the existing `Evaluation` contract. Soft labels are rejected for ROC AUC. Pairwise comparison deltas are `null` whenever either candidate lacks that metric; comparison artifacts include each candidate's supervised, positive, and negative counts per target.

Outputs are content-addressed JSON artifacts: the complete OOF aggregate, target AUC table, pairwise comparison report, and one existing `Evaluation` contract per candidate. Input identity includes the experiment, fold plan, label record IDs, prediction artifact IDs, and metric implementation version. The CLI response records aggregation rows per second and peak Python traced memory separately from deterministic artifacts.

No training engine or model architecture is required. Adapters may create `PredictionArtifact`s from any external inference system; this evaluator consumes the established artifact contract and does not require a `TrainingResult`.
