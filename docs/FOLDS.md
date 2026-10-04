# Fold plans

A `FoldPlan` is an experiment input. Reuse the same saved plan for every baseline you intend to compare: results trained on different splits are not directly comparable.

## Recommended workflow

```text
dataset index
   ↓
generate 5-fold
   ↓
validate
   ↓
inspect stats
   ↓
lock
   ↓
use the same FoldPlan for every baseline
```

Example:

```bash
python -m rsna folds generate --dataset-manifest artifacts/dataset-index/manifest.json \
  --output artifacts/folds/foldplan-v1.json --n-folds 5 --strategy group --seed 42
python -m rsna folds validate --fold-plan artifacts/folds/foldplan-v1.json \
  --dataset-manifest artifacts/dataset-index/manifest.json --reproduce
python -m rsna folds inspect --fold-plan artifacts/folds/foldplan-v1.json --dataset-manifest artifacts/dataset-index/manifest.json
python -m rsna folds lock --fold-plan artifacts/folds/foldplan-v1.json
```

Generation refuses to overwrite an existing output. `--force` records an explicit overwrite of an unlocked plan; locked plans refuse overwrite. Dry runs validate and report statistics without writing. The plan identity covers dataset version, strategy, seed, configuration, and assignments, and does not depend on timestamps.

The Issue 8 index-only manifest is accepted without scanning DICOM files. Such a manifest is bound to a deterministic `DatasetVersion` derived from its exact bytes. A manifest wrapper may instead provide a `dataset_version` object (the normal `DatasetVersion` contract) or a `dataset_version_id`. Validation fails when that ID differs from the plan.

Validation checks plan identity/schema, dataset identity, unknown and missing studies, patient leakage, and coverage. Fold statistics include patients, studies, series, slices, and binary label counts when labels are present. Balance reports fold sizes and per-target prevalence ranges and population standard deviations; these interpretable diagnostics are not combined into a quality score. Study assignments imply that each study's series remain together.

Diff reports exact assignment equality and semantic partition equality. Semantic equality treats a consistent permutation of fold labels as the same partition. Imported CSV plans are revalidated against dataset membership and patient leakage before they are saved.

Every command supports `--json`; JSON mode writes one JSON document to stdout and sends errors to stderr. Exit codes: `0` success/valid, `1` validation failure, `2` usage or missing output/conflict, `3` corrupt input/artifact, `4` dataset mismatch.

This CLI manages fold artifacts only. It does not train models, create OOF predictions, tune hyperparameters, augment data, run TTA/ensembles, or create submissions.
