# Reproducible fold plans

Fold assignment is a versioned dataset artifact. Generate the plan before comparing
models, and reuse the same saved file for every experiment in that comparison.
Changing folds changes the validation population and makes scores incomparable.

## Identity and grouping

`DatasetVersion` binds a plan to the dataset version. The folds command accepts a
`dataset_version_id` at the top level or via `--dataset-version-id`; when neither
exists it derives a stable SHA-256 dataset identity from the manifest's `index_id`
and canonical study records. PatientID is the preferred indivisible unit. Studies
without PatientID fall back to study grouping and are called out in warnings. A
reused SeriesInstanceUID links its studies into one indivisible group. Conflicting
PatientID values within one study fail closed.

The default `group` algorithm sorts input records canonically, orders larger groups
first, and places each group in the currently smallest fold. Seeded SHA-256 tie
breakers make the assignment independent of input order and Python's random module.
The optional `multilabel_group_stratified` strategy greedily minimizes normalized
positive and negative mass while keeping whole patient/study groups intact. Labels
are normalized by the central `TargetRegistry`; target aliases map to canonical names
and the manifest records the official target order and registry identity. Hard
`LabelRecord`s accept 0, 1, or missing values. Soft `LabelRecord`s require the
explicit `allow_soft` opt-in, contribute their probabilities to prevalence and
stratification, and remain distinguishable in fold statistics and plan identity.
Missing labels are counted separately and are never imputed. This is an approximation
and is not enabled by default.

## Lifecycle

```text
DatasetVersion
      ↓
group extraction (Patient → Study → Series)
      ↓
fold strategy
      ↓
assignment manifest + statistics
      ↓
leakage validation
      ↓
locked FoldPlan for experiment reuse
```

The manifest's `fold_plan_id` hashes the dataset binding, strategy, seed, grouping,
actual study and group assignments, statistics, warnings, and lock state. It excludes
output path, host, and timestamp. Loading verifies this content hash and never
regenerates assignments. `--locked` makes the saved plan immutable through the save
API; locked manifests cannot be overwritten with a different plan.

Applying a plan checks DatasetVersion and rejects new studies under the default
`strict` policy. Missing expected studies are returned as `application_warnings` so
the caller can see the incomplete dataset. Creating an extension requires generating
and saving a new plan explicitly.

## CLI

```bash
python -m rsna folds \
  --dataset-manifest artifacts/dataset-index.json \
  --output artifacts/folds/folds-v1.json \
  --n-folds 5 --strategy group --seed 42
```

When study labels are available, provide a JSON object keyed by study ID whose values
map canonical target names or registered aliases to `0`, `1`, or `null`, and select
`--strategy multilabel-group-stratified`. To pass soft labels, use serialized
`LabelRecord` objects with `label_type: "soft"` and `allow_soft: true`. The report
includes per-fold and total study, patient, series, and group counts; target positive,
negative, soft, missing, and prevalence counts; size and prevalence imbalance
diagnostics; and warnings for rare targets.

## Assignment management commands

The extended CLI uses the canonical Issue 6 FoldPlanManifest and generator, and calls the Issue 7 Leakage Guard for canonical index validation. It adds dataset-aware validation, inspect, stats, diff, CSV/JSON export, CSV import, and locking.

Example: python -m rsna folds generate --dataset-manifest artifacts/dataset-index/manifest.json --output artifacts/folds/foldplan-v1.json --n-folds 5 --strategy group --seed 42
