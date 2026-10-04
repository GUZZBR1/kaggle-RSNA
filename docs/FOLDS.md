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
positive and known-label count imbalance while keeping whole patient/study groups
intact. Missing labels are excluded from positive and negative counts; no imputation
is performed. It is an approximation and is not enabled by default.

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
map target names to `0`, `1`, or `null`, and select `--strategy multilabel-group-stratified`. The report includes per-fold and total study, patient,
series, and group counts; target positive, negative, missing, and prevalence counts;
size and prevalence imbalance diagnostics; and warnings for rare targets.
