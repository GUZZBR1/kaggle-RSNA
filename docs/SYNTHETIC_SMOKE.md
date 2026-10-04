# Synthetic end-to-end data smoke

## Why it exists

This CPU-only check provides a runnable integration proof for the metadata and dataset
preparation path without downloading competition data. It creates small synthetic DICOM
series and labels, then exercises discovery, the metadata cache, geometry ordering,
orientation and laterality, slice selection, the official TargetRegistry order with
synthetic labels, patient-grouped folds,
leakage checks, and the `DatasetVersion`/`FoldPlan` contracts. It does not train a model.

## Run it

```bash
python -m rsna synthetic smoke
python -m rsna synthetic smoke --seed 42
python -m rsna synthetic smoke --keep
python -m rsna synthetic smoke --keep ./debug-run --seed 42
```

The default data has eight patients, two studies per patient, two series per study, and
24–32 small slices per series across sagittal, coronal, and axial planes and both
laterality values. `--keep` writes `raw/`, `index/`, `folds/`, `reports/`,
`prepared/`, and `smoke-summary.json` under `synthetic-smoke/` (or the supplied directory).
Without `--keep`, temporary files are removed when the command exits. Generated DICOMs
are runtime artifacts and are not committed. A retained output directory must be empty so
old DICOMs cannot mix with a new seed; choose a new directory for each kept run.

The final JSON reports `status: READY`, material IDs, entity counts, cache behavior, label
missingness, and named checks. `READY` means the metadata-only dataset artifacts passed
these infrastructure checks; it does not mean a model has been trained or clinically
validated.

## Scenarios covered

- The default happy path indexes valid DICOMs, resolves sagittal/coronal/axial orientation
  and LEFT/RIGHT metadata, exercises the deliberately misleading `InstanceNumber`, selects
  with `physical_span / 24`, generates five patient-grouped folds, and reports zero
  patient/study/series/slice leakage.
- The same dataset runs `uniform / 16` and `physical_span / 32` to check preprocessing IDs.
  The raw source identity remains the same while preprocessing-bound dataset identities
  change.
- The cache runs cold then warm. The warm run must reuse parsed metadata and reparse zero
  files.
- A fold-seed change preserves the dataset identity and changes the `FoldPlan` ID.
- JSON index and assignment artifacts are reloaded and selection, labels, fold assignments,
  target order, and leakage results are compared.
- The data is copied to a second absolute output directory; source, dataset, fold, and
  selection identities must remain the same.

## Expected failures and fallback

`--inject` creates one intentional fault and succeeds only when the expected detector
finds it:

```bash
python -m rsna synthetic smoke --inject patient-leakage
python -m rsna synthetic smoke --inject duplicate-sop
python -m rsna synthetic smoke --inject orientation-conflict
python -m rsna synthetic smoke --inject missing-position
python -m rsna synthetic smoke --inject missing-metadata
python -m rsna synthetic smoke --inject spacing-irregular
python -m rsna synthetic smoke --inject corrupted-cache
```

Detected faults produce `status: EXPECTED_FAILURE` and an explanation in JSON, with exit
code 0. Missing positions exercise a documented uniform fallback and warning rather than
claiming physical-span ordering.

## Validating future changes

Run the command and the integration tests when changing discovery, cache, geometry,
orientation/laterality, slice selection, targets, folds, leakage, or preprocessing identity.
This smoke should become a required pull-request gate before substantial changes to those
areas. The default is deliberately metadata-only and CPU-based; it does not depend on a
GPU, network access, Kaggle data, or neural-network packages.

## Limits

The target schema uses the repository's official TargetRegistry; label values and their
missingness are synthetic. Synthetic pixel arrays are tiny
constant-valued fixtures used only to make valid DICOM image instances; this pipeline reads
headers and does not perform image preprocessing. The current `PreparedDataset` equivalent
is a JSON artifact combining the existing `DatasetVersion`, `FoldPlan`, target schema,
preprocessing identity, and leakage report.
