# Dataset CLI

The CLI reads the canonical `rsna.data.DatasetIndex` manifest without decoding DICOM pixel arrays. `data index` discovers metadata and writes that manifest. The earlier Issue 1 `data-index` spelling remains a compatibility alias to the same command.

```bash
python -m rsna --help
python -m rsna data index --input /data/rsna --output artifacts/dataset-index --json
python -m rsna data summary --manifest artifacts/dataset-index/manifest.json --json
python -m rsna data validate --manifest artifacts/dataset-index/manifest.json
python -m rsna data validate --manifest artifacts/dataset-index/manifest.json --level full --json
python -m rsna data inspect-study --manifest manifest.json --study-id 1.2.840...
python -m rsna data inspect-series --manifest manifest.json --series-id 1.2.840...
python -m rsna data inspect-slice --manifest manifest.json --sop-id 1.2.840...
python -m rsna data inspect-manifest --manifest manifest.json --json
python -m rsna data sample --manifest manifest.json --entity series --count 20 --seed 42
python -m rsna data stats --manifest manifest.json --plane sagittal --laterality left --json
python -m rsna artifact validate artifacts/dataset-index/manifest.json
```

Every inspection command supports `--format human|json` (or `--json`). Validation reports distinguish errors, warnings, and informational findings. Exit status is `0` for a pass, `1` for validation failure or a missing entity, `2` for invalid CLI usage, and `3` for unreadable or corrupt input. JSON results go to stdout; operational errors go to stderr. `--debug` adds a traceback for operational errors.

A missing DatasetVersion binding is reported as unavailable; counts absent from the manifest are omitted instead of guessed. Optional DICOM metadata produces warnings and does not fail validation unless `--warnings-as-errors` is set. Remote references that cannot be read locally are marked `UNVERIFIED`.

An optional TOML config can provide defaults in a `[cli]` section, such as `output_format = "json"`, `manifest = "manifest.json"`, and `validation_level = "basic"`. Explicit command line flags take precedence. `--quiet` suppresses informational logging; result output remains available for automation.

The original smoke invocation remains supported:

```bash
python -m rsna configs/experiments/smoke.toml
```

## Fold plans

Fold-plan operations are available under `python -m rsna folds`:

```text
generate  create, validate, summarize, and optionally save assignments
validate  validate plan identity, dataset match, coverage, and leakage
inspect   show plan metadata, fold statistics, or study/patient lookup
stats     report per-fold counts and balance diagnostics
diff      compare assignments and detect semantic split equality
export    write assignments as CSV or JSON
import    validate external assignments and bind them to a dataset
lock      mark a plan as frozen for experiment reuse
```

Examples and the recommended freeze-and-reuse workflow are in [FOLDS.md](FOLDS.md). Add `--json` to any folds command for machine-readable output. Fold commands return `0` for success, `1` when a validation gate fails, and `2` for invalid command/configuration or output conflicts. With `--json`, the command writes one JSON document to stdout; warnings and errors go to stderr. Fold commands read the existing dataset index manifest and do not scan DICOM files or load pixels.

## CNN-224 baseline

The CNN-224 run reuses a saved DatasetVersion, DatasetIndex, canonical FoldPlan, and LabelRecords. Copy and edit `configs/models/cnn224-baseline.toml` to point to those files, the DICOM root, and an explicit `StudyInstanceUID` to `SeriesInstanceUID` map. Then run:

```bash
python -m rsna baseline cnn224 smoke --json
python -m rsna baseline cnn224 run --config configs/models/cnn224-baseline.toml --json
```

The smoke is CPU-only and synthetic. The real run requires five folds and the optional PyTorch training dependencies; its configured `device = "cuda:0"` is strict and fails if CUDA is unavailable. The command validates the existing LeakageGuard boundary, trains one model per held-out fold with `TrainingEngine`, reloads each fold checkpoint, emits validation-only OOF `PredictionArtifact`s, and calls the OOF Evaluation Engine. Details and input policies are in [CNN224_BASELINE.md](CNN224_BASELINE.md).
