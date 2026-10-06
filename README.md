# kaggle-RSNA

Reproducible competition codebase for the **RSNA Knee Abnormality Detection** MRI classification challenge.

The repository now includes the full foundation needed to move from metadata preparation to real model evaluation:

- DICOM discovery/indexing and cache
- geometry, orientation/laterality and slice-selection contracts
- official target/label handling with missing-label masks
- deterministic fold generation and leakage guards
- auditable `PreparedDataset` preparation
- deterministic synthetic end-to-end smoke tests
- reproducible PyTorch Training Engine
- OOF aggregation and per-target/macro AUC evaluation
- a first reproducible **CNN-224** study-level baseline wired to the canonical training/evaluation stack

The current competitive milestone is **not infrastructure work**: it is to run the CNN-224 baseline on the real RSNA data, complete all five folds, and establish the first full real OOF result.

## Current status

| Area | Status |
| --- | --- |
| Data/index/geometry/orientation/selection | READY |
| Targets, labels, folds and leakage | READY |
| PreparedDataset pipeline | READY |
| Synthetic E2E smoke | READY |
| Training Engine | READY |
| OOF Evaluation | READY |
| CNN-224 implementation | READY |
| Real RSNA validation | PENDING |
| Real CNN-224 5-fold OOF | PENDING |
| Competitive ablations/model arms | PENDING |
| Kaggle inference workflow | PENDING |

See [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for the canonical execution order and [`docs/CNN224_BASELINE.md`](docs/CNN224_BASELINE.md) for the baseline contract and commands.

## Quick start

Python 3.11 or newer.

```bash
uv sync --extra training --locked
python -m unittest discover -s tests -v
python -m rsna synthetic smoke
python -m rsna baseline cnn224 smoke --json
```

Dataset metadata can be indexed without decoding image pixels:

```bash
python -m rsna data index --input /data/rsna --output artifacts/dataset-index
python -m rsna data summary --manifest artifacts/dataset-index/manifest.json
```

Fold generation is available with:

```bash
python -m rsna folds
```

For a real CNN-224 run, prepare the canonical dataset/fold/label/series-map inputs and use:

```bash
python -m rsna baseline cnn224 run --config configs/models/cnn224-baseline.toml --json
```

## Canonical architecture

```text
ExperimentSpec
  -> DatasetVersion
  -> FoldPlan
  -> ModelCandidate
  -> TrainingJob
  -> CheckpointArtifact
  -> PredictionArtifact
  -> Evaluation
  -> SubmissionArtifact
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Project status and critical path](docs/PROJECT_STATUS.md)
- [CNN-224 baseline](docs/CNN224_BASELINE.md)
- [Data pipeline](docs/DATA_PIPELINE.md)
- [Dataset indexing](docs/DATA_INDEX.md)
- [DICOM geometry](docs/DICOM_GEOMETRY.md)
- [Orientation/laterality](docs/ORIENTATION_LATERALITY.md)
- [Slice selection](docs/SLICE_SELECTION.md)
- [Targets and labels](docs/TARGETS_AND_LABELS.md)
- [Folds](docs/FOLDS.md)
- [Leakage guard](docs/LEAKAGE_GUARD.md)
- [Synthetic smoke](docs/SYNTHETIC_SMOKE.md)
- [OOF evaluation](docs/OOF_EVALUATION.md)
- [CLI usage](docs/CLI.md)

## Current constraints

- The synthetic smoke proves wiring and contracts; it is not a competition score.
- No real five-fold RSNA OOF result has been established yet.
- Real pixel decoding may require additional pydicom decoder plugins depending on the dataset transfer syntaxes.
- Pretrained model weights must be explicitly supplied and provenance-tracked; the current CNN-224 baseline does not implicitly download weights.
- Cloud/GPU availability is an execution concern, not a reason to fork the training/evaluation contracts.
