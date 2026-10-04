# kaggle-RSNA

Architecture foundation for the **RSNA Knee Abnormality Detection** MRI classification
competition. It defines reproducible dataset, fold-plan, model-candidate, training,
prediction, evaluation, and submission-artifact contracts, and includes a metadata-only
synthetic data smoke. It does not implement or train a neural model or provide a final
Kaggle notebook.

## Quick start

Python 3.11 or newer; DICOM metadata indexing uses `pydicom`.

```bash
python -m unittest discover -s tests -v
python -m rsna configs/experiments/smoke.toml
python -m rsna synthetic smoke
```

The CLI uses an explicitly synthetic mock provider and synthetic target names. Its
checkpoint URI and metrics are placeholders for contract smoke tests, not model results.
Real datasets and predictions use the centralized official target order.

Ray is optional: `pip install -e '.[ray]'`. Cloud and Ray providers adapt injected execution
boundaries and do not provision or assume a particular vendor or cluster.

See [architecture](docs/ARCHITECTURE.md), [target and label contracts](docs/TARGETS_AND_LABELS.md),
and [migration notes](docs/MIGRATION.md).

Dataset metadata can be indexed without decoding image pixels:

```bash
python -m rsna data-index --input /data/rsna --output artifacts/dataset-index
```

See [dataset indexing](docs/DATA_INDEX.md) for discovery rules, manifest loading, and
the `DatasetVersion` binding.

Run `python -m rsna synthetic smoke --keep` to retain generated DICOMs and audit artifacts.
See [synthetic smoke documentation](docs/SYNTHETIC_SMOKE.md).
