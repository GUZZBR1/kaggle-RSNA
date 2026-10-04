# Canonical data preparation pipeline

The data preparation API connects source discovery, metadata indexing, geometry,
orientation, slice selection, targets, folds, and leakage validation. Its output is
one reproducible reference for a future training consumer. Preparation reads metadata
and labels; it does not train a network or decode and transform image tensors.

```text
RAW RSNA
  ↓
DISCOVERY / INDEX
  ↓
DICOM GEOMETRY
  ↓
ORIENTATION / LATERALITY
  ↓
SLICE SELECTION
  ↓
TARGETS
  ↓
FOLDS
  ↓
LEAKAGE GATE
  ↓
PREPARED DATASET
  ↓
FUTURE TRAINING CONSUMER
```

## Integration baseline

The integration audit was repeated against current `main` (`f0e0938`). It includes
canonical discovery/indexing and SQLite incremental cache (Issues 1 and 8), DICOM
geometry/order (Issue 2), orientation/laterality (Issue 3), slice selection (Issue 4),
official targets/labels (Issue 5), canonical fold generation and hierarchy validation
(Issue 6 and follow-up fixes), leakage validation (Issue 7), metadata inspection CLI
services (Issue 9), and the resumable Training Engine. Issue 11 reuses those APIs and
supplies orchestration plus the final preparation manifest.

| Component | Main status and API | Inputs → outputs | Integration decision |
| --- | --- | --- | --- |
| Dataset/training contracts | Present: `DatasetVersion`, `FoldPlan`, `TrainingJob` | Material config and IDs → immutable contracts | Reuse contracts; bind FoldPlan directly to final DatasetVersion |
| Identity | Present: `rsna.identity.digest`, canonical JSON, freeze helpers | JSON-safe values → stable SHA-256 | Reuse; keep paths and telemetry out of material specs |
| Orientation/laterality | Present: `describe_series_orientation`, `resolve_series_laterality`, `build_orientation_provenance` | Ordered metadata mappings → evidence and declarative normalization plan | Reuse after geometry ordering; do not claim pixel transforms |
| Discovery/index | Present: `discover_dataset`, canonical records and manifests | DICOM root → `DatasetIndex` with Study/Series/Slice records | Reuse discovery and record identities; adapt synthetic fixtures to those records |
| Geometry/order | Present: `order_series_slices` | Series metadata → ordering result, geometry diagnostics and fallbacks | Reuse; selection must consume its ordered slice references |
| Slice selection | Present: `SliceSelector`, `SliceSelectionConfig`, `select_slices` | Geometry `OrderingResult` and selection policy → references, positions, fallback and coverage diagnostics | Reuse after ordering and bind its config/output into preprocessing identity |
| Targets/labels | Present: `TargetRegistry`, `LabelRecord`, CSV conversion and schemas | Registry plus label rows → validated target schema and records | Reuse official registry and label contracts; synthetic registry remains explicit |
| Fold assignment | Present: `generate_fold_plan`, `FoldPlanManifest`, `load_fold_plan`, `save_fold_plan` | Dataset records, grouping/stratification policy, labels and seed → validated assignments and persisted manifest | Reuse generation/cache; adapt its assignments to the training `FoldPlan` bound to final `DatasetVersion` |
| Leakage | Present: `validate_leakage`, `require_valid_leakage_report` | Canonical index/records and fold assignments → version-bound report | Reuse as mandatory real-data gate; only explicit synthetic mode may bypass |
| Cache/artifacts | Present: incremental SQLite index cache, atomic JSON manifest, content-addressed JSON artifact store | Source root → cold build, warm load or incremental refresh | Use cache report modes; write final preparation manifest atomically |
| CLI/schemas | Data inspection/validation, index, folds, slice selection, leakage check, and experiment smoke | Dataset manifests/config → inspectable JSON or human summaries | Preserve existing commands; add `prepare-data`, JSON mode, and PreparedDataset schema |

The only missing service is orchestration plus conversion of the fold generator's
manifest into the existing training `FoldPlan` bound to the final `DatasetVersion`.
Existing discovery, geometry, orientation, selection, targets/labels, fold
generation, leakage, cache, contracts, and identity utilities remain authoritative.

The pipeline reuses `DatasetVersion`, `FoldPlan`, immutable identity helpers, and
`rsna.data` orientation/laterality provenance. Geometry answers where slices are;
orientation describes plane and side; selection chooses among ordered slices.
See [orientation/laterality](ORIENTATION_LATERALITY.md) for the evidence rules.

## Public API and CLI

```python
from rsna.preparation import PreparationConfig, prepare_dataset

config = PreparationConfig.from_toml("configs/data/baseline.toml")
prepared = prepare_dataset(config)

# Future TrainingJobs use these existing contracts.
dataset = prepared.dataset_version
folds = prepared.fold_plan
```

```bash
python -m rsna prepare-data --config configs/data/baseline.toml
python -m rsna prepare-data --config configs/data/baseline.toml --json
```

The human output reports each stage, warnings, and final identities. JSON output
provides machine-readable preparation status and lineage. The existing experiment
smoke command remains separate:

```bash
python -m rsna configs/experiments/smoke.toml
```

## Sources and targets

The explicit metadata-fixture boundary is `source_manifest.json` in the configured
source root. It records Study/Series/Slice metadata, stable source identifiers,
relative file locations, patient grouping, and target maps. Synthetic fixtures
must set `synthetic=true` in both the manifest and config; real fixtures must use
real mode and existing files whose content can be checked. Equivalent input order
does not change material outputs.

For real DICOM directories, preparation reuses main's `discover_dataset` and
`load_or_refresh` APIs and joins `train.csv` or `labels.csv` by `StudyInstanceUID`.
The official `TargetRegistry` is the default and a supplied real registry must
match its canonical order. `pydicom` is already a project dependency. Metadata
discovery reads headers and does not decode pixels or create training tensors.

Targets are validated for count, order, schema, and label values. Class names flow
into the existing `DatasetVersion.class_names` contract in registry order. Future
training and predictions must preserve that order.


The manifest shape is explicit (this shortened synthetic example illustrates one
study; a runnable fold fixture needs at least as many patients as folds):

```json
{
  "schema_version": 1,
  "synthetic": true,
  "studies": [
    {
      "patient_id": "synthetic_patient_01",
      "study_id": "synthetic_study_01",
      "targets": {"class_01": 0, "class_02": 1},
      "series": [
        {
          "series_id": "synthetic_series_01",
          "slices": [
            {
              "slice_id": "synthetic_slice_01",
              "path": "synthetic_study_01/synthetic_series_01/1.dcm",
              "metadata": {
                "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
                "ImagePositionPatient": [0, 0, 1],
                "InstanceNumber": 1,
                "Laterality": "L"
              }
            }
          ]
        }
      ]
    }
  ]
}
```

An optional synthetic `target_registry.json` can declare fixture target names:

```json
{
  "schema_version": 1,
  "target_names": ["class_01", "class_02"],
  "official": false,
  "synthetic": true
}
```

Synthetic target names are allowed only in explicit synthetic mode. Real preparation
uses the canonical official registry and binary `0/1` labels per study; the pipeline
does not fetch or certify source labels. Multi-class severity encoding requires a
future target adapter.

For directory discovery, `labels.csv` or `train.csv` is keyed by `StudyInstanceUID`,
followed by the registry's binary target columns. DICOM metadata must contain
`PatientID`, `StudyInstanceUID`, and `SeriesInstanceUID`. `SOPInstanceUID` supplies
the slice identifier when available. The reader uses `stop_before_pixels=True`.

## Stages and diagnostics

| Stage | Input | Output |
| --- | --- | --- |
| `DISCOVER` | Configured source root and input manifest or DICOM boundary | Canonical source manifest and source identity |
| `INDEX` | Source records | Stable Study/Series/Slice index |
| `GEOMETRY` | Indexed slice metadata | Physical ordering and geometry diagnostics |
| `ORIENTATION` | Ordered series and orientation configuration | Existing plane/laterality/normalization provenance |
| `SELECT` | Ordered series and slice policy | Selected slice references |
| `LABELS` | Source target maps and registry | Validated targets and ordered schema |
| `FOLDS` | Prepared dataset identity, patient groups, fold config | Dataset-bound `FoldPlan` and assignment manifest |
| `LEAKAGE` | Index, selected references, labels, and fold assignments | Leakage report and preparation gate |
| `FINALIZE` | Validated outputs from preceding stages | Versioned prepared artifact with complete lineage |

Every stage records input/output references, status, warnings, elapsed duration,
reuse, and processed record count. Diagnostics are explicit run data, not hidden
global state. Duration and runtime timestamps do not participate in material IDs.

Geometry ordering precedes slice selection. Missing geometry is either rejected
in strict mode or handled by an explicitly reported deterministic fallback.
`physical_span` selection uses physical positions; missing positions produce a
recorded fallback rather than a claim that physical sampling occurred.

The default orientation mode is `preserve_native`. `left_canonical` records a
future reflection plan and participates in preprocessing identity, but does not
apply a pixel flip. A future tensor loader must implement and validate any actual
transform before claiming canonical image orientation.

## Configuration and identity

[The baseline configuration](../configs/data/baseline.toml) centralizes source,
index/cache, geometry strictness, orientation mode, slice strategy/count, target
schema, fold strategy/seed, leakage policy, and artifact output settings.
`reuse_valid_artifacts` defaults to true.

Material identity uses the canonical SHA-256 helpers already used by contracts.
The source manifest anchors dataset lineage; the preprocessing specification
includes orientation, selection policy, target schema, relevant index identity,
fold configuration (including the seed), and the assignment manifest hash.
`DatasetVersion.preprocessing` carries these references. The `FoldPlan` is bound
to exactly that `DatasetVersion.dataset_version_id`; a supplied plan for a different
dataset fails validation. The assignment manifest and fold configuration determine
fold identity. The final preparation lineage joins all those IDs.

| Change | Identity consequence |
| --- | --- |
| Source content or source metadata | Source/index and dependent identities change |
| Slice count, such as 24 → 32 | Preprocessing and dependent dataset/preparation identities change |
| Slice strategy, such as uniform → physical_span | Preprocessing identity changes |
| Orientation mode | Preprocessing identity changes |
| Ordered target schema | Registry, preprocessing, and dataset identities change |
| Fold seed or assignment content | Preprocessing, DatasetVersion, FoldPlan, and preparation lineage identities change |
| Artifact directory or source root relocation with identical relative content | Material identities remain stable |
| Hostname, workers, logging, duration, timestamp | Material identities remain stable |
| Equivalent source record ordering | Material identities remain stable |

The existing contracts keep dataset and fold identities distinct. The pipeline also
records fold seed and assignments in material preprocessing, so changing either
changes preprocessing and DatasetVersion identity as well as the bound FoldPlan
and final preparation reference. Paths locate artifacts, while checksums validate
their contents.

## Reuse, failure, and resume

Existing artifacts are eligible for reuse only when their version, source/config
identity, references, and checksums validate. A corrupt index is detected before
reuse. Cache diagnostics distinguish `cold_build`, `warm_load`, and, when supported,
`incremental_refresh`; a warm load must not claim an incremental refresh. Source
byte checksums are cached while file size and modification time match. Replacing a
file with different bytes while preserving both values can evade this fast path;
remove the source-content cache entry or force a cold rebuild when that replacement
is possible.

Reusing a valid index or fold artifact avoids reconstructing that stage. This is
a bounded resume mechanism, not a workflow engine. Re-running unchanged material
inputs produces the same IDs, while new stage telemetry can differ.

A failed stage records its failure and prevents finalization. Valid artifacts from
previous stages are retained. Official artifacts are published with atomic writes,
so incomplete files do not become valid preparation references. Removing a final
manifest or bypassing a failed gate does not make the underlying data valid.

## Status and leakage gate

The canonical statuses are `READY`, `INVALID`, `FAILED`, and `SYNTHETIC`.
`READY` means real preparation completed with passing checks. Every successful
synthetic run has status `SYNTHETIC`, including runs with no leakage violations.
Synthetic status alone does not imply a leakage bypass; `leakage_bypassed` on the
PreparedDataset records whether an explicit synthetic bypass accepted violations.
The canonical `LeakageReport.passed` value remains unchanged. `INVALID` means
validation or leakage rejected the dataset. `FAILED` means a stage could not complete.
Failures raise `PreparationError` with status and stage diagnostics; they do not
return a valid `PreparedDataset`.

Real preparation cannot finalize when `LeakageReport.passed` is false. Critical
overlap found by the canonical validator blocks real preparation. A bypass is
permitted only through explicit synthetic/test configuration and remains visible
on the result. A passing synthetic end-to-end fixture exercises the preparation API;
it is not evidence of valid medical labels or a trained model.

## Prepared output and lineage

`PreparedDataset` bundles the existing dataset/fold contracts with the dataset
index, target registry, leakage report, preprocessing specification, artifact
references, statistics, stage diagnostics, and final status. The versioned final
artifact records their identities and configuration together, so consumers need
one coherent preparation reference rather than rebuilding stage order themselves.
Reloading verifies serialized IDs and cross-references before use.

```text
source manifest → dataset index → preprocessing + targets + assignments → DatasetVersion
                                                                ↓
                                              assignments → bound FoldPlan
                                                                ↓
                                                 leakage report → PreparedDataset
```

The preparation reference can later join a TrainingJob/checkpoint's existing
`dataset_version_id` and `fold_plan_id` lineage. No checkpoint is created here.
Artifact URIs are execution locations and are excluded from material identity.

## Before the first real training run

Provide an authoritative target registry, complete real labels, stable patient
identifiers, and accessible source files. Review geometry and orientation warnings,
validate the chosen slice policy, and obtain a passing real leakage report. The
metadata pipeline does not validate a future image tensor loader, intensity
processing, applied orientation transforms, or model input shape.

Network architectures, PyTorch training loops, augmentation, OOF inference, TTA,
ensembles, and a final Kaggle notebook are outside this integration. Existing mock
provider outputs remain explicitly synthetic.
