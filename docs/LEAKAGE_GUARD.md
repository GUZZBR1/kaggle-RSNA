# RSNA Leakage Guard

The guard is a dataset quality and experiment-validity gate. It does not train models or decode DICOM pixels.

```text
DatasetIndex / dataset manifest     FoldPlan / explicit split collections
                \                                  /
                 └────── Leakage Validator ────────┘
                                  ↓
                           LeakageReport
                         ↙             ↘
                  PASS → training     FAIL → blocked
```

## Calling the validator

```python
from rsna.leakage import require_valid_leakage_report, validate_leakage

report = validate_leakage(dataset_index, fold_plan, policy="strict")
validated_report_id = require_valid_leakage_report(report)
# A TrainingJob may record validated_leakage_report_id for provenance.
```

`validate_leakage(dataset_index, fold_plan)` accepts the existing `FoldPlan` contract and a sequence of mapping records. The records are the minimal adapter when a richer `DatasetIndex` is not available. Recognized identifiers include `patient_id`/`PatientID`, `study_uid`/`StudyInstanceUID`, `series_uid`/`SeriesInstanceUID`, `sop_uid`/`SOPInstanceUID`, `file_hash`/`sha256`, and `fold_id`/`split`/`partition`. DICOM metadata may live in a nested `metadata` mapping. File paths are evidence only; they are never identity keys.

The existing `FoldPlan` stores its dataset binding and fold IDs, but does not contain per-study assignments. Supply assignments separately as an identity-to-fold mapping, or put the resolved `fold_id` on each record. JSON CLI input may use `{ "fold_plan": { ... }, "assignments": { "study-uid": "fold_0" } }`.

Without a FoldPlan, validate named collections directly:

```python
from rsna.leakage import validate_split
report = validate_split(train_records, validation_records,
                        test_records=test_records, holdout_records=holdout_records)
```

## Policies and outcomes

`strict` is the default. Reports with error-severity issues have `passed == False`; the CLI exits with status 1 and `require_valid_leakage_report` raises `LeakageValidationError` carrying the report. Training code should gate on the report before creating or running jobs. `audit` returns the complete report without raising; its `passed` value still reflects whether critical leakage was found.

Critical by default: patient, study, series, and slice cross-fold identity; duplicate file hashes across splits; conflicting PatientIDs for one Study UID; unbound series/slices; unknown groups/folds; dataset version mismatch; inconsistent parent relationships; and cross-split source lineage or label provenance. Repeated UIDs and repeated file hashes within one fold are warnings for audit. Metadata near-duplicates are warnings when callers provide `metadata_identity`; the guard does not guess identities from arbitrary metadata fields.

| Issue type | Meaning |
| --- | --- |
| `PATIENT_CROSS_FOLD` | One trusted PatientID appears in multiple folds/splits. |
| `STUDY_CROSS_FOLD` | One StudyInstanceUID appears in multiple folds/splits. |
| `SERIES_CROSS_FOLD` | One SeriesInstanceUID appears in multiple folds/splits. |
| `SLICE_CROSS_FOLD` | One SOPInstanceUID appears in multiple folds/splits. |
| `DUPLICATE_FILE_HASH` | Identical file bytes occur across splits (error) or within one split (warning). |
| `CONFLICTING_PATIENT_ID` | A StudyInstanceUID is associated with multiple PatientIDs. |
| `UNBOUND_SERIES` / `UNBOUND_SLICE` | Explicit series/slice entity lacks its required parent relation. |
| `LABEL_PROVENANCE_LEAKAGE` | A label's declared source split differs from the consuming record's split. |

Every report includes schema version, validator version, dataset and fold-plan IDs, policy, issue details, entity statistics, timestamp metadata, and deterministic `report_id`. The ID hashes normalized issue/config/material data and excludes timestamp, host and paths.

## CLI

```bash
python -m rsna leakage-check \
  --dataset-manifest artifacts/index.json \
  --fold-plan artifacts/folds.json \
  --output artifacts/leakage-report.json
```

The dataset manifest is JSON with `records` (or `entities`) and optional `dataset_version_id`. Fold assignments may be included on each record or in the FoldPlan wrapper described above. `--policy audit` writes the report and returns zero even when it contains errors; the report's `passed` remains false. `--policy strict` returns nonzero when any error is present.

## Records and derived data

For slices, include parent `series_uid`; include `study_uid` on series records or slice rows when available. Derived records may supply `source_entity_ids` and `parent_artifact_ids`; shared lineage across splits is treated as an error. Label producers can supply `label_provenance.source_fold_id` (or `source_split`) for a source/consumer split check. Timestamp is report metadata only. `TrainingJob.validated_leakage_report_id` is optional to preserve the synthetic foundation workflow; production orchestration should populate it after passing the strict gate.

## Performance and limits

Identity lookup, lineage lookup and issue aggregation use hash maps/sets and are O(n) expected time and O(n) memory for n records. Hash calculation is intentionally outside this validator; callers provide content hashes. Near-duplicate metadata requires a caller-defined stable `metadata_identity`, because broad fuzzy comparison risks conflating unrelated studies. This issue does not generate folds, decode DICOM, inspect image pixels, extract labels, or implement pseudo-labeling.
