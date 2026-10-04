# Targets and labels

## Official targets and source

Canonical names and positions follow the **Submission File** example on the
[official Kaggle competition page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview):

| Index | Submission column | Accepted descriptive aliases |
|---:|---|---|
| 0 | `ACL` | Anterior Cruciate Ligament; ACL injury |
| 1 | `MCL` | Medial Collateral Ligament; MCL injury |
| 2 | `Medial Meniscus` | Medial Meniscus Tear |
| 3 | `Lateral Meniscus` | Lateral Meniscus Tear |
| 4 | `Medial OA` | Medial Osteoarthritis; Medial-compartment osteoarthritis |
| 5 | `Lateral OA` | Lateral Osteoarthritis; Lateral-compartment osteoarthritis |
| 6 | `PF OA` | Patellofemoral Osteoarthritis; Patellofemoral OA |
| 7 | `Effusion` | Joint Effusion |
| 8 | `Synovitis` | — |
| 9 | `Baker's` | Baker's Cyst; Bakers; Baker Cyst |
| 10 | `Contusion` | Bone Contusion; Bone Bruise |
| 11 | `Fracture` | — |

The competition specifies twelve confidence scores and macro ROC AUC over those
twelve labels. The submission identifier column is `StudyInstanceUID` and the
file is `submission.csv`. The page does not define additional aliases; aliases
above are explicit conveniences for dataset/report terminology. Unknown names
and aliases that collide are rejected.

## Registry to downstream artifacts

`rsna.targets.TARGET_REGISTRY` is the single source of truth. Its target schema
version is independent from artifact schema versions. Indexes are stable and
`target_index`/`target_name` are the only name-position mapping API.

```text
official submission headers
  → TargetRegistry canonical order (schema version 1)
  → LabelRecord.values + mask (same order)
  → DatasetVersion / PredictionArtifact columns (same order)
  → Evaluation AUC keys and macro AUC (same targets)
  → SubmissionArtifact columns (StudyInstanceUID, then targets)
```

## LabelRecord

Each `StudyDatasetRecord` links a `StudyMetadata` object to one `LabelRecord` and
a `DatasetVersion` identity while keeping exam metadata separate from labels.
Each label record describes one study, separate from optional patient/exam metadata.
`values` is canonicalized to all twelve official keys. An unprovided label is
`None` and its corresponding `mask` entry is false; a known negative is `0`
and its mask is true. Missing labels are never filled with zero. Partial labels
are allowed by default and can be disabled with `allow_partial=False`.

Hard labels accept only `0`, `1`, or `None`. Soft labels accept finite numbers
from 0 through 1 and require `allow_soft=True`. Label type applies to the full
record. Provenance is explicit (`official_gold`, `report_regex`, `report_llm`,
`pseudo_label`, or `manual_review`); this contract does not implement extraction.
JSON serialization includes canonical targets, values, mask, provenance, label
type, and target schema version. `to_row()` provides lightweight CSV/DataFrame
interop without adding pandas as a dependency.

## Synthetic data

Real `DatasetVersion`, `PredictionArtifact`, and `Evaluation` instances must use
the official registry order. Custom names, including `class_01` placeholders,
are accepted only when `synthetic=True`; that flag is included in artifact
identity hashes. Dataset identity also includes target schema version.

## JSON schemas

`schemas/target-registry.schema.json`, `schemas/label-record.schema.json`,
`schemas/study-metadata.schema.json`, `schemas/study-dataset-record.schema.json`, and
the updated DatasetVersion, PredictionArtifact, Evaluation, and
SubmissionArtifact schemas describe these serialized contracts.
