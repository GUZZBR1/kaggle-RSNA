# Architecture

This repository is a foundation for the RSNA Knee Abnormality Detection competition, a
multi-label MRI classification workflow and removes the unrelated prior domain.
No model architecture, DICOM reader, real fold generation, heavy training, or Kaggle
notebook is implemented here.

## Data flow

```text
ExperimentSpec -> DatasetVersion -> FoldPlan -> ModelCandidate -> TrainingJob
  -> CheckpointArtifact -> PredictionArtifact -> Evaluation -> SubmissionArtifact
```

Contracts are immutable, schema-versioned records with deterministic SHA-256 identities.
URIs are locations and do not alter content identities. Dataset manifests and preprocessing
configuration are identity-bearing. FoldPlan records a reproducible declaration only; it
 does not split patient data. Candidate contracts describe configuration without bundling a
network implementation.

`TrainingJob` carries candidate, dataset, fold, CPU/GPU resources, and configuration.
Providers expose submission and result retrieval: LocalProvider is synchronous and suited
to smoke tasks; MockProvider is explicitly synthetic; RayProvider uses per-job resource
requests on an existing Ray runtime; CloudProvider delegates to injected callbacks. None
assumes a specific cloud vendor or provisions infrastructure.

`TrainingResult` records status, metrics, execution/provider provenance, and a bound
checkpoint on success. The `TargetRegistry` defines the official twelve submission
targets, order, aliases, and target schema version. `LabelRecord` encodes hard or soft
per-study labels, missingness mask, and provenance. `PredictionArtifact`, `Evaluation`,
DatasetVersion, and SubmissionArtifact validate against the same registry order. Evaluation
requires one AUC entry per target and derives or validates macro AUC; numerical metric
computation is intentionally outside this contract layer.
`SubmissionBuilder` is only a future interface for producing a SubmissionArtifact.

The content-addressed JSON artifact store verifies payload SHA-256 on read. Telemetry is
provider-neutral. Configuration and artifact schemas are versioned. The smoke config
explicitly opts into synthetic placeholder names. See
[TARGETS_AND_LABELS.md](TARGETS_AND_LABELS.md) for the official source, canonical order,
missing-label policy, and provenance contract.

## Dataset discovery

The independent `rsna.data` layer discovers DICOM headers, records Study/Series/Slice
metadata, and writes a deterministic JSON manifest. It does not decode pixels or perform
anatomical ordering; see [DATA_INDEX.md](DATA_INDEX.md).

## DICOM metadata audit

`rsna.data` provides metadata-only orientation and laterality descriptions. DICOM
ImageOrientationPatient geometry drives anatomical plane classification; structured
Laterality/ImageLaterality fields take precedence over conservative text-token fallback.
Conflicting evidence remains ambiguous and is retained with warnings and confidence in
serializable provenance. Normalization defaults to `preserve_native`; the optional
left-canonical mode describes a future operation but does not change pixels. Any future
applied transform must be included in `DatasetVersion.preprocessing` so dataset identity
reflects the changed preprocessing. See [ORIENTATION_LATERALITY.md](ORIENTATION_LATERALITY.md).
