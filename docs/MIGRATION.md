# Domain correction

The foundation is aligned to the RSNA Knee Abnormality Detection MRI classification task.
Its contracts cover DatasetVersion, FoldPlan, ModelCandidate, ExperimentSpec, TrainingJob,
TrainingResult, CheckpointArtifact, PredictionArtifact, Evaluation, and SubmissionArtifact.

Generic stable hashing, canonical JSON, artifact manifests, provider and cloud boundaries,
telemetry, and configuration-driven smoke execution remain. The former domain-specific
workflow and submission packaging have been removed rather than carried forward under new
names.

No actual model, pixel decoder, official class mapping, patient fold assignment, or Kaggle
notebook is included. DICOM header discovery and metadata manifests are now available; see
`docs/DATA_INDEX.md`. The example class names, fold declaration, checkpoint, and metrics are
explicitly synthetic placeholders.

DatasetVersion schema 1 payloads without target-registry fields retain their original identity
and can still be loaded. Payloads already bound to the target registry also keep their schema 1
identity. Newly created DatasetVersion records use schema 2, which binds the official target
registry into identity material; this makes the transition explicit without rewriting stored IDs.
