# Domain correction

The foundation is aligned to the RSNA Knee Abnormality Detection MRI classification task.
Its contracts cover DatasetVersion, FoldPlan, ModelCandidate, ExperimentSpec, TrainingJob,
TrainingResult, CheckpointArtifact, PredictionArtifact, Evaluation, and SubmissionArtifact.

Generic stable hashing, canonical JSON, artifact manifests, provider and cloud boundaries,
telemetry, and configuration-driven smoke execution remain. The former domain-specific
workflow and submission packaging have been removed rather than carried forward under new
names.

No actual model, DICOM pipeline, official class mapping, patient fold assignment, or Kaggle
notebook is included. The example class names, fold declaration, checkpoint, and metrics are
explicitly synthetic placeholders.
