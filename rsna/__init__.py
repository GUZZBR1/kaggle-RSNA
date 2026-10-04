"""Reproducible experiment infrastructure for Kaggle RSNA classification workflows."""

from .contracts import (
    ArtifactReference,
    DatasetVersion,
    Evaluation,
    ExperimentSpec,
    FoldPlan,
    ModelCandidate,
    PredictionArtifact,
    SubmissionArtifact,
    TrainingJob,
    TrainingResult,
)
from .labels import (LabelRecord, StudyDatasetRecord, StudyMetadata, label_records_to_rows,
                     rows_to_label_records, validate_label_record)
from .targets import (OFFICIAL_TARGETS, TARGETS, TARGET_REGISTRY, TargetRegistry,
                      validate_prediction_columns, validate_submission_columns, validate_targets)

__all__ = [
    "ArtifactReference", "DatasetVersion", "Evaluation", "ExperimentSpec", "FoldPlan", "ModelCandidate",
    "PredictionArtifact", "SubmissionArtifact", "TrainingJob", "TrainingResult",
    "LabelRecord", "StudyMetadata", "StudyDatasetRecord", "validate_label_record", "OFFICIAL_TARGETS",
    "rows_to_label_records", "label_records_to_rows",
    "TARGETS", "TARGET_REGISTRY", "TargetRegistry", "validate_targets",
    "validate_prediction_columns", "validate_submission_columns",
]
"""Reproducible workflow contracts for RSNA knee MRI classification."""

from .contracts import (ArtifactReference, CheckpointArtifact, DatasetVersion,
    Evaluation, ExperimentSpec, FoldPlan, ModelCandidate, PredictionArtifact,
    SubmissionArtifact, TrainingJob, TrainingResult)
from .preparation import PreparedDataset, PreparationConfig, PreparationError, prepare_dataset

__all__ = ["ArtifactReference", "CheckpointArtifact", "DatasetVersion", "Evaluation",
    "ExperimentSpec", "FoldPlan", "ModelCandidate", "PredictionArtifact",
    "SubmissionArtifact", "TrainingJob", "TrainingResult"]
__all__ += ["PreparedDataset", "PreparationConfig", "PreparationError", "prepare_dataset"]
