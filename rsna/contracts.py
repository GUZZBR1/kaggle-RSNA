"""Versioned contracts for dataset, folds, training, prediction and evaluation provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import fmean
from typing import Any, Mapping

from .identity import digest, freeze_json, jsonable
from .targets import (TARGETS, TARGET_REGISTRY_ID,
                      TARGET_SCHEMA_VERSION, validate_targets)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetVersion:
    name: str
    version: str
    source_manifest_sha256: str
    preprocessing_version: str
    class_names: tuple[str, ...]
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    uri: str = ""
    schema_version: int = SCHEMA_VERSION
    dataset_version_id: str = ""
    synthetic: bool = False
    target_schema_version: int = TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.name, "dataset name")
        _required_text(self.version, "dataset version")
        _required_text(self.preprocessing_version, "preprocessing version")
        _sha256(self.source_manifest_sha256, "source_manifest_sha256")
        _unique_names(self.class_names, "class_names")
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic must be a boolean")
        validate_targets(self.class_names, allow_synthetic=self.synthetic)
        _target_schema(self.target_schema_version)
        _schema(self.schema_version)
        object.__setattr__(self, "class_names", tuple(self.class_names))
        object.__setattr__(self, "preprocessing", freeze_json(self.preprocessing))
        expected = digest({"schema_version": self.schema_version, "name": self.name,
                           "version": self.version,
                           "source_manifest_sha256": self.source_manifest_sha256,
                           "preprocessing_version": self.preprocessing_version,
                           "preprocessing": self.preprocessing,
                           "class_names": self.class_names, "synthetic": self.synthetic,
                           "target_schema_version": self.target_schema_version,
                           "target_registry_id": TARGET_REGISTRY_ID})
        _match_id(self.dataset_version_id, expected, "dataset_version_id")
        object.__setattr__(self, "dataset_version_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version,
                "source_manifest_sha256": self.source_manifest_sha256,
                "preprocessing_version": self.preprocessing_version,
                "class_names": list(self.class_names),
                "preprocessing": jsonable(self.preprocessing), "uri": self.uri,
                "synthetic": self.synthetic, "target_schema_version": self.target_schema_version,
                "target_registry_id": TARGET_REGISTRY_ID,
                "schema_version": self.schema_version,
                "dataset_version_id": self.dataset_version_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetVersion":
        _verify_target_registry(data)
        payload = dict(data)
        payload.pop("target_registry_id")
        return cls(**payload)


@dataclass(frozen=True)
class FoldPlan:
    dataset_version_id: str
    strategy: str
    fold_ids: tuple[str, ...]
    random_state: int
    version: str = "1"
    configuration: Mapping[str, Any] = field(default_factory=dict)
    assignment_manifest_sha256: str | None = None
    synthetic: bool = False
    schema_version: int = SCHEMA_VERSION
    fold_plan_id: str = ""

    def __post_init__(self) -> None:
        _sha256(self.dataset_version_id, "dataset_version_id")
        _required_text(self.strategy, "fold strategy")
        _required_text(self.version, "fold plan version")
        _unique_names(self.fold_ids, "fold_ids")
        if type(self.random_state) is not int or self.random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        if self.assignment_manifest_sha256 is not None:
            _sha256(self.assignment_manifest_sha256, "assignment_manifest_sha256")
        _schema(self.schema_version)
        object.__setattr__(self, "fold_ids", tuple(self.fold_ids))
        object.__setattr__(self, "configuration", freeze_json(self.configuration))
        expected = digest({"schema_version": self.schema_version,
                           "dataset_version_id": self.dataset_version_id,
                           "strategy": self.strategy, "version": self.version,
                           "fold_ids": self.fold_ids, "random_state": self.random_state,
                           "configuration": self.configuration,
                           "assignment_manifest_sha256": self.assignment_manifest_sha256,
                           "synthetic": self.synthetic})
        _match_id(self.fold_plan_id, expected, "fold_plan_id")
        object.__setattr__(self, "fold_plan_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"dataset_version_id": self.dataset_version_id, "strategy": self.strategy,
                "fold_ids": list(self.fold_ids), "random_state": self.random_state,
                "version": self.version, "configuration": jsonable(self.configuration),
                "assignment_manifest_sha256": self.assignment_manifest_sha256,
                "synthetic": self.synthetic, "schema_version": self.schema_version,
                "fold_plan_id": self.fold_plan_id}


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    model_configuration: Mapping[str, Any]
    training_configuration: Mapping[str, Any] = field(default_factory=dict)
    implementation_sha256: str | None = None
    implementation_uri: str = ""
    schema_version: int = SCHEMA_VERSION
    model_candidate_id: str = ""

    def __post_init__(self) -> None:
        _required_text(self.name, "candidate name")
        if self.implementation_sha256 is not None:
            _sha256(self.implementation_sha256, "implementation_sha256")
        _schema(self.schema_version)
        object.__setattr__(self, "model_configuration", freeze_json(self.model_configuration))
        object.__setattr__(self, "training_configuration", freeze_json(self.training_configuration))
        expected = digest({"schema_version": self.schema_version, "name": self.name,
                           "model_configuration": self.model_configuration,
                           "training_configuration": self.training_configuration,
                           "implementation_sha256": self.implementation_sha256})
        _match_id(self.model_candidate_id, expected, "model_candidate_id")
        object.__setattr__(self, "model_candidate_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "model_configuration": jsonable(self.model_configuration),
                "training_configuration": jsonable(self.training_configuration),
                "implementation_sha256": self.implementation_sha256,
                "implementation_uri": self.implementation_uri,
                "schema_version": self.schema_version,
                "model_candidate_id": self.model_candidate_id}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    dataset_version_id: str
    fold_plan_id: str
    model_candidate_ids: tuple[str, ...]
    fold_ids: tuple[str, ...]
    class_names: tuple[str, ...]
    resources: Mapping[str, Any] = field(default_factory=lambda: {"cpus": 1, "gpus": 0})
    training_configuration: Mapping[str, Any] = field(default_factory=dict)
    evaluation_policy: Mapping[str, Any] = field(default_factory=lambda: {"primary_metric": "macro_auc"})
    schema_version: int = SCHEMA_VERSION
    experiment_id: str = ""
    synthetic: bool = False
    target_schema_version: int = TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.name, "experiment name")
        _sha256(self.dataset_version_id, "dataset_version_id")
        _sha256(self.fold_plan_id, "fold_plan_id")
        _unique_sha256(self.model_candidate_ids, "model_candidate_ids")
        _unique_names(self.fold_ids, "fold_ids")
        _unique_names(self.class_names, "class_names")
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic must be a boolean")
        validate_targets(self.class_names, allow_synthetic=self.synthetic)
        _target_schema(self.target_schema_version)
        _schema(self.schema_version)
        _validate_resources(self.resources)
        object.__setattr__(self, "model_candidate_ids", tuple(self.model_candidate_ids))
        object.__setattr__(self, "fold_ids", tuple(self.fold_ids))
        object.__setattr__(self, "class_names", tuple(self.class_names))
        object.__setattr__(self, "resources", freeze_json(self.resources))
        object.__setattr__(self, "training_configuration", freeze_json(self.training_configuration))
        object.__setattr__(self, "evaluation_policy", freeze_json(self.evaluation_policy))
        expected = digest({"schema_version": self.schema_version, "name": self.name,
                           "dataset_version_id": self.dataset_version_id,
                           "fold_plan_id": self.fold_plan_id,
                           "model_candidate_ids": self.model_candidate_ids,
                           "fold_ids": self.fold_ids, "class_names": self.class_names,
                           "synthetic": self.synthetic,
                           "target_schema_version": self.target_schema_version,
                           "target_registry_id": TARGET_REGISTRY_ID,
                           "resources": self.resources,
                           "training_configuration": self.training_configuration,
                           "evaluation_policy": self.evaluation_policy})
        _match_id(self.experiment_id, expected, "experiment_id")
        object.__setattr__(self, "experiment_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dataset_version_id": self.dataset_version_id,
                "fold_plan_id": self.fold_plan_id,
                "model_candidate_ids": list(self.model_candidate_ids),
                "fold_ids": list(self.fold_ids), "class_names": list(self.class_names),
                "synthetic": self.synthetic,
                "target_schema_version": self.target_schema_version,
                "target_registry_id": TARGET_REGISTRY_ID,
                "resources": jsonable(self.resources),
                "training_configuration": jsonable(self.training_configuration),
                "evaluation_policy": jsonable(self.evaluation_policy),
                "schema_version": self.schema_version, "experiment_id": self.experiment_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentSpec":
        _verify_target_registry(data)
        payload = dict(data)
        payload.pop("target_registry_id")
        return cls(**payload)


@dataclass(frozen=True)
class TrainingJob:
    experiment_id: str
    dataset_version_id: str
    fold_plan_id: str
    model_candidate_id: str
    fold_id: str
    random_state: int
    resources: Mapping[str, Any]
    configuration: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    training_job_id: str = ""

    def __post_init__(self) -> None:
        for field_name in ("experiment_id", "dataset_version_id", "fold_plan_id", "model_candidate_id"):
            _sha256(getattr(self, field_name), field_name)
        _required_text(self.fold_id, "fold_id")
        if type(self.random_state) is not int or self.random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        _validate_resources(self.resources)
        _schema(self.schema_version)
        object.__setattr__(self, "resources", freeze_json(self.resources))
        object.__setattr__(self, "configuration", freeze_json(self.configuration))
        expected = digest({"schema_version": self.schema_version,
                           "experiment_id": self.experiment_id,
                           "dataset_version_id": self.dataset_version_id,
                           "fold_plan_id": self.fold_plan_id,
                           "model_candidate_id": self.model_candidate_id,
                           "fold_id": self.fold_id, "random_state": self.random_state,
                           "resources": self.resources, "configuration": self.configuration})
        _match_id(self.training_job_id, expected, "training_job_id")
        object.__setattr__(self, "training_job_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id,
                "dataset_version_id": self.dataset_version_id,
                "fold_plan_id": self.fold_plan_id,
                "model_candidate_id": self.model_candidate_id, "fold_id": self.fold_id,
                "random_state": self.random_state, "resources": jsonable(self.resources),
                "configuration": jsonable(self.configuration),
                "schema_version": self.schema_version, "training_job_id": self.training_job_id}


@dataclass(frozen=True)
class ArtifactReference:
    artifact_id: str
    media_type: str
    uri: str
    sha256: str
    manifest: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _sha256(self.artifact_id, "artifact_id")
        _sha256(self.sha256, "sha256")
        if self.artifact_id != self.sha256:
            raise ValueError("artifact_id must equal the content SHA-256")
        _required_text(self.media_type, "media_type")
        _required_text(self.uri, "artifact uri")
        _schema(self.schema_version)
        object.__setattr__(self, "manifest", freeze_json(self.manifest))

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_id": self.artifact_id, "media_type": self.media_type,
                "uri": self.uri, "sha256": self.sha256,
                "manifest": jsonable(self.manifest), "schema_version": self.schema_version}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactReference":
        return cls(**dict(data))


@dataclass(frozen=True)
class CheckpointArtifact:
    artifact: ArtifactReference
    model_candidate_id: str
    dataset_version_id: str
    fold_plan_id: str
    fold_id: str
    training_job_id: str
    format: str
    schema_version: int = SCHEMA_VERSION
    checkpoint_id: str = ""

    def __post_init__(self) -> None:
        for field_name in ("model_candidate_id", "dataset_version_id", "fold_plan_id", "training_job_id"):
            _sha256(getattr(self, field_name), field_name)
        _required_text(self.fold_id, "fold_id")
        _required_text(self.format, "checkpoint format")
        _schema(self.schema_version)
        expected = digest({"schema_version": self.schema_version,
                           "artifact": self.artifact.to_dict(),
                           "model_candidate_id": self.model_candidate_id,
                           "dataset_version_id": self.dataset_version_id,
                           "fold_plan_id": self.fold_plan_id, "fold_id": self.fold_id,
                           "training_job_id": self.training_job_id, "format": self.format})
        _match_id(self.checkpoint_id, expected, "checkpoint_id")
        object.__setattr__(self, "checkpoint_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"artifact": self.artifact.to_dict(), "model_candidate_id": self.model_candidate_id,
                "dataset_version_id": self.dataset_version_id, "fold_plan_id": self.fold_plan_id,
                "fold_id": self.fold_id, "training_job_id": self.training_job_id,
                "format": self.format, "schema_version": self.schema_version,
                "checkpoint_id": self.checkpoint_id}


@dataclass(frozen=True)
class TrainingResult:
    training_job_id: str
    experiment_id: str
    dataset_version_id: str
    fold_plan_id: str
    model_candidate_id: str
    fold_id: str
    status: str
    metrics: Mapping[str, float]
    provider: str
    execution_id: str
    checkpoint: CheckpointArtifact | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    failure: str | None = None
    schema_version: int = SCHEMA_VERSION
    training_result_id: str = ""

    def __post_init__(self) -> None:
        for field_name in ("training_job_id", "experiment_id", "dataset_version_id",
                           "fold_plan_id", "model_candidate_id"):
            _sha256(getattr(self, field_name), field_name)
        _required_text(self.fold_id, "fold_id")
        _required_text(self.provider, "provider")
        _required_text(self.execution_id, "execution_id")
        if self.status not in {"succeeded", "failed"}:
            raise ValueError("training status must be succeeded or failed")
        if self.status == "succeeded" and (self.checkpoint is None or self.failure):
            raise ValueError("successful training requires a checkpoint and no failure")
        if self.status == "failed" and not self.failure:
            raise ValueError("failed training requires a failure description")
        if self.checkpoint is not None:
            checkpoint_lineage = (self.checkpoint.training_job_id, self.checkpoint.dataset_version_id,
                                  self.checkpoint.fold_plan_id, self.checkpoint.model_candidate_id,
                                  self.checkpoint.fold_id)
            result_lineage = (self.training_job_id, self.dataset_version_id, self.fold_plan_id,
                              self.model_candidate_id, self.fold_id)
            if checkpoint_lineage != result_lineage:
                raise ValueError("checkpoint provenance must match TrainingResult")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name.strip() or not _finite(value):
                raise ValueError("training metrics must have names and finite numeric values")
        _schema(self.schema_version)
        object.__setattr__(self, "metrics", freeze_json(self.metrics))
        object.__setattr__(self, "provenance", freeze_json(self.provenance))
        expected = digest({"schema_version": self.schema_version,
                           "training_job_id": self.training_job_id,
                           "experiment_id": self.experiment_id,
                           "dataset_version_id": self.dataset_version_id,
                           "fold_plan_id": self.fold_plan_id,
                           "model_candidate_id": self.model_candidate_id,
                           "fold_id": self.fold_id, "status": self.status,
                           "metrics": self.metrics, "provider": self.provider,
                           "execution_id": self.execution_id,
                           "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
                           "provenance": self.provenance, "failure": self.failure})
        _match_id(self.training_result_id, expected, "training_result_id")
        object.__setattr__(self, "training_result_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"training_job_id": self.training_job_id, "experiment_id": self.experiment_id,
                "dataset_version_id": self.dataset_version_id,
                "fold_plan_id": self.fold_plan_id,
                "model_candidate_id": self.model_candidate_id, "fold_id": self.fold_id,
                "status": self.status, "metrics": jsonable(self.metrics),
                "provider": self.provider, "execution_id": self.execution_id,
                "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
                "provenance": jsonable(self.provenance), "failure": self.failure,
                "schema_version": self.schema_version,
                "training_result_id": self.training_result_id}


@dataclass(frozen=True)
class PredictionArtifact:
    artifact: ArtifactReference
    kind: str
    model_candidate_id: str
    dataset_version_id: str
    checkpoint_id: str
    class_names: tuple[str, ...]
    row_count: int
    fold_plan_id: str | None = None
    fold_id: str | None = None
    schema_version: int = SCHEMA_VERSION
    prediction_artifact_id: str = ""
    synthetic: bool = False
    target_schema_version: int = TARGET_SCHEMA_VERSION
    provenance: str = "model_inference"

    def __post_init__(self) -> None:
        if self.kind not in {"oof", "inference"}:
            raise ValueError("prediction kind must be oof or inference")
        for field_name in ("model_candidate_id", "dataset_version_id", "checkpoint_id"):
            _sha256(getattr(self, field_name), field_name)
        _unique_names(self.class_names, "class_names")
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic must be a boolean")
        validate_targets(self.class_names, allow_synthetic=self.synthetic)
        _target_schema(self.target_schema_version)
        _required_text(self.provenance, "prediction provenance")
        if type(self.row_count) is not int or self.row_count < 1:
            raise ValueError("prediction row_count must be positive")
        if (self.kind == "oof") != (self.fold_plan_id is not None and self.fold_id is not None):
            raise ValueError("OOF predictions need a fold plan and fold; inference predictions do not")
        if self.fold_plan_id is not None:
            _sha256(self.fold_plan_id, "fold_plan_id")
        if self.fold_id is not None:
            _required_text(self.fold_id, "fold_id")
        _schema(self.schema_version)
        object.__setattr__(self, "class_names", tuple(self.class_names))
        expected = digest({"schema_version": self.schema_version,
                           "artifact": self.artifact.to_dict(), "kind": self.kind,
                           "model_candidate_id": self.model_candidate_id,
                           "dataset_version_id": self.dataset_version_id,
                           "checkpoint_id": self.checkpoint_id,
                           "class_names": self.class_names, "row_count": self.row_count,
                           "fold_plan_id": self.fold_plan_id, "fold_id": self.fold_id,
                           "synthetic": self.synthetic,
                           "target_schema_version": self.target_schema_version,
                           "target_registry_id": TARGET_REGISTRY_ID,
                           "provenance": self.provenance})
        _match_id(self.prediction_artifact_id, expected, "prediction_artifact_id")
        object.__setattr__(self, "prediction_artifact_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"artifact": self.artifact.to_dict(), "kind": self.kind,
                "model_candidate_id": self.model_candidate_id,
                "dataset_version_id": self.dataset_version_id,
                "checkpoint_id": self.checkpoint_id, "class_names": list(self.class_names),
                "row_count": self.row_count, "fold_plan_id": self.fold_plan_id,
                "synthetic": self.synthetic, "target_schema_version": self.target_schema_version,
                "target_registry_id": TARGET_REGISTRY_ID,
                "provenance": self.provenance,
                "fold_id": self.fold_id, "schema_version": self.schema_version,
                "prediction_artifact_id": self.prediction_artifact_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PredictionArtifact":
        _verify_target_registry(data)
        payload = dict(data)
        payload.pop("target_registry_id")
        payload["artifact"] = ArtifactReference.from_dict(payload["artifact"])
        return cls(**payload)


@dataclass(frozen=True)
class Evaluation:
    experiment_id: str
    dataset_version_id: str
    model_candidate_id: str
    prediction_artifact_ids: tuple[str, ...]
    class_names: tuple[str, ...]
    auc_by_class: Mapping[str, float | None] = field(default_factory=dict)
    macro_auc: float | None = None
    additional_metrics: Mapping[str, float] = field(default_factory=dict)
    synthetic: bool = False
    schema_version: int = SCHEMA_VERSION
    evaluation_id: str = ""
    target_schema_version: int = TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("experiment_id", "dataset_version_id", "model_candidate_id"):
            _sha256(getattr(self, field_name), field_name)
        _unique_sha256(self.prediction_artifact_ids, "prediction_artifact_ids")
        _unique_names(self.class_names, "class_names")
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic must be a boolean")
        validate_targets(self.class_names, allow_synthetic=self.synthetic)
        _target_schema(self.target_schema_version)
        if set(self.auc_by_class) != set(self.class_names):
            raise ValueError("AUC must be supplied for exactly every declared target")
        for value in self.auc_by_class.values():
            if value is not None and (not _finite(value) or not 0 <= value <= 1):
                raise ValueError("per-class AUC values must be between 0 and 1")
        known_auc = list(self.auc_by_class.values())
        if self.macro_auc is None and known_auc and all(value is not None for value in known_auc):
            object.__setattr__(self, "macro_auc", fmean(known_auc))
        if self.macro_auc is not None:
            if (not _finite(self.macro_auc) or not 0 <= self.macro_auc <= 1
                    or len(self.auc_by_class) != len(self.class_names)
                    or any(value is None for value in known_auc)):
                raise ValueError("macro AUC requires finite AUC for every declared class")
            if not math.isclose(self.macro_auc, fmean(known_auc), abs_tol=1e-12):
                raise ValueError("macro AUC must equal the mean of the class AUC values")
        for name, value in self.additional_metrics.items():
            if not name.strip() or not _finite(value):
                raise ValueError("additional metrics must have names and finite values")
        _schema(self.schema_version)
        object.__setattr__(self, "prediction_artifact_ids", tuple(self.prediction_artifact_ids))
        object.__setattr__(self, "class_names", tuple(self.class_names))
        object.__setattr__(self, "auc_by_class", freeze_json(self.auc_by_class))
        object.__setattr__(self, "additional_metrics", freeze_json(self.additional_metrics))
        expected = digest({"schema_version": self.schema_version,
                           "experiment_id": self.experiment_id,
                           "dataset_version_id": self.dataset_version_id,
                           "model_candidate_id": self.model_candidate_id,
                           "prediction_artifact_ids": self.prediction_artifact_ids,
                           "class_names": self.class_names, "auc_by_class": self.auc_by_class,
                           "macro_auc": self.macro_auc,
                           "additional_metrics": self.additional_metrics,
                           "synthetic": self.synthetic,
                           "target_schema_version": self.target_schema_version,
                           "target_registry_id": TARGET_REGISTRY_ID})
        _match_id(self.evaluation_id, expected, "evaluation_id")
        object.__setattr__(self, "evaluation_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id,
                "dataset_version_id": self.dataset_version_id,
                "model_candidate_id": self.model_candidate_id,
                "prediction_artifact_ids": list(self.prediction_artifact_ids),
                "class_names": list(self.class_names), "auc_by_class": jsonable(self.auc_by_class),
                "macro_auc": self.macro_auc,
                "additional_metrics": jsonable(self.additional_metrics),
                "synthetic": self.synthetic, "schema_version": self.schema_version,
                "target_schema_version": self.target_schema_version,
                "target_registry_id": TARGET_REGISTRY_ID,
                "evaluation_id": self.evaluation_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Evaluation":
        _verify_target_registry(data)
        payload = dict(data)
        payload.pop("target_registry_id")
        return cls(**payload)


@dataclass(frozen=True)
class SubmissionArtifact:
    artifact: ArtifactReference
    model_candidate_id: str
    dataset_version_id: str
    evaluation_id: str
    prediction_artifact_ids: tuple[str, ...]
    format: str
    schema_version: int = SCHEMA_VERSION
    submission_artifact_id: str = ""
    class_names: tuple[str, ...] = TARGETS
    target_schema_version: int = TARGET_SCHEMA_VERSION

    @property
    def column_order(self) -> tuple[str, ...]:
        return ("StudyInstanceUID", *self.class_names)

    def __post_init__(self) -> None:
        for field_name in ("model_candidate_id", "dataset_version_id", "evaluation_id"):
            _sha256(getattr(self, field_name), field_name)
        _unique_sha256(self.prediction_artifact_ids, "prediction_artifact_ids")
        _required_text(self.format, "submission format")
        _target_schema(self.target_schema_version)
        validate_targets(self.class_names)
        _schema(self.schema_version)
        object.__setattr__(self, "prediction_artifact_ids", tuple(self.prediction_artifact_ids))
        object.__setattr__(self, "class_names", tuple(self.class_names))
        expected = digest({"schema_version": self.schema_version,
                           "artifact": self.artifact.to_dict(),
                           "model_candidate_id": self.model_candidate_id,
                           "dataset_version_id": self.dataset_version_id,
                           "evaluation_id": self.evaluation_id,
                           "prediction_artifact_ids": self.prediction_artifact_ids,
                           "format": self.format, "class_names": self.class_names,
                           "column_order": self.column_order,
                           "target_schema_version": self.target_schema_version,
                           "target_registry_id": TARGET_REGISTRY_ID})
        _match_id(self.submission_artifact_id, expected, "submission_artifact_id")
        object.__setattr__(self, "submission_artifact_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"artifact": self.artifact.to_dict(),
                "model_candidate_id": self.model_candidate_id,
                "dataset_version_id": self.dataset_version_id,
                "evaluation_id": self.evaluation_id,
                "prediction_artifact_ids": list(self.prediction_artifact_ids),
                "class_names": list(self.class_names),
                "column_order": list(self.column_order),
                "target_schema_version": self.target_schema_version,
                "target_registry_id": TARGET_REGISTRY_ID,
                "format": self.format, "schema_version": self.schema_version,
                "submission_artifact_id": self.submission_artifact_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubmissionArtifact":
        _verify_target_registry(data)
        if tuple(data.get("column_order", ())) != ("StudyInstanceUID", *data.get("class_names", ())):
            raise ValueError("serialized submission column order does not match target order")
        payload = dict(data)
        payload.pop("target_registry_id")
        payload.pop("column_order")
        payload["artifact"] = ArtifactReference.from_dict(payload["artifact"])
        return cls(**payload)


def _validate_resources(resources: Mapping[str, Any]) -> None:
    if not isinstance(resources, Mapping):
        raise ValueError("resources must be an object")
    for name in ("cpus", "gpus"):
        value = resources.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"resource {name} must be a nonnegative number")
    if resources["cpus"] == 0:
        raise ValueError("at least one CPU is required")


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _required_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")


def _unique_names(values: tuple[str, ...], label: str) -> None:
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _unique_sha256(values: tuple[str, ...], label: str) -> None:
    _unique_names(values, label)
    for value in values:
        _sha256(value, label)


def _sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _schema(value: int) -> None:
    if value != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {value}")


def _target_schema(value: int) -> None:
    if type(value) is not int or value != TARGET_SCHEMA_VERSION:
        raise ValueError(f"unsupported target schema version: {value}")


def _verify_target_registry(data: Mapping[str, Any]) -> None:
    if data.get("target_registry_id") != TARGET_REGISTRY_ID:
        raise ValueError("serialized target registry identity does not match this version")


def _match_id(supplied: str, expected: str, label: str) -> None:
    if supplied and supplied != expected:
        raise ValueError(f"{label} does not match contract contents")
