"""Synchronous local provider intended for smoke checks and lightweight work."""

from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from ..contracts import ArtifactReference, CheckpointArtifact, TrainingJob, TrainingResult
from ..identity import digest
from .base import TrainingProvider


class LocalProvider:
    def __init__(self, trainer: Callable[[TrainingJob], dict[str, Any]]):
        self.trainer = trainer
        self._results: dict[str, TrainingResult] = {}

    def submit_training(self, job: TrainingJob) -> str:
        execution_id = f"local-{uuid4().hex}"
        try:
            run = getattr(self.trainer, "run", None)
            outcome = run(job, execution_id=execution_id) if callable(run) else self.trainer(job)
            if isinstance(outcome, TrainingResult):
                expected = (job.training_job_id, job.experiment_id, job.dataset_version_id,
                            job.fold_plan_id, job.model_candidate_id, job.fold_id)
                actual = (outcome.training_job_id, outcome.experiment_id, outcome.dataset_version_id,
                          outcome.fold_plan_id, outcome.model_candidate_id, outcome.fold_id)
                if actual != expected:
                    raise ValueError("trainer TrainingResult lineage does not match TrainingJob")
                self._results[execution_id] = outcome
                return execution_id
            if not isinstance(outcome, dict) or "checkpoint_uri" not in outcome:
                raise ValueError("trainer must return checkpoint_uri and optional metrics")
            uri = str(outcome["checkpoint_uri"])
            checkpoint_sha = outcome.get("checkpoint_sha256") or digest({"uri": uri, "job": job.training_job_id})
            artifact = ArtifactReference(checkpoint_sha,
                str(outcome.get("checkpoint_media_type", "application/octet-stream")), uri,
                checkpoint_sha, outcome.get("checkpoint_manifest", {"training_job_id": job.training_job_id}))
            checkpoint = CheckpointArtifact(artifact, job.model_candidate_id,
                job.dataset_version_id, job.fold_plan_id, job.fold_id, job.training_job_id,
                str(outcome.get("format", "pytorch" if "checkpoint_media_type" in outcome else "opaque")))
            result = TrainingResult(job.training_job_id, job.experiment_id,
                job.dataset_version_id, job.fold_plan_id, job.model_candidate_id,
                job.fold_id, "succeeded", outcome.get("metrics", {}), "local",
                execution_id, checkpoint, outcome.get("provenance", {}))
        except Exception as exc:
            result = TrainingResult(job.training_job_id, job.experiment_id,
                job.dataset_version_id, job.fold_plan_id, job.model_candidate_id,
                job.fold_id, "failed", {}, "local", execution_id,
                failure=f"{type(exc).__name__}: {exc}")
        self._results[execution_id] = result
        return execution_id

    def get_training_result(self, execution_id: str) -> TrainingResult:
        try:
            return self._results[execution_id]
        except KeyError as exc:
            raise KeyError(f"unknown local execution: {execution_id}") from exc
