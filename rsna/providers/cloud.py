"""Cloud execution boundary; concrete infrastructure is supplied by the caller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from ..contracts import TrainingJob, TrainingResult
from ..identity import jsonable


def build_cloud_training_spec(
    job: TrainingJob,
    *,
    input_uris: Mapping[str, str],
    output_uri: str,
    environment: Mapping[str, str],
    command: Sequence[str],
    resume_checkpoint_uri: str | None = None,
) -> dict[str, Any]:
    """Build a serializable cloud job description without submitting it."""
    if not isinstance(job, TrainingJob):
        raise TypeError("job must be a TrainingJob")
    if not isinstance(input_uris, Mapping) or not input_uris:
        raise ValueError("input_uris must be a non-empty mapping")
    normalized_inputs = {}
    for name, uri in input_uris.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("input names must be non-empty strings")
        if not isinstance(uri, str) or not uri.strip():
            raise ValueError(f"input URI for {name!r} must be a non-empty string")
        normalized_inputs[name] = uri
    if not isinstance(output_uri, str) or not output_uri.strip():
        raise ValueError("output_uri must be a non-empty string")
    if not isinstance(environment, Mapping):
        raise ValueError("environment must be a mapping")
    normalized_environment = {}
    for key, value in environment.items():
        if not isinstance(key, str) or not key.strip() or "=" in key or "\x00" in key:
            raise ValueError("environment variable names must be non-empty valid strings")
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"environment value for {key!r} must be a string")
        normalized_environment[key] = value
    if (isinstance(command, (str, bytes)) or not isinstance(command, Sequence)
            or not command or any(not isinstance(part, str) or not part.strip() for part in command)):
        raise ValueError("command must be a non-empty sequence of non-empty strings")
    if resume_checkpoint_uri is not None and (
            not isinstance(resume_checkpoint_uri, str) or not resume_checkpoint_uri.strip()):
        raise ValueError("resume_checkpoint_uri must be a non-empty string when provided")

    spec = {
        "training_job_id": job.training_job_id,
        "experiment_id": job.experiment_id,
        "dataset_version_id": job.dataset_version_id,
        "fold_plan_id": job.fold_plan_id,
        "model_candidate_id": job.model_candidate_id,
        "fold_id": job.fold_id,
        "random_state": job.random_state,
        "resources": jsonable(job.resources),
        "configuration": jsonable(job.configuration),
        "validated_leakage_report_id": job.validated_leakage_report_id,
        "inputs": normalized_inputs,
        "output_uri": output_uri,
        "environment": normalized_environment,
        "command": list(command),
    }
    if resume_checkpoint_uri is not None:
        spec["resume_checkpoint_uri"] = resume_checkpoint_uri
    return spec


class CloudProvider:
    def __init__(self, submit: Callable[[TrainingJob], str],
                 fetch_result: Callable[[str], TrainingResult]):
        self._submit = submit
        self._fetch_result = fetch_result

    def submit_training(self, job: TrainingJob) -> str:
        return self._submit(job)

    def get_training_result(self, execution_id: str) -> TrainingResult:
        return self._fetch_result(execution_id)
