"""Optional Ray adapter, loaded only when instantiated."""

from __future__ import annotations

from typing import Callable

from ..contracts import TrainingJob, TrainingResult


class RayProvider:
    def __init__(self, trainer: Callable[[TrainingJob], TrainingResult]):
        try:
            import ray
        except ImportError as exc:
            raise RuntimeError("RayProvider requires the optional 'ray' dependency") from exc
        self._ray = ray
        self._trainer = trainer
        self._refs: dict[str, object] = {}

    def submit_training(self, job: TrainingJob) -> str:
        remote = self._ray.remote(self._trainer)
        resources = dict(job.resources)
        ref = remote.options(num_cpus=resources["cpus"], num_gpus=resources["gpus"]).remote(job)
        execution_id = f"ray-{job.training_job_id}"
        self._refs[execution_id] = ref
        return execution_id

    def get_training_result(self, execution_id: str) -> TrainingResult:
        try:
            ref = self._refs[execution_id]
        except KeyError as exc:
            raise KeyError(f"unknown Ray execution: {execution_id}") from exc
        return self._ray.get(ref)
