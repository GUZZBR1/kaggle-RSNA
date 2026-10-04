"""Optional Ray backend for an already provisioned Ray cluster."""

from __future__ import annotations

from collections.abc import Callable

from ..contracts import SimulationJob, SimulationResult


class RayProvider:
    name = "ray"

    def __init__(self, simulate: Callable[[dict], dict], *, num_cpus: float = 1,
                 num_gpus: float = 0) -> None:
        try:
            import ray
        except ImportError as exc:
            raise RuntimeError("RayProvider requires the optional 'ray' dependency") from exc
        self._ray = ray
        self._remote = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(simulate)
        self._references: dict[str, object] = {}

    def submit_simulation(self, job: SimulationJob) -> str:
        execution_id = job.job_id
        self._references[execution_id] = self._remote.remote(job.to_dict())
        return execution_id

    def get_result(self, execution_id: str) -> SimulationResult:
        ref = self._references[execution_id]
        payload = self._ray.get(ref)
        return SimulationResult(**payload)
