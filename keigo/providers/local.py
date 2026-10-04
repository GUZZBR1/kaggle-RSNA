"""Inline provider for short smoke tests and development runs."""

from collections.abc import Callable
import uuid

from ..contracts import SimulationJob, SimulationResult


class LocalProvider:
    name = "local"

    def __init__(self, simulate: Callable[[SimulationJob], dict]):
        self._simulate = simulate
        self._results: dict[str, SimulationResult] = {}

    def submit_simulation(self, job: SimulationJob) -> str:
        execution_id = f"local-{uuid.uuid4().hex}"
        try:
            metrics = self._simulate(job)
            result = SimulationResult(
                job_id=job.job_id, candidate_id=job.candidate_id,
                opponent_id=job.opponent_id, seed=job.seed, seat=job.seat,
                status="succeeded", metrics=metrics, provider=self.name,
                execution_id=execution_id,
            )
        except Exception as exc:
            result = SimulationResult(
                job_id=job.job_id, candidate_id=job.candidate_id,
                opponent_id=job.opponent_id, seed=job.seed, seat=job.seat,
                status="failed", metrics={}, provider=self.name,
                execution_id=execution_id, failure=f"{type(exc).__name__}: {exc}",
            )
        self._results[execution_id] = result
        return execution_id

    def get_result(self, execution_id: str) -> SimulationResult:
        try:
            return self._results[execution_id]
        except KeyError as exc:
            raise KeyError(f"unknown local execution ID: {execution_id}") from exc
