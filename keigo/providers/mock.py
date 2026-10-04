"""Deterministic mock provider for contract and wiring validation."""

from ..contracts import SimulationJob, SimulationResult
from ..identity import digest


class MockProvider:
    name = "mock"

    def __init__(self) -> None:
        self._results: dict[str, SimulationResult] = {}

    def submit_simulation(self, job: SimulationJob) -> str:
        execution_id = f"mock-{job.job_id}"
        score = int(digest({"candidate": job.candidate_id, "seed": job.seed,
                            "seat": job.seat})[:8], 16) / 0xFFFFFFFF
        self._results[execution_id] = SimulationResult(
            job_id=job.job_id, candidate_id=job.candidate_id,
            opponent_id=job.opponent_id, seed=job.seed, seat=job.seat,
            status="succeeded", metrics={"score": score}, provider=self.name,
            execution_id=execution_id,
        )
        return execution_id

    def get_result(self, execution_id: str) -> SimulationResult:
        return self._results[execution_id]
