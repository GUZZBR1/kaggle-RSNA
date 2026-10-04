"""Tournament planning primitives for future candidate populations."""

from ..contracts import SimulationJob


def round_robin_jobs(jobs: list[SimulationJob]) -> list[SimulationJob]:
    """Return a stable order independent of candidate registration order."""
    return sorted(jobs, key=lambda job: (job.seed, job.opponent_id, job.candidate_id,
                                         job.seat, job.job_id))
