"""Expand an experiment declaration into deterministic simulation jobs."""

from collections.abc import Mapping

from ..contracts import Candidate, ExperimentSpec, SimulationJob


def plan_jobs(spec: ExperimentSpec, candidates: Mapping[str, Candidate]) -> list[SimulationJob]:
    if set(candidates) != set(spec.candidate_ids):
        raise ValueError("candidate manifest IDs must match the ExperimentSpec")
    if any(candidates[candidate_id].candidate_id != candidate_id
           for candidate_id in spec.candidate_ids):
        raise ValueError("candidate manifest key does not match its candidate_id")
    jobs = [SimulationJob(
        experiment_id=spec.experiment_id, candidate_id=candidate_id,
        candidate_artifact_sha256=candidates[candidate_id].artifact_sha256,
        candidate_artifact_uri=candidates[candidate_id].artifact_uri,
        opponent_id=opponent, seed=seed, seat=seat,
        simulator=spec.simulator, simulator_version=spec.simulator_version,
        configuration=dict(spec.configuration),
    ) for candidate_id in sorted(spec.candidate_ids)
      for opponent in sorted(spec.opponents)
      for seed in sorted(spec.seeds)
      for seat in (0, 1)]
    return sorted(jobs, key=lambda job: job.job_id)
