"""Execute the first contract-to-report pipeline using any SimulationProvider."""

from collections.abc import Mapping

from ..artifacts import JsonArtifactStore
from ..contracts import Candidate, ExperimentSpec
from ..evaluation import evaluate
from ..providers.base import SimulationProvider
from .plan import plan_jobs


def run_experiment(spec: ExperimentSpec, candidates: Mapping[str, Candidate],
                   provider: SimulationProvider, artifacts: JsonArtifactStore):
    jobs = plan_jobs(spec, candidates)
    submissions = [(job, provider.submit_simulation(job)) for job in jobs]
    results = [provider.get_result(execution_id) for _, execution_id in submissions]
    for job, result in zip((job for job, _ in submissions), results):
        if (result.job_id, result.candidate_id, result.opponent_id, result.seed, result.seat) != (
                job.job_id, job.candidate_id, job.opponent_id, job.seed, job.seat):
            raise ValueError("provider returned a result for a different job")
        if result.provider != provider.name:
            raise ValueError("result provenance does not match its provider")
    evaluation = evaluate(spec, results)
    artifact = artifacts.put_json(evaluation.to_dict(), manifest={
        "experiment_id": spec.experiment_id,
        "result_ids": list(evaluation.result_ids),
        "provider": provider.name,
    })
    return evaluation, artifact
