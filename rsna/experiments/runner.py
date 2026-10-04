"""Run planned jobs through an abstract training provider."""

from __future__ import annotations

from ..contracts import TrainingJob, TrainingResult
from ..providers.base import TrainingProvider


def run_jobs(jobs: list[TrainingJob], provider: TrainingProvider) -> list[TrainingResult]:
    results = []
    for job in jobs:
        execution_id = provider.submit_training(job)
        result = provider.get_training_result(execution_id)
        expected = (job.training_job_id, job.experiment_id, job.dataset_version_id,
                    job.fold_plan_id, job.model_candidate_id, job.fold_id)
        actual = (result.training_job_id, result.experiment_id, result.dataset_version_id,
                  result.fold_plan_id, result.model_candidate_id, result.fold_id)
        if actual != expected or result.execution_id != execution_id:
            raise ValueError("provider result provenance does not match submitted TrainingJob")
        results.append(result)
    return results
