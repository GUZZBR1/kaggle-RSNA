"""Create deterministic training jobs for declared candidate/fold pairs."""

from __future__ import annotations

from ..contracts import DatasetVersion, ExperimentSpec, FoldPlan, ModelCandidate, TrainingJob


def plan_jobs(spec: ExperimentSpec, dataset: DatasetVersion, fold_plan: FoldPlan,
              candidates: dict[str, ModelCandidate]) -> list[TrainingJob]:
    if dataset.dataset_version_id != spec.dataset_version_id:
        raise ValueError("experiment dataset does not match DatasetVersion")
    if fold_plan.fold_plan_id != spec.fold_plan_id or fold_plan.dataset_version_id != dataset.dataset_version_id:
        raise ValueError("experiment FoldPlan does not match dataset or spec")
    if tuple(spec.class_names) != tuple(dataset.class_names):
        raise ValueError("experiment class ordering does not match DatasetVersion")
    if spec.synthetic != dataset.synthetic:
        raise ValueError("experiment synthetic mode does not match DatasetVersion")
    if not set(spec.fold_ids).issubset(fold_plan.fold_ids):
        raise ValueError("experiment references folds absent from FoldPlan")
    if set(candidates) != set(spec.model_candidate_ids):
        raise ValueError("candidate map must exactly match experiment candidate IDs")
    jobs = []
    for candidate_id in spec.model_candidate_ids:
        for fold_id in spec.fold_ids:
            candidate = candidates[candidate_id]
            jobs.append(TrainingJob(spec.experiment_id, dataset.dataset_version_id,
                fold_plan.fold_plan_id, candidate.model_candidate_id, fold_id,
                fold_plan.random_state, spec.resources,
                {**dict(spec.training_configuration), **dict(candidate.training_configuration)}))
    return jobs
