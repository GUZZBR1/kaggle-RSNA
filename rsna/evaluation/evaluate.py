"""Validate the provenance of predictions before metrics are recorded."""

from __future__ import annotations

from ..contracts import Evaluation, ExperimentSpec, PredictionArtifact


def validate_evaluation_inputs(spec: ExperimentSpec, predictions: list[PredictionArtifact],
                               evaluation: Evaluation) -> None:
    if evaluation.experiment_id != spec.experiment_id:
        raise ValueError("evaluation belongs to a different experiment")
    if evaluation.dataset_version_id != spec.dataset_version_id:
        raise ValueError("evaluation belongs to a different dataset")
    if set(evaluation.prediction_artifact_ids) != {item.prediction_artifact_id for item in predictions}:
        raise ValueError("evaluation prediction references do not match supplied predictions")
    for item in predictions:
        if item.dataset_version_id != spec.dataset_version_id:
            raise ValueError("prediction belongs to a different dataset")
        if item.model_candidate_id != evaluation.model_candidate_id:
            raise ValueError("prediction belongs to a different model candidate")
        if tuple(item.class_names) != tuple(evaluation.class_names):
            raise ValueError("prediction class ordering differs from evaluation")
