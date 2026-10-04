"""Deterministic aggregation and evaluation of externally produced OOF predictions."""

from __future__ import annotations

from itertools import combinations
import json
import math
from pathlib import Path
import time
import tracemalloc
from typing import Any, Mapping, Sequence

from ..artifacts import JsonArtifactStore
from ..contracts import Evaluation, ExperimentSpec, PredictionArtifact
from ..folds.models import FoldPlanManifest
from ..folds.manifest import load_fold_plan
from ..identity import digest
from ..labels import LabelRecord
from ..leakage import LeakagePolicy, require_valid_leakage_report, validate_leakage
from ..targets import TARGETS

OOF_PREDICTIONS_SCHEMA_VERSION = 1
OOF_METRIC_IMPLEMENTATION = "rank_auc_v1"


def _load_prediction_rows(prediction: PredictionArtifact) -> list[dict[str, Any]]:
    """Load and validate the versioned JSON rows stored by a PredictionArtifact."""
    if prediction.artifact.media_type != "application/json":
        raise ValueError(
            f"prediction artifact {prediction.prediction_artifact_id} must use application/json"
        )
    try:
        payload = JsonArtifactStore.load_json(prediction.artifact)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            f"cannot read prediction artifact {prediction.prediction_artifact_id}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "class_names", "rows"}:
        raise ValueError(
            f"prediction artifact {prediction.prediction_artifact_id} must contain only "
            "schema_version, class_names, and rows"
        )
    if payload["schema_version"] != OOF_PREDICTIONS_SCHEMA_VERSION:
        raise ValueError("unsupported OOF prediction payload schema version")
    if tuple(payload["class_names"]) != TARGETS:
        raise ValueError("prediction payload target order must exactly match TARGETS")
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError("prediction payload rows must be a list")
    if len(rows) != prediction.row_count:
        raise ValueError(
            f"prediction row_count mismatch for fold {prediction.fold_id}: "
            f"metadata={prediction.row_count}, payload={len(rows)}"
        )
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"StudyInstanceUID", "scores"}:
            raise ValueError(f"prediction row {index} must contain only StudyInstanceUID and scores")
        study_id = row["StudyInstanceUID"]
        if not isinstance(study_id, str) or not study_id.strip():
            raise ValueError(f"prediction row {index} has an empty study_id")
        scores = row["scores"]
        if not isinstance(scores, list) or len(scores) != len(TARGETS):
            raise ValueError(
                f"prediction row {study_id!r} must have {len(TARGETS)} ordered scores"
            )
        for target, score in zip(TARGETS, scores, strict=True):
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError(
                    f"prediction score for study {study_id!r}, target {target!r} "
                    "must be a finite probability in [0, 1]"
                )
        validated.append({"study_id": study_id, "scores": [float(score) for score in scores]})
    return validated


def binary_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute tie-aware binary ROC AUC; return None when either class is absent."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError("ROC AUC labels must be hard binary 0/1 values")
    if any(isinstance(score, bool) or not isinstance(score, (int, float))
           or not math.isfinite(score) for score in scores):
        raise ValueError("ROC AUC scores must be finite numeric values")
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None

    # Average ranks give tied positive/negative scores half credit, without sklearn.
    order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    positive_rank_sum = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        positive_rank_sum += average_rank * sum(labels[index] for index in order[start:end])
        start = end
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def _evaluate_oof_impl(spec: ExperimentSpec, fold_plan: FoldPlanManifest,
                       labels: Sequence[LabelRecord], predictions: Sequence[PredictionArtifact],
                       artifact_store: JsonArtifactStore) -> dict[str, Any]:
    """Validate complete OOF coverage, calculate per-target AUC, and persist reports.

    Each prediction artifact stores JSON with ``schema_version``, canonical
    ``class_names``, and ``rows`` containing ``StudyInstanceUID`` plus an ordered
    score vector. Every experiment candidate must provide exactly one artifact
    per fold.
    """
    if spec.dataset_version_id != fold_plan.dataset_version_id:
        raise ValueError("ExperimentSpec DatasetVersion mismatch with FoldPlan")
    if spec.fold_plan_id != fold_plan.fold_plan_id:
        raise ValueError("ExperimentSpec FoldPlan mismatch with supplied FoldPlan")
    if tuple(spec.class_names) != TARGETS:
        raise ValueError("OOF evaluation requires the official TARGETS order")
    expected_fold_ids = tuple(f"fold_{index}" for index in range(fold_plan.n_folds))
    if set(fold_plan.assignments.values()) != set(expected_fold_ids):
        raise ValueError("FoldPlan assignments must include every declared fold")
    if tuple(spec.fold_ids) != expected_fold_ids:
        raise ValueError(
            f"ExperimentSpec fold_ids must match FoldPlan assignments: {expected_fold_ids}"
        )
    if not predictions:
        raise ValueError("at least one OOF PredictionArtifact is required")
    if len({item.prediction_artifact_id for item in predictions}) != len(predictions):
        raise ValueError("duplicate PredictionArtifact supplied")

    label_by_study: dict[str, LabelRecord] = {}
    patient_bindings: dict[str, tuple[str, str]] = {}
    for label in labels:
        if label.study_id in label_by_study:
            raise ValueError(f"duplicate label row for study {label.study_id!r}")
        if label.patient_id is not None and label.study_id in fold_plan.assignments:
            binding = (fold_plan.study_groups[label.study_id],
                       fold_plan.assignments[label.study_id])
            previous = patient_bindings.setdefault(label.patient_id, binding)
            if previous != binding:
                raise ValueError(
                    f"PatientID {label.patient_id!r} spans multiple FoldPlan groups or folds"
                )
        label_by_study[label.study_id] = label
    expected_studies = set(fold_plan.assignments)
    if set(label_by_study) != expected_studies:
        missing = sorted(expected_studies - set(label_by_study))
        foreign = sorted(set(label_by_study) - expected_studies)
        raise ValueError(f"label study coverage mismatch; missing={missing[:5]}, foreign={foreign[:5]}")
    for study_id, label in label_by_study.items():
        if label.label_type != "hard":
            raise ValueError(f"label type {label.label_type!r} is not supported for ROC AUC")
        if tuple(label.values) != TARGETS:
            raise ValueError(f"label target order mismatch for study {study_id!r}")
        for target, value in label.values.items():
            if value is not None and (isinstance(value, bool) or value not in (0, 1)):
                raise ValueError(
                    f"label for study {study_id!r}, target {target!r} is not hard binary"
                )

    leakage_records = [
        {"study_uid": study_id, "fold_id": fold_plan.assignments[study_id],
         "patient_id": label_by_study[study_id].patient_id}
        for study_id in sorted(expected_studies)
    ]
    leakage_report = validate_leakage(
        leakage_records, assignments=fold_plan.assignments,
        policy=LeakagePolicy.STRICT, dataset_version_id=spec.dataset_version_id,
    )
    leakage_report_id = require_valid_leakage_report(leakage_report)

    by_candidate_fold: dict[tuple[str, str], PredictionArtifact] = {}
    for prediction in predictions:
        if prediction.kind != "oof":
            raise ValueError("inference PredictionArtifact cannot be used for OOF evaluation")
        if prediction.model_candidate_id not in spec.model_candidate_ids:
            raise ValueError("prediction model candidate is not declared by ExperimentSpec")
        if prediction.dataset_version_id != spec.dataset_version_id:
            raise ValueError("prediction DatasetVersion mismatch with ExperimentSpec")
        if prediction.fold_plan_id != fold_plan.fold_plan_id:
            raise ValueError("prediction FoldPlan mismatch with supplied FoldPlan")
        if tuple(prediction.class_names) != TARGETS:
            raise ValueError("PredictionArtifact target order must exactly match TARGETS")
        if prediction.synthetic is not spec.synthetic:
            raise ValueError("prediction synthetic flag differs from ExperimentSpec")
        if prediction.target_schema_version != spec.target_schema_version:
            raise ValueError("prediction target schema version differs from ExperimentSpec")
        if prediction.fold_id not in expected_fold_ids:
            raise ValueError(f"prediction has unknown fold_id {prediction.fold_id!r}")
        key = (prediction.model_candidate_id, prediction.fold_id)
        if key in by_candidate_fold:
            raise ValueError(
                f"multiple prediction artifacts for candidate {key[0]} fold {key[1]}"
            )
        by_candidate_fold[key] = prediction
    expected_pairs = {
        (candidate, fold_id)
        for candidate in spec.model_candidate_ids
        for fold_id in expected_fold_ids
    }
    if set(by_candidate_fold) != expected_pairs:
        missing = sorted(expected_pairs - set(by_candidate_fold))
        extra = sorted(set(by_candidate_fold) - expected_pairs)
        raise ValueError(f"prediction fold coverage mismatch; missing={missing[:5]}, extra={extra[:5]}")

    all_candidate_rows: dict[str, dict[str, dict[str, Any]]] = {}
    candidate_evaluations: dict[str, Evaluation] = {}
    target_tables: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    source_ids_by_candidate: dict[str, list[str]] = {}

    for candidate_id in spec.model_candidate_ids:
        candidate_rows: dict[str, dict[str, Any]] = {}
        candidate_sources: list[str] = []
        for fold_id in expected_fold_ids:
            prediction = by_candidate_fold[(candidate_id, fold_id)]
            candidate_sources.append(prediction.prediction_artifact_id)
            rows = _load_prediction_rows(prediction)
            seen_in_artifact: set[str] = set()
            for row in rows:
                study_id = row["study_id"]
                if study_id in seen_in_artifact:
                    raise ValueError(
                        f"duplicate study {study_id!r} in prediction artifact fold {fold_id}"
                    )
                seen_in_artifact.add(study_id)
                if study_id not in expected_studies:
                    raise ValueError(f"foreign prediction study {study_id!r} in fold {fold_id}")
                assigned_fold = fold_plan.assignments[study_id]
                if assigned_fold != fold_id:
                    raise ValueError(
                        f"prediction study {study_id!r} is assigned to {assigned_fold}, "
                        f"not artifact fold {fold_id}"
                    )
                if study_id in candidate_rows:
                    raise ValueError(f"duplicate OOF prediction for study {study_id!r}")
                candidate_rows[study_id] = {"fold_id": fold_id, "scores": row["scores"]}
        missing = sorted(expected_studies - set(candidate_rows))
        foreign = sorted(set(candidate_rows) - expected_studies)
        if missing or foreign:
            raise ValueError(
                f"OOF study coverage mismatch for candidate {candidate_id}; "
                f"missing={missing[:5]}, foreign={foreign[:5]}"
            )
        all_candidate_rows[candidate_id] = candidate_rows
        source_ids_by_candidate[candidate_id] = sorted(candidate_sources)

        auc_by_target: dict[str, float | None] = {}
        for target_index, target in enumerate(TARGETS):
            observed = [study_id for study_id in sorted(expected_studies)
                        if label_by_study[study_id].values[target] is not None]
            y_true = [int(label_by_study[study_id].values[target]) for study_id in observed]
            y_score = [candidate_rows[study_id]["scores"][target_index] for study_id in observed]
            auc = binary_roc_auc(y_true, y_score)
            auc_by_target[target] = auc
            target_tables.append({"model_candidate_id": candidate_id, "target": target,
                                  "auc": auc, "observed_count": len(observed),
                                  "positive_count": sum(y_true),
                                  "negative_count": len(y_true) - sum(y_true),
                                  "status": ("defined" if auc is not None else
                                      "NO_SUPERVISION" if not observed else
                                      "NO_POSITIVES" if sum(y_true) == 0 else
                                      "NO_NEGATIVES")})
        candidate_evaluations[candidate_id] = Evaluation(
            experiment_id=spec.experiment_id,
            dataset_version_id=spec.dataset_version_id,
            model_candidate_id=candidate_id,
            prediction_artifact_ids=tuple(source_ids_by_candidate[candidate_id]),
            class_names=TARGETS,
            auc_by_class=auc_by_target,
            synthetic=spec.synthetic,
            target_schema_version=spec.target_schema_version,
        )

    for candidate_id in sorted(all_candidate_rows):
        for study_id in sorted(expected_studies):
            record = all_candidate_rows[candidate_id][study_id]
            aggregate_rows.append({"model_candidate_id": candidate_id,
                                   "StudyInstanceUID": study_id,
                                   "fold_id": record["fold_id"],
                                   "scores": record["scores"]})

    evaluation_values = [candidate_evaluations[candidate].to_dict()
                         for candidate in spec.model_candidate_ids]
    evaluation_refs = {candidate: artifact_store.put_json(
        candidate_evaluations[candidate].to_dict()).to_dict()
        for candidate in spec.model_candidate_ids}
    comparison_pairs = []
    for left_id, right_id in combinations(sorted(candidate_evaluations), 2):
        left, right = candidate_evaluations[left_id], candidate_evaluations[right_id]
        comparison_pairs.append({
            "left_model_candidate_id": left_id,
            "right_model_candidate_id": right_id,
            "macro_auc_delta": (right.macro_auc - left.macro_auc
                                if left.macro_auc is not None and right.macro_auc is not None else None),
            "auc_delta_by_target": {
                target: (right.auc_by_class[target] - left.auc_by_class[target]
                         if right.auc_by_class[target] is not None
                         and left.auc_by_class[target] is not None else None)
                for target in TARGETS
            },
        })
    input_identity = digest({
        "experiment_id": spec.experiment_id,
        "fold_plan_id": fold_plan.fold_plan_id,
        "label_record_ids": [label_by_study[study].record_id for study in sorted(expected_studies)],
        "prediction_artifact_ids": sorted(item.prediction_artifact_id for item in predictions),
        "leakage_report_id": leakage_report_id,
        "metric_implementation": OOF_METRIC_IMPLEMENTATION,
    })
    aggregate_ref = artifact_store.put_json({
        "schema_version": OOF_PREDICTIONS_SCHEMA_VERSION,
        "experiment_id": spec.experiment_id,
        "dataset_version_id": spec.dataset_version_id,
        "fold_plan_id": fold_plan.fold_plan_id,
        "target_order": list(TARGETS),
        "model_candidate_ids": sorted(all_candidate_rows),
        "source_prediction_artifact_ids": sorted(item.prediction_artifact_id for item in predictions),
        "leakage_report_id": leakage_report_id,
        "rows": aggregate_rows,
        "input_identity": input_identity,
    }).to_dict()
    table_ref = artifact_store.put_json({
        "schema_version": 1, "metric": "roc_auc", "metric_implementation": OOF_METRIC_IMPLEMENTATION,
        "input_identity": input_identity, "rows": target_tables,
    }).to_dict()
    candidate_metrics = {}
    for candidate_id in sorted(candidate_evaluations):
        evaluation = candidate_evaluations[candidate_id]
        rows_by_target = {row["target"]: row for row in target_tables
                          if row["model_candidate_id"] == candidate_id}
        candidate_metrics[candidate_id] = {
            "evaluation_id": evaluation.evaluation_id,
            "macro_auc": evaluation.macro_auc,
            "unavailable_targets": [target for target in TARGETS
                                    if evaluation.auc_by_class[target] is None],
            "auc_by_target": {
                target: {
                    "auc": rows_by_target[target]["auc"],
                    "supervised_count": rows_by_target[target]["observed_count"],
                    "positive_count": rows_by_target[target]["positive_count"],
                    "negative_count": rows_by_target[target]["negative_count"],
                    "unavailable_reason": (rows_by_target[target]["status"]
                                           if rows_by_target[target]["auc"] is None else None),
                }
                for target in TARGETS
            },
        }
    comparison_ref = artifact_store.put_json({
        "schema_version": 1, "experiment_id": spec.experiment_id,
        "dataset_version_id": spec.dataset_version_id,
        "metric_implementation": OOF_METRIC_IMPLEMENTATION,
        "input_identity": input_identity,
        "model_candidate_ids": sorted(candidate_evaluations),
        "evaluations": {candidate: candidate_evaluations[candidate].evaluation_id
                        for candidate in sorted(candidate_evaluations)},
        "candidate_metrics": candidate_metrics,
        "pairwise_comparisons": comparison_pairs,
    }).to_dict()

    return {
        "status": "ready",
        "experiment_id": spec.experiment_id,
        "dataset_version_id": spec.dataset_version_id,
        "fold_plan_id": fold_plan.fold_plan_id,
        "input_identity": input_identity,
        "metric_implementation": OOF_METRIC_IMPLEMENTATION,
        "evaluations": evaluation_values,
        "artifacts": {
            "oof_aggregate": aggregate_ref,
            "target_auc_table": table_ref,
            "comparison_report": comparison_ref,
            "evaluations": evaluation_refs,
        },
        "_aggregated_row_count": len(aggregate_rows),
    }


def evaluate_oof(spec: ExperimentSpec, fold_plan: FoldPlanManifest,
                 labels: Sequence[LabelRecord], predictions: Sequence[PredictionArtifact],
                 artifact_store: JsonArtifactStore) -> dict[str, Any]:
    """Measure the aggregation while keeping timing out of deterministic artifacts."""
    owns_tracemalloc = not tracemalloc.is_tracing()
    if owns_tracemalloc:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        report = _evaluate_oof_impl(spec, fold_plan, labels, predictions, artifact_store)
        elapsed = max(time.perf_counter() - started, 1e-12)
        _, peak_bytes = tracemalloc.get_traced_memory()
        row_count = report.pop("_aggregated_row_count")
        report["performance"] = {
            "rows": row_count,
            "elapsed_seconds": elapsed,
            "rows_per_second": row_count / elapsed,
            "peak_traced_bytes": peak_bytes,
        }
        return report
    finally:
        if owns_tracemalloc:
            tracemalloc.stop()


def evaluate_oof_job(job_path: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Run the public JSON job format used by ``rsna evaluate oof``."""
    path = Path(job_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("OOF job must be a JSON object")
    spec = ExperimentSpec.from_dict(document["experiment"])
    fold_plan = load_fold_plan(document["fold_plan"])
    labels_data = json.loads(Path(document["labels"]).read_text(encoding="utf-8"))
    if isinstance(labels_data, dict):
        labels_data = labels_data.get("labels")
    if not isinstance(labels_data, list):
        raise ValueError("labels file must contain a list or an object with a labels list")
    labels = [LabelRecord.from_dict(item) for item in labels_data]
    predictions_data = document["prediction_artifacts"]
    if isinstance(predictions_data, str):
        predictions_data = json.loads(Path(predictions_data).read_text(encoding="utf-8"))
    if isinstance(predictions_data, dict):
        predictions_data = predictions_data.get("prediction_artifacts")
    if not isinstance(predictions_data, list):
        raise ValueError("prediction_artifacts must be a list or a JSON file containing a list")
    predictions = [PredictionArtifact.from_dict(item) for item in predictions_data]
    store = JsonArtifactStore(output_dir or document.get("output_dir", "artifacts/evaluation"))
    return evaluate_oof(spec, fold_plan, labels, predictions, store)
