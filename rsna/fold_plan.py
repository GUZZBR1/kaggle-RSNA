"""Adapters around the canonical Issue 6 fold-manifest API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from .contracts import DatasetVersion
from .data.models import index_from_dict
from .folds import FoldPlanManifest, generate_fold_plan, load_fold_plan, save_fold_plan
from .folds.generate import _dataset_version_id, _records
from .identity import digest
from .leakage import validate_leakage


class UsageError(ValueError):
    """A valid command shape with invalid command values or constraints."""


class FoldValidationReport(TypedDict):
    passed: bool
    fold_plan_id: str
    dataset_version_id: str
    errors: list[str]
    warnings: list[str]
    fold_statistics: dict[str, Any]
    leakage: dict[str, Any]
    balance: dict[str, Any]
    coverage: dict[str, Any]


def _load_dataset(path: str | Path) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    raw_bytes = Path(path).read_bytes()
    data = json.loads(raw_bytes)
    if not isinstance(data, dict):
        raise ValueError("dataset manifest must be a JSON object")
    index = data.get("dataset_index", data.get("index", data))
    if not isinstance(index, dict) or not isinstance(index.get("studies"), list):
        raise ValueError("dataset manifest must be an object containing studies")
    if isinstance(index.get("index_id"), str) and len(index["index_id"]) == 64:
        index_from_dict(index)
    version = data.get("dataset_version")
    if isinstance(version, dict):
        dataset_id = DatasetVersion.from_dict(version).dataset_version_id
    elif data.get("dataset_version_id") or index.get("dataset_version_id"):
        dataset_id = data.get("dataset_version_id") or index["dataset_version_id"]
        if (not isinstance(dataset_id, str) or len(dataset_id) != 64
                or any(char not in "0123456789abcdef" for char in dataset_id)):
            raise ValueError("dataset_version_id must be a SHA-256 digest")
    else:
        dataset_id = _dataset_version_id(index, _records(index))

    studies: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in index["studies"]:
        if not isinstance(item, dict):
            raise ValueError("study records must be objects")
        study_id = item.get("study_instance_uid") or item.get("study_id")
        if not isinstance(study_id, str) or not study_id:
            raise ValueError("every study needs study_instance_uid or study_id")
        if study_id in seen_ids:
            raise ValueError(f"duplicate study ID in dataset manifest: {study_id}")
        seen_ids.add(study_id)
        series = item.get("series", [])
        if not isinstance(series, list):
            series = []
        for series_record in series:
            if (isinstance(series_record, dict) and series_record.get("study_instance_uid")
                    and item.get("study_instance_uid")
                    and series_record["study_instance_uid"] != item["study_instance_uid"]):
                raise ValueError(f"series belongs to a different study: {series_record.get('series_instance_uid')}")
        labels = item.get("labels", {})
        if not isinstance(labels, dict):
            labels = {}
        labels = {name: int(value) if isinstance(value, float) and value in (0.0, 1.0) else value
                  for name, value in labels.items()}
        studies.append({"study_id": study_id, "study_instance_uid": study_id,
                        "patient_id": item.get("patient_id"),
                        "n_series": len(series) if series else int(item.get("n_series", 0)),
                        "n_slices": (sum(len(s.get("slices", [])) for s in series if isinstance(s, dict))
                                     if series else int(item.get("n_slices", 0))),
                        "series_ids": [s.get("series_instance_uid") or s.get("series_id")
                                       for s in series if isinstance(s, dict)
                                       and (s.get("series_instance_uid") or s.get("series_id"))],
                        "series": series,
                        "labels": labels})
    return dataset_id, studies, index


def load_dataset(path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    dataset_id, studies, _ = _load_dataset(path)
    return dataset_id, studies


def _manifest(plan: dict[str, Any]) -> FoldPlanManifest:
    allowed = set(FoldPlanManifest.__dataclass_fields__) - {"application_warnings", "fold_plan_id"}
    return FoldPlanManifest(**{key: plan[key] for key in allowed if key in plan})


def _with_cli_fields(plan: dict[str, Any]) -> dict[str, Any]:
    return {**plan, "fold_ids": [f"fold_{i}" for i in range(plan["n_folds"])]}


def read_plan(path: str | Path) -> dict[str, Any]:
    try:
        plan = load_fold_plan(path)
    except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
        raise ValueError(f"corrupt fold plan: {exc}") from exc
    return _with_cli_fields(plan.to_dict())


def save_plan(plan: dict[str, Any], path: str | Path) -> str:
    return save_fold_plan(_manifest(plan), path)


def make_plan(dataset_id: str, studies: list[dict[str, Any]], n_folds: int, strategy: str,
              seed: int, grouping_key: str, *, locked: bool = False) -> dict[str, Any]:
    if grouping_key != "patient_id":
        raise UsageError("grouping_key must be patient_id")
    labels = {row["study_id"]: {
        name: int(value) if isinstance(value, float) and value in (0.0, 1.0) else value
        for name, value in row.get("labels", {}).items()
    } for row in studies if row.get("labels")}
    try:
        manifest = generate_fold_plan({"dataset_version_id": dataset_id, "studies": studies},
                                      n_folds=n_folds, strategy=strategy,
                                      random_state=seed, labels=labels or None, locked=locked)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    return _with_cli_fields(manifest.to_dict())


def _leakage_report(index: dict[str, Any], studies: list[dict[str, Any]], plan: dict[str, Any],
                     dataset_id: str | None) -> dict[str, Any]:
    if isinstance(index.get("index_id"), str) and len(index["index_id"]) == 64:
        report = validate_leakage(index, assignments=plan["assignments"],
                                  dataset_version_id=dataset_id, policy="strict")
    else:
        records: list[dict[str, Any]] = []
        for study in studies:
            fold = plan["assignments"].get(study["study_id"])
            records.append({"patient_id": study.get("patient_id"),
                            "study_uid": study["study_id"], "fold_id": fold})
            for series_id in study.get("series_ids", []):
                records.append({"patient_id": study.get("patient_id"),
                                "study_uid": study["study_id"], "series_uid": series_id,
                                "fold_id": fold})
        report = validate_leakage(records, assignments=plan["assignments"],
                                  dataset_version_id=dataset_id, policy="strict")
    return report.to_dict()


def validate(plan: dict[str, Any], dataset_id: str | None = None,
             studies: list[dict[str, Any]] | None = None, *, require_coverage: bool = True,
             dataset_index: dict[str, Any] | None = None) -> FoldValidationReport:
    errors: list[str] = []
    if dataset_id and plan["dataset_version_id"] != dataset_id:
        errors.append("dataset_version_id does not match dataset manifest")
    assignments = plan["assignments"]
    known = {study["study_id"]: study for study in (studies or [])}
    missing = sorted(set(known) - set(assignments)) if studies is not None else []
    unknown = sorted(set(assignments) - set(known)) if studies is not None else []
    if unknown:
        errors.append("assignments contain unknown studies")
    if missing and require_coverage:
        errors.append("studies are missing assignments")
    if dataset_index is not None:
        records = _records({"studies": studies or []})
        current_fingerprints = {row["study_id"]: digest(
            {"patient_id": row["patient_id"], "series_ids": row["series_ids"],
             "series_available": row["series_available"]}) for row in records}
        changed = sorted(study for study in set(current_fingerprints) & set(plan.get("study_fingerprints", {}))
                         if current_fingerprints[study] != plan["study_fingerprints"][study])
        if changed:
            errors.append(f"study identity changed since FoldPlan generation: {changed[:5]}")
    fold_ids = plan["fold_ids"]
    invalid_folds = sorted(set(assignments.values()) - set(fold_ids))
    if invalid_folds:
        errors.append("assignments contain invalid fold IDs")
    leakage = _leakage_report(dataset_index or {"studies": studies or []},
                              studies or [], plan, dataset_id)
    if not leakage["passed"]:
        errors.append("Leakage Guard validation failed")
    fold_stats = plan.get("statistics", {})
    folds = fold_stats.get("folds", {})
    warnings = list(plan.get("warnings", []))
    if fold_stats.get("total", {}).get("targets") and not any(
            (target.get("positive", 0) + target.get("negative", 0) + target.get("soft_count", 0))
            for target in fold_stats["total"]["targets"].values()):
        warnings.append("no observed labels are present; target prevalence balance is unavailable")
    sizes = [fold.get("n_studies", 0) for fold in folds.values()]
    total = fold_stats.get("total", {})
    prevalence = {}
    for target in total.get("targets", {}):
        values = [fold.get("targets", {}).get(target, {}).get("prevalence") for fold in folds.values()]
        known_values = [value for value in values if value is not None]
        prevalence[target] = {"prevalence_range": [min(known_values), max(known_values)] if known_values else None}
    balance = {"study_distribution": sizes,
               "patient_distribution": [fold.get("n_patients", 0) for fold in folds.values()],
               "target_prevalence": prevalence,
               **fold_stats.get("diagnostics", {})}
    expected = len(known) if studies is not None else len(assignments)
    covered = len(set(assignments) & set(known)) if studies is not None else len(assignments)
    coverage = {"expected_studies": expected, "assigned_studies": len(assignments),
                "unassigned_studies": missing, "unknown_assigned_studies": unknown,
                "coverage_fraction": covered / expected if expected else 1.0}
    return {"passed": not errors, "fold_plan_id": plan["fold_plan_id"],
            "dataset_version_id": plan["dataset_version_id"], "errors": errors,
            "warnings": warnings, "fold_statistics": fold_stats,
            "leakage": {"guard_report": leakage,
                        "patient_leakage_groups": [issue["entity_id"] for issue in leakage["issues"]
                                                   if issue["type"] == "PATIENT_CROSS_FOLD"],
                        "series_containment": {"passed": not any(
                            issue["type"] in {"SERIES_CROSS_FOLD", "SERIES_STUDY_MISMATCH"}
                            for issue in leakage["issues"]) }},
            "balance": balance, "coverage": coverage}


def diff_plans(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    a, b = left["assignments"], right["assignments"]
    common = set(a) & set(b)
    moved = {key: {"from": a[key], "to": b[key]} for key in sorted(common) if a[key] != b[key]}
    added, removed = sorted(set(b) - set(a)), sorted(set(a) - set(b))
    partition = lambda values: frozenset(frozenset(k for k, v in values.items() if v == fold)
                                         for fold in set(values.values()))
    exact = a == b
    semantic = set(a) == set(b) and partition(a) == partition(b)
    before_groups = left.get("study_groups", {})
    after_groups = right.get("study_groups", {})
    moved_groups = sum(1 for group in set(before_groups.values()) & set(after_groups.values())
                       if len({a[s] for s, g in before_groups.items() if g == group and s in common}) == 1
                       and len({b[s] for s, g in after_groups.items() if g == group and s in common}) == 1
                       and {a[s] for s, g in before_groups.items() if g == group and s in common}
                       != {b[s] for s, g in after_groups.items() if g == group and s in common})
    return {"same_dataset_version": left["dataset_version_id"] == right["dataset_version_id"],
            "same_strategy": left["strategy"] == right["strategy"],
            "same_seed": left["random_state"] == right["random_state"],
            "same_number_of_folds": left["n_folds"] == right["n_folds"],
            "exact_equality": exact, "semantic_partition_equality": semantic,
            "studies_moved": len(moved), "patients_moved": moved_groups,
            "added_studies": added, "removed_studies": removed, "moved": moved}
