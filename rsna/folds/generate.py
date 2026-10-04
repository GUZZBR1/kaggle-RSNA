"""Deterministic group assignment algorithms and public generation API."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping

from ..identity import digest
from .models import FoldPlanManifest
from .statistics import summarize
from .validate import validate_leakage


def generate_fold_plan(dataset: Any, n_folds: int = 5, strategy: str = "group",
                       random_state: int = 42, labels: Mapping[str, Mapping[str, Any]] | None = None,
                       *, dataset_version_id: str | None = None, locked: bool = False,
                       size_deviation_threshold: float = 0.20,
                       prevalence_range_threshold: float = 0.20) -> FoldPlanManifest:
    """Generate a stable plan from index records or simple study mappings.

    Patient IDs are the indivisible unit when available, with study-level fallback for
    missing IDs. Reused SeriesInstanceUIDs connect otherwise separate units so they
    cannot cross folds. Group assignments are deterministic for a fixed input and seed.
    """
    if type(n_folds) is not int or n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if type(random_state) is not int or random_state < 0:
        raise ValueError("random_state must be a nonnegative integer")
    if strategy not in {"group", "multilabel_group_stratified"}:
        raise ValueError("strategy must be 'group' or 'multilabel_group_stratified'")
    for threshold_name, threshold in (("size_deviation_threshold", size_deviation_threshold),
                                      ("prevalence_range_threshold", prevalence_range_threshold)):
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold < 0:
            raise ValueError(f"{threshold_name} must be a nonnegative number")
    records = _records(dataset)
    study_ids = [row["study_id"] for row in records]
    if len(set(study_ids)) != len(study_ids):
        raise ValueError("duplicate study IDs in dataset")
    if n_folds > len(study_ids):
        raise ValueError(f"n_folds ({n_folds}) cannot exceed number of studies ({len(study_ids)})")
    for row in records:
        if any("conflicting PatientID" in warning for warning in row["warnings"]):
            raise ValueError(f"inconsistent PatientID values for study {row['study_id']!r}")
    records.sort(key=lambda row: row["study_id"])
    labels = _normalize_labels(labels, set(study_ids))
    if strategy == "multilabel_group_stratified" and not any(
            value is not None for row in labels.values() for value in row.values()):
        raise ValueError("multilabel_group_stratified requires at least one observed label")

    group_for_study, group_members, grouping_key = _groups(records)
    if n_folds > len(group_members):
        raise ValueError(f"n_folds ({n_folds}) cannot exceed number of indivisible groups ({len(group_members)})")
    seed = random_state
    if strategy == "group":
        group_folds = _balanced_group_assignment(group_members, n_folds, seed)
    else:
        group_folds = _stratified_group_assignment(group_members, records, labels, n_folds, seed)
    assignments = {study: group_folds[group] for study, group in group_for_study.items()}
    validate_leakage(records, assignments)

    identity = dataset_version_id or _dataset_version_id(dataset, records)
    stats, stat_warnings = summarize(records, assignments, n_folds, group_for_study, labels,
                                     size_deviation_threshold=size_deviation_threshold,
                                     prevalence_range_threshold=prevalence_range_threshold)
    warnings = []
    if any(not row.get("patient_id") for row in records):
        warnings.append("some studies lack PatientID; those studies use Study grouping")
    if not labels:
        warnings.append("no study-level labels supplied; multilabel statistics are unavailable")
    warnings.extend(stat_warnings)
    fingerprints = {row["study_id"]: digest({"patient_id": row["patient_id"], "series_ids": row["series_ids"],
                                               "series_available": row["series_available"]})
                    for row in records}
    return FoldPlanManifest(dataset_version_id=identity, strategy=strategy, n_folds=n_folds,
                            random_state=seed, grouping_key=grouping_key, assignments=assignments,
                            group_assignments=group_folds, study_groups=group_for_study,
                            study_fingerprints=fingerprints,
                            configuration={"algorithm": ("deterministic_group_greedy_v1" if strategy == "group"
                                                        else "deterministic_multilabel_greedy_v1"),
                                           "size_deviation_threshold": size_deviation_threshold,
                                           "prevalence_range_threshold": prevalence_range_threshold,
                                           "labels_sha256": digest(labels) if labels else None},
                            statistics=stats,
                            warnings=tuple(warnings), locked=locked)


def _records(dataset: Any) -> list[dict[str, Any]]:
    if hasattr(dataset, "studies"):
        source = dataset.studies
    elif isinstance(dataset, Mapping) and "studies" in dataset:
        source = dataset["studies"]
    else:
        source = dataset
    result = []
    for item in source:
        row = _mapping(item)
        study_id = row.get("study_instance_uid") or row.get("study_id") or row.get("id")
        if not study_id:
            raise ValueError("every study needs study_instance_uid, study_id, or id")
        series_available = "series" in row or "series_ids" in row
        series = row.get("series", row.get("series_ids", ())) or ()
        series_ids = []
        for value in series:
            value = _mapping(value) if not isinstance(value, str) else {"series_instance_uid": value}
            series_ids.append(value.get("series_instance_uid") or value.get("series_id"))
        result.append({"study_id": str(study_id).strip(),
                       "patient_id": _clean_id(row.get("patient_id", row.get("PatientID"))),
                       "series_ids": tuple(sorted({str(v).strip() for v in series_ids if v and str(v).strip()})),
                       "series_available": series_available,
                       "warnings": tuple(row.get("warnings", ()))})
    if not result:
        raise ValueError("dataset contains no studies")
    return result


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return {name: getattr(value, name) for name in
            ("study_instance_uid", "study_id", "patient_id", "series", "warnings")
            if hasattr(value, name)}


def _clean_id(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _groups(records: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, list[str]], str]:
    # Union patient/study units whenever a repeated series identifier crosses them.
    parents: dict[str, str] = {}
    study_unit: dict[str, str] = {}
    series_owner: dict[str, str] = {}
    def find(value: str) -> str:
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]
    def union(a: str, b: str) -> None:
        left, right = find(a), find(b)
        if left != right:
            low, high = sorted((left, right))
            parents[high] = low
    has_patient = any(row.get("patient_id") for row in records)
    for row in records:
        study = row["study_id"]
        unit = f"patient:{row['patient_id']}" if row.get("patient_id") else f"study:{study}"
        study_unit[study] = unit
        find(unit)
        for series in row["series_ids"]:
            if series in series_owner:
                union(unit, series_owner[series])
            else:
                series_owner[series] = unit
    components: dict[str, list[str]] = defaultdict(list)
    for study, unit in study_unit.items():
        components[find(unit)].append(study)
    study_group: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for members in components.values():
        atomic = sorted({study_unit[study] for study in members})
        key = atomic[0] if len(atomic) == 1 else "linked:" + digest(atomic)
        groups[key] = sorted(members)
        for study in members:
            study_group[study] = key
    return study_group, dict(sorted(groups.items())), "patient_id_with_study_fallback" if has_patient else "study_id"


def _balanced_group_assignment(groups: Mapping[str, list[str]], n_folds: int, seed: int) -> dict[str, str]:
    def tie(key: str) -> str:
        return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    ordered = sorted(groups, key=lambda key: (-len(groups[key]), tie(key), key))
    sizes = [0] * n_folds
    counts = [0] * n_folds
    result = {}
    for key in ordered:
        fold = min(range(n_folds), key=lambda i: (sizes[i], counts[i], hashlib.sha256(f"{seed}:{key}:fold_{i}".encode()).hexdigest()))
        result[key] = f"fold_{fold}"
        sizes[fold] += len(groups[key])
        counts[fold] += 1
    return result


def _stratified_group_assignment(groups: Mapping[str, list[str]], records: list[dict[str, Any]],
                                 labels: Mapping[str, Mapping[str, Any]], n_folds: int,
                                 seed: int) -> dict[str, str]:
    """Greedy group allocation minimizing normalized positive/known-count imbalance.

    Study labels are aggregated at the indivisible group level; null labels contribute
    neither positive nor negative counts. This is a deterministic approximation, not an
    exact multilabel stratification solver.
    """
    targets = sorted({name for values in labels.values() for name in values})
    vectors: dict[str, list[float]] = {}
    for group, studies in groups.items():
        vector = [float(len(studies))]
        for target in targets:
            values = [labels.get(study, {}).get(target) for study in studies]
            vector.extend((sum(value in (1, True) for value in values),
                           sum(value in (0, False) for value in values)))
        vectors[group] = vector
    totals = [sum(vector[i] for vector in vectors.values()) for i in range(len(next(iter(vectors.values()))))]
    def tie(key: str) -> str:
        return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    ordered = sorted(groups, key=lambda key: (-sum(vectors[key][1::2]), -len(groups[key]), tie(key), key))
    fold_values = [[0.0] * len(totals) for _ in range(n_folds)]
    result = {}
    for group in ordered:
        empty_folds = [index for index, values in enumerate(fold_values) if values[0] == 0]
        eligible_folds = empty_folds if empty_folds and len(ordered) - len(result) >= len(empty_folds) else range(n_folds)
        def cost(fold: int) -> tuple[float, str]:
            scores = []
            for index, value in enumerate(vectors[group]):
                target = totals[index] / n_folds
                scale = max(target, 1.0)
                scores.append(((fold_values[fold][index] + value - target) / scale) ** 2)
            return sum(scores), hashlib.sha256(f"{seed}:{group}:fold_{fold}".encode()).hexdigest()
        chosen = min(eligible_folds, key=cost)
        result[group] = f"fold_{chosen}"
        for index, value in enumerate(vectors[group]):
            fold_values[chosen][index] += value
    return result


def _normalize_labels(labels: Mapping[str, Mapping[str, Any]] | None,
                      study_ids: set[str]) -> dict[str, dict[str, int | None]]:
    if labels is None:
        return {}
    if "labels" in labels and isinstance(labels["labels"], Mapping):
        labels = labels["labels"]  # type: ignore[assignment]
    result = {}
    for study, targets in labels.items():
        if study not in study_ids:
            raise ValueError(f"labels reference unknown study {study!r}")
        if not isinstance(targets, Mapping):
            raise ValueError(f"labels for study {study!r} must be an object")
        result[str(study)] = {}
        for name, value in targets.items():
            if not isinstance(name, str) or not name:
                raise ValueError("label target names must be nonempty strings")
            if value is None:
                normalized = None
            elif value in (True, 1) and type(value) in (bool, int):
                normalized = 1
            elif value in (False, 0) and type(value) in (bool, int):
                normalized = 0
            else:
                raise ValueError(f"label {name!r} for study {study!r} must be 0, 1, or null")
            result[str(study)][name] = normalized
    return result


def _dataset_version_id(dataset: Any, records: list[dict[str, Any]]) -> str:
    if hasattr(dataset, "dataset_version_id"):
        return dataset.dataset_version_id
    if isinstance(dataset, Mapping):
        supplied = dataset.get("dataset_version_id")
        if supplied:
            return supplied
        version = dataset.get("dataset_version")
        if isinstance(version, Mapping) and version.get("dataset_version_id"):
            return version["dataset_version_id"]
        identity = dataset.get("index_id", dataset.get("dataset_index_id"))
    else:
        identity = getattr(dataset, "index_id", None)
    return digest({"source": identity, "studies": records})
