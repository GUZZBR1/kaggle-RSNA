"""Summaries over the canonical DICOM metadata index."""
from __future__ import annotations

from collections import Counter
import math
from typing import Any

from ..data import DatasetIndex, describe_series_orientation, resolve_series_laterality
from .query import filter_entities


def _counts(index: DatasetIndex, *, plane: str | None, laterality: str | None,
            warning_code: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_studies = filter_entities(index, "study")
    series = filter_entities(index, "series", plane=plane, laterality=laterality, warning_code=warning_code)
    series_ids = {row["series_id"] for row in series}
    studies = all_studies if all(value is None for value in (plane, laterality, warning_code)) else [
        study for study in all_studies if any(row["series_id"] in series_ids for row in study["series"])
    ]
    slices = [item for row in series for item in row.get("slices", [])]
    return studies, series, slices

def _warnings(index: DatasetIndex, records: list[dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    all_warnings = list(index.warnings)
    for row in records:
        all_warnings.extend(row.get("warnings", []))
    counts = Counter(str(warning) for warning in all_warnings)
    return all_warnings, dict(sorted(counts.items()))


def build_dataset_summary(index: DatasetIndex, *, plane: str | None = None,
                          laterality: str | None = None, warning_code: str | None = None) -> dict[str, Any]:
    studies, series, slices = _counts(index, plane=plane, laterality=laterality, warning_code=warning_code)
    filtered = any(value is not None for value in (plane, laterality, warning_code))
    result: dict[str, Any] = {"dataset_index_id": index.index_id,
                              "dataset_version_binding": "unavailable",
                              "studies": len(studies), "series": len(series), "slices": len(slices)}
    patient_ids = {row.get("patient_id") for row in studies if row.get("patient_id") is not None}
    if patient_ids:
        result["patients"] = len(patient_ids)
    elif studies and all(row.get("patient_id") is not None for row in studies):
        result["patients"] = 0
    # Resolve these with the canonical domain helpers; the CLI does not duplicate their rules.
    if series:
        result["planes"] = dict(sorted(Counter(describe_series_orientation(row.get("metadata", {})).plane for row in series).items()))
        result["laterality_distribution"] = dict(sorted(Counter(resolve_series_laterality(row.get("metadata", {})).resolved for row in series).items()))
    stats = dict(index.statistics)
    if not filtered:
        result.update({key: stats[key] for key in ("invalid_files", "missing_uid_count", "duplicate_sop_uid_count") if key in stats})
    duplicate_names = ("duplicate_sop_uid_count", "duplicate_series_uid_count", "conflicting_patient_id_count")
    duplicates = {key: stats[key] for key in duplicate_names if key in stats}
    if duplicates:
        result["duplicate_conflict_counts"] = duplicates
    warnings, frequency = _warnings(index, studies + series + slices)
    if warnings:
        result["warnings"] = warnings
        result["warning_frequency"] = frequency
    return result


def numerical_summary(values: list[float | int]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values if isinstance(value, (int, float))
                     and not isinstance(value, bool) and math.isfinite(value))
    if not ordered:
        return {"count": 0}
    def percentile(fraction: float) -> float:
        pos = (len(ordered) - 1) * fraction
        lower = math.floor(pos); upper = math.ceil(pos)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)
    return {"count": len(ordered), "min": ordered[0], "max": ordered[-1],
            "mean": sum(ordered) / len(ordered), "p25": percentile(.25), "p50": percentile(.5),
            "p75": percentile(.75), "p90": percentile(.9), "p95": percentile(.95)}


def build_dataset_stats(index: DatasetIndex, *, plane: str | None = None,
                        laterality: str | None = None, warning_code: str | None = None) -> dict[str, Any]:
    studies, series, slices = _counts(index, plane=plane, laterality=laterality, warning_code=warning_code)
    series_ids = {row["series_id"] for row in series}
    result: dict[str, Any] = {
        "series_per_study": numerical_summary([sum(item["series_id"] in series_ids
            for item in study["series"]) for study in studies]),
        "slices_per_series": numerical_summary([row["n_slices"] for row in series]),
    }
    warning_count = Counter()
    for row in studies + series + slices:
        warning_count.update(str(item) for item in row.get("warnings", []))
    if warning_count:
        result["warning_frequency"] = dict(sorted(warning_count.items()))
    return result
