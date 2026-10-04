"""Dataset summaries and numerical statistics over metadata indexes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
from typing import Any

from .query import (DatasetIndex, TABLE_IDS, distribution, ensure_index,
                    filter_entities, warning_codes)


def _selected(index: DatasetIndex, *, plane: str | None = None,
              laterality: str | None = None, warning_code: str | None = None) -> DatasetIndex:
    if plane is None and laterality is None and warning_code is None:
        return index
    series = filter_entities(index, "series", plane=plane, laterality=laterality,
                             warning_code=warning_code)
    series_ids = {str(row["series_uid"]) for row in series if row.get("series_uid") is not None}
    study_ids = {str(row["study_uid"]) for row in series if row.get("study_uid") is not None}
    slices = [row for row in index.slices if str(row.get("series_uid")) in series_ids]
    studies = [row for row in index.studies if str(row.get("study_uid")) in study_ids]
    result = DatasetIndex(metadata=dict(index.metadata), studies=studies, series=series, slices=slices,
                          source=index.source, entities_present=set(index.entities_present))
    # Global counts cannot describe a filtered subset.
    for key in ("statistics", "counts", "invalid_files", "missing_metadata", "warnings", "duplicate_uids"):
        result.metadata.pop(key, None)
    return result


def _warnings(index: DatasetIndex) -> dict[str, int]:
    codes = Counter(warning_codes(index.metadata))
    for table in TABLE_IDS:
        for row in getattr(index, table):
            codes.update(warning_codes(row))
    return dict(sorted(codes.items()))


def build_dataset_summary(index: DatasetIndex | Mapping[str, Any], *, plane: str | None = None,
                          laterality: str | None = None,
                          warning_code: str | None = None) -> dict[str, Any]:
    index = _selected(ensure_index(index), plane=plane, laterality=laterality, warning_code=warning_code)
    result: dict[str, Any] = {}
    for key in ("dataset_id", "dataset_version", "dataset_version_id", "schema_version"):
        if key in index.metadata:
            result[key] = index.metadata[key]
    # Declared statistics are retained only when no loaded table supersedes them.
    for key in ("statistics", "counts"):
        declared = index.metadata.get(key)
        if isinstance(declared, Mapping):
            for name in ("studies", "patients", "series", "slices", "invalid_files", "missing_metadata", "duplicate_uids"):
                if name in declared:
                    result[name] = declared[name]
    for name in ("invalid_files", "missing_metadata"):
        if name in index.metadata:
            value = index.metadata[name]
            result[name] = len(value) if isinstance(value, list) else value
    for table in index.entities_present:
        result[table] = len(getattr(index, table))
    patient_ids = {str(row["patient_id"]) for table in TABLE_IDS for row in getattr(index, table)
                   if row.get("patient_id") is not None}
    if patient_ids:
        result["patients"] = len(patient_ids)
    elif "studies" in index.entities_present and not index.studies:
        result["patients"] = 0
    if index.entities_present:
        duplicates: dict[str, int] = {}
        for table in sorted(index.entities_present):
            key = TABLE_IDS[table]
            counts = Counter(str(row[key]) for row in getattr(index, table) if row.get(key) is not None)
            duplicates[table] = sum(count - 1 for count in counts.values() if count > 1)
        result["duplicate_uids"] = duplicates
    records = index.series if "series" in index.entities_present else index.slices
    planes = distribution(records, "plane")
    laterality = distribution(records, "laterality")
    if planes:
        result["planes"] = planes
    if laterality:
        result["laterality_distribution"] = laterality
    warnings = _warnings(index)
    if warnings or any("warnings" in row for table in TABLE_IDS for row in getattr(index, table)) or "warnings" in index.metadata:
        declared_warnings = index.metadata.get("warnings")
        result["warnings"] = declared_warnings if isinstance(declared_warnings, int) else sum(warnings.values())
        if warnings:
            result["warning_frequency"] = warnings
    return result


def numerical_summary(values: list[float | int]) -> dict[str, Any]:
    """Describe observed finite numbers; an empty observation has no fake extrema."""
    ordered = sorted(float(value) for value in values if isinstance(value, (int, float))
                     and not isinstance(value, bool) and math.isfinite(value))
    if not ordered:
        return {"count": 0}

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"count": len(ordered), "min": ordered[0], "max": ordered[-1],
            "mean": sum(ordered) / len(ordered), "p25": percentile(0.25),
            "p50": percentile(0.5), "p75": percentile(0.75), "p90": percentile(0.9),
            "p95": percentile(0.95)}


def build_dataset_stats(index: DatasetIndex | Mapping[str, Any], *, plane: str | None = None,
                        laterality: str | None = None,
                        warning_code: str | None = None) -> dict[str, Any]:
    index = _selected(ensure_index(index), plane=plane, laterality=laterality, warning_code=warning_code)
    result: dict[str, Any] = {}
    if {"studies", "series"}.issubset(index.entities_present):
        counts = [len(index._series_by_study.get(str(row["study_uid"]), []))
                  for row in index.studies if row.get("study_uid") is not None]
        result["series_per_study"] = numerical_summary(counts)
    if "series" in index.entities_present:
        if "slices" in index.entities_present:
            counts = [len(index._slices_by_series.get(str(row["series_uid"]), []))
                      for row in index.series if row.get("series_uid") is not None]
            result["slices_per_series"] = numerical_summary(counts)
        elif any(isinstance(row.get("n_slices"), (int, float)) for row in index.series):
            result["slices_per_series"] = numerical_summary([row["n_slices"] for row in index.series
                                                             if isinstance(row.get("n_slices"), (int, float))])
    spacing = [row["slice_spacing"] for row in index.slices + index.series
               if isinstance(row.get("slice_spacing"), (int, float))]
    if spacing:
        result["slice_spacing"] = numerical_summary(spacing)
    records = index.series if "series" in index.entities_present else index.slices
    for field_name, output_name in (("plane", "plane_counts"), ("laterality", "laterality_counts")):
        counts = distribution(records, field_name)
        if counts:
            result[output_name] = counts
    warnings = _warnings(index)
    if warnings:
        result["warning_frequency"] = warnings
    return result
