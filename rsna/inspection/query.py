"""Queries over the canonical metadata-only dataset index."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ..data import (DatasetIndex, SeriesRecord, SliceRecord, StudyRecord, build_orientation_provenance,
describe_series_orientation, load_manifest, resolve_series_laterality)


class DatasetReadError(ValueError):
    """The dataset manifest could not be read or parsed."""


class EntityNotFoundError(ValueError):
    """The requested UID is absent from the canonical index."""


def ensure_index(index: DatasetIndex) -> DatasetIndex:
    if not isinstance(index, DatasetIndex):
        raise TypeError("expected rsna.data.DatasetIndex")
    return index


def load_dataset_index(path: str | Path) -> DatasetIndex:
    """Load canonical records without rescanning source DICOM files."""
    try:
        return load_manifest(path)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise DatasetReadError(f"cannot load dataset manifest {path}: {exc}") from exc


def _study(index: DatasetIndex, uid: str) -> StudyRecord:
    matches = [item for item in index.studies if item.study_instance_uid == uid or item.study_id == uid]
    if not matches:
        raise EntityNotFoundError(f"study UID {uid!r} was not found")
    if len(matches) > 1:
        raise EntityNotFoundError(f"study UID {uid!r} is ambiguous")
    return matches[0]


def _series(index: DatasetIndex, uid: str) -> tuple[SeriesRecord, StudyRecord | None]:
    matches = [(study, series) for study in index.studies for series in study.series
               if series.series_instance_uid == uid or series.series_id == uid]
    if not matches:
        raise EntityNotFoundError(f"series UID {uid!r} was not found")
    if len(matches) > 1:
        raise EntityNotFoundError(f"series UID {uid!r} is ambiguous")
    study, series = matches[0]
    return series, study


def _slice(index: DatasetIndex, uid: str) -> tuple[SliceRecord, SeriesRecord, StudyRecord | None]:
    matches = [(study, series, item) for study in index.studies for series in study.series for item in series.slices
               if item.metadata.get("SOPInstanceUID") == uid or item.slice_id == uid]
    if not matches:
        raise EntityNotFoundError(f"slice UID {uid!r} was not found")
    if len(matches) > 1:
        raise EntityNotFoundError(f"slice UID {uid!r} is ambiguous")
    study, series, item = matches[0]
    return item, series, study


def warning_codes(record: Any) -> list[str]:
    warnings = getattr(record, "warnings", ())
    return [str(item.get("code", item.get("message", "UNKNOWN"))) if isinstance(item, dict) else str(item)
            for item in warnings]


def inspect_study(index: DatasetIndex, study_id: str) -> dict[str, Any]:
    study = _study(ensure_index(index), study_id)
    return {"study_instance_uid": study.study_instance_uid, "patient_id": study.patient_id,
            "study_id": study.study_id, "n_series": len(study.series),
            "n_slices": sum(series.n_slices for series in study.series),
            "series": [{"series_instance_uid": item.series_instance_uid,
                        "series_id": item.series_id, "n_slices": item.n_slices} for item in study.series],
            "warnings": list(study.warnings), "provenance": None}


def inspect_series(index: DatasetIndex, series_id: str) -> dict[str, Any]:
    series, study = _series(ensure_index(index), series_id)
    metadata = series.metadata
    keys = ("SeriesDescription", "ProtocolName", "Modality", "ImageOrientationPatient",
            "PixelSpacing", "Rows", "Columns", "ImagePositionPatient")
    result = {key: metadata[key] for key in keys if key in metadata and metadata[key] is not None}
    result.update({"series_instance_uid": series.series_instance_uid, "series_id": series.series_id,
                   "study_instance_uid": series.study_instance_uid,
                   "study_id": study.study_id if study else None, "n_slices": series.n_slices,
                   "warnings": list(series.warnings)})
    result["orientation"] = describe_series_orientation(metadata).to_dict()
    result["laterality"] = resolve_series_laterality(metadata).to_dict()
    result["provenance"] = build_orientation_provenance(metadata).to_dict()
    return result


def inspect_slice(index: DatasetIndex, sop_id: str) -> dict[str, Any]:
    item, series, study = _slice(ensure_index(index), sop_id)
    metadata = item.metadata
    fields = ("SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID", "InstanceNumber",
              "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns")
    result = {key: metadata[key] for key in fields if key in metadata}
    result.update({"relative_path": item.relative_path, "file_size": item.file_size,
                   "slice_id": item.slice_id, "series_instance_uid": series.series_instance_uid,
                   "study_instance_uid": study.study_instance_uid if study else series.study_instance_uid,
                   "warnings": list(item.warnings)})
    return result


def inspect_manifest(index: DatasetIndex) -> dict[str, Any]:
    index = ensure_index(index)
    return {"index_id": index.index_id, "root_identity": index.root_identity,
            "discovery_version": index.discovery_version,
            "manifest_schema_version": 1, "statistics": dict(index.statistics),
            "metadata_files": list(index.metadata_files), "warnings": list(index.warnings)}


def _metadata(record: Any) -> dict[str, Any]:
    if isinstance(record, StudyRecord):
        return {"study_instance_uid": record.study_instance_uid, "patient_id": record.patient_id}
    if isinstance(record, SeriesRecord):
        return record.metadata
    return dict(record.metadata)


def filter_entities(index: DatasetIndex, entity: str = "series", *, plane: str | None = None,
                    laterality: str | None = None, warning_code: str | None = None) -> list[dict[str, Any]]:
    index = ensure_index(index)
    rows: list[Any]
    if entity == "study":
        rows = list(index.studies)
    elif entity == "series":
        rows = [series for study in index.studies for series in study.series]
    elif entity == "slice":
        rows = [item for study in index.studies for series in study.series for item in series.slices]
    else:
        raise ValueError(f"unknown entity type: {entity}")
    result = []
    for row in rows:
        metadata = _metadata(row)
        codes = warning_codes(row)
        if plane is not None:
            value = describe_series_orientation(metadata).plane if isinstance(row, SeriesRecord) else metadata.get("plane", "")
            if str(value).casefold() != plane.casefold():
                continue
        if laterality is not None:
            value = resolve_series_laterality(metadata).resolved if isinstance(row, SeriesRecord) else metadata.get("Laterality", metadata.get("ImageLaterality", ""))
            if str(value).casefold() != laterality.casefold():
                continue
        if warning_code is not None and warning_code not in codes:
            continue
        result.append(row.to_dict())
    return result


def sample_entities(index: DatasetIndex, entity: str = "series", count: int = 20, seed: int = 42, **filters: Any) -> dict[str, Any]:
    if count < 0:
        raise ValueError("sample count must be non-negative")
    records = filter_entities(index, entity, **filters)
    records.sort(key=lambda row: (str(row.get("series_instance_uid", row.get("study_instance_uid", row.get("slice_id", "")))),
                                  json.dumps(row, sort_keys=True)))
    selected = random.Random(seed).sample(records, min(count, len(records)))
    return {"entity": entity, "seed": seed, "requested_count": count,
            "available_count": len(records), "count": len(selected), "records": selected}
