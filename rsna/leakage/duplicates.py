"""Linear-time identity and duplicate grouping primitives for the validator."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


ALIASES: dict[str, tuple[str, ...]] = {
    "patient_id": ("patient_id", "PatientID", "patientId"),
    "study_uid": ("study_uid", "StudyInstanceUID", "study_instance_uid"),
    "series_uid": ("series_uid", "SeriesInstanceUID", "series_instance_uid"),
    "sop_uid": ("sop_uid", "SOPInstanceUID", "sop_instance_uid"),
    "file_hash": ("file_hash", "sha256", "content_sha256"),
    "path": ("path", "uri", "file_path", "relative_path"),
    "fold_id": ("fold_id", "split", "partition"),
}


def value(record: Mapping[str, Any], field: str) -> Any:
    for alias in ALIASES.get(field, (field,)):
        result = record.get(alias)
        if result is not None and result != "":
            return result
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        for alias in ALIASES.get(field, (field,)):
            result = metadata.get(alias)
            if result is not None and result != "":
                return result
    return None


def records_of(index: Any) -> list[Mapping[str, Any]]:
    if isinstance(index, Mapping):
        if "records" in index or "entities" in index:
            records = index.get("records", index.get("entities", ()))
        else:
            wrapped_index = index.get("dataset_index", index.get("index", index))
            records = _adapt_dataset_index(wrapped_index)
    else:
        records = _adapt_dataset_index(index) if hasattr(index, "studies") else getattr(index, "records", index)
    if isinstance(records, Mapping):
        records = records.values()
    result = list(records)
    if any(not isinstance(record, Mapping) for record in result):
        raise TypeError("dataset records must be mappings")
    return result


def _adapt_dataset_index(index: Any) -> list[Mapping[str, Any]]:
    """Expose existing StudyRecord/SeriesRecord/SliceRecord objects as validator rows."""
    if isinstance(index, Mapping) and "studies" in index:
        from ..data.models import index_from_dict
        index = index_from_dict(dict(index))
    studies = getattr(index, "studies", None)
    if studies is None:
        return ()
    result: list[Mapping[str, Any]] = []
    for study in studies:
        patient_id = study.patient_id
        study_uid = study.study_instance_uid
        result.append({"entity_type": "study", "patient_id": patient_id, "study_uid": study_uid})
        for series in study.series:
            series_uid = series.series_instance_uid
            series_study_uid = series.study_instance_uid or study_uid
            result.append({"entity_type": "series", "patient_id": patient_id,
                           "study_uid": series_study_uid, "series_uid": series_uid,
                           "metadata": series.metadata})
            for slice_record in series.slices:
                metadata = slice_record.metadata
                result.append({"entity_type": "slice", "patient_id": patient_id,
                               "study_uid": series_study_uid, "series_uid": series_uid,
                               "sop_uid": metadata.get("SOPInstanceUID"),
                               "relative_path": slice_record.relative_path,
                               "file_size": slice_record.file_size, "metadata": metadata})
    return result


def groups(records: Iterable[Mapping[str, Any]], field: str) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        item = value(record, field)
        if item is not None:
            result[str(item)].append(record)
    return dict(result)
