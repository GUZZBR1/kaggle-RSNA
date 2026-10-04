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
    "path": ("path", "uri", "file_path"),
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
        records = index.get("records", index.get("entities", ()))
    else:
        records = getattr(index, "records", index)
    if isinstance(records, Mapping):
        records = records.values()
    result = list(records)
    if any(not isinstance(record, Mapping) for record in result):
        raise TypeError("dataset records must be mappings")
    return result


def groups(records: Iterable[Mapping[str, Any]], field: str) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        item = value(record, field)
        if item is not None:
            result[str(item)].append(record)
    return dict(result)
