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
    """Flatten canonical records while retaining the external mapping adapter."""
    from ..data.models import DatasetIndex, SeriesRecord, SliceRecord, StudyRecord, index_from_dict

    canonical_types = (DatasetIndex, StudyRecord, SeriesRecord, SliceRecord)
    if isinstance(index, canonical_types):
        index = index.to_dict()
    if isinstance(index, Mapping):
        if "records" in index or "entities" in index:
            index = index.get("records", index.get("entities", ()))
            if isinstance(index, Mapping):
                index = index.values()
        elif "dataset_index" in index or "index" in index:
            return records_of(index.get("dataset_index", index.get("index")))
        elif "studies" in index:
            # Verify the canonical manifest's schema and content identities.
            index_from_dict(dict(index))
            return [row for study in index["studies"] for row in _adapt_record(study)]
        else:
            return _adapt_record(index)
    if hasattr(index, "records"):
        index = index.records
    result = []
    for record in index:
        if isinstance(record, canonical_types):
            record = record.to_dict()
        if not isinstance(record, Mapping):
            raise TypeError("dataset records must be canonical records or mappings")
        result.extend(_adapt_record(record))
    return result


def _adapt_record(record: Mapping[str, Any], parent: Mapping[str, Any] | None = None,
                  ancestors: tuple[str, ...] = ()) -> list[Mapping[str, Any]]:
    """Inherit absent relations, keeping declared child metadata for mismatch checks."""
    parent = parent or {}
    if "series" in record:
        kind, identity_field, children = "study", "study_id", record["series"]
    elif "slices" in record:
        kind, identity_field, children = "series", "series_id", record["slices"]
    elif "relative_path" in record and "slice_id" in record:
        kind, identity_field, children = "slice", "slice_id", ()
    else:
        return [dict(record)]
    identity = record.get(identity_field)
    assignment_ids = (*ancestors, str(identity)) if identity else ancestors
    row = {key: item for key, item in record.items()
           if key not in {"series", "slices", "metadata", "relative_paths", "n_slices"}}
    row.update(entity_type=kind, _assignment_ids=assignment_ids)
    # Parent aggregates include SOP/hash metadata when they contain one slice;
    # those values describe the child and must not create phantom duplicates.
    if kind == "slice":
        row["metadata"] = record.get("metadata", {})
    for field in ("patient_id", "study_uid", "series_uid"):
        declared = value(record, field) if kind == "slice" else value(row, field)
        inherited = value(parent, field)
        if declared is not None or inherited is not None:
            row[field] = declared if declared is not None else inherited
    if kind == "series":
        row["_parent_study_uid"] = value(parent, "study_uid")
    elif kind == "slice":
        row["_parent_series_uid"] = value(parent, "series_uid")
    inherited_fold = value(parent, "fold_id")
    if value(row, "fold_id") is None and inherited_fold is not None:
        row["fold_id"] = inherited_fold
    rows = [row]
    for child in children:
        rows.extend(_adapt_record(child, row, assignment_ids))
    return rows


def groups(records: Iterable[Mapping[str, Any]], field: str) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        item = value(record, field)
        if item is not None:
            result[str(item)].append(record)
    return dict(result)
