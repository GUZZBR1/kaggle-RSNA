"""Metadata-only dataset adapters and reusable entity inspection services.

The minimal format is a JSON object with ``studies``, ``series`` and ``slices``
tables, each a list of objects or a mapping keyed by UID. DICOM keyword names
and common snake-case aliases are accepted without requiring a DICOM library.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
import random
from pathlib import Path
from typing import Any


class DatasetReadError(ValueError):
    """The dataset manifest or a metadata table could not be read."""


class EntityNotFoundError(ValueError):
    """The requested UID is absent from the dataset index."""


ALIASES = {
    "study_uid": ("study_uid", "study_id", "study_instance_uid", "StudyInstanceUID"),
    "series_uid": ("series_uid", "series_id", "series_instance_uid", "SeriesInstanceUID"),
    "sop_uid": ("sop_uid", "sop_id", "slice_id", "sop_instance_uid", "SOPInstanceUID"),
    "patient_id": ("patient_id", "PatientID"),
    "description": ("description", "series_description", "SeriesDescription"),
    "protocol": ("protocol", "protocol_name", "ProtocolName"),
    "modality": ("modality", "Modality"),
    "laterality": ("laterality", "Laterality", "ImageLaterality"),
    "plane": ("plane", "anatomical_plane"),
    "path": ("path", "relative_path", "file_path"),
    "instance_number": ("instance_number", "InstanceNumber"),
    "position": ("position", "image_position_patient", "ImagePositionPatient"),
    "orientation": ("orientation", "image_orientation_patient", "ImageOrientationPatient"),
    "spacing": ("spacing", "pixel_spacing", "PixelSpacing"),
    "slice_spacing": ("slice_spacing", "spacing_between_slices", "SpacingBetweenSlices"),
    "rows": ("rows", "Rows"),
    "columns": ("columns", "Columns"),
    "n_slices": ("n_slices", "slice_count", "num_slices"),
}
TABLE_IDS = {"studies": "study_uid", "series": "series_uid", "slices": "sop_uid"}
ENTITY_TABLES = {"study": "studies", "series": "series", "slice": "slices"}


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Retain source metadata and provide stable names for known fields."""
    result = dict(record)
    # Metadata may be nested in a record, but pixels are never accessed.
    nested = record.get("metadata")
    values = {**nested, **record} if isinstance(nested, Mapping) else record
    for name, aliases in ALIASES.items():
        for alias in aliases:
            if alias in values and values[alias] is not None:
                result[name] = values[alias]
                break
    return result


def _records(value: Any, table: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, Mapping):
        rows = []
        for uid, row in value.items():
            if not isinstance(row, Mapping):
                raise DatasetReadError(f"{table} record {uid!r} must be an object")
            normalized = normalize_record(row)
            if not any(normalized.get(alias) is not None for alias in ALIASES[TABLE_IDS[table]]):
                normalized[TABLE_IDS[table]] = str(uid)
            rows.append(normalized)
    else:
        raise DatasetReadError(f"{table} must be a list or a UID-keyed object")
    if any(not isinstance(row, Mapping) for row in rows):
        raise DatasetReadError(f"each {table} record must be an object")
    return [normalize_record(row) for row in rows]


def _uid(record: Mapping[str, Any], key: str) -> str | None:
    value = record.get(key)
    return str(value) if value is not None and str(value) else None


@dataclass
class DatasetIndex:
    metadata: dict[str, Any]
    studies: list[dict[str, Any]] = field(default_factory=list)
    series: list[dict[str, Any]] = field(default_factory=list)
    slices: list[dict[str, Any]] = field(default_factory=list)
    source: Path | None = None
    entities_present: set[str] = field(default_factory=set)
    _lookup: dict[str, dict[str, list[dict[str, Any]]]] = field(init=False, repr=False)
    _series_by_study: dict[str, list[dict[str, Any]]] = field(init=False, repr=False)
    _slices_by_series: dict[str, list[dict[str, Any]]] = field(init=False, repr=False)
    _slices_by_study: dict[str, list[dict[str, Any]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lookup = {}
        for table, key in TABLE_IDS.items():
            lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in getattr(self, table):
                uid = _uid(record, key)
                if uid is not None:
                    lookup[uid].append(record)
            self._lookup[table] = dict(lookup)
        self._series_by_study = self._group(self.series, "study_uid")
        self._slices_by_series = self._group(self.slices, "series_uid")
        self._slices_by_study = self._group(self.slices, "study_uid")

    @staticmethod
    def _group(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            uid = _uid(record, key)
            if uid is not None:
                groups[uid].append(record)
        return dict(groups)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], source: str | Path | None = None) -> DatasetIndex:
        if not isinstance(mapping, Mapping):
            raise DatasetReadError("dataset manifest must be a JSON object")
        tables = mapping
        for key in ("index", "data", "payload"):
            nested = tables.get(key)
            if isinstance(nested, Mapping) and any(name in nested for name in TABLE_IDS):
                tables = nested
                break
        present = {name for name in TABLE_IDS if name in tables}
        records = {name: _records(tables[name], name) for name in present}
        metadata = {key: value for key, value in mapping.items() if key not in TABLE_IDS}
        if tables is not mapping:
            metadata.update({key: value for key, value in tables.items() if key not in TABLE_IDS})
        return cls(metadata=metadata, source=Path(source) if source else None,
                   entities_present=present, **records)

    def lookup(self, entity: str, uid: str) -> dict[str, Any]:
        table = ENTITY_TABLES.get(entity, entity)
        if table not in TABLE_IDS:
            raise ValueError(f"unknown entity type: {entity}")
        matches = self._lookup[table].get(str(uid), [])
        if not matches:
            raise EntityNotFoundError(f"{entity} UID {uid!r} was not found")
        if len(matches) > 1:
            raise EntityNotFoundError(f"{entity} UID {uid!r} is ambiguous: {len(matches)} records")
        return dict(matches[0])


def ensure_index(index: DatasetIndex | Mapping[str, Any]) -> DatasetIndex:
    return index if isinstance(index, DatasetIndex) else DatasetIndex.from_mapping(index)


def load_dataset_index(path: str | Path) -> DatasetIndex:
    """Read JSON metadata, including explicitly referenced JSON index payloads."""
    source = Path(path)
    if source.is_dir():
        source = source / "manifest.json"
    try:
        mapping = json.loads(source.read_text(encoding="utf-8"), parse_constant=_invalid_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetReadError(f"cannot read dataset manifest {source}: {exc}") from exc
    if not isinstance(mapping, Mapping):
        raise DatasetReadError("dataset manifest must be a JSON object")
    if not any(name in mapping for name in TABLE_IDS):
        reference = mapping.get("index_path") or mapping.get("payload_path")
        if reference is None and isinstance(mapping.get("payload"), str):
            reference = mapping["payload"]
        if reference is not None:
            if not isinstance(reference, str):
                raise DatasetReadError("metadata index path must be a string")
            payload = Path(reference)
            if not payload.is_absolute():
                payload = source.parent / payload
            try:
                records = json.loads(payload.read_text(encoding="utf-8"), parse_constant=_invalid_constant)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise DatasetReadError(f"cannot read dataset metadata index {payload}: {exc}") from exc
            if not isinstance(records, Mapping):
                raise DatasetReadError("dataset metadata index must be a JSON object")
            mapping = {**mapping, "index": records}
    return DatasetIndex.from_mapping(mapping, source)


def _invalid_constant(value: str) -> None:
    raise DatasetReadError(f"dataset JSON contains non-finite number {value}")


def warning_codes(record: Mapping[str, Any]) -> list[str]:
    warnings = record.get("warnings", [])
    if isinstance(warnings, (str, Mapping)):
        warnings = [warnings]
    if not isinstance(warnings, list):
        return []
    return [str(item.get("code", item.get("message", "UNKNOWN")))
            if isinstance(item, Mapping) else str(item) for item in warnings]


def distribution(records: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field_name]) for row in records
                               if row.get(field_name) is not None).items()))


def inspect_study(index: DatasetIndex | Mapping[str, Any], study_id: str) -> dict[str, Any]:
    index = ensure_index(index)
    result = index.lookup("study", study_id)
    series = index._series_by_study.get(str(study_id), [])
    slices = list(index._slices_by_study.get(str(study_id), []))
    # Some indexes declare the parent only on series records.
    if "slices" in index.entities_present:
        seen = {id(row) for row in slices}
        for row in series:
            for item in index._slices_by_series.get(_uid(row, "series_uid") or "", []):
                if id(item) not in seen:
                    slices.append(item)
                    seen.add(id(item))
        result["n_slices"] = len(slices)
    elif series and all(isinstance(row.get("n_slices"), int) for row in series):
        result["n_slices"] = sum(row["n_slices"] for row in series)
    if "series" in index.entities_present:
        result["n_series"] = len(series)
        result["series_uids"] = [_uid(row, "series_uid") for row in series]
    for field_name, output_name in (("laterality", "laterality_counts"), ("plane", "plane_counts")):
        counts = distribution(series or slices, field_name)
        if counts:
            result[output_name] = counts
    orientations = [row["orientation"] for row in slices if row.get("orientation") is not None]
    if orientations:
        result["orientations"] = [json.loads(value) for value in sorted({json.dumps(value) for value in orientations})]
    if "provenance" not in result and "provenance" in index.metadata:
        result["provenance"] = index.metadata["provenance"]
    return result


def inspect_series(index: DatasetIndex | Mapping[str, Any], series_id: str) -> dict[str, Any]:
    index = ensure_index(index)
    result = index.lookup("series", series_id)
    slices = index._slices_by_series.get(str(series_id), [])
    if "slices" in index.entities_present:
        result["n_slices"] = len(slices)
    geometric = _geometry(slices)
    for key, value in geometric.items():
        result.setdefault(key, value)
    return result


def inspect_slice(index: DatasetIndex | Mapping[str, Any], sop_id: str) -> dict[str, Any]:
    result = ensure_index(index).lookup("slice", sop_id)
    if result.get("rows") is not None and result.get("columns") is not None:
        result.setdefault("dimensions", [result["rows"], result["columns"]])
    return result


def _vector(value: Any, length: int) -> list[float] | None:
    if isinstance(value, str):
        value = value.split("\\")
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _geometry(slices: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive a span only for a consistent, fully observed plane geometry."""
    if len(slices) < 2:
        return {}
    positions = [_vector(row.get("position"), 3) for row in slices]
    orientations = [_vector(row.get("orientation"), 6) for row in slices]
    if any(value is None for value in positions + orientations):
        return {}
    orientation = orientations[0]
    if any(any(abs(a - b) > 1e-5 for a, b in zip(orientation, value)) for value in orientations[1:]):
        return {}
    a, b = orientation[:3], orientation[3:]
    normal = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    norm = math.sqrt(sum(value * value for value in normal))
    if norm < 1e-8:
        return {}
    locations = sorted(sum(value * direction / norm for value, direction in zip(position, normal))
                       for position in positions)
    distances = [right - left for left, right in zip(locations, locations[1:])]
    return {"physical_span_mm": locations[-1] - locations[0],
            "spacing_stats_mm": {"count": len(distances), "min": min(distances),
                                 "max": max(distances), "mean": sum(distances) / len(distances)}}


def inspect_manifest(index: DatasetIndex | Mapping[str, Any]) -> dict[str, Any]:
    index = ensure_index(index)
    result = dict(index.metadata)
    # Inline payloads can be large; summarize their entity tables separately.
    for key in ("index", "data", "payload"):
        if isinstance(result.get(key), Mapping) and any(name in result[key] for name in TABLE_IDS):
            result.pop(key)
    result["entity_tables"] = {name: len(getattr(index, name)) for name in sorted(index.entities_present)}
    if index.source is not None:
        result["manifest_path"] = str(index.source)
    return result


def filter_entities(index: DatasetIndex | Mapping[str, Any], entity: str = "series", *,
                    plane: str | None = None, laterality: str | None = None,
                    warning_code: str | None = None) -> list[dict[str, Any]]:
    index = ensure_index(index)
    table = ENTITY_TABLES.get(entity, entity)
    if table not in TABLE_IDS:
        raise ValueError(f"unknown entity type: {entity}")
    rows = getattr(index, table)
    return [dict(row) for row in rows
            if (plane is None or str(row.get("plane", "")).casefold() == plane.casefold())
            and (laterality is None or str(row.get("laterality", "")).casefold() == laterality.casefold())
            and (warning_code is None or warning_code in warning_codes(row))]


def sample_entities(index: DatasetIndex | Mapping[str, Any], entity: str = "series", count: int = 20,
                    seed: int = 42, *, plane: str | None = None, laterality: str | None = None,
                    warning_code: str | None = None) -> dict[str, Any]:
    if count < 0:
        raise ValueError("sample count must be non-negative")
    rows = filter_entities(index, entity, plane=plane, laterality=laterality, warning_code=warning_code)
    table = ENTITY_TABLES.get(entity, entity)
    # Stable order makes the seed independent of serialization order.
    rows.sort(key=lambda row: (_uid(row, TABLE_IDS[table]) or "", json.dumps(row, sort_keys=True)))
    selected = random.Random(seed).sample(rows, min(count, len(rows)))
    return {"entity": entity, "seed": seed, "requested_count": count,
            "available_count": len(rows), "count": len(selected), "records": selected}
