"""Immutable, JSON-friendly records for MRI dataset metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping

from ..identity import digest, freeze_json, jsonable


@dataclass(frozen=True)
class SliceRecord:
    relative_path: str
    file_size: int
    metadata: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    slice_id: str = ""

    def __post_init__(self) -> None:
        # Keep a detached JSON-shaped dict so downstream metadata utilities can
        # normalize a working copy while the record attributes themselves stay frozen.
        object.__setattr__(self, "metadata", jsonable(self.metadata))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        expected = digest({"relative_path": self.relative_path, "file_size": self.file_size,
                           "metadata": self.metadata, "warnings": self.warnings})
        _identity(self.slice_id, expected, "slice_id")
        object.__setattr__(self, "slice_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"relative_path": self.relative_path, "file_size": self.file_size,
                "metadata": jsonable(self.metadata), "warnings": list(self.warnings), "slice_id": self.slice_id}


@dataclass(frozen=True)
class SeriesRecord:
    series_instance_uid: str | None
    study_instance_uid: str | None
    slices: tuple[SliceRecord, ...]
    warnings: tuple[str, ...] = ()
    series_id: str = ""

    def __post_init__(self) -> None:
        slices = tuple(sorted(self.slices, key=lambda item: item.relative_path))
        object.__setattr__(self, "slices", slices)
        object.__setattr__(self, "warnings", tuple(self.warnings))
        expected = digest({"series_instance_uid": self.series_instance_uid,
                           "study_instance_uid": self.study_instance_uid,
                           "slice_ids": [item.slice_id for item in slices], "warnings": self.warnings})
        _identity(self.series_id, expected, "series_id")
        object.__setattr__(self, "series_id", expected)

    @property
    def n_slices(self) -> int:
        return len(self.slices)

    @property
    def metadata(self) -> dict[str, Any]:
        """Values shared by every slice; differing values are represented as null."""
        if not self.slices:
            return {}
        keys = sorted({key for item in self.slices for key in item.metadata})
        result = {}
        for key in keys:
            first = self.slices[0].metadata.get(key)
            result[key] = first if all(item.metadata.get(key) == first for item in self.slices) else None
        return result

    @property
    def relative_paths(self) -> tuple[str, ...]:
        return tuple(item.relative_path for item in self.slices)

    def to_dict(self) -> dict[str, Any]:
        return {"series_instance_uid": self.series_instance_uid,
                "study_instance_uid": self.study_instance_uid,
                "slices": [item.to_dict() for item in self.slices], "n_slices": self.n_slices,
                "metadata": self.metadata, "relative_paths": list(self.relative_paths),
                "warnings": list(self.warnings), "series_id": self.series_id}


@dataclass(frozen=True)
class StudyRecord:
    study_instance_uid: str | None
    patient_id: str | None
    series: tuple[SeriesRecord, ...]
    warnings: tuple[str, ...] = ()
    study_id: str = ""

    def __post_init__(self) -> None:
        series = tuple(sorted(self.series, key=lambda item: item.series_id))
        object.__setattr__(self, "series", series)
        object.__setattr__(self, "warnings", tuple(self.warnings))
        expected = digest({"study_instance_uid": self.study_instance_uid, "patient_id": self.patient_id,
                           "series_ids": [item.series_id for item in series], "warnings": self.warnings})
        _identity(self.study_id, expected, "study_id")
        object.__setattr__(self, "study_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"study_instance_uid": self.study_instance_uid, "patient_id": self.patient_id,
                "series": [item.to_dict() for item in self.series],
                "metadata": {"study_instance_uid": self.study_instance_uid,
                             "patient_id": self.patient_id, "n_series": len(self.series)},
                "relative_paths": list(self.relative_paths),
                "warnings": list(self.warnings), "study_id": self.study_id}

    @property
    def relative_paths(self) -> tuple[str, ...]:
        return tuple(sorted(path for series in self.series for path in series.relative_paths))


@dataclass(frozen=True)
class DatasetIndex:
    root_identity: str
    discovery_version: str
    studies: tuple[StudyRecord, ...]
    warnings: tuple[str, ...]
    statistics: Mapping[str, int]
    metadata_files: tuple[str, ...] = ()
    index_id: str = ""
    _study_lookup: Mapping[str, StudyRecord] = field(init=False, repr=False, compare=False)
    _series_lookup: Mapping[str, tuple[SeriesRecord, ...]] = field(init=False, repr=False, compare=False)
    _path_lookup: Mapping[str, SliceRecord] = field(init=False, repr=False, compare=False)
    _sop_lookup: Mapping[str, tuple[SliceRecord, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        studies = tuple(sorted(self.studies, key=lambda item: item.study_id))
        study_uids = [study.study_instance_uid for study in studies if study.study_instance_uid]
        if len(study_uids) != len(set(study_uids)):
            raise ValueError("DatasetIndex contains duplicate StudyInstanceUID values")
        object.__setattr__(self, "studies", studies)
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "statistics", freeze_json(self.statistics))
        object.__setattr__(self, "metadata_files", tuple(sorted(self.metadata_files)))
        study_lookup = {s.study_instance_uid: s for s in studies if s.study_instance_uid}
        series_lookup: dict[str, list[SeriesRecord]] = {}
        path_lookup = {}
        sop_lookup: dict[str, list[SliceRecord]] = {}
        for study in studies:
            for series in study.series:
                if series.series_instance_uid:
                    series_lookup.setdefault(series.series_instance_uid, []).append(series)
                for item in series.slices:
                    path_lookup[item.relative_path] = item
                    sop = item.metadata.get("SOPInstanceUID")
                    if sop:
                        sop_lookup.setdefault(str(sop), []).append(item)
        object.__setattr__(self, "_study_lookup", MappingProxyType(study_lookup))
        object.__setattr__(self, "_series_lookup", MappingProxyType({k: tuple(v) for k, v in series_lookup.items()}))
        object.__setattr__(self, "_path_lookup", MappingProxyType(path_lookup))
        object.__setattr__(self, "_sop_lookup", MappingProxyType({k: tuple(v) for k, v in sop_lookup.items()}))
        expected = digest({"root_identity": self.root_identity, "discovery_version": self.discovery_version,
                           "study_ids": [item.study_id for item in studies], "warnings": self.warnings,
                           "statistics": self.statistics, "metadata_files": self.metadata_files})
        _identity(self.index_id, expected, "index_id")
        object.__setattr__(self, "index_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"manifest_schema_version": 1, "root_identity": self.root_identity,
                "discovery_version": self.discovery_version,
                "studies": [item.to_dict() for item in self.studies],
                "warnings": list(self.warnings), "statistics": jsonable(self.statistics),
                "metadata_files": list(self.metadata_files),
                "index_id": self.index_id}

    def study_by_uid(self, uid: str) -> StudyRecord | None:
        return self._study_lookup.get(uid)

    def series_by_uid(self, uid: str) -> tuple[SeriesRecord, ...]:
        return self._series_lookup.get(uid, ())

    def slice_by_path(self, relative_path: str) -> SliceRecord | None:
        return self._path_lookup.get(relative_path)

    def slices_by_sop_uid(self, uid: str) -> tuple[SliceRecord, ...]:
        return self._sop_lookup.get(uid, ())


def index_from_dict(data: dict[str, Any]) -> DatasetIndex:
    _manifest_record(data, {
        "manifest_schema_version": int, "root_identity": str, "discovery_version": str,
        "studies": list, "warnings": list, "statistics": dict,
        "metadata_files": list, "index_id": str,
    }, "manifest")
    if type(data["manifest_schema_version"]) is not int or data["manifest_schema_version"] != 1:
        raise ValueError("unsupported dataset manifest schema version")
    if any(type(count) is not int or count < 0 for count in data["statistics"].values()):
        raise ValueError("manifest.statistics values must be nonnegative integers")
    if any(not isinstance(path, str) for path in data["metadata_files"]):
        raise ValueError("manifest.metadata_files must contain strings")
    studies = []
    for study in data["studies"]:
        _manifest_record(study, {"study_instance_uid": (str, type(None)),
            "patient_id": (str, type(None)), "series": list, "warnings": list, "study_id": str}, "study")
        series = []
        for item in study["series"]:
            _manifest_record(item, {"series_instance_uid": (str, type(None)),
                "study_instance_uid": (str, type(None)), "slices": list,
                "warnings": list, "series_id": str}, "series")
            for slice_data in item["slices"]:
                _manifest_record(slice_data, {"relative_path": str, "file_size": int,
                    "metadata": dict, "warnings": list, "slice_id": str}, "slice")
                if type(slice_data["file_size"]) is not int or slice_data["file_size"] < 0:
                    raise ValueError("slice.file_size must be a nonnegative integer")
                _json_metadata(slice_data["metadata"])
            slices = [SliceRecord(**slice_data) for slice_data in item["slices"]]
            series.append(SeriesRecord(item.get("series_instance_uid"),
                item.get("study_instance_uid"), tuple(slices), tuple(item.get("warnings", ())),
                item.get("series_id", "")))
        studies.append(StudyRecord(study.get("study_instance_uid"), study.get("patient_id"),
            tuple(series), tuple(study.get("warnings", ())), study.get("study_id", "")))
    index = DatasetIndex(data["root_identity"], data["discovery_version"], tuple(studies),
        tuple(data.get("warnings", ())), data["statistics"], tuple(data.get("metadata_files", ())),
        data.get("index_id", ""))
    # Derived fields are persisted for readers, but must agree with the canonical records.
    canonical = index.to_dict()
    for raw_study in data["studies"]:
        study = next(value for value in canonical["studies"] if value["study_id"] == raw_study["study_id"])
        _derived_fields(raw_study, study, ("metadata", "relative_paths"), "study")
        for raw_series in raw_study["series"]:
            series = next(value for value in study["series"] if value["series_id"] == raw_series["series_id"])
            _derived_fields(raw_series, series, ("metadata", "relative_paths", "n_slices"), "series")
    return index


def _manifest_record(value: Any, fields: dict[str, Any], entity: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{entity} must be an object")
    for name, kind in fields.items():
        if name not in value or not isinstance(value[name], kind):
            raise ValueError(f"{entity}.{name} is required and has an invalid type")
        if name in {"slice_id", "series_id", "study_id", "index_id"} and (not value[name] or len(value[name]) != 64):
            raise ValueError(f"{entity}.{name} must be a content SHA-256 ID")
    if any(not isinstance(warning, str) for warning in value.get("warnings", ())):
        raise ValueError(f"{entity}.warnings must contain strings")


def _json_metadata(value: Any) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json_metadata(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_metadata(item)
        return
    raise ValueError("slice.metadata must contain finite JSON values")


def _derived_fields(raw: dict[str, Any], canonical: dict[str, Any], fields: tuple[str, ...], entity: str) -> None:
    for name in fields:
        if name in raw and (type(raw[name]) is not type(canonical[name]) or raw[name] != canonical[name]):
            raise ValueError(f"{entity}.{name} does not match its canonical records")


def _identity(actual: str, expected: str, name: str) -> None:
    if actual and actual != expected:
        raise ValueError(f"{name} does not match its content")
