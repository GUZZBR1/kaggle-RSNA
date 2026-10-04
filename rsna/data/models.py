"""Immutable, JSON-friendly records for MRI dataset metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    if data.get("manifest_schema_version") != 1:
        raise ValueError("unsupported dataset manifest schema version")
    studies = []
    for study in data["studies"]:
        series = []
        for item in study["series"]:
            slices = [SliceRecord(**slice_data) for slice_data in item["slices"]]
            series.append(SeriesRecord(item.get("series_instance_uid"),
                item.get("study_instance_uid"), tuple(slices), tuple(item.get("warnings", ())),
                item.get("series_id", "")))
        studies.append(StudyRecord(study.get("study_instance_uid"), study.get("patient_id"),
            tuple(series), tuple(study.get("warnings", ())), study.get("study_id", "")))
    return DatasetIndex(data["root_identity"], data["discovery_version"], tuple(studies),
        tuple(data.get("warnings", ())), data["statistics"], tuple(data.get("metadata_files", ())),
        data.get("index_id", ""))


def _identity(actual: str, expected: str, name: str) -> None:
    if actual and actual != expected:
        raise ValueError(f"{name} does not match its content")
