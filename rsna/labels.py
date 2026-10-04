"""Validated per-study supervision with explicit missingness and provenance."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .identity import digest, freeze_json, jsonable
from .targets import TARGETS, TARGET_REGISTRY, TARGET_REGISTRY_ID, TARGET_SCHEMA_VERSION

LABEL_PROVENANCE = ("official_gold", "report_regex", "report_llm", "pseudo_label", "manual_review")


@dataclass(frozen=True)
class StudyMetadata:
    study_id: str
    patient_id: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.study_id, str) or not self.study_id.strip():
            raise ValueError("study_id cannot be empty")
        if self.patient_id is not None and (not isinstance(self.patient_id, str) or not self.patient_id.strip()):
            raise ValueError("patient_id must be nonempty when supplied")
        object.__setattr__(self, "metadata", freeze_json(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {"study_id": self.study_id, "patient_id": self.patient_id,
                "metadata": jsonable(self.metadata)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StudyMetadata":
        return cls(**dict(data))


@dataclass(frozen=True)
class StudyDatasetRecord:
    """Study metadata and labels kept in separate, explicitly linked records."""

    study: StudyMetadata
    labels: "LabelRecord"
    dataset_version_id: str

    def __post_init__(self) -> None:
        if self.study.study_id != self.labels.study_id:
            raise ValueError("study metadata and label record must identify the same study")
        if (not isinstance(self.dataset_version_id, str) or len(self.dataset_version_id) != 64
                or any(char not in "0123456789abcdef" for char in self.dataset_version_id)):
            raise ValueError("dataset_version_id must be a lowercase SHA-256 digest")

    @property
    def record_id(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"study": self.study.to_dict(), "labels": self.labels.to_dict(),
                "dataset_version_id": self.dataset_version_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StudyDatasetRecord":
        payload = dict(data)
        payload["study"] = StudyMetadata.from_dict(payload["study"])
        payload["labels"] = LabelRecord.from_dict(payload["labels"])
        return cls(**payload)


@dataclass(frozen=True)
class LabelRecord:
    study_id: str
    values: Mapping[str, int | float | None]
    provenance: str
    label_type: str = "hard"
    allow_partial: bool = False
    allow_soft: bool = False
    target_schema_version: int = TARGET_SCHEMA_VERSION
    patient_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.study_id, str) or not self.study_id.strip():
            raise ValueError("study_id cannot be empty")
        if self.patient_id is not None and (not isinstance(self.patient_id, str) or not self.patient_id.strip()):
            raise ValueError("patient_id must be nonempty when supplied")
        if type(self.target_schema_version) is not int or self.target_schema_version != TARGET_SCHEMA_VERSION:
            raise ValueError(f"unsupported target schema version: {self.target_schema_version}")
        if self.label_type not in {"hard", "soft"}:
            raise ValueError("label_type must be 'hard' or 'soft'")
        if type(self.allow_partial) is not bool or type(self.allow_soft) is not bool:
            raise ValueError("allow_partial and allow_soft must be booleans")
        if self.label_type == "soft" and not self.allow_soft:
            raise ValueError("soft labels are disabled by this schema")
        if self.provenance not in LABEL_PROVENANCE:
            raise ValueError(f"unknown label provenance: {self.provenance!r}")
        canonical: dict[str, int | float | None] = {}
        for supplied, value in self.values.items():
            name = TARGET_REGISTRY.canonical_name(supplied)
            if name in canonical:
                raise ValueError(f"duplicate target after alias normalization: {name}")
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"label for {name} must be finite numeric or None")
                if self.label_type == "hard" and value not in (0, 1):
                    raise ValueError(f"hard label for {name} must be 0, 1, or None")
                if self.label_type == "soft" and not 0 <= value <= 1:
                    raise ValueError(f"soft label for {name} must be between 0 and 1")
            canonical[name] = value
        extras = set(canonical) - set(TARGETS)
        if extras:
            raise ValueError(f"unexpected targets: {sorted(extras)}")
        if set(canonical) != set(TARGETS):
            if not self.allow_partial:
                raise ValueError("all official targets are required when partial labels are disabled")
            canonical = {name: canonical.get(name) for name in TARGETS}
        elif not self.allow_partial and any(value is None for value in canonical.values()):
            raise ValueError("missing labels are disabled by this schema")
        object.__setattr__(self, "values", canonical)

    @property
    def mask(self) -> tuple[bool, ...]:
        return tuple(self.values[name] is not None for name in TARGETS)

    @property
    def record_id(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"study_id": self.study_id, "patient_id": self.patient_id,
                "target_schema_version": self.target_schema_version,
                "target_registry_id": TARGET_REGISTRY_ID,
                "targets": list(TARGETS), "values": dict(self.values),
                "mask": list(self.mask), "provenance": self.provenance,
                "label_type": self.label_type, "allow_partial": self.allow_partial,
                "allow_soft": self.allow_soft}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabelRecord":
        if tuple(data.get("targets", ())) != TARGETS:
            raise ValueError("serialized label target order does not match the registry")
        if data.get("target_registry_id") != TARGET_REGISTRY_ID:
            raise ValueError("serialized target registry identity does not match this version")
        payload = {key: data[key] for key in (
            "study_id", "values", "provenance", "label_type", "allow_partial",
            "allow_soft", "target_schema_version", "patient_id") if key in data}
        record = cls(**payload)
        if tuple(data.get("mask", ())) != record.mask:
            raise ValueError("serialized label mask does not match label values")
        return record

    def to_row(self) -> dict[str, Any]:
        return {"StudyInstanceUID": self.study_id, "PatientID": self.patient_id, **self.values,
                "label_mask": list(self.mask), "label_provenance": self.provenance,
                "label_type": self.label_type}

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "LabelRecord":
        values = {name: _missing_to_none(row.get(name)) for name in TARGETS}
        return cls(study_id=row.get("StudyInstanceUID", row.get("study_id", "")),
                   patient_id=row.get("PatientID"), values=values,
                   provenance=row.get("label_provenance", ""),
                   label_type=row.get("label_type", "hard"), allow_partial=True,
                   allow_soft=row.get("label_type", "hard") == "soft")


def validate_label_record(record: LabelRecord | Mapping[str, Any]) -> LabelRecord:
    if isinstance(record, LabelRecord):
        return record
    return LabelRecord(**record)


def rows_to_label_records(rows: Any) -> list[LabelRecord]:
    """Convert row mappings or a pandas-like DataFrame without requiring pandas."""
    if hasattr(rows, "to_dict") and not isinstance(rows, Mapping):
        rows = rows.to_dict(orient="records")
    if not isinstance(rows, (list, tuple)):
        raise ValueError("rows must be a sequence of row mappings or a DataFrame")
    return [LabelRecord.from_row(row) for row in rows]


def label_records_to_rows(records: Any) -> list[dict[str, Any]]:
    return [validate_label_record(record).to_row() for record in records]


def _missing_to_none(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
