"""Versioned, serializable contracts for dataset leakage validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping

from ..identity import digest, freeze_json, jsonable

REPORT_SCHEMA_VERSION = 1
VALIDATOR_VERSION = "1.0.0"


class LeakageIssueType(StrEnum):
    PATIENT_CROSS_FOLD = "PATIENT_CROSS_FOLD"
    STUDY_CROSS_FOLD = "STUDY_CROSS_FOLD"
    SERIES_CROSS_FOLD = "SERIES_CROSS_FOLD"
    SLICE_CROSS_FOLD = "SLICE_CROSS_FOLD"
    DUPLICATE_SOP_UID = "DUPLICATE_SOP_UID"
    DUPLICATE_SERIES_UID = "DUPLICATE_SERIES_UID"
    DUPLICATE_STUDY_UID = "DUPLICATE_STUDY_UID"
    DUPLICATE_FILE_HASH = "DUPLICATE_FILE_HASH"
    CONFLICTING_PATIENT_ID = "CONFLICTING_PATIENT_ID"
    UNBOUND_SERIES = "UNBOUND_SERIES"
    UNBOUND_SLICE = "UNBOUND_SLICE"
    UNKNOWN_GROUP_ID = "UNKNOWN_GROUP_ID"
    DATASET_VERSION_MISMATCH = "DATASET_VERSION_MISMATCH"
    SERIES_STUDY_MISMATCH = "SERIES_STUDY_MISMATCH"
    SLICE_SERIES_MISMATCH = "SLICE_SERIES_MISMATCH"
    NEAR_DUPLICATE_METADATA = "NEAR_DUPLICATE_METADATA"
    UNKNOWN_FOLD_ID = "UNKNOWN_FOLD_ID"
    LABEL_PROVENANCE_LEAKAGE = "LABEL_PROVENANCE_LEAKAGE"
    PROVENANCE_CROSS_SPLIT = "PROVENANCE_CROSS_SPLIT"


class LeakagePolicy(StrEnum):
    STRICT = "strict"
    AUDIT = "audit"


DEFAULT_SEVERITIES: dict[LeakageIssueType, str] = {
    issue_type: "error" for issue_type in (
        LeakageIssueType.PATIENT_CROSS_FOLD, LeakageIssueType.STUDY_CROSS_FOLD,
        LeakageIssueType.SERIES_CROSS_FOLD, LeakageIssueType.SLICE_CROSS_FOLD,
        LeakageIssueType.CONFLICTING_PATIENT_ID, LeakageIssueType.UNBOUND_SERIES,
        LeakageIssueType.UNBOUND_SLICE, LeakageIssueType.UNKNOWN_GROUP_ID,
        LeakageIssueType.DATASET_VERSION_MISMATCH, LeakageIssueType.SERIES_STUDY_MISMATCH,
        LeakageIssueType.SLICE_SERIES_MISMATCH, LeakageIssueType.UNKNOWN_FOLD_ID,
        LeakageIssueType.LABEL_PROVENANCE_LEAKAGE, LeakageIssueType.PROVENANCE_CROSS_SPLIT,
    )
}


@dataclass(frozen=True)
class LeakageIssue:
    type: LeakageIssueType
    severity: str
    entity_id: str
    folds_involved: tuple[str, ...] = ()
    related_ids: tuple[str, ...] = ()
    message: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported LeakageIssue schema version")
        if self.severity not in {"error", "warning", "info"}:
            raise ValueError("severity must be error, warning, or info")
        object.__setattr__(self, "folds_involved", tuple(sorted(set(self.folds_involved))))
        object.__setattr__(self, "related_ids", tuple(sorted(set(self.related_ids))))
        object.__setattr__(self, "evidence", freeze_json(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "severity": self.severity, "entity_id": self.entity_id,
                "folds_involved": list(self.folds_involved), "related_ids": list(self.related_ids),
                "message": self.message, "evidence": jsonable(self.evidence),
                "schema_version": self.schema_version}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LeakageIssue":
        return cls(LeakageIssueType(value["type"]), value["severity"], value.get("entity_id", ""),
                   tuple(value.get("folds_involved", ())), tuple(value.get("related_ids", ())),
                   value.get("message", ""), value.get("evidence", {}), value.get("schema_version", 1))


@dataclass(frozen=True)
class LeakageReport:
    passed: bool
    issues: tuple[LeakageIssue, ...]
    counts: Mapping[str, int]
    checked_entities: Mapping[str, int]
    fold_plan_id: str | None
    dataset_version_id: str | None
    policy: LeakagePolicy = LeakagePolicy.STRICT
    validator_version: str = VALIDATOR_VERSION
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = REPORT_SCHEMA_VERSION
    report_id: str = ""
    input_material_sha256: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported LeakageReport schema version")
        issues = tuple(sorted(self.issues, key=lambda i: (i.type.value, i.entity_id, i.folds_involved,
                                                          i.related_ids, i.severity)))
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "counts", freeze_json(self.counts))
        object.__setattr__(self, "checked_entities", freeze_json(self.checked_entities))
        object.__setattr__(self, "provenance", freeze_json(self.provenance))
        object.__setattr__(self, "policy", LeakagePolicy(self.policy))
        material_sha = self.input_material_sha256 or digest({"records": [], "assignments": {}, "overrides": {}})
        if len(material_sha) != 64 or any(char not in "0123456789abcdef" for char in material_sha):
            raise ValueError("input_material_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "input_material_sha256", material_sha)
        has_errors = any(issue.severity == "error" for issue in issues)
        if self.passed != (not has_errors):
            raise ValueError("passed must be false exactly when the report contains errors")
        expected = digest({"schema_version": self.schema_version, "passed": self.passed,
                           "issues": [_material_issue(i) for i in issues], "counts": self.counts,
                           "checked_entities": self.checked_entities,
                           "fold_plan_id": self.fold_plan_id, "dataset_version_id": self.dataset_version_id,
                           "policy": self.policy.value, "validator_version": self.validator_version,
                           "input_material_sha256": material_sha})
        if self.report_id and self.report_id != expected:
            raise ValueError("report_id does not match report contents")
        object.__setattr__(self, "report_id", expected)

    @property
    def n_errors(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def n_warnings(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "issues": [i.to_dict() for i in self.issues],
                "counts": jsonable(self.counts), "checked_entities": jsonable(self.checked_entities),
                "fold_plan_id": self.fold_plan_id, "dataset_version_id": self.dataset_version_id,
                "policy": self.policy.value, "validator_version": self.validator_version,
                "timestamp": self.timestamp, "schema_version": self.schema_version,
                "report_id": self.report_id, "input_material_sha256": self.input_material_sha256,
                "provenance": jsonable(self.provenance)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LeakageReport":
        if value.get("schema_version") != REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported LeakageReport schema version")
        return cls(value["passed"], tuple(LeakageIssue.from_dict(i) for i in value["issues"]),
                   value["counts"], value["checked_entities"], value.get("fold_plan_id"),
                   value.get("dataset_version_id"), LeakagePolicy(value["policy"]),
                   value["validator_version"], value["timestamp"], value["schema_version"],
                   value.get("report_id", ""), value.get("input_material_sha256", ""),
                   value.get("provenance", {}))


def _material_issue(issue: LeakageIssue) -> dict[str, Any]:
    data = issue.to_dict()
    if _looks_like_location(data["entity_id"]):
        data["entity_id"] = "<location>"
    data["related_ids"] = [item for item in data["related_ids"] if not _looks_like_location(item)]
    evidence = data["evidence"]
    data["evidence"] = {key: value for key, value in evidence.items()
                        if key not in {"path", "paths", "uri", "uris", "absolute_path"}}
    return data


def _looks_like_location(value: str) -> bool:
    return value.startswith(("/", "\\\\")) or "://" in value or (len(value) > 2 and value[1:3] == ":\\")
