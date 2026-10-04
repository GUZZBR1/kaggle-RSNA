"""Fold planning contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from ..identity import digest, freeze_json, jsonable

SCHEMA_VERSION = 2
GENERATOR_VERSION = "0.1.0"


@dataclass(frozen=True)
class FoldGenerationConfig:
    """Validated parameters for the canonical fold generator."""

    n_folds: int = 5
    strategy: str = "group"
    random_state: int = 42
    size_deviation_threshold: float = 0.20
    prevalence_range_threshold: float = 0.20

    def __post_init__(self) -> None:
        if type(self.n_folds) is not int or self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if self.strategy not in {"group", "multilabel_group_stratified"}:
            raise ValueError("unsupported fold strategy")
        if type(self.random_state) is not int or self.random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        for name in ("size_deviation_threshold", "prevalence_range_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")


@dataclass(frozen=True)
class FoldPlanManifest:
    dataset_version_id: str
    generator_version: str
    input_fingerprint: str
    provenance: Mapping[str, Any]
    strategy: str
    n_folds: int
    random_state: int
    grouping_key: str
    assignments: Mapping[str, str]
    group_assignments: Mapping[str, str]
    study_groups: Mapping[str, str]
    study_fingerprints: Mapping[str, str]
    configuration: Mapping[str, Any] = field(default_factory=dict)
    statistics: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    locked: bool = False
    schema_version: int = SCHEMA_VERSION
    fold_plan_id: str = ""
    application_warnings: tuple[str, ...] = field(default=(), compare=False, repr=False)

    def __post_init__(self) -> None:
        if not _sha(self.dataset_version_id):
            raise ValueError("dataset_version_id must be a lowercase SHA-256 digest")
        if not isinstance(self.generator_version, str) or not self.generator_version.strip():
            raise ValueError("generator_version cannot be empty")
        if not _sha(self.input_fingerprint):
            raise ValueError("input_fingerprint must be a lowercase SHA-256 digest")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported fold manifest schema version: {self.schema_version}")
        if not isinstance(self.grouping_key, str) or not self.grouping_key.strip():
            raise ValueError("grouping_key cannot be empty")
        if type(self.locked) is not bool:
            raise ValueError("locked must be a boolean")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be an object")
        if not isinstance(self.configuration, Mapping) or not isinstance(self.statistics, Mapping):
            raise ValueError("configuration and statistics must be objects")
        if self.strategy not in {"group", "multilabel_group_stratified", "imported"}:
            raise ValueError(f"unsupported fold strategy: {self.strategy}")
        if type(self.n_folds) is not int or self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if type(self.random_state) is not int or self.random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        if (not isinstance(self.assignments, Mapping) or not self.assignments
                or any(not isinstance(k, str) or not k for k in self.assignments)
                or any(not isinstance(v, str) for v in self.assignments.values())):
            raise ValueError("assignments must map nonempty study IDs to fold IDs")
        expected_folds = {f"fold_{i}" for i in range(self.n_folds)}
        if set(self.assignments.values()) - expected_folds:
            raise ValueError("assignments contain fold IDs outside the declared range")
        if set(self.group_assignments.values()) != expected_folds:
            raise ValueError("every declared fold must have at least one indivisible group")
        if (any(not isinstance(k, str) or not k for k in self.study_groups)
                or any(not isinstance(v, str) or not v for v in self.study_groups.values())):
            raise ValueError("study_groups must map study IDs to nonempty groups")
        if any(not _sha(value) for value in self.study_fingerprints.values()):
            raise ValueError("study_fingerprints must contain lowercase SHA-256 digests")
        if set(self.assignments) != set(self.study_groups) or set(self.assignments) != set(self.study_fingerprints):
            raise ValueError("assignments, study_groups, and study_fingerprints must cover identical studies")
        if set(self.study_groups.values()) != set(self.group_assignments):
            raise ValueError("study_groups must reference every group assignment exactly")
        for study, group in self.study_groups.items():
            if self.assignments[study] != self.group_assignments[group]:
                raise ValueError(f"study {study!r} assignment conflicts with its indivisible group")
        object.__setattr__(self, "assignments", freeze_json(dict(sorted(self.assignments.items()))))
        object.__setattr__(self, "group_assignments", freeze_json(dict(sorted(self.group_assignments.items()))))
        object.__setattr__(self, "study_groups", freeze_json(dict(sorted(self.study_groups.items()))))
        object.__setattr__(self, "study_fingerprints", freeze_json(dict(sorted(self.study_fingerprints.items()))))
        object.__setattr__(self, "configuration", freeze_json(self.configuration))
        object.__setattr__(self, "statistics", freeze_json(self.statistics))
        object.__setattr__(self, "provenance", freeze_json(self.provenance))
        if any(not isinstance(item, str) for item in self.warnings):
            raise ValueError("warnings must contain strings")
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "application_warnings", tuple(self.application_warnings))
        expected = digest(self._identity_payload())
        if self.fold_plan_id and self.fold_plan_id != expected:
            raise ValueError("fold_plan_id does not match manifest contents")
        object.__setattr__(self, "fold_plan_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "dataset_version_id": self.dataset_version_id,
                "generator_version": self.generator_version,
                "input_fingerprint": self.input_fingerprint, "provenance": self.provenance,
                "strategy": self.strategy, "n_folds": self.n_folds, "random_state": self.random_state,
                "grouping_key": self.grouping_key, "assignments": self.assignments,
                "group_assignments": self.group_assignments, "study_groups": self.study_groups,
                "study_fingerprints": self.study_fingerprints, "configuration": self.configuration,
                "statistics": self.statistics,
                "warnings": self.warnings, "locked": self.locked}

    def fold_for_study(self, study_id: str) -> str:
        try:
            return self.assignments[study_id]
        except KeyError as exc:
            raise KeyError(f"study {study_id!r} is not assigned in FoldPlan {self.fold_plan_id}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {**jsonable(self._identity_payload()), "fold_plan_id": self.fold_plan_id}


def _sha(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
