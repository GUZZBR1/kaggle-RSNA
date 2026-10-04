"""Canonical target definitions for the RSNA Knee Abnormality Detection competition."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from .identity import digest

TARGET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Target:
    name: str
    aliases: tuple[str, ...] = ()


OFFICIAL_TARGETS = (
    Target("ACL", ("Anterior Cruciate Ligament", "ACL injury", "anterior cruciate ligament injury")),
    Target("MCL", ("Medial Collateral Ligament", "MCL injury", "medial collateral ligament injury")),
    Target("Medial Meniscus", ("Medial Meniscus Tear",)),
    Target("Lateral Meniscus", ("Lateral Meniscus Tear",)),
    Target("Medial OA", ("Medial Osteoarthritis", "Medial-compartment osteoarthritis")),
    Target("Lateral OA", ("Lateral Osteoarthritis", "Lateral-compartment osteoarthritis")),
    Target("PF OA", ("Patellofemoral Osteoarthritis", "Patellofemoral OA")),
    Target("Effusion", ("Joint Effusion",)),
    Target("Synovitis"),
    Target("Baker's", ("Baker's Cyst", "Bakers", "Baker Cyst")),
    Target("Contusion", ("Bone Contusion", "Bone Bruise")),
    Target("Fracture"),
)


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.casefold())


@dataclass(frozen=True)
class TargetRegistry:
    targets: tuple[Target, ...] = OFFICIAL_TARGETS
    schema_version: int = TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != TARGET_SCHEMA_VERSION:
            raise ValueError(f"unsupported target schema version: {self.schema_version}")
        names = [target.name for target in self.targets]
        if (not names or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("canonical target names must be nonempty and unique")
        lookup: dict[str, str] = {}
        for target in self.targets:
            for alias in (target.name, *target.aliases):
                if not isinstance(alias, str) or not alias.strip():
                    raise ValueError("target aliases must be nonempty strings")
                key = _key(alias)
                previous = lookup.get(key)
                if previous is not None and previous != target.name:
                    raise ValueError(f"ambiguous target alias: {alias!r}")
                lookup[key] = target.name
        object.__setattr__(self, "_lookup", lookup)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(target.name for target in self.targets)

    @property
    def count(self) -> int:
        return len(self.targets)

    @property
    def registry_id(self) -> str:
        return digest(self.to_dict())

    def target_index(self, name: str) -> int:
        canonical = self.canonical_name(name)
        return self.names.index(canonical)

    def target_name(self, index: int) -> str:
        if type(index) is not int or not 0 <= index < self.count:
            raise ValueError(f"target index must be between 0 and {self.count - 1}")
        return self.targets[index].name

    def canonical_name(self, name: str) -> str:
        if not isinstance(name, str) or _key(name) not in self._lookup:
            raise ValueError(f"unknown target name or alias: {name!r}")
        return self._lookup[_key(name)]

    def validate_target_order(self, names: Sequence[str]) -> tuple[str, ...]:
        actual = tuple(names)
        if actual != self.names:
            raise ValueError(f"target order mismatch: expected {self.names!r}, got {actual!r}")
        return actual

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version,
                "targets": [{"name": item.name, "index": index,
                             "aliases": list(item.aliases)}
                            for index, item in enumerate(self.targets)]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetRegistry":
        if not isinstance(data, Mapping) or not isinstance(data.get("targets"), list):
            raise ValueError("serialized target registry must contain a targets array")
        targets = data["targets"]
        for index, item in enumerate(targets):
            if not isinstance(item, Mapping) or type(item.get("index")) is not int or item["index"] != index:
                raise ValueError("serialized target registry indexes do not match canonical order")
        return cls(tuple(Target(item["name"], tuple(item["aliases"])) for item in targets),
                   data.get("schema_version"))


TARGET_REGISTRY = TargetRegistry()
TARGETS = TARGET_REGISTRY.names
TARGET_REGISTRY_ID = TARGET_REGISTRY.registry_id


def validate_targets(names: Sequence[str], *, allow_synthetic: bool = False) -> tuple[str, ...]:
    values = tuple(names)
    if allow_synthetic:
        if (not values or any(not isinstance(value, str) or not value.strip() for value in values)
                or len(set(values)) != len(values)):
            raise ValueError("synthetic target names must be nonempty and unique")
        return values
    return TARGET_REGISTRY.validate_target_order(values)


def validate_prediction_columns(columns: Sequence[str], *, allow_synthetic: bool = False) -> tuple[str, ...]:
    return validate_targets(columns, allow_synthetic=allow_synthetic)


def validate_submission_columns(columns: Sequence[str]) -> tuple[str, ...]:
    expected = ("StudyInstanceUID", *TARGETS)
    actual = tuple(columns)
    if actual != expected:
        raise ValueError(f"submission columns mismatch: expected {expected!r}, got {actual!r}")
    return actual
