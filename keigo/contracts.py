"""Versioned, strategy-neutral contracts for the first experiment flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .identity import digest, freeze_json, jsonable

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Candidate:
    name: str
    artifact_sha256: str
    entrypoint: str = "agent"
    artifact_uri: str = ""
    configuration: Mapping[str, Any] = field(default_factory=dict)
    candidate_id: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("candidate name cannot be empty")
        _validate_sha256(self.artifact_sha256, "artifact_sha256")
        if not self.entrypoint.strip():
            raise ValueError("candidate entrypoint cannot be empty")
        object.__setattr__(self, "configuration", freeze_json(self.configuration))
        expected = digest({"name": self.name, "artifact_sha256": self.artifact_sha256,
                           "entrypoint": self.entrypoint,
                           "configuration": self.configuration})
        if self.candidate_id and self.candidate_id != expected:
            raise ValueError("candidate_id does not match candidate contents")
        object.__setattr__(self, "candidate_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "artifact_sha256": self.artifact_sha256,
                "entrypoint": self.entrypoint, "artifact_uri": self.artifact_uri,
                "configuration": jsonable(self.configuration),
                "candidate_id": self.candidate_id}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    candidate_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    seed_split: str = "development"
    opponents: tuple[str, ...] = ("baseline",)
    simulator: str = "kaggriculture"
    simulator_version: str = "unconfigured"
    configuration: Mapping[str, Any] = field(default_factory=dict)
    evaluation_policy: Mapping[str, Any] = field(default_factory=dict)
    resource_request: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported ExperimentSpec schema: {self.schema_version}")
        if not self.name.strip():
            raise ValueError("experiment name cannot be empty")
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("ExperimentSpec requires unique candidate IDs")
        for candidate_id in self.candidate_ids:
            _validate_sha256(candidate_id, "candidate_id")
        if not self.seeds or any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise ValueError("ExperimentSpec requires nonnegative integer seeds")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("ExperimentSpec seeds must be unique")
        if self.seed_split not in {"development", "validation", "holdout"}:
            raise ValueError("seed_split must be development, validation or holdout")
        if not self.opponents or len(set(self.opponents)) != len(self.opponents):
            raise ValueError("ExperimentSpec requires unique opponents")
        if any(not isinstance(opponent, str) or not opponent.strip()
               for opponent in self.opponents):
            raise ValueError("opponent IDs cannot be empty")
        if not self.simulator.strip() or not self.simulator_version.strip():
            raise ValueError("simulator and simulator_version must be declared")
        object.__setattr__(self, "configuration", freeze_json(self.configuration))
        object.__setattr__(self, "evaluation_policy", freeze_json(self.evaluation_policy))
        object.__setattr__(self, "resource_request", freeze_json(self.resource_request))

    @property
    def experiment_id(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "candidate_ids": list(self.candidate_ids),
                "seeds": list(self.seeds), "seed_split": self.seed_split,
                "opponents": list(self.opponents), "simulator": self.simulator,
                "simulator_version": self.simulator_version,
                "configuration": jsonable(self.configuration),
                "evaluation_policy": jsonable(self.evaluation_policy),
                "resource_request": jsonable(self.resource_request),
                "schema_version": self.schema_version}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentSpec":
        data = dict(value)
        data["candidate_ids"] = tuple(data["candidate_ids"])
        data["seeds"] = tuple(data["seeds"])
        data["seed_split"] = data.get("seed_split", "development")
        data["opponents"] = tuple(data.get("opponents", ("baseline",)))
        return cls(**data)


@dataclass(frozen=True)
class SimulationJob:
    experiment_id: str
    candidate_id: str
    candidate_artifact_sha256: str
    candidate_artifact_uri: str
    opponent_id: str
    seed: int
    seat: int
    simulator: str
    simulator_version: str
    configuration: Mapping[str, Any] = field(default_factory=dict)
    job_id: str = ""

    def __post_init__(self) -> None:
        _validate_sha256(self.experiment_id, "experiment_id")
        _validate_sha256(self.candidate_id, "candidate_id")
        _validate_sha256(self.candidate_artifact_sha256, "candidate_artifact_sha256")
        if (not self.opponent_id.strip() or not self.simulator.strip()
                or not self.simulator_version.strip()):
            raise ValueError("job opponent and simulator references cannot be empty")
        if type(self.seed) is not int or self.seed < 0 or type(self.seat) is not int or self.seat not in (0, 1):
            raise ValueError("job seed must be nonnegative and seat must be 0 or 1")
        object.__setattr__(self, "configuration", freeze_json(self.configuration))
        expected = digest({"experiment_id": self.experiment_id,
                           "candidate_id": self.candidate_id,
                           "candidate_artifact_sha256": self.candidate_artifact_sha256,
                           "opponent_id": self.opponent_id, "seed": self.seed,
                           "seat": self.seat, "simulator": self.simulator,
                           "simulator_version": self.simulator_version,
                           "configuration": self.configuration})
        if self.job_id and self.job_id != expected:
            raise ValueError("job_id does not match job contents")
        object.__setattr__(self, "job_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id, "candidate_id": self.candidate_id,
                "candidate_artifact_sha256": self.candidate_artifact_sha256,
                "candidate_artifact_uri": self.candidate_artifact_uri,
                "opponent_id": self.opponent_id, "seed": self.seed, "seat": self.seat,
                "simulator": self.simulator, "simulator_version": self.simulator_version,
                "configuration": jsonable(self.configuration), "job_id": self.job_id}


@dataclass(frozen=True)
class SimulationResult:
    job_id: str
    candidate_id: str
    opponent_id: str
    seed: int
    seat: int
    status: str
    metrics: Mapping[str, float]
    provider: str
    execution_id: str
    failure: str | None = None
    outputs: Mapping[str, str] = field(default_factory=dict)
    result_id: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_sha256(self.job_id, "job_id")
        _validate_sha256(self.candidate_id, "candidate_id")
        if (not self.opponent_id.strip() or type(self.seed) is not int or self.seed < 0
                or type(self.seat) is not int or self.seat not in (0, 1)):
            raise ValueError("result opponent, seed or seat is invalid")
        if self.status not in {"succeeded", "failed"}:
            raise ValueError("result status must be succeeded or failed")
        if self.status == "failed" and not self.failure:
            raise ValueError("failed results must explain the failure")
        if self.status == "succeeded" and self.failure:
            raise ValueError("successful results cannot carry a failure")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported SimulationResult schema: {self.schema_version}")
        if not self.provider.strip() or not self.execution_id.strip():
            raise ValueError("result must declare provider and execution ID")
        object.__setattr__(self, "metrics", freeze_json(self.metrics))
        object.__setattr__(self, "outputs", freeze_json(self.outputs))
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in self.metrics.values()):
            raise ValueError("result metrics must be numeric")
        expected = digest({"job_id": self.job_id, "candidate_id": self.candidate_id,
                           "opponent_id": self.opponent_id, "seed": self.seed,
                           "seat": self.seat, "status": self.status,
                           "metrics": self.metrics, "provider": self.provider,
                           "execution_id": self.execution_id, "failure": self.failure,
                           "outputs": self.outputs, "schema_version": self.schema_version})
        if self.result_id and self.result_id != expected:
            raise ValueError("result_id does not match result contents")
        object.__setattr__(self, "result_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {"job_id": self.job_id, "candidate_id": self.candidate_id,
                "opponent_id": self.opponent_id, "seed": self.seed, "seat": self.seat,
                "status": self.status, "metrics": jsonable(self.metrics),
                "provider": self.provider, "execution_id": self.execution_id,
                "failure": self.failure, "outputs": jsonable(self.outputs),
                "result_id": self.result_id, "schema_version": self.schema_version}


def _validate_sha256(value: str, label: str) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
