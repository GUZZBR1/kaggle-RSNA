"""Small generic evaluator; game-specific metrics can be supplied by workloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Sequence

from ..contracts import ExperimentSpec, SimulationResult
from ..identity import digest


@dataclass(frozen=True)
class Evaluation:
    experiment_id: str
    result_ids: tuple[str, ...]
    candidate_means: dict[str, dict[str, float]]
    status: str
    evaluation_id: str

    def to_dict(self) -> dict:
        return {"experiment_id": self.experiment_id,
                "result_ids": list(self.result_ids),
                "candidate_means": self.candidate_means,
                "status": self.status,
                "evaluation_id": self.evaluation_id}


def evaluate(spec: ExperimentSpec, results: Sequence[SimulationResult]) -> Evaluation:
    expected = {(candidate, opponent, seed, seat)
                for candidate in spec.candidate_ids for opponent in spec.opponents
                for seed in spec.seeds for seat in (0, 1)}
    observed = set()
    grouped: dict[str, dict[str, list[float]]] = {}
    for result in results:
        if result.status != "succeeded":
            raise ValueError(f"failed result {result.result_id} cannot be evaluated")
        key = (result.candidate_id, result.opponent_id, result.seed, result.seat)
        if key in observed:
            raise ValueError(f"duplicate result for {key}")
        observed.add(key)
        if result.candidate_id not in spec.candidate_ids:
            raise ValueError("result candidate is outside the ExperimentSpec")
        if "score" not in result.metrics:
            raise ValueError("each result must include a score metric")
        grouped.setdefault(result.candidate_id, {}).setdefault("score", []).append(
            float(result.metrics["score"]))
    if observed != expected:
        missing, extra = expected - observed, observed - expected
        raise ValueError(f"result set is incomplete or unexpected (missing={len(missing)}, "
                         f"extra={len(extra)})")
    means = {candidate: {metric: fmean(values) for metric, values in metrics.items()}
             for candidate, metrics in sorted(grouped.items())}
    ordered_ids = tuple(sorted(result.result_id for result in results))
    body = {"experiment_id": spec.experiment_id, "result_ids": ordered_ids,
            "candidate_means": means, "status": "complete"}
    return Evaluation(**body, evaluation_id=digest(body))
