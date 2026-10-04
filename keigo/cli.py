"""CLI smoke flow. Real experiments select their provider in deployment configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

from .artifacts import JsonArtifactStore
from .contracts import Candidate, ExperimentSpec
from .experiments.runner import run_experiment
from .experiments.seeds import SeedRegistry
from .providers.mock import MockProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Keigo mock experiment")
    parser.add_argument("config", type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    candidate_data = config["candidate"]
    candidate = Candidate(**candidate_data)
    experiment_data = dict(config["experiment"])
    seed_registry = experiment_data.pop("seed_registry", "../seeds/registry.json")
    experiment_data["candidate_ids"] = [candidate.candidate_id]
    spec = ExperimentSpec.from_dict(experiment_data)
    seed_registry_path = (args.config.parent / seed_registry).resolve()
    SeedRegistry(seed_registry_path).validate(spec.seeds, spec.seed_split)
    evaluation, artifact = run_experiment(
        spec, {candidate.candidate_id: candidate}, MockProvider(),
        JsonArtifactStore(args.artifacts))
    print(f"experiment={spec.experiment_id}")
    print(f"evaluation={evaluation.evaluation_id}")
    print(f"artifact={artifact.uri}")
