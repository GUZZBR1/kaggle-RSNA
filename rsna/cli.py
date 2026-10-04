"""Command line smoke path for the RSNA experiment contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

from .contracts import DatasetVersion, ExperimentSpec, FoldPlan, ModelCandidate
from .experiments.plan import plan_jobs
from .experiments.runner import run_jobs
from .providers.mock import MockProvider


def run_config(path: str | Path) -> dict:
    config = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    data = config["dataset"]
    dataset = DatasetVersion(**data)
    folds = FoldPlan(dataset.dataset_version_id, **config["fold_plan"])
    candidates = [ModelCandidate(**item) for item in config["model_candidates"]]
    spec_data = dict(config["experiment"])
    spec_data.update(dataset_version_id=dataset.dataset_version_id,
                     fold_plan_id=folds.fold_plan_id,
                     model_candidate_ids=[c.model_candidate_id for c in candidates])
    spec = ExperimentSpec(**spec_data)
    jobs = plan_jobs(spec, dataset, folds, {c.model_candidate_id: c for c in candidates})
    results = run_jobs(jobs, MockProvider())
    return {"experiment": spec.to_dict(), "jobs": [job.to_dict() for job in jobs],
            "results": [result.to_dict() for result in results], "synthetic": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kaggle-rsna")
    parser.add_argument("config", help="TOML smoke configuration")
    args = parser.parse_args(argv)
    print(json.dumps(run_config(args.config), indent=2, sort_keys=True))
    return 0
