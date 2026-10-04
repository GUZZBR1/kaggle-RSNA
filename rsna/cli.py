"""Command line smoke path for the RSNA experiment contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib

from .contracts import DatasetVersion, ExperimentSpec, FoldPlan, ModelCandidate
from .experiments.plan import plan_jobs
from .experiments.runner import run_jobs
from .providers.mock import MockProvider
from .data.selection import SliceSelector, SliceSelectionConfig


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
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "select-slices":
        return _select_slices_cli(argv[1:])
    parser = argparse.ArgumentParser(prog="kaggle-rsna")
    parser.add_argument("config", help="TOML smoke configuration")
    args = parser.parse_args(argv)
    print(json.dumps(run_config(args.config), indent=2, sort_keys=True))
    return 0


def _select_slices_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="kaggle-rsna select-slices")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--strategy", choices=("uniform", "center", "physical_span"), default="uniform")
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--short-series-policy", choices=("keep_all", "repeat_nearest", "pad_reference", "strict"), default="keep_all")
    parser.add_argument("--physical-position-tolerance-mm", type=float, default=1e-3)
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args(argv)
    data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    config = SliceSelectionConfig(args.strategy, args.count, args.short_series_policy,
                                  args.physical_position_tolerance_mm, not args.no_fallback)
    selector = SliceSelector(config)
    summaries = []
    for study in data.get("studies", []):
        for series in study.get("series", []):
            result = selector.select(series)
            summaries.append(result.to_dict())
    print(json.dumps({"series_count": len(summaries), "selections": summaries}, indent=2, sort_keys=True))
    return 0
