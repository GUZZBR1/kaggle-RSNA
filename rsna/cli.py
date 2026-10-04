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
from .leakage import LeakagePolicy, validate_leakage


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
    if argv and argv[0] == "leakage-check":
        return _leakage_check(argv[1:])
    parser = argparse.ArgumentParser(prog="kaggle-rsna")
    parser.add_argument("config", help="TOML smoke configuration")
    args = parser.parse_args(argv)
    print(json.dumps(run_config(args.config), indent=2, sort_keys=True))
    return 0


def _leakage_check(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="kaggle-rsna leakage-check")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--fold-plan")
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", choices=[policy.value for policy in LeakagePolicy],
                        default=LeakagePolicy.STRICT.value)
    args = parser.parse_args(argv)
    dataset = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.fold_plan).read_text(encoding="utf-8")) if args.fold_plan else None
    report = validate_leakage(dataset, plan, policy=args.policy)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Dataset: {report.dataset_version_id or 'unbound'}")
    print(f"FoldPlan: {report.fold_plan_id or 'explicit splits'}")
    labels = (("Patient leakage", "PATIENT_CROSS_FOLD"), ("Study leakage", "STUDY_CROSS_FOLD"),
              ("Series leakage", "SERIES_CROSS_FOLD"), ("Slice leakage", "SLICE_CROSS_FOLD"),
              ("Duplicate hashes across folds", "DUPLICATE_FILE_HASH"))
    for label, key in labels:
        print(f"{label}: {report.counts['issues_by_type'].get(key, 0)}")
    if report.passed:
        print("RESULT: PASS")
        return 0
    print(f"RESULT: FAIL\nCritical issues: {report.n_errors}")
    return 1 if args.policy == LeakagePolicy.STRICT.value else 0
