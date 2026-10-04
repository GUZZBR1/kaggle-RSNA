"""Command line smoke path for the RSNA experiment contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

from .contracts import DatasetVersion, ExperimentSpec, FoldPlan, ModelCandidate
from .data import discover_dataset, save_manifest
from .experiments.plan import plan_jobs
from .experiments.runner import run_jobs
from .folds import generate_fold_plan, load_fold_plan, save_fold_plan
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


def run_folds(dataset_manifest: str | Path, output: str | Path, *, n_folds: int = 5,
              strategy: str = "group", seed: int = 42,
              labels_path: str | Path | None = None,
              dataset_version_id: str | None = None, locked: bool = False) -> dict:
    dataset = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8")) if labels_path else None
    plan = generate_fold_plan(dataset, n_folds, strategy, seed, labels,
                              dataset_version_id=dataset_version_id, locked=locked)
    save_fold_plan(plan, output)
    reloaded = load_fold_plan(output, dataset_version_id=plan.dataset_version_id, dataset=dataset)
    return {"plan": reloaded.to_dict(), "output": str(output)}


def _folds_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="kaggle-rsna folds")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--strategy", choices=("group", "multilabel-group-stratified"), default="group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels")
    parser.add_argument("--dataset-version-id")
    parser.add_argument("--locked", action="store_true")
    args = parser.parse_args(argv)
    strategy = "multilabel_group_stratified" if args.strategy == "multilabel-group-stratified" else args.strategy
    result = run_folds(args.dataset_manifest, args.output, n_folds=args.n_folds,
                       strategy=strategy, seed=args.seed, labels_path=args.labels,
                       dataset_version_id=args.dataset_version_id, locked=args.locked)
    plan = result["plan"]
    print(f"FoldPlan ID: {plan['fold_plan_id']}")
    print(f"DatasetVersion: {plan['dataset_version_id']}")
    print(f"Strategy: {plan['strategy']}\nFolds: {plan['n_folds']}")
    for fold, stats in plan["statistics"]["folds"].items():
        print(f"{fold}: {stats['n_studies']} studies")
    print("Leakage: PASS")
    for warning in plan["warnings"]:
        print(f"Warning: {warning}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "data-index":
        return _data_index(argv[1:])
    if argv and argv[0] == "folds":
        return _folds_cli(argv[1:])
    parser = argparse.ArgumentParser(prog="kaggle-rsna")
    parser.add_argument("config", help="TOML smoke configuration")
    args = parser.parse_args(argv)
    print(json.dumps(run_config(args.config), indent=2, sort_keys=True))
    return 0


def _data_index(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="kaggle-rsna data-index")
    parser.add_argument("--input", required=True, help="Dataset root directory")
    parser.add_argument("--output", required=True, help="Output directory for manifest.json")
    parser.add_argument("--on-invalid", choices=("strict", "warn", "skip-invalid"), default="warn")
    args = parser.parse_args(argv)
    index = discover_dataset(args.input, on_invalid=args.on_invalid)
    manifest_path = Path(args.output) / "manifest.json"
    manifest_hash = save_manifest(index, manifest_path)
    stats = index.statistics
    print(f"Studies: {stats['n_studies']}, Series: {stats['n_series']}, Slices: {stats['n_slices']}")
    print(f"Invalid DICOMs: {stats['invalid_files']}, Duplicate SOPInstanceUIDs: {stats['duplicate_sop_uid_count']}")
    print(f"Manifest: {manifest_path}")
    print(f"Manifest SHA-256: {manifest_hash}")
    print(f"Dataset index ID: {index.index_id}")
    return 0
