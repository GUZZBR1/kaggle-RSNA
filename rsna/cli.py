"""Command line smoke path for the RSNA experiment contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib

from .contracts import DatasetVersion, ExperimentSpec, FoldPlan, ModelCandidate
from .data import load_or_refresh, save_manifest
from .experiments.plan import plan_jobs
from .experiments.runner import run_jobs
from .folds import generate_fold_plan, load_fold_plan, save_fold_plan
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
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "leakage-check":
        return _leakage_check(argv[1:])
    if argv and argv[0] == "data-index":
        return _data_index(argv[1:])
    if argv and argv[0] == "folds":
        return _folds_cli(argv[1:])
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
    report = validate_leakage(dataset, plan, policy=args.policy,
                              dataset_version=dataset.get("dataset_version"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Dataset: {report.dataset_version_id or 'unbound'}")
    print(f"FoldPlan: {report.fold_plan_id or 'explicit splits'}")
    labels = (("Patient leakage", "PATIENT_CROSS_FOLD"), ("Study leakage", "STUDY_CROSS_FOLD"),
              ("Series leakage", "SERIES_CROSS_FOLD"), ("Slice leakage", "SLICE_CROSS_FOLD"),
              ("Duplicate hashes across folds", "DUPLICATE_FILE_HASH"))
    for label, key in labels:
        if key == "DUPLICATE_FILE_HASH":
            count = sum(issue.type.value == key and len(issue.folds_involved) > 1 for issue in report.issues)
        else:
            count = report.counts["issues_by_type"].get(key, 0)
        print(f"{label}: {count}")
    if report.passed:
        print("RESULT: PASS")
        return 0
    print(f"RESULT: FAIL\nCritical issues: {report.n_errors}")
    return 1 if args.policy == LeakagePolicy.STRICT.value else 0


def _data_index(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="kaggle-rsna data-index")
    parser.add_argument("--input", required=True, help="Dataset root directory")
    parser.add_argument("--output", default="artifacts/dataset-index",
                        help="Output directory (default: artifacts/dataset-index)")
    parser.add_argument("--on-invalid", choices=("strict", "warn", "skip-invalid"), default="warn")
    parser.add_argument("--refresh", action="store_true", help="Force a source scan and incremental refresh")
    parser.add_argument("--rebuild", action="store_true", help="Discard and rebuild the cache")
    parser.add_argument("--validate-only", action="store_true", help="Validate cache and source without writing")
    parser.add_argument("--cache-policy", choices=("strict", "rebuild"), default="rebuild")
    parser.add_argument("--dataset-version-id", help="Optional DatasetVersion identity binding")
    args = parser.parse_args(argv)
    cache_path = Path(args.output) / "index.sqlite3"
    result = load_or_refresh(args.input, cache_path,
        dataset_version_id=args.dataset_version_id, refresh=args.refresh,
        rebuild=args.rebuild, validate_only=args.validate_only,
        cache_policy=args.cache_policy, on_invalid=args.on_invalid)
    index = result.index
    # Retain the original inspectable manifest alongside the SQLite refresh cache.
    manifest_path = Path(args.output) / "manifest.json"
    if not args.validate_only:
        save_manifest(index, manifest_path)
    stats = index.statistics
    print(f"Mode: {result.report.mode}")
    print(f"Files discovered: {result.report.added + result.report.modified + result.report.reused}")
    print(f"Files parsed: {result.report.reparsed}")
    print(f"Cache reused: {result.report.reused}")
    print(f"Added: {result.report.added}, Modified: {result.report.modified}, Removed: {result.report.removed}")
    print(f"Studies: {stats['n_studies']}, Series: {stats['n_series']}, Slices: {stats['n_slices']}")
    print(f"Invalid DICOMs: {stats['invalid_files']}, Duplicate SOPInstanceUIDs: {stats['duplicate_sop_uid_count']}")
    print(f"Index cache: {cache_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Source fingerprint: {result.source_fingerprint}")
    print(f"Dataset index ID: {index.index_id}")
    for warning in result.report.warnings:
        print(f"Warning: {warning}")
    return 0
