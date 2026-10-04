"""Command line smoke path for the RSNA experiment contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

from .contracts import DatasetVersion, ExperimentSpec, FoldPlan, ModelCandidate
from .data import load_or_refresh, save_manifest
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
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "data-index":
        return _data_index(argv[1:])
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
