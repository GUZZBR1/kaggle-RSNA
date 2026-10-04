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
    if len(argv) >= 2 and argv[:2] == ["synthetic", "smoke"]:
        return _synthetic_smoke_cli(argv[2:])
    if argv and argv[0] == "select-slices":
        return _select_slices_cli(argv[1:])
    if argv and argv[0] == "leakage-check":
        return _leakage_check(argv[1:])
    if argv and argv[0] == "data-index":
        return _data_index(argv[1:])
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


def _synthetic_smoke_cli(argv: list[str]) -> int:
    import tempfile
    parser = argparse.ArgumentParser(prog="kaggle-rsna synthetic smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inject", choices=("patient-leakage", "duplicate-sop",
        "orientation-conflict", "missing-position", "missing-metadata", "spacing-irregular",
        "corrupted-cache"))
    parser.add_argument("--keep", nargs="?", const="synthetic-smoke", metavar="DIR",
                        help="keep generated DICOMs and reports in DIR (default: ./synthetic-smoke)")
    args = parser.parse_args(argv)
    from .data.smoke import run_data_smoke
    if args.keep:
        result = run_data_smoke(Path(args.keep), seed=args.seed, injection=args.inject)
    else:
        with tempfile.TemporaryDirectory(prefix="rsna-synthetic-smoke-") as temp:
            result = run_data_smoke(temp, seed=args.seed, injection=args.inject)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _select_slices_cli(argv: list[str]) -> int:
    from .data.selection import SliceSelectionConfig, SliceSelector
    parser = argparse.ArgumentParser(prog="kaggle-rsna select-slices")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--strategy", choices=("uniform", "center", "physical_span"), default="uniform")
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--short-series-policy", choices=("keep_all", "repeat_nearest",
        "pad_reference", "strict"), default="keep_all")
    parser.add_argument("--physical-position-tolerance-mm", type=float, default=1e-3)
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args(argv)
    data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    config = SliceSelectionConfig(args.strategy, args.count, args.short_series_policy,
                                  args.physical_position_tolerance_mm, not args.no_fallback)
    selector = SliceSelector(config)
    selections = [selector.select(series).to_dict()
        for study in data.get("studies", []) for series in study.get("series", [])]
    print(json.dumps({"series_count": len(selections), "selections": selections},
                     indent=2, sort_keys=True))
    return 0
