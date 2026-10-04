"""Command line entry point for smoke experiments and dataset operations."""

from __future__ import annotations

import argparse
import json
import importlib
from pathlib import Path
import sys
import tomllib
from typing import Any

from .contracts import DatasetVersion, ExperimentSpec, FoldPlan, ModelCandidate
from .data import load_or_refresh, save_manifest
from .experiments.plan import plan_jobs
from .experiments.runner import run_jobs
from .data.geometry import GeometryConfig
from .data.selection import SliceSelector, SliceSelectionConfig
from .leakage import LeakagePolicy, validate_leakage
from .providers.mock import MockProvider


def run_config(path: str | Path) -> dict:
    """Run the existing synthetic TOML smoke configuration."""
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rsna", description="Inspect and validate RSNA dataset manifests.")
    commands = parser.add_subparsers(dest="group", required=True)
    data = commands.add_parser("data", help="index, inspect, and validate dataset metadata")
    data_commands = data.add_subparsers(dest="command", required=True)
    index_command = data_commands.add_parser("index", help="discover DICOM metadata and write a manifest")
    index_command.add_argument("--input", required=True, help="dataset root directory")
    index_command.add_argument("--output", default="artifacts/dataset-index", help="output directory")
    index_command.add_argument("--on-invalid", choices=("strict", "warn", "skip-invalid"), default="warn")
    index_command.add_argument("--refresh", action="store_true", help="force an incremental source check")
    index_command.add_argument("--rebuild", action="store_true", help="discard and rebuild the cache")
    index_command.add_argument("--validate-only", action="store_true", help="validate cache and source without writing")
    index_command.add_argument("--cache-policy", choices=("strict", "rebuild"), default="rebuild")
    index_command.add_argument("--dataset-version-id", help="optional DatasetVersion identity binding")
    _output_options(index_command)
    for name, help_text in (("summary", "summarize a dataset manifest"),
                            ("inspect-study", "inspect one Study"),
                            ("inspect-series", "inspect one Series"),
                            ("inspect-slice", "inspect one DICOM instance"),
                            ("inspect-manifest", "inspect manifest metadata and tables"),
                            ("validate", "validate dataset structure"),
                            ("sample", "select a deterministic entity sample"),
                            ("stats", "summarize dataset distributions")):
        command = data_commands.add_parser(name, help=help_text)
        _common_command_options(command)
        command.add_argument("--manifest", help="path to a canonical dataset manifest JSON")
        _output_options(command)
        if name in {"inspect-study", "inspect-series", "inspect-slice"}:
            entity_arg = {"inspect-study": "--study-id", "inspect-series": "--series-id",
                          "inspect-slice": "--sop-id"}[name]
            command.add_argument(entity_arg, required=True)
        if name == "validate":
            command.add_argument("--level", choices=("basic", "full"), default=None)
            command.add_argument("--dataset-root")
            command.add_argument("--warnings-as-errors", action="store_true")
        if name == "sample":
            command.add_argument("--entity", choices=("study", "series", "slice"), required=True)
            command.add_argument("--count", type=int, required=True)
            command.add_argument("--seed", type=int, default=0)
        if name in {"summary", "stats"}:
            command.add_argument("--plane")
            command.add_argument("--laterality")
            command.add_argument("--warning-code")
    artifact = commands.add_parser("artifact", help="inspect stored artifacts")
    artifact_commands = artifact.add_subparsers(dest="command", required=True)
    validate = artifact_commands.add_parser("validate", help="validate an artifact or manifest file")
    _common_command_options(validate)
    validate.add_argument("path", help="artifact JSON path")
    _output_options(validate)
    validate.add_argument("--level", choices=("basic", "full"), default="full")
    validate.add_argument("--warnings-as-errors", action="store_true")
    synthetic = commands.add_parser("synthetic", help="run synthetic data pipeline checks")
    synthetic_commands = synthetic.add_subparsers(dest="command", required=True)
    smoke = synthetic_commands.add_parser("smoke", help="exercise the metadata-only data pipeline")
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--inject", choices=("patient-leakage", "duplicate-sop",
        "orientation-conflict", "missing-position", "missing-metadata", "spacing-irregular",
        "corrupted-cache", "hierarchy-mismatch"))
    smoke.add_argument("--keep", nargs="?", const="synthetic-smoke", metavar="DIR",
                       help="keep generated DICOMs and reports in DIR (default: ./synthetic-smoke)")
    _output_options(smoke)
    smoke.set_defaults(format="json")
    train = commands.add_parser("train", help="run model-agnostic PyTorch training")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    train_smoke = train_commands.add_parser("smoke", help="run a deterministic synthetic training job")
    train_smoke.add_argument("--output-dir", help="directory for checkpoint and telemetry artifacts")
    train_smoke.add_argument("--resume-from", help="resume a prior smoke checkpoint")
    train_smoke.add_argument("--stop-after-optimizer-steps", type=int,
                             help="stop at an optimizer boundary and print the resume checkpoint")
    _output_options(train_smoke)
    train_smoke.set_defaults(format="json")
    train_run = train_commands.add_parser("run", help="run a declared job using importable data/model factories")
    train_run.add_argument("--job", required=True, help="JSON job document")
    _output_options(train_run)
    parser.add_argument("--config", help="optional TOML defaults")
    parser.add_argument("--debug", action="store_true", help="show traceback for operational errors")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("human", "json"), default=None)
    parser.add_argument("--json", dest="format", action="store_const", const="json")


def _common_command_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--debug", action="store_true", default=argparse.SUPPRESS,
                        help="show traceback for operational errors")
    parser.add_argument("--verbose", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS)


def _config_defaults(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    return tomllib.loads(Path(path).read_text(encoding="utf-8")).get("cli", {})


def _render_human(value: Any, *, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            label = key.replace("_", " ").capitalize()
            if isinstance(item, (dict, list)):
                lines.extend((f"{prefix}{label}:", _render_human(item, indent=indent + 2)))
            elif item is not None:
                lines.append(f"{prefix}{label}: {item}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{prefix}(none)"
        return "\n".join(f"{prefix}- {item}" if not isinstance(item, dict) else
                         f"{prefix}- " + _render_human(item, indent=indent + 2).lstrip() for item in value)
    return f"{prefix}{value}"


def _emit(value: Any, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(_render_human(value))


def _dispatch(args: argparse.Namespace, config: dict[str, Any] | None = None) -> tuple[Any, int]:
    config = config or {}
    if args.group == "synthetic":
        import tempfile
        from .data.smoke import run_data_smoke
        if args.keep:
            result = run_data_smoke(Path(args.keep), seed=args.seed, injection=args.inject)
        else:
            with tempfile.TemporaryDirectory(prefix="rsna-synthetic-smoke-") as temp:
                result = run_data_smoke(temp, seed=args.seed, injection=args.inject)
        return result, 0
    if args.group == "train":
        if args.train_command == "smoke":
            from .training.smoke import run_training_smoke
            result = run_training_smoke(args.output_dir, resume_from=args.resume_from,
                                       stop_after_optimizer_steps=args.stop_after_optimizer_steps)
            return result, 0 if result["status"] in {"succeeded", "interrupted"} else 1
        result = _run_training_job(Path(args.job))
        return result, 0 if result["status"] == "succeeded" else 1
    from .inspection.query import (inspect_manifest, inspect_series, inspect_slice,
                                   inspect_study, load_dataset_index, sample_entities)
    from .inspection.summary import build_dataset_stats, build_dataset_summary
    from .inspection.validate import validate_artifact, validate_dataset

    if args.group == "artifact":
        level = args.level or config.get("validation_level", "full")
        path = Path(args.path)
        if config.get("artifact_directory") and not path.is_absolute():
            path = Path(config["artifact_directory"]) / path
        report = validate_artifact(path, level=level, warnings_as_errors=args.warnings_as_errors)
        return report.to_dict(), report.exit_code
    if args.command == "index":
        result = load_or_refresh(args.input, Path(args.output) / "index.sqlite3",
            dataset_version_id=args.dataset_version_id, refresh=args.refresh, rebuild=args.rebuild,
            validate_only=args.validate_only, cache_policy=args.cache_policy, on_invalid=args.on_invalid)
        manifest = Path(args.output) / "manifest.json"
        manifest_hash = None if args.validate_only else save_manifest(result.index, manifest)
        return {"mode": result.report.mode, "files_discovered": result.report.added + result.report.modified + result.report.reused,
                "files_parsed": result.report.reparsed, "cache_reused": result.report.reused,
                "added": result.report.added, "modified": result.report.modified,
                "removed": result.report.removed, "index_cache": str(Path(args.output) / "index.sqlite3"),
                "manifest": str(manifest) if manifest_hash is not None else None,
                "manifest_sha256": manifest_hash, "source_fingerprint": result.source_fingerprint,
                "dataset_index_id": result.index.index_id, "statistics": dict(result.index.statistics),
                "warnings": list(result.report.warnings)}, 0
    manifest = args.manifest
    if args.command == "validate":
        report = validate_dataset(manifest, level=args.level or config.get("validation_level", "basic"),
            dataset_root=args.dataset_root or config.get("dataset_root"),
            warnings_as_errors=args.warnings_as_errors)
        return report.to_dict(), report.exit_code
    index = load_dataset_index(manifest)
    if args.command == "summary":
        return build_dataset_summary(index, plane=args.plane, laterality=args.laterality,
                                     warning_code=args.warning_code), 0
    if args.command == "stats":
        return build_dataset_stats(index, plane=args.plane, laterality=args.laterality,
                                   warning_code=args.warning_code), 0
    if args.command == "inspect-study":
        return inspect_study(index, args.study_id), 0
    if args.command == "inspect-series":
        return inspect_series(index, args.series_id), 0
    if args.command == "inspect-slice":
        return inspect_slice(index, args.sop_id), 0
    if args.command == "inspect-manifest":
        return inspect_manifest(index), 0
    if args.command == "sample":
        if args.count < 0:
            raise ValueError("--count must be nonnegative")
        return sample_entities(index, args.entity, args.count, args.seed), 0
    raise ValueError(f"unsupported command: {args.command}")


def _leakage_check(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="kaggle-rsna leakage-check")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--fold-plan")
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", choices=[policy.value for policy in LeakagePolicy],
                        default=LeakagePolicy.STRICT.value)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    dataset = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.fold_plan).read_text(encoding="utf-8")) if args.fold_plan else None
    report = validate_leakage(dataset, plan, policy=args.policy,
                              dataset_version=dataset.get("dataset_version"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"Dataset: {report.dataset_version_id or 'unbound'}")
        print(f"FoldPlan: {report.fold_plan_id or 'explicit splits'}")
        for label, key in (("Patient leakage", "PATIENT_CROSS_FOLD"), ("Study leakage", "STUDY_CROSS_FOLD"),
                           ("Series leakage", "SERIES_CROSS_FOLD"), ("Slice leakage", "SLICE_CROSS_FOLD")):
            print(f"{label}: {report.counts['issues_by_type'].get(key, 0)}")
        print(f"Duplicate hashes across folds: {sum(i.type.value == 'DUPLICATE_FILE_HASH' and len(i.folds_involved) > 1 for i in report.issues)}")
        print("RESULT: PASS" if report.passed else f"RESULT: FAIL\nCritical issues: {report.n_errors}")
    return 0 if report.passed or args.policy != LeakagePolicy.STRICT.value else 1


def _load_callable(spec: str):
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError("factory must use module:callable syntax")
    module_name, attribute = spec.split(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"configured factory {spec!r} is not callable")
    return value


def _run_training_job(path: Path) -> dict[str, Any]:
    """Run a JSON-declared job through user-supplied importable factories."""
    from .contracts import ModelCandidate, TrainingJob
    from .training.engine import TrainingEngine
    from .training.telemetry import JsonlTelemetrySink

    document = json.loads(path.read_text(encoding="utf-8"))
    job = TrainingJob(**document["job"])
    candidate = ModelCandidate(**document["candidate"])
    context = _load_callable(document["dataset_factory"])(job, document.get("dataset_configuration", {}))
    required = {"dataset_version", "dataset_index", "fold_plan", "fold_loader"}
    if not isinstance(context, dict) or not required.issubset(context):
        raise TypeError(f"dataset_factory must return a mapping with {sorted(required)}")
    artifact_dir = Path(document.get("artifact_dir", "artifacts/training"))
    event_sink = JsonlTelemetrySink(document["telemetry_path"]) if document.get("telemetry_path") else None
    candidate_factory = _load_callable(document["model_factory"])
    def model_factory(training_job):
        if training_job.model_candidate_id != candidate.model_candidate_id:
            raise ValueError("model factory candidate does not match TrainingJob")
        return candidate_factory(training_job, candidate)
    engine = TrainingEngine(dataset_version=context["dataset_version"],
        dataset_index=context["dataset_index"], fold_plan=context["fold_plan"],
        fold_loader=context["fold_loader"], model_factory=model_factory,
        artifact_dir=artifact_dir, event_sink=event_sink)
    result = engine.run(job, resume_from=document.get("resume_from"),
                        stop_after_optimizer_steps=document.get("stop_after_optimizer_steps"))
    return result.to_dict()


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "folds":
        from .folds.cli import main as folds_main
        return folds_main(args_list[1:])
    if args_list and args_list[0] == "select-slices":
        return _select_slices_cli(args_list[1:])
    if args_list and args_list[0] == "leakage-check":
        return _leakage_check(args_list[1:])
    # Preserve the Issue 1 flat spelling as an alias, not as a second parser hierarchy.
    if args_list and args_list[0] == "data-index":
        args_list = ["data", "index", *args_list[1:]]
    # Preserve: python -m rsna configs/experiments/smoke.toml
    if args_list and not args_list[0].startswith("-") and args_list[0] not in {"data", "artifact", "synthetic", "train"}:
        try:
            print(json.dumps(run_config(args_list[0]), indent=2, sort_keys=True))
            return 0
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3 if isinstance(exc, OSError) else 1
    parser = _parser()
    try:
        args = parser.parse_args(args_list)
        config = _config_defaults(args.config)
        output_format = getattr(args, "format", None) or config.get("output_format", "human")
        if args.quiet:
            output_format = "json" if getattr(args, "format", None) == "json" else "human"
        if args.group == "data" and hasattr(args, "manifest") and args.command != "index":
            args.manifest = args.manifest or config.get("manifest")
            if not args.manifest:
                parser.error("--manifest is required (or set cli.manifest in --config)")
        result, code = _dispatch(args, config)
        _emit(result, output_format)
        return code
    except SystemExit as exc:
        return int(exc.code or 0)
    except (OSError, json.JSONDecodeError) as exc:
        result = {"passed": False, "errors": [{"code": "UNREADABLE_ARTIFACT", "message": str(exc)}],
                  "warnings": [], "exit_code": 3}
        if "--json" in args_list or "--format" in args_list and "json" in args_list:
            _emit(result, "json")
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        if "--debug" in args_list:
            raise
        return 3
    except (ValueError, KeyError, LookupError) as exc:
        if "--debug" in args_list:
            raise
        from .inspection.query import DatasetReadError
        code = 3 if isinstance(exc, DatasetReadError) else 1
        error = {"passed": False, "errors": [{"code": "UNREADABLE_ARTIFACT" if code == 3 else "INVALID_INPUT",
                                                  "message": str(exc)}], "warnings": [], "exit_code": code}
        if "--json" in args_list or "--format" in args_list and "json" in args_list:
            _emit(error, "json")
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return code


def _select_slices_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="kaggle-rsna select-slices")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--strategy", choices=("uniform", "center", "physical_span"), default="uniform")
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--short-series-policy", choices=("keep_all", "repeat_nearest", "pad_reference", "strict"), default="keep_all")
    parser.add_argument("--duplicate-position-tolerance-mm", "--physical-position-tolerance-mm",
                        dest="duplicate_position_tolerance_mm", type=float, default=1e-3)
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args(argv)
    data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    config = SliceSelectionConfig(strategy=args.strategy, count=args.count,
        short_series_policy=args.short_series_policy, allow_fallback=not args.no_fallback,
        geometry_config=GeometryConfig(duplicate_position_tolerance_mm=args.duplicate_position_tolerance_mm))
    selector = SliceSelector(config)
    summaries = []
    for study in data.get("studies", []):
        for series in study.get("series", []):
            result = selector.select(series, study=study)
            summaries.append(result.to_dict())
    print(json.dumps({"series_count": len(summaries), "selections": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
