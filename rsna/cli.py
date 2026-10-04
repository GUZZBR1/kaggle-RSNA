"""Command line entry point for smoke experiments and dataset operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib
from typing import Any

from .contracts import DatasetVersion, ExperimentSpec, FoldPlan, ModelCandidate
from .experiments.plan import plan_jobs
from .experiments.runner import run_jobs
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
    data = commands.add_parser("data", help="inspect and validate dataset metadata")
    data_commands = data.add_subparsers(dest="command", required=True)
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
        command.add_argument("--manifest", help="path to a dataset manifest JSON")
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
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("cli", {})


def _render_human(value: Any, *, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            label = key.replace("_", " ").capitalize()
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{label}:")
                lines.append(_render_human(item, indent=indent + 2))
            elif item is not None:
                lines.append(f"{prefix}{label}: {item}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{prefix}(none)"
        return "\n".join(f"{prefix}- {item}" if not isinstance(item, dict) else
                         f"{prefix}- " + _render_human(item, indent=indent + 2).lstrip()
                         for item in value)
    return f"{prefix}{value}"


def _emit(value: Any, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(_render_human(value))


def _dispatch(args: argparse.Namespace, config: dict[str, Any] | None = None) -> tuple[Any, int]:
    config = config or {}
    from .inspection.summary import build_dataset_summary, build_dataset_stats
    from .inspection.query import (inspect_manifest, inspect_series, inspect_slice,
                                   inspect_study, load_dataset_index, sample_entities)
    from .inspection.validate import validate_dataset, validate_artifact

    if args.group == "artifact":
        level = args.level or config.get("validation_level", "full")
        artifact_path = Path(args.path)
        artifact_directory = config.get("artifact_directory")
        if artifact_directory and not artifact_path.is_absolute():
            artifact_path = Path(artifact_directory) / artifact_path
        report = validate_artifact(artifact_path, level=level,
                                   warnings_as_errors=args.warnings_as_errors)
        return report.to_dict(), report.exit_code
    manifest = args.manifest
    if args.command == "validate":
        level = args.level or config.get("validation_level", "basic")
        report = validate_dataset(manifest, level=level,
                                  dataset_root=args.dataset_root or config.get("dataset_root"),
                                  warnings_as_errors=args.warnings_as_errors)
        return report.to_dict(), report.exit_code
    if args.command == "summary":
        index = load_dataset_index(manifest)
        return build_dataset_summary(index, plane=args.plane, laterality=args.laterality,
                                     warning_code=args.warning_code), 0
    if args.command == "stats":
        index = load_dataset_index(manifest)
        return build_dataset_stats(index, plane=args.plane, laterality=args.laterality,
                                   warning_code=args.warning_code), 0
    if args.command == "inspect-study":
        return inspect_study(load_dataset_index(manifest), args.study_id), 0
    if args.command == "inspect-series":
        return inspect_series(load_dataset_index(manifest), args.series_id), 0
    if args.command == "inspect-slice":
        return inspect_slice(load_dataset_index(manifest), args.sop_id), 0
    if args.command == "inspect-manifest":
        return inspect_manifest(load_dataset_index(manifest)), 0
    if args.command == "sample":
        if args.count < 0:
            raise ValueError("--count must be nonnegative")
        return sample_entities(load_dataset_index(manifest), args.entity, args.count, args.seed), 0
    raise ValueError(f"unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    # Preserve the original positional smoke command during the CLI transition.
    if args_list and not args_list[0].startswith("-") and args_list[0] not in {"data", "artifact"}:
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
        if args.group == "data" and hasattr(args, "manifest"):
            args.manifest = getattr(args, "manifest", None) or config.get("manifest")
            if not args.manifest:
                parser.error("--manifest is required (or set cli.manifest in --config)")
        result, code = _dispatch(args, config)
        _emit(result, output_format)
        return code
    except SystemExit as exc:
        return int(exc.code or 0)
    except (OSError, json.JSONDecodeError) as exc:
        message = {"passed": False, "errors": [{"code": "UNREADABLE_ARTIFACT", "message": str(exc)}],
                   "warnings": [], "exit_code": 3}
        if "--json" in args_list or "--format" in args_list and "json" in args_list:
            _emit(message, "json")
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        if "--debug" in args_list:
            raise
        return 3
    except (ValueError, KeyError, LookupError) as exc:
        if "--debug" in args_list:
            raise
        from .inspection.query import DatasetReadError
        if isinstance(exc, DatasetReadError):
            code = 3
            error_code = "UNREADABLE_ARTIFACT"
        else:
            code = 1
            error_code = "INVALID_INPUT"
        if "--json" in args_list or "--format" in args_list and "json" in args_list:
            _emit({"passed": False, "errors": [{"code": error_code, "message": str(exc)}],
                   "warnings": [], "exit_code": code}, "json")
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
