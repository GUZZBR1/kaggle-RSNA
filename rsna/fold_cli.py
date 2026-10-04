"""Assignment management commands built on canonical fold and leakage contracts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .fold_plan import (FoldValidationReport, UsageError, _load_dataset, _manifest,
                       diff_plans, make_plan, read_plan, save_plan, validate)
from .folds import FoldPlanManifest, generate_fold_plan
from .folds.generate import _groups, _records
from .folds.statistics import summarize


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rsna folds")
    parser.add_argument("--json", action="store_true", dest="json_output")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--dataset-manifest", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--n-folds", type=int, default=5)
    generate.add_argument("--strategy", choices=("group", "multilabel_group_stratified"), default="group")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--grouping-key", choices=("patient_id",), default="patient_id")
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--force", action="store_true")
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--fold-plan", required=True)
    validate_parser.add_argument("--dataset-manifest", required=True)
    validate_parser.add_argument("--allow-incomplete", action="store_true")
    validate_parser.add_argument("--reproduce", action="store_true")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--fold-plan", required=True)
    inspect.add_argument("--fold")
    inspect.add_argument("--study-id")
    inspect.add_argument("--patient-id")
    inspect.add_argument("--limit", type=int, default=20)
    inspect.add_argument("--dataset-manifest")
    stats = sub.add_parser("stats")
    stats.add_argument("--fold-plan", required=True)
    stats.add_argument("--dataset-manifest", required=True)
    diff = sub.add_parser("diff")
    diff.add_argument("left")
    diff.add_argument("right")
    diff.add_argument("--show-moved", action="store_true")
    export = sub.add_parser("export")
    export.add_argument("--fold-plan", required=True)
    export.add_argument("--format", choices=("csv", "json"), required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--dataset-manifest")
    lock = sub.add_parser("lock")
    lock.add_argument("--fold-plan", required=True)
    import_parser = sub.add_parser("import")
    import_parser.add_argument("--assignments", required=True)
    import_parser.add_argument("--dataset-manifest", required=True)
    import_parser.add_argument("--output", required=True)
    import_parser.add_argument("--force", action="store_true")
    return parser


def _emit(value: Any, json_mode: bool, status: int) -> int:
    print(json.dumps(value, sort_keys=True, allow_nan=False,
                     indent=None if json_mode else 2))
    return status


def _validation_exit(report: FoldValidationReport) -> int:
    if report["passed"]:
        return 0
    return 4 if any("dataset_version_id" in error for error in report["errors"]) else 1


def _legacy_generate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m rsna folds")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--strategy", choices=("group", "multilabel_group_stratified"), default="group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-version-id")
    parser.add_argument("--locked", action="store_true")
    parser.add_argument("--labels")
    args = parser.parse_args(argv)
    dataset_id, studies, index = _load_dataset(args.dataset_manifest)
    if args.dataset_version_id:
        dataset_id = args.dataset_version_id
    if args.labels:
        labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
        if not isinstance(labels, dict):
            raise ValueError("labels file must contain a study-to-target mapping")
        for study in studies:
            study["labels"] = labels.get(study["study_id"], study.get("labels", {}))
    plan = make_plan(dataset_id, studies, args.n_folds, args.strategy, args.seed,
                     "patient_id", locked=args.locked)
    report = validate(plan, dataset_id, studies, dataset_index=index)
    if not report["passed"]:
        print("Leakage: FAIL")
        print(json.dumps(report, sort_keys=True))
        return _validation_exit(report)
    save_plan(plan, args.output)
    print("Leakage: PASS")
    print(json.dumps({"fold_plan_id": plan["fold_plan_id"], "output": args.output,
                      "n_studies": len(studies)}, sort_keys=True))
    return 0


def _existing_output(path: Path, force: bool) -> dict[str, Any] | None:
    if not path.exists():
        return None
    previous = read_plan(path)
    if previous["locked"]:
        raise FileExistsError(f"fold plan is locked: {path}")
    if not force:
        raise FileExistsError(f"output exists: {path} (use --force)")
    return previous


def _label_map(studies: list[dict[str, Any]]) -> dict[str, dict[str, Any]] | None:
    labels = {row["study_id"]: row.get("labels", {}) for row in studies if row.get("labels")}
    return labels or None


def _import_manifest(dataset_id: str, studies: list[dict[str, Any]],
                     assignments: dict[str, str]) -> FoldPlanManifest:
    records = _records({"studies": studies})
    study_groups, group_members, grouping_key = _groups(records)
    group_assignments: dict[str, str] = {}
    for group, members in group_members.items():
        assigned_folds = {assignments[study] for study in members if study in assignments}
        if len(assigned_folds) != 1:
            raise ValueError(f"assignments split indivisible patient/series group: {group}")
        group_assignments[group] = next(iter(assigned_folds))
    labels = _label_map(studies)
    fold_ids = {f"fold_{i}" for i in range(len(set(assignments.values())))}
    if set(group_assignments.values()) != fold_ids:
        raise ValueError("each declared fold must contain at least one indivisible group")
    statistics, warnings = summarize(records, assignments, len(fold_ids), study_groups, labels)
    generated = generate_fold_plan({"dataset_version_id": dataset_id, "studies": studies},
                                   n_folds=len(fold_ids), strategy="group", random_state=0,
                                   labels=labels)
    configuration = dict(generated.configuration)
    configuration.update({"algorithm": "external_assignment_import_v1", "source": "csv"})
    return replace(generated, assignments=assignments, group_assignments=group_assignments,
                   configuration=configuration, statistics=statistics,
                   warnings=tuple(warnings), fold_plan_id="")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    json_mode = "--json" in argv
    argv = [arg for arg in argv if arg != "--json"]
    try:
        if argv and argv[0].startswith("-"):
            return _legacy_generate(argv)
        args = _parser().parse_args(argv)
        if args.command == "generate":
            dataset_id, studies, index = _load_dataset(args.dataset_manifest)
            plan = make_plan(dataset_id, studies, args.n_folds, args.strategy, args.seed, args.grouping_key)
            report = validate(plan, dataset_id, studies, dataset_index=index)
            if not report["passed"]:
                return _emit(report, json_mode, 1)
            if not args.dry_run:
                target = Path(args.output)
                previous = _existing_output(target, args.force)
                save_plan(plan, target)
                report["output"] = str(target)
                report["overwrote"] = previous is not None
            report["dry_run"] = args.dry_run
            return _emit(report, json_mode, 0)
        if args.command in {"validate", "stats"}:
            plan = read_plan(args.fold_plan)
            dataset_id, studies, index = _load_dataset(args.dataset_manifest)
            report = validate(plan, dataset_id, studies,
                              require_coverage=not getattr(args, "allow_incomplete", False),
                              dataset_index=index)
            if getattr(args, "reproduce", False):
                if plan.get("configuration", {}).get("source") == "csv":
                    report["warnings"].append("reproduction unavailable for externally imported assignments")
                    report["reproduction"] = {"supported": False, "passed": None}
                else:
                    regenerated = make_plan(dataset_id, studies, plan["n_folds"],
                                            plan["strategy"], plan["random_state"],
                                            "patient_id", locked=plan["locked"])
                    report["reproduction"] = {"supported": True,
                                              "passed": regenerated["fold_plan_id"] == plan["fold_plan_id"],
                                              "regenerated_fold_plan_id": regenerated["fold_plan_id"]}
                    if not report["reproduction"]["passed"]:
                        report["errors"].append("reproduced assignments do not match fold plan")
                        report["passed"] = False
            return _emit(report, json_mode, _validation_exit(report))
        if args.command == "inspect":
            if args.limit < 0:
                raise UsageError("--limit must be nonnegative")
            plan = read_plan(args.fold_plan)
            result: dict[str, Any] = {key: plan.get(key) for key in
                                      ("fold_plan_id", "dataset_version_id", "strategy", "n_folds",
                                       "random_state", "grouping_key", "locked", "fold_ids", "warnings")}
            result["assignments_count"] = len(plan["assignments"])
            result["validation_status"] = "not_run"
            patient_by_study = {}
            if args.dataset_manifest:
                dataset_id, studies, index = _load_dataset(args.dataset_manifest)
                result["validation"] = validate(plan, dataset_id, studies, dataset_index=index)
                result["validation_status"] = "passed" if result["validation"]["passed"] else "failed"
                patient_by_study = {study["study_id"]: study["patient_id"] for study in studies}
            if args.study_id:
                result["study"] = {"study_id": args.study_id,
                                   "fold": plan["assignments"].get(args.study_id),
                                   "patient_id": patient_by_study.get(args.study_id),
                                   "group_id": plan["study_groups"].get(args.study_id)}
            if args.patient_id:
                if not args.dataset_manifest:
                    raise UsageError("--patient-id requires --dataset-manifest")
                folds = sorted({plan["assignments"][study_id] for study_id, patient in patient_by_study.items()
                                if patient == args.patient_id and study_id in plan["assignments"]})
                result["patient"] = {"patient_id": args.patient_id, "folds": folds}
            if args.fold:
                if args.fold not in plan["fold_ids"]:
                    raise ValueError(f"unknown fold: {args.fold}")
                result["fold"] = args.fold
                result["studies"] = [study for study, fold in plan["assignments"].items()
                                     if fold == args.fold][:args.limit]
                result["statistics"] = plan["statistics"].get("folds", {}).get(args.fold)
            return _emit(result, json_mode,
                         _validation_exit(result["validation"]) if "validation" in result else 0)
        if args.command == "diff":
            result = diff_plans(read_plan(args.left), read_plan(args.right))
            if not args.show_moved:
                result.pop("moved")
            return _emit(result, json_mode, 0)
        if args.command == "export":
            plan = read_plan(args.fold_plan)
            patient_by_study = {}
            if args.dataset_manifest:
                dataset_id, studies, index = _load_dataset(args.dataset_manifest)
                report = validate(plan, dataset_id, studies, dataset_index=index)
                if not report["passed"]:
                    return _emit(report, json_mode, _validation_exit(report))
                patient_by_study = {study["study_id"]: study["patient_id"] for study in studies}
            rows = [{"study_id": study, "patient_id": patient_by_study.get(study), "fold": fold}
                    for study, fold in sorted(plan["assignments"].items())]
            target = Path(args.output)
            if target.resolve() == Path(args.fold_plan).resolve():
                raise FileExistsError("export output cannot replace its source fold plan")
            if target.exists():
                try:
                    if read_plan(target)["locked"]:
                        raise FileExistsError(f"fold plan is locked: {target}")
                except ValueError:
                    pass
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.format == "json":
                target.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            else:
                with target.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["study_id", "patient_id", "fold"])
                    writer.writeheader()
                    writer.writerows(rows)
            return _emit({"output": str(target), "assignments": len(rows), "format": args.format}, json_mode, 0)
        if args.command == "import":
            dataset_id, studies, index = _load_dataset(args.dataset_manifest)
            patient_ids = {study["study_id"]: study["patient_id"] for study in studies}
            assignments: dict[str, str] = {}
            with Path(args.assignments).open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if not {"study_id", "fold"}.issubset(reader.fieldnames or []):
                    raise ValueError("assignments CSV requires study_id and fold columns")
                for row in reader:
                    if row is None or not row.get("study_id") or not row.get("fold"):
                        raise ValueError("study_id and fold cannot be empty")
                    study_id, fold = row["study_id"], row["fold"]
                    if study_id in assignments:
                        raise ValueError(f"duplicate assignment: {study_id}")
                    if row.get("patient_id") and row["patient_id"] != patient_ids.get(study_id):
                        raise ValueError(f"patient_id mismatch for study: {study_id}")
                    assignments[study_id] = fold
            folds = sorted(set(assignments.values()), key=lambda fold: int(fold[5:])
                           if fold.startswith("fold_") and fold[5:].isdigit() else -1)
            if len(folds) < 2 or folds != [f"fold_{i}" for i in range(len(folds))]:
                raise ValueError("assignments must use contiguous fold_0 through fold_n IDs with at least two folds")
            plan_manifest = _import_manifest(dataset_id, studies, assignments)
            plan = {**plan_manifest.to_dict(), "fold_ids": folds}
            report = validate(plan, dataset_id, studies, dataset_index=index)
            if not report["passed"]:
                return _emit(report, json_mode, _validation_exit(report))
            target = Path(args.output)
            previous = _existing_output(target, args.force)
            save_plan(plan, target)
            report["output"] = str(target)
            report["overwrote"] = previous is not None
            return _emit(report, json_mode, 0)
        if args.command == "lock":
            path = Path(args.fold_plan)
            plan = read_plan(path)
            if plan["locked"]:
                return _emit({"locked": True, "fold_plan_id": plan["fold_plan_id"]}, json_mode, 0)
            save_plan(replace(_manifest(plan), locked=True, fold_plan_id="").to_dict(), path)
            locked = read_plan(path)
            return _emit({"locked": True, "fold_plan_id": locked["fold_plan_id"]}, json_mode, 0)
    except SystemExit as exc:
        return int(exc.code)
    except UsageError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (FileExistsError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 0
