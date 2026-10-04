"""Operational fold-plan creation, validation, comparison, and export."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import pstdev as population_stdev
from typing import Any, TypedDict

from ..contracts import DatasetVersion
from ..identity import digest, jsonable
from ..data.models import index_from_dict
from ..leakage import validate_leakage
from ..targets import TARGETS
from .generate import create_fold_plan_from_assignments, generate_fold_plan
from .manifest import load_fold_plan, lock_fold_plan, save_fold_plan
from .models import FoldPlanManifest


class UsageError(ValueError):
    """A valid command shape with invalid command values or constraints."""


class AssignmentValidationError(ValueError):
    """An external assignment artifact fails a dataset or leakage gate."""


class FoldValidationReport(TypedDict):
    passed: bool
    fold_plan_id: str
    dataset_version_id: str
    errors: list[str]
    warnings: list[str]
    fold_statistics: dict[str, Any]
    leakage: dict[str, Any]
    balance: dict[str, Any]
    coverage: dict[str, Any]


def load_dataset(path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    """Load existing index records without DICOM access or source traversal."""
    raw_bytes = Path(path).read_bytes()
    data = json.loads(raw_bytes)
    if not isinstance(data, dict):
        raise ValueError("dataset manifest must be a JSON object")
    index = data.get("dataset_index", data.get("index", data))
    if not isinstance(index, dict) or not isinstance(index.get("studies"), list):
        raise ValueError("dataset manifest must be an object containing studies")
    version = data.get("dataset_version")
    if isinstance(version, dict):
        version = (DatasetVersion.from_dict(version) if "target_registry_id" in version
                   else DatasetVersion(**version))
        dataset_id = version.dataset_version_id
    elif data.get("dataset_version_id") or index.get("dataset_version_id"):
        dataset_id = data.get("dataset_version_id") or index["dataset_version_id"]
        if not isinstance(dataset_id, str) or len(dataset_id) != 64 or any(c not in "0123456789abcdef" for c in dataset_id):
            raise ValueError("dataset_version_id must be a SHA-256 digest")
    else:
        if "index_id" in index:
            index_from_dict(index)  # Verify the canonical DatasetIndex identity.
            dataset_id = digest({"dataset_index_id": index["index_id"]})
        else:
            identity_records = sorted(({
                "study_id": item.get("study_instance_uid") or item.get("study_id"),
                "patient_id": item.get("patient_id"),
                "series": sorted((series.get("series_instance_uid") or series.get("series_id") or "")
                                 for series in item.get("series", []) if isinstance(series, dict)),
            } for item in index["studies"] if isinstance(item, dict)), key=lambda item: str(item["study_id"]))
            dataset_id = digest({"studies": identity_records})
    studies = []
    seen_ids: set[str] = set()
    if "index_id" in index:
        index_from_dict(index)
    for item in index["studies"]:
        if not isinstance(item, dict):
            raise ValueError("study records must be objects")
        assignment_id = item.get("study_instance_uid") or item.get("study_id")
        patient_id = item.get("patient_id")
        if not isinstance(assignment_id, str) or not assignment_id:
            raise ValueError("every study needs study_instance_uid or study_id")
        if assignment_id in seen_ids:
            raise ValueError(f"duplicate study ID in dataset manifest: {assignment_id}")
        seen_ids.add(assignment_id)
        series = item.get("series", [])
        if not isinstance(series, list):
            series = []
        normalized_series = []
        for index_in_study, raw_series in enumerate(series):
            if not isinstance(raw_series, dict):
                continue
            current = dict(raw_series)
            current.setdefault("series_id", f"series-fallback:{assignment_id}:{index_in_study}")
            series_study_uid = current.get("study_instance_uid")
            if series_study_uid is None:
                current["study_instance_uid"] = assignment_id
            elif series_study_uid != assignment_id:
                series_identity = current.get("series_instance_uid") or current["series_id"]
                raise ValueError(
                    f"series {series_identity!r} StudyInstanceUID {series_study_uid!r} "
                    f"conflicts with parent StudyInstanceUID {assignment_id!r}")
            current_slices = current.get("slices", [])
            current["slices"] = [dict(row) for row in current_slices if isinstance(row, dict)] if isinstance(current_slices, list) else []
            normalized_series.append(current)
        n_series = len(normalized_series) or int(item.get("n_series", 0))
        n_slices = sum(len(s.get("slices", [])) for s in normalized_series) or int(item.get("n_slices", 0))
        series_ids = [s.get("series_instance_uid") or s["series_id"] for s in normalized_series]
        labels = item.get("labels", {})
        studies.append({"study_id": assignment_id, "study_instance_uid": assignment_id,
                        "patient_id": patient_id, "warnings": list(item.get("warnings", [])),
                        "series": normalized_series,
                        "n_series": n_series, "n_slices": n_slices,
                        "series_ids": series_ids,
                        "labels": labels if isinstance(labels, dict) else {}})
    return dataset_id, studies


def read_plan(path: str | Path) -> dict[str, Any]:
    """Return the canonical FoldPlan JSON view after native identity validation."""
    return load_fold_plan(path).to_dict()


def statistics(studies: list[dict[str, Any]], plan: FoldPlanManifest) -> dict[str, Any]:
    """Add index-only slice counts to the canonical generator's statistics."""
    result = jsonable(plan.statistics.get("folds", {}))
    for fold, values in result.items():
        chosen = [study for study in studies if plan.assignments.get(study["study_id"]) == fold]
        values["n_series"] = sum(study["n_series"] for study in chosen)
        values["n_slices"] = sum(study["n_slices"] for study in chosen)
        for target in values.get("targets", {}).values():
            target["supervision_count"] = target["positive"] + target["negative"] + target.get("soft_count", 0)
    return result


def validate(plan: dict[str, Any] | FoldPlanManifest, dataset_id: str | None = None,
             studies: list[dict[str, Any]] | None = None, *, require_coverage: bool = True) -> FoldValidationReport:
    manifest = plan if isinstance(plan, FoldPlanManifest) else FoldPlanManifest(**plan)
    plan_data = manifest.to_dict()
    errors: list[str] = []
    if dataset_id and manifest.dataset_version_id != dataset_id:
        errors.append("dataset_version_id does not match dataset manifest")
    assigned = manifest.assignments
    known = {study["study_id"]: study for study in (studies or [])}
    missing = sorted(set(known) - set(assigned)) if studies is not None else []
    unknown = sorted(set(assigned) - set(known)) if studies is not None else []
    if unknown:
        errors.append("assignments contain unknown studies")
    if missing and require_coverage:
        errors.append("studies are missing assignments")
    guard_report = None
    if studies is not None:
        guard_report = validate_leakage(_leakage_records(studies, assigned),
                                        dataset_version_id=dataset_id)
        if guard_report.counts["n_errors"]:
            errors.append("Leakage Guard rejected the fold assignments")
    stats = statistics(studies or [], manifest)
    sizes = [value["n_studies"] for value in stats.values()]
    nonzero = [count for count in sizes if count]
    balance = {"largest_fold": max(sizes, default=0), "smallest_fold": min(sizes, default=0),
               "size_ratio": max(nonzero) / min(nonzero) if nonzero else None,
               "study_distribution": sizes,
               "patient_distribution": [value["n_patients"] for value in stats.values()]}
    target_balance = {}
    for target in TARGETS:
        prevalence = [value["targets"][target]["prevalence"] for value in stats.values()
                      if value["targets"][target]["prevalence"] is not None]
        target_balance[target] = {
            "prevalence_range": [min(prevalence), max(prevalence)] if prevalence else None,
            "prevalence_std": population_stdev(prevalence) if len(prevalence) > 1 else 0.0 if prevalence else None,
        }
    balance["target_prevalence"] = target_balance
    warnings = list(manifest.warnings)
    if guard_report:
        warnings.extend(issue.message for issue in guard_report.issues if issue.severity == "warning")
    if not any(study.get("labels") for study in studies or []):
        warnings.append("no labels are present; target prevalence balance is unavailable")
    expected = len(known) if studies is not None else len(assigned)
    covered = len(set(assigned) & set(known)) if studies is not None else len(assigned)
    coverage = {"expected_studies": expected, "assigned_studies": len(assigned),
                "unassigned_studies": missing, "unknown_assigned_studies": unknown,
                "coverage_fraction": covered / expected if expected else 1.0}
    return {"passed": not errors, "fold_plan_id": manifest.fold_plan_id,
            "dataset_version_id": manifest.dataset_version_id, "errors": errors,
            "warnings": warnings, "fold_statistics": stats,
            "leakage": guard_report.to_dict() if guard_report else {"passed": None},
            "balance": balance, "coverage": coverage}


def _leakage_records(studies: list[dict[str, Any]], assignments: dict[str, str] | Any = None) -> list[dict[str, Any]]:
    """Pass existing manifest hierarchy to Leakage Guard without scanning source files."""
    records = []
    for study in studies:
        fold = assignments.get(study["study_id"]) if assignments is not None else None
        normalized = {key: study.get(key) for key in
                      ("study_id", "study_instance_uid", "patient_id", "warnings") if study.get(key) is not None}
        normalized["study_instance_uid"] = study["study_id"]
        normalized["study_id"] = study["study_id"]
        normalized["fold_id"] = fold
        series_rows = []
        for series in study.get("series", []):
            item = dict(series)
            item["fold_id"] = fold
            item["slices"] = [{**row, "fold_id": fold} for row in item.get("slices", [])
                              if isinstance(row, dict) and
                              (row.get("slice_id") or row.get("SOPInstanceUID") or
                               row.get("sop_instance_uid") or row.get("relative_path"))]
            series_rows.append(item)
        normalized["series"] = series_rows
        records.append(normalized)
    return records


def make_plan(dataset_id: str, studies: list[dict[str, Any]], n_folds: int, strategy: str,
              seed: int, grouping_key: str) -> dict[str, Any]:
    if grouping_key != "patient_id":
        raise UsageError("grouping-key must be patient_id; studies without PatientID use Study fallback")
    labels = {study["study_id"]: study["labels"] for study in studies if study.get("labels")}
    dataset = {"dataset_version_id": dataset_id, "studies": studies}
    return generate_fold_plan(dataset, n_folds=n_folds, strategy=strategy,
                              random_state=seed, labels=labels or None,
                              dataset_version_id=dataset_id).to_dict()


def diff_plans(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    a, b = left["assignments"], right["assignments"]
    common = set(a) & set(b)
    moved = {key: {"from": a[key], "to": b[key]} for key in sorted(common) if a[key] != b[key]}
    added, removed = sorted(set(b) - set(a)), sorted(set(a) - set(b))
    partition = lambda assignments: frozenset(frozenset(k for k, v in assignments.items() if v == fold)
                                                for fold in set(assignments.values()))
    exact = a == b
    semantic = set(a) == set(b) and partition(a) == partition(b)
    patient_maps = [plan.get("configuration", {}).get("patient_by_study", {}) for plan in (left, right)]
    common_patients: dict[str, list[str]] = defaultdict(list)
    for study_id in common:
        patient = patient_maps[0].get(study_id) or patient_maps[1].get(study_id)
        if patient:
            common_patients[patient].append(study_id)
    moved_patients = {}
    for patient, ids in common_patients.items():
        before = {a[i] for i in ids}
        after = {b[i] for i in ids}
        if len(before) == len(after) == 1 and before != after:
            moved_patients[patient] = {"from": next(iter(before)), "to": next(iter(after))}
    patients_moved = len(moved_patients)
    same_dataset = left["dataset_version_id"] == right["dataset_version_id"]
    same_identity = left.get("fold_plan_id") == right.get("fold_plan_id")
    return {"same_dataset_version": same_dataset,
            "same_fold_plan_identity": same_identity,
            "same_strategy": left["strategy"] == right["strategy"],
            "same_seed": left["random_state"] == right["random_state"],
            "same_number_of_folds": left["n_folds"] == right["n_folds"],
            "exact_equality": exact, "semantic_partition_equality": semantic,
            "comparison": ("identical" if same_identity else
                           "equivalent_partition" if semantic else "different_assignments"),
            "studies_moved": len(moved), "patients_moved": patients_moved,
            "added_studies": added, "removed_studies": removed,
            "moved_patients": moved_patients, "moved": moved}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m rsna folds")
    p.add_argument("--json", action="store_true", dest="json_output")
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate"); g.add_argument("--dataset-manifest", required=True); g.add_argument("--output", required=True); g.add_argument("--n-folds", type=int, default=5); g.add_argument("--strategy", choices=("group", "multilabel-group-stratified"), default="group"); g.add_argument("--seed", type=int, default=42); g.add_argument("--grouping-key", choices=("patient_id",), default="patient_id"); g.add_argument("--dry-run", action="store_true"); g.add_argument("--force", action="store_true")
    v = sub.add_parser("validate"); v.add_argument("--fold-plan", required=True); v.add_argument("--dataset-manifest", required=True); v.add_argument("--allow-incomplete", action="store_true"); v.add_argument("--reproduce", action="store_true")
    i = sub.add_parser("inspect"); i.add_argument("--fold-plan", required=True); i.add_argument("--fold"); i.add_argument("--study-id"); i.add_argument("--patient-id"); i.add_argument("--group-id"); i.add_argument("--limit", type=int, default=20); i.add_argument("--dataset-manifest")
    s = sub.add_parser("stats"); s.add_argument("--fold-plan", required=True); s.add_argument("--dataset-manifest", required=True)
    d = sub.add_parser("diff"); d.add_argument("left"); d.add_argument("right"); d.add_argument("--show-moved", action="store_true")
    e = sub.add_parser("export"); e.add_argument("--fold-plan", required=True); e.add_argument("--format", choices=("csv", "json"), required=True); e.add_argument("--output", required=True); e.add_argument("--dataset-manifest")
    l = sub.add_parser("lock"); l.add_argument("--fold-plan", required=True)
    m = sub.add_parser("import"); m.add_argument("--assignments", required=True); m.add_argument("--dataset-manifest", required=True); m.add_argument("--output", required=True); m.add_argument("--dataset-version-id"); m.add_argument("--force", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Keep the Issue 6 one-shot invocation as a shorthand for `folds generate`.
    if argv and argv[0].startswith("-"):
        argv.insert(0, "generate")
    json_mode = "--json" in argv
    argv = [arg for arg in argv if arg != "--json"]
    try:
        args = _parser().parse_args(argv)
        if args.command == "generate":
            dataset_id, studies = load_dataset(args.dataset_manifest)
            strategy = args.strategy.replace("-", "_")
            plan = FoldPlanManifest(**make_plan(dataset_id, studies, args.n_folds, strategy,
                                                args.seed, args.grouping_key))
            report = validate(plan, dataset_id, studies)
            if not report["passed"]:
                return _emit(report, json_mode, 1)
            if not args.dry_run:
                target = Path(args.output)
                existed = target.exists()
                if existed and load_fold_plan(target).locked:
                    raise FileExistsError(f"fold plan is locked: {target}")
                if existed and not args.force:
                    raise FileExistsError(f"output exists: {target} (use --force)")
                save_fold_plan(plan, target)
                report["output"] = str(target)
                report["overwrote"] = existed and args.force
            report["dry_run"] = args.dry_run
            return _emit(report, json_mode, 0)
        if args.command in {"validate", "stats"}:
            plan = load_fold_plan(args.fold_plan)
            dataset_id, studies = load_dataset(args.dataset_manifest)
            report = validate(plan, dataset_id, studies,
                              require_coverage=not getattr(args, "allow_incomplete", False))
            if getattr(args, "reproduce", False):
                if plan.strategy == "imported":
                    report["warnings"].append("reproduction unavailable for externally imported assignments")
                    report["reproduction"] = {"supported": False, "passed": None}
                else:
                    regenerated = FoldPlanManifest(**make_plan(dataset_id, studies, plan.n_folds,
                                                                 plan.strategy, plan.random_state,
                                                                 "patient_id"))
                    same = regenerated.fold_plan_id == plan.fold_plan_id
                    report["reproduction"] = {"supported": True, "passed": same,
                                              "regenerated_fold_plan_id": regenerated.fold_plan_id}
                    if not same:
                        report["errors"].append("reproduced assignments do not match FoldPlan identity")
                        report["passed"] = False
            return _emit(report, json_mode, _validation_exit(report))
        if args.command == "inspect":
            plan = load_fold_plan(args.fold_plan)
            plan_data = plan.to_dict()
            result = {key: plan_data.get(key) for key in
                      ("fold_plan_id", "dataset_version_id", "strategy", "n_folds", "random_state", "grouping_key", "locked")}
            result["assignments_count"] = len(plan.assignments)
            result["validation_status"] = "not_run"
            studies = None
            if args.dataset_manifest:
                dataset_id, studies = load_dataset(args.dataset_manifest)
                result["validation"] = validate(plan, dataset_id, studies)
                result["validation_status"] = "passed" if result["validation"]["passed"] else "failed"
            if args.study_id:
                row = next((study for study in studies or [] if study["study_id"] == args.study_id), {})
                result["study"] = {"study_id": args.study_id,
                                    "fold": plan.assignments.get(args.study_id),
                                    "group_id": plan.study_groups.get(args.study_id),
                                    "patient_id": row.get("patient_id")}
            if args.patient_id:
                patient_map = {study["study_id"]: study["patient_id"] for study in studies or []}
                folds = sorted({plan.assignments[study_id] for study_id, patient in patient_map.items()
                                if patient == args.patient_id})
                result["patient"] = {"patient_id": args.patient_id, "folds": folds}
            if args.group_id:
                members = [study for study, group in plan.study_groups.items() if group == args.group_id]
                result["group"] = {"group_id": args.group_id, "studies": members,
                                    "fold": plan.group_assignments.get(args.group_id)}
            if args.fold:
                fold_ids = {f"fold_{index}" for index in range(plan.n_folds)}
                if args.fold not in fold_ids:
                    raise ValueError(f"unknown fold: {args.fold}")
                member_ids = [study for study, fold in plan.assignments.items() if fold == args.fold]
                result.update(fold=args.fold, studies=member_ids[:args.limit],
                              groups=sorted({plan.study_groups[study] for study in member_ids}),
                              statistics=statistics(studies or [], plan).get(args.fold))
            return _emit(result, json_mode, _validation_exit(result["validation"]) if "validation" in result else 0)
        if args.command == "diff":
            result = diff_plans(read_plan(args.left), read_plan(args.right))
            if not args.show_moved:
                result.pop("moved")
            return _emit(result, json_mode, 0)
        if args.command == "export":
            plan = load_fold_plan(args.fold_plan)
            patient_by_study = {}
            if args.dataset_manifest:
                dataset_id, studies = load_dataset(args.dataset_manifest)
                report = validate(plan, dataset_id, studies)
                if not report["passed"]:
                    return _emit(report, json_mode, _validation_exit(report))
                patient_by_study = {study["study_id"]: study["patient_id"] for study in studies}
            rows = [{"study_id": study, "patient_id": patient_by_study.get(study), "fold": fold}
                    for study, fold in sorted(plan.assignments.items())]
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.format == "json":
                payload = {"schema_version": 1, "dataset_version_id": plan.dataset_version_id,
                           "source_fold_plan_id": plan.fold_plan_id, "identity_preserved": False,
                           "assignments": rows}
                target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            else:
                fields = ["study_id", "patient_id", "fold", "dataset_version_id", "source_fold_plan_id"]
                with target.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows({**row, "dataset_version_id": plan.dataset_version_id,
                                      "source_fold_plan_id": plan.fold_plan_id} for row in rows)
            return _emit({"output": str(target), "assignments": len(rows), "format": args.format,
                          "source_fold_plan_id": plan.fold_plan_id, "identity_preserved": False}, json_mode, 0)
        if args.command == "import":
            dataset_id, studies = load_dataset(args.dataset_manifest)
            if args.dataset_version_id and args.dataset_version_id != dataset_id:
                raise AssignmentValidationError("DatasetVersion mismatch for imported assignments")
            expected_ids = {study["study_id"] for study in studies}
            source_format = Path(args.assignments).suffix.lower().lstrip(".")
            source_fold_plan_id = None
            if source_format == "json":
                payload = json.loads(Path(args.assignments).read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    source_fold_plan_id = payload.get("source_fold_plan_id")
                    if payload.get("dataset_version_id") not in (None, dataset_id):
                        raise AssignmentValidationError("DatasetVersion mismatch in assignment file")
                    rows = payload.get("assignments")
                else:
                    rows = payload
            else:
                with Path(args.assignments).open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
            if not isinstance(rows, list):
                raise ValueError("assignment file must contain a list of assignments")
            external: dict[str, str] = {}
            patient_ids = {study["study_id"]: study["patient_id"] for study in studies}
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("each assignment must be an object")
                study_id, fold = row.get("study_id"), row.get("fold")
                if not isinstance(study_id, str) or not study_id or not isinstance(fold, str) or not fold:
                    raise ValueError("study_id and fold cannot be empty")
                if study_id in external:
                    raise AssignmentValidationError(f"duplicate assignment: {study_id}")
                if study_id not in expected_ids:
                    raise AssignmentValidationError(f"unknown Study in imported assignments: {study_id}")
                if row.get("dataset_version_id") and row["dataset_version_id"] != dataset_id:
                    raise AssignmentValidationError("DatasetVersion mismatch in assignment row")
                if row.get("patient_id") and row["patient_id"] != patient_ids.get(study_id):
                    raise AssignmentValidationError(f"patient_id mismatch for study: {study_id}")
                external[study_id] = fold
            if set(external) != expected_ids:
                raise AssignmentValidationError(f"import must cover all Studies; missing={sorted(expected_ids - set(external))[:5]}")
            external_folds = sorted(set(external.values()))
            if len(external_folds) < 2:
                raise AssignmentValidationError("import needs at least two folds")
            fold_map = {value: f"fold_{index}" for index, value in enumerate(external_folds)}
            canonical_assignments = {study: fold_map[fold] for study, fold in external.items()}
            labels = {study["study_id"]: study["labels"] for study in studies if study.get("labels")}
            try:
                plan = create_fold_plan_from_assignments(
                    {"dataset_version_id": dataset_id, "studies": studies}, canonical_assignments,
                    dataset_version_id=dataset_id, labels=labels or None,
                    configuration={"source_format": source_format, "source_fold_plan_id": source_fold_plan_id,
                                   "identity_preserved": False})
            except ValueError as exc:
                raise AssignmentValidationError(str(exc)) from exc
            report = validate(plan, dataset_id, studies)
            if not report["passed"]:
                return _emit(report, json_mode, _validation_exit(report))
            target = Path(args.output)
            existed = target.exists()
            if existed and load_fold_plan(target).locked:
                raise FileExistsError(f"fold plan is locked: {target}")
            if existed and not args.force:
                raise FileExistsError(f"output exists: {target} (use --force)")
            save_fold_plan(plan, target)
            report.update(output=str(target), identity_preserved=False,
                          source_fold_plan_id=source_fold_plan_id, imported_fold_plan_id=plan.fold_plan_id)
            return _emit(report, json_mode, 0)
        if args.command == "lock":
            plan = lock_fold_plan(args.fold_plan)
            return _emit({"locked": plan.locked, "fold_plan_id": plan.fold_plan_id}, json_mode, 0)
    except SystemExit as exc:
        return int(exc.code)
    except UsageError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except AssignmentValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (FileExistsError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _emit(value: Any, json_mode: bool, status: int) -> int:
    warnings = value.get("warnings", []) if isinstance(value, dict) else []
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if json_mode:
        print(json.dumps(value, sort_keys=True, allow_nan=False))
    else:
        if "passed" in value and "leakage" in value:
            print(f"FoldPlan: {value.get('fold_plan_id', 'unknown')}")
            print(f"DatasetVersion: {value.get('dataset_version_id', 'unknown')}")
            print(f"Validation: {'PASS' if value['passed'] else 'FAIL'}")
            print(f"Leakage: {'PASS' if value.get('leakage', {}).get('passed') else 'FAIL'}")
        print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return status


def _validation_exit(report: FoldValidationReport) -> int:
    return 0 if report["passed"] else 1
