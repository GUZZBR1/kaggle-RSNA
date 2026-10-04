"""Operational fold-plan creation, validation, comparison, and export."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import pstdev as population_stdev
from typing import Any, TypedDict

from .contracts import DatasetVersion, FoldPlan
from .identity import digest
from .data.models import index_from_dict


class UsageError(ValueError):
    """A valid command shape with invalid command values or constraints."""


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
    """Load index JSON without DICOM access; accepts an explicit DatasetVersion wrapper."""
    raw_bytes = Path(path).read_bytes()
    data = json.loads(raw_bytes)
    if not isinstance(data, dict):
        raise ValueError("dataset manifest must be a JSON object")
    index = data.get("index", data)
    if not isinstance(index, dict) or not isinstance(index.get("studies"), list):
        raise ValueError("dataset manifest must be an object containing studies")
    version = data.get("dataset_version")
    if isinstance(version, dict):
        dataset_id = DatasetVersion(**version).dataset_version_id
    elif data.get("dataset_version_id") or index.get("dataset_version_id"):
        dataset_id = data.get("dataset_version_id") or index["dataset_version_id"]
        if (not isinstance(dataset_id, str) or len(dataset_id) != 64
                or any(char not in "0123456789abcdef" for char in dataset_id)):
            raise ValueError("dataset_version_id must be a SHA-256 digest")
    else:
        # Issue 10 must remain usable with the Issue 8 index-only manifest.
        # Bind it deterministically to its exact bytes until a DatasetVersion
        # wrapper is available; no timestamp or filesystem location is involved.
        dataset_id = DatasetVersion(
            name="dataset-index", version="manifest-v1",
            source_manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            preprocessing_version="unspecified", class_names=("unspecified",),
        ).dataset_version_id
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
        n_series = len(series) if isinstance(series, list) else int(item.get("n_series", 0))
        n_slices = sum(len(s.get("slices", [])) for s in series if isinstance(s, dict)) if isinstance(series, list) else int(item.get("n_slices", 0))
        series_ids = [s["series_instance_uid"] for s in series
                      if isinstance(s, dict) and s.get("series_instance_uid")]
        labels = item.get("labels", {})
        studies.append({"study_id": assignment_id, "patient_id": patient_id,
                        "n_series": n_series, "n_slices": n_slices,
                        "series_ids": series_ids,
                        "labels": labels if isinstance(labels, dict) else {}})
    return dataset_id, studies


def read_plan(path: str | Path) -> dict[str, Any]:
    try:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"corrupt fold plan: {exc}") from exc
    if (not isinstance(plan, dict) or type(plan.get("schema_version")) is not int
            or plan.get("schema_version") != 1):
        raise ValueError("corrupt fold plan: unsupported or missing schema_version")
    required = {"dataset_version_id", "strategy", "n_folds", "random_state",
                "grouping_key", "fold_ids", "assignments", "fold_plan_id"}
    if not required.issubset(plan):
        raise ValueError(f"corrupt fold plan: missing {sorted(required - set(plan))}")
    assignments = plan["assignments"]
    configuration = plan.get("configuration", {})
    if not isinstance(configuration, dict) or not isinstance(plan.get("statistics"), dict):
        raise ValueError("corrupt fold plan: configuration and statistics must be objects")
    if (not isinstance(plan["fold_ids"], list)
            or any(not isinstance(fold, str) or not fold for fold in plan["fold_ids"])
            or type(plan["n_folds"]) is not int):
        raise ValueError("corrupt fold plan: invalid fold IDs or n_folds")
    if not isinstance(assignments, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in assignments.items()):
        raise ValueError("corrupt fold plan: assignments must map study IDs to fold IDs")
    if (len(set(plan["fold_ids"])) != len(plan["fold_ids"])
            or set(assignments.values()) - set(plan["fold_ids"])
            or plan["n_folds"] != len(plan["fold_ids"])):
        raise ValueError("corrupt fold plan: invalid fold IDs")
    core = FoldPlan(plan["dataset_version_id"], plan["strategy"], tuple(plan["fold_ids"]),
                    plan["random_state"], configuration=configuration,
                    assignment_manifest_sha256=digest(assignments),
                    fold_plan_id=plan["fold_plan_id"])
    if plan.get("assignment_manifest_sha256") != digest(assignments):
        raise ValueError("corrupt fold plan: assignment manifest hash mismatch")
    if plan.get("configuration", {}).get("statistics_sha256") != digest(plan.get("statistics")):
        raise ValueError("corrupt fold plan: statistics hash mismatch")
    if core.fold_plan_id != plan["fold_plan_id"]:
        raise ValueError("corrupt fold plan: fold_plan_id does not match contents")
    return plan


def statistics(studies: list[dict[str, Any]], assignments: dict[str, str], fold_ids: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    targets = sorted({target for study in studies for target in study.get("labels", {})})
    for fold in fold_ids:
        chosen = [s for s in studies if assignments.get(s["study_id"]) == fold]
        patients = {s["patient_id"] for s in chosen if s["patient_id"]}
        result[fold] = {"n_patients": len(patients), "n_studies": len(chosen),
                        "n_series": sum(s["n_series"] for s in chosen),
                        "n_slices": sum(s["n_slices"] for s in chosen)}
        label_stats = {}
        for target in targets:
            values = [s.get("labels", {}).get(target) for s in chosen]
            observed = [int(value) for value in values if isinstance(value, (bool, int)) and value in (0, 1)]
            positive = sum(observed)
            label_stats[target] = {"positive": positive, "negative": len(observed) - positive,
                                   "missing": len(values) - len(observed),
                                   "prevalence": positive / len(observed) if observed else None}
        result[fold]["targets"] = label_stats
    return result


def validate(plan: dict[str, Any], dataset_id: str | None = None,
             studies: list[dict[str, Any]] | None = None, *, require_coverage: bool = True) -> FoldValidationReport:
    errors: list[str] = []
    if dataset_id and plan["dataset_version_id"] != dataset_id:
        errors.append("dataset_version_id does not match dataset manifest")
    known = {s["study_id"]: s for s in (studies or [])}
    assigned = plan["assignments"]
    missing = sorted(set(known) - set(assigned)) if studies is not None else []
    unknown = sorted(set(assigned) - set(known)) if studies is not None else []
    if unknown:
        errors.append("assignments contain unknown studies")
    if missing and require_coverage:
        errors.append("studies are missing assignments")
    groups: dict[str, set[str]] = defaultdict(set)
    for study_id, fold in assigned.items():
        patient_id = known.get(study_id, {}).get("patient_id")
        groups[patient_id or f"study:{study_id}"].add(fold)
    leaked = sorted(group for group, folds in groups.items() if len(folds) > 1)
    if leaked:
        errors.append("patient leakage across folds")
    series_folds: dict[str, set[str]] = defaultdict(set)
    for study in studies or []:
        fold = assigned.get(study["study_id"])
        for series_id in study.get("series_ids", []):
            if fold is not None:
                series_folds[series_id].add(fold)
    split_series = sorted(series_id for series_id, folds in series_folds.items() if len(folds) > 1)
    if split_series:
        errors.append("series are split across folds")
    fold_ids = plan["fold_ids"]
    stats = statistics(studies or [], assigned, fold_ids)
    if studies is not None and plan.get("statistics") != stats:
        errors.append("fold statistics do not match dataset assignments")
    sizes = [v["n_studies"] for v in stats.values()]
    nonzero = [n for n in sizes if n]
    balance = {"largest_fold": max(sizes, default=0), "smallest_fold": min(sizes, default=0),
               "size_ratio": (max(nonzero) / min(nonzero)) if nonzero else None,
               "study_distribution": sizes,
               "patient_distribution": [v["n_patients"] for v in stats.values()]}
    target_balance = {}
    for target in sorted({name for fold_stats in stats.values() for name in fold_stats.get("targets", {})}):
        prevalence = [fold_stats["targets"][target]["prevalence"] for fold_stats in stats.values()
                      if fold_stats["targets"].get(target, {}).get("prevalence") is not None]
        target_balance[target] = {
            "prevalence_range": [min(prevalence), max(prevalence)] if prevalence else None,
            "prevalence_std": population_stdev(prevalence) if len(prevalence) > 1 else 0.0 if prevalence else None,
        }
    balance["target_prevalence"] = target_balance
    warnings = ["no labels are present; target prevalence balance is unavailable"] if not target_balance else []
    expected = len(known) if studies is not None else len(assigned)
    covered = len(set(assigned) & set(known)) if studies is not None else len(assigned)
    coverage = {"expected_studies": expected, "assigned_studies": len(assigned),
                "unassigned_studies": missing, "unknown_assigned_studies": unknown,
                "coverage_fraction": covered / expected if expected else 1.0}
    return {"passed": not errors, "fold_plan_id": plan["fold_plan_id"],
            "dataset_version_id": plan["dataset_version_id"], "errors": errors,
            "warnings": warnings, "fold_statistics": stats,
            "leakage": {"patient_leakage_groups": leaked, "study_leakage": False,
                        "series_containment": {"passed": not split_series, "split_series": split_series}},
            "balance": balance, "coverage": coverage}


def make_plan(dataset_id: str, studies: list[dict[str, Any]], n_folds: int, strategy: str,
              seed: int, grouping_key: str) -> dict[str, Any]:
    if n_folds < 2 or n_folds > len(studies):
        raise UsageError("n_folds must be at least 2 and no greater than study count")
    if strategy != "group":
        raise UsageError("only group strategy is supported")
    if grouping_key != "patient_id":
        raise UsageError("grouping_key must be patient_id")
    if seed < 0:
        raise UsageError("seed must be nonnegative")
    grouped: dict[str, list[str]] = defaultdict(list)
    for study in studies:
        group = study.get("patient_id")
        grouped[group or f"study:{study['study_id']}"] .append(study["study_id"])
    if len(grouped) < n_folds:
        raise UsageError("n_folds cannot exceed the number of patient groups")
    fold_ids = [f"fold_{i}" for i in range(n_folds)]
    rng = random.Random(seed)
    groups = list(grouped.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: -len(item[1]))
    load = [0] * n_folds
    assigned: dict[str, str] = {}
    for _, ids in groups:
        idx = min(range(n_folds), key=lambda i: (load[i], i))
        for study_id in ids:
            assigned[study_id] = fold_ids[idx]
        load[idx] += len(ids)
    plan_statistics = statistics(studies, assigned, fold_ids)
    config = {"grouping_key": grouping_key,
              "statistics_sha256": digest(plan_statistics)}
    config["patient_by_study"] = {s["study_id"]: s["patient_id"] for s in studies if s["patient_id"]}
    core = FoldPlan(dataset_id, strategy, tuple(fold_ids), seed,
                    configuration=config, assignment_manifest_sha256=digest(assigned))
    return {**core.to_dict(), "n_folds": n_folds, "grouping_key": grouping_key,
            "assignments": dict(sorted(assigned.items())),
            "statistics": plan_statistics, "locked": False, "created_with": "0.1.0"}


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
    return {"same_dataset_version": left["dataset_version_id"] == right["dataset_version_id"],
            "same_strategy": left["strategy"] == right["strategy"],
            "same_seed": left["random_state"] == right["random_state"],
            "same_number_of_folds": left["n_folds"] == right["n_folds"],
            "exact_equality": exact, "semantic_partition_equality": semantic,
            "studies_moved": len(moved), "patients_moved": patients_moved,
            "added_studies": added, "removed_studies": removed,
            "moved_patients": moved_patients, "moved": moved}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m rsna folds")
    p.add_argument("--json", action="store_true", dest="json_output")
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate"); g.add_argument("--dataset-manifest", required=True); g.add_argument("--output", required=True); g.add_argument("--n-folds", type=int, default=5); g.add_argument("--strategy", choices=("group",), default="group"); g.add_argument("--seed", type=int, default=42); g.add_argument("--grouping-key", choices=("patient_id",), default="patient_id"); g.add_argument("--dry-run", action="store_true"); g.add_argument("--force", action="store_true")
    v = sub.add_parser("validate"); v.add_argument("--fold-plan", required=True); v.add_argument("--dataset-manifest", required=True); v.add_argument("--allow-incomplete", action="store_true"); v.add_argument("--reproduce", action="store_true")
    i = sub.add_parser("inspect"); i.add_argument("--fold-plan", required=True); i.add_argument("--fold"); i.add_argument("--study-id"); i.add_argument("--patient-id"); i.add_argument("--limit", type=int, default=20); i.add_argument("--dataset-manifest")
    s = sub.add_parser("stats"); s.add_argument("--fold-plan", required=True); s.add_argument("--dataset-manifest", required=True)
    d = sub.add_parser("diff"); d.add_argument("left"); d.add_argument("right"); d.add_argument("--show-moved", action="store_true")
    e = sub.add_parser("export"); e.add_argument("--fold-plan", required=True); e.add_argument("--format", choices=("csv", "json"), required=True); e.add_argument("--output", required=True); e.add_argument("--dataset-manifest")
    l = sub.add_parser("lock"); l.add_argument("--fold-plan", required=True)
    m = sub.add_parser("import"); m.add_argument("--assignments", required=True); m.add_argument("--dataset-manifest", required=True); m.add_argument("--output", required=True); m.add_argument("--force", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # argparse accepts global options before a subcommand; also permit --json at the end.
    json_mode = "--json" in argv
    argv = [arg for arg in argv if arg != "--json"]
    try:
        args = _parser().parse_args(argv)
        if args.command == "generate":
            dataset_id, studies = load_dataset(args.dataset_manifest)
            plan = make_plan(dataset_id, studies, args.n_folds, args.strategy, args.seed, args.grouping_key)
            report = validate(plan, dataset_id, studies)
            if not report["passed"]:
                return _emit(report, json_mode, 1)
            if not args.dry_run:
                target = Path(args.output)
                existed = target.exists()
                previous_plan = read_plan(target) if existed else None
                if existed and previous_plan.get("locked"):
                    raise FileExistsError(f"fold plan is locked: {target}")
                if existed and not args.force:
                    raise FileExistsError(f"output exists: {target} (use --force)")
                if existed:
                    plan["overwrite"] = {"explicit": True,
                                         "previous_fold_plan_id": previous_plan["fold_plan_id"]}
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                report["output"] = str(target)
                report["overwrote"] = bool(existed and args.force)
            report["dry_run"] = args.dry_run
            return _emit(report, json_mode, 0)
        if args.command in {"validate", "stats"}:
            plan = read_plan(args.fold_plan); dataset_id, studies = load_dataset(args.dataset_manifest)
            report = validate(plan, dataset_id, studies, require_coverage=not getattr(args, "allow_incomplete", False))
            if getattr(args, "reproduce", False):
                if plan.get("strategy") == "imported":
                    report["warnings"].append("reproduction unavailable for externally imported assignments")
                    report["reproduction"] = {"supported": False, "passed": None}
                else:
                    regenerated = make_plan(dataset_id, studies, plan["n_folds"], plan["strategy"],
                                            plan["random_state"], plan["grouping_key"])
                    report["reproduction"] = {"supported": True,
                                              "passed": regenerated["fold_plan_id"] == plan["fold_plan_id"],
                                              "regenerated_fold_plan_id": regenerated["fold_plan_id"]}
                    if not report["reproduction"]["passed"]:
                        report["errors"].append("reproduced assignments do not match fold plan")
                        report["passed"] = False
            return _emit(report, json_mode, _validation_exit(report))
        if args.command == "inspect":
            plan = read_plan(args.fold_plan)
            result: dict[str, Any] = {k: plan.get(k) for k in ("fold_plan_id", "dataset_version_id", "strategy", "n_folds", "random_state", "grouping_key", "locked", "created_with", "fold_ids")}
            result["assignments_count"] = len(plan["assignments"])
            result["validation_status"] = "not_run"
            if args.dataset_manifest:
                dataset_id, studies = load_dataset(args.dataset_manifest)
                result["validation"] = validate(plan, dataset_id, studies)
                result["validation_status"] = "passed" if result["validation"]["passed"] else "failed"
            if args.study_id:
                patient = plan.get("configuration", {}).get("patient_by_study", {}).get(args.study_id)
                result["study"] = {"study_id": args.study_id,
                                    "fold": plan["assignments"].get(args.study_id),
                                    "patient_id": patient}
            if args.patient_id:
                patient_map = plan.get("configuration", {}).get("patient_by_study", {})
                if args.dataset_manifest:
                    _, studies = load_dataset(args.dataset_manifest)
                    patient_map = {s["study_id"]: s["patient_id"] for s in studies}
                folds = sorted({plan["assignments"][study_id] for study_id, patient in patient_map.items()
                                if patient == args.patient_id and study_id in plan["assignments"]})
                result["patient"] = {"patient_id": args.patient_id, "folds": folds}
            if args.fold:
                if args.fold not in plan["fold_ids"]: raise ValueError(f"unknown fold: {args.fold}")
                result["fold"] = args.fold; result["studies"] = [k for k, v in plan["assignments"].items() if v == args.fold][:args.limit]
                patient_map = plan.get("configuration", {}).get("patient_by_study", {})
                result["patients"] = sorted({patient_map[study_id] for study_id in result["studies"]
                                              if study_id in patient_map})
                result["statistics"] = plan.get("statistics", {}).get(args.fold)
            return _emit(result, json_mode, _validation_exit(result["validation"]) if "validation" in result else 0)
        if args.command == "diff":
            result = diff_plans(read_plan(args.left), read_plan(args.right))
            if not args.show_moved: result.pop("moved")
            return _emit(result, json_mode, 0)
        if args.command == "export":
            plan = read_plan(args.fold_plan)
            patient_by_study = {}
            if args.dataset_manifest:
                dataset_id, studies = load_dataset(args.dataset_manifest)
                report = validate(plan, dataset_id, studies)
                if not report["passed"]:
                    return _emit(report, json_mode, _validation_exit(report))
                patient_by_study = {s["study_id"]: s["patient_id"] for s in studies}
            rows = [{"study_id": s, "patient_id": patient_by_study.get(s), "fold": f} for s, f in sorted(plan["assignments"].items())]
            target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
            if args.format == "json": target.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            else:
                with target.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["study_id", "patient_id", "fold"] if args.dataset_manifest else ["study_id", "fold"]); writer.writeheader(); writer.writerows(rows)
            return _emit({"output": str(target), "assignments": len(rows), "format": args.format}, json_mode, 0)
        if args.command == "import":
            dataset_id, studies = load_dataset(args.dataset_manifest)
            patient_ids = {study["study_id"]: study["patient_id"] for study in studies}
            assignments: dict[str, str] = {}
            with Path(args.assignments).open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if not {"study_id", "fold"}.issubset(reader.fieldnames or []):
                    raise ValueError("assignments CSV requires study_id and fold columns")
                for row in reader:
                    study_id, fold = row["study_id"], row["fold"]
                    if not study_id or not fold:
                        raise ValueError("study_id and fold cannot be empty")
                    if study_id in assignments:
                        raise ValueError(f"duplicate assignment: {study_id}")
                    if row.get("patient_id") and row["patient_id"] != patient_ids.get(study_id):
                        raise ValueError(f"patient_id mismatch for study: {study_id}")
                    assignments[study_id] = fold
            fold_ids = sorted(set(assignments.values()))
            plan_statistics = statistics(studies, assignments, fold_ids)
            core = FoldPlan(dataset_id, "imported", tuple(fold_ids), 0,
                            configuration={"grouping_key": "patient_id", "source": "csv",
                                           "statistics_sha256": digest(plan_statistics),
                                           "patient_by_study": {s["study_id"]: s["patient_id"] for s in studies if s["patient_id"]}},
                            assignment_manifest_sha256=digest(assignments))
            plan = {**core.to_dict(), "n_folds": len(fold_ids), "grouping_key": "patient_id",
                    "assignments": dict(sorted(assignments.items())), "locked": False,
                    "statistics": plan_statistics, "created_with": "0.1.0"}
            report = validate(plan, dataset_id, studies)
            if not report["passed"]:
                return _emit(report, json_mode, _validation_exit(report))
            target = Path(args.output)
            existed = target.exists()
            if existed and read_plan(target).get("locked"):
                raise FileExistsError(f"fold plan is locked: {target}")
            if existed and not args.force:
                raise FileExistsError(f"output exists: {target} (use --force)")
            if existed:
                previous = read_plan(target)
                plan["overwrite"] = {"explicit": True,
                                     "previous_fold_plan_id": previous["fold_plan_id"]}
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            report["output"] = str(target)
            report["overwrote"] = existed
            return _emit(report, json_mode, 0)
        if args.command == "lock":
            path = Path(args.fold_plan); plan = read_plan(path)
            if plan.get("locked"): return _emit({"locked": True, "fold_plan_id": plan["fold_plan_id"]}, json_mode, 0)
            plan["locked"] = True; path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return _emit({"locked": True, "fold_plan_id": plan["fold_plan_id"]}, json_mode, 0)
    except SystemExit as exc:
        return int(exc.code)
    except UsageError as exc:
        print(str(exc), file=sys.stderr); return 2
    except (FileExistsError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr); return 2
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        print(str(exc), file=sys.stderr); return 3
    return 0


def _emit(value: Any, json_mode: bool, status: int) -> int:
    if json_mode: print(json.dumps(value, sort_keys=True, allow_nan=False))
    else: print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return status


def _validation_exit(report: FoldValidationReport) -> int:
    if report["passed"]:
        return 0
    return 4 if any("dataset_version_id" in error for error in report["errors"]) else 1
