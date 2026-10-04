"""CPU-only end-to-end check for synthetic DICOM data contracts."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
from typing import Any

from ..contracts import DatasetVersion
from ..folds import generate_fold_plan, load_fold_plan, save_fold_plan
from ..identity import digest
from ..leakage import (LeakagePolicy, LeakageValidationError,
                       require_valid_leakage_report, validate_leakage)
from ..targets import TARGET_REGISTRY, TARGET_REGISTRY_ID
from .cache import load_or_refresh
from .geometry import order_series_slices
from .laterality import resolve_series_laterality
from .manifest import load_manifest, save_manifest
from .orientation import describe_series_orientation
from .selection import SliceSelectionConfig, SliceSelector
from .synthetic import SyntheticConfig, TARGETS, generate_synthetic_dataset


INJECTIONS = {"patient-leakage", "duplicate-sop", "orientation-conflict",
              "missing-position", "missing-metadata", "spacing-irregular", "corrupted-cache"}


def run_data_smoke(output_dir: str | Path, *, seed: int = 42,
                   injection: str | None = None,
                   synthetic_config: SyntheticConfig | None = None) -> dict[str, Any]:
    """Run generation, indexing, geometry, labels, folds and artifact round-trip checks."""
    if injection is not None and injection not in INJECTIONS:
        raise ValueError(f"unknown injection {injection!r}; choose from {sorted(INJECTIONS)}")
    started = time.perf_counter()
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"smoke output directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "raw"
    config = synthetic_config or SyntheticConfig(seed=seed, injection=injection)
    if config.seed != seed or config.injection != injection:
        raise ValueError("synthetic_config seed and injection must match smoke arguments")
    generated = generate_synthetic_dataset(config, raw)
    cache_path = root / "index" / "index.sqlite3"
    cold = load_or_refresh(raw, cache_path, on_invalid="strict")
    if injection == "corrupted-cache":
        cache_path.write_bytes(b"intentionally corrupted synthetic cache")
        try:
            load_or_refresh(raw, cache_path, cache_policy="strict")
        except ValueError as exc:
            if "cache" not in str(exc).lower():
                raise AssertionError("corrupted cache failed for an unexpected reason") from exc
            return _injected_summary(root, generated, injection, "cache validation rejected corruption")
        raise AssertionError("corrupted cache was accepted")
    warm = load_or_refresh(raw, cache_path, on_invalid="strict")
    if cold.report.mode != "cold-build" or warm.report.mode != "warm-load" or warm.report.reparsed:
        raise AssertionError("cache did not transition from cold build to a zero-reparse warm load")
    index = warm.index
    if index.index_id != cold.index.index_id:
        raise AssertionError("cold and warm index identities differ")
    save_manifest(index, root / "index" / "manifest.json")
    target_rows = _load_targets(raw / "targets.csv")
    if len(target_rows) != index.statistics["n_studies"]:
        raise AssertionError("synthetic labels do not cover every indexed study")
    for target in TARGETS:
        observed = {row[target] for row in target_rows.values() if row[target] is not None}
        if not observed.issubset({0, 1}) or len(observed) != 2:
            raise AssertionError(f"target {target} does not contain both hard-label values")
    target_schema_id = TARGET_REGISTRY_ID
    label_identity = digest(target_rows)
    if injection == "duplicate-sop":
        if index.statistics["duplicate_sop_uid_count"] == 0:
            raise AssertionError("duplicate SOP injection was not detected by the index")
        return _injected_summary(root, generated, injection, "duplicate SOPInstanceUID detected")
    if injection == "missing-metadata":
        missing = sum(item.metadata.get("PixelSpacing") is None
            for study in index.studies for series in study.series for item in series.slices)
        if not missing:
            raise AssertionError("missing PixelSpacing injection was not detected")
        return _injected_summary(root, generated, injection,
                                 "missing PixelSpacing metadata detected", details={"missing_fields": missing})
    if index.statistics["duplicate_sop_uid_count"]:
        raise AssertionError("happy path contains duplicate SOPInstanceUIDs")

    series_records = [series for study in index.studies for series in study.series]
    geometry_orderings = [order_series_slices(series) for series in series_records]
    orientation_results = []
    laterality_results = []
    selector24 = SliceSelector(SliceSelectionConfig("physical_span", 24))
    selector16 = SliceSelector(SliceSelectionConfig("uniform", 16))
    selector32 = SliceSelector(SliceSelectionConfig("physical_span", 32))
    ordered_series = [{"slices": ordered.slices,
                       "series_instance_uid": source.series_instance_uid,
                       "warnings": source.warnings}
                      for source, ordered in zip(series_records, geometry_orderings)]
    selections24 = [selector24.select(series) for series in ordered_series]
    selections16 = [selector16.select(series) for series in ordered_series]
    selections32 = [selector32.select(series) for series in ordered_series]
    for series in series_records:
        metadata = [item.metadata for item in series.slices]
        orientation_results.append(describe_series_orientation(metadata).to_dict())
        laterality_results.append(resolve_series_laterality(metadata).to_dict())
    plane_counts = {plane: sum(item["plane"] == plane for item in orientation_results)
                    for plane in ("sagittal", "coronal", "axial")}
    laterality_counts = {side: sum(item["resolved"] == side for item in laterality_results)
                         for side in ("LEFT", "RIGHT")}
    geometry_ordering_proven = all(ordered.method == "geometry" for ordered in geometry_orderings) and all(
        result.selected_positions_mm == tuple(sorted(result.selected_positions_mm))
        for result in selections24 if all(p is not None for p in result.selected_positions_mm)
    )
    instance_number_trap_proven = all(
        [int(item.metadata["InstanceNumber"]) for item in result.selected if item is not None]
        == sorted([int(item.metadata["InstanceNumber"]) for item in result.selected if item is not None],
                  reverse=True)
        for result in selections24 if result.actual_count > 1 and result.fallback is None
    )
    if injection == "missing-position":
        fallback = [result for result in selections24 if result.fallback == "uniform"]
        ordering_fallback = [result for result in geometry_orderings if result.method != "geometry"
                             and any(w.code == "geometry_incomplete" for w in result.warnings)]
        if not fallback or not ordering_fallback or not any("used uniform" in " ".join(result.warnings) for result in fallback):
            raise AssertionError("missing-position injection did not produce an explicit fallback warning")
        return _injected_summary(root, generated, injection, "physical-span fallback warning recorded",
                                 details={"fallback_series": len(fallback),
                                          "ordering_method": ordering_fallback[0].method})
    if injection == "spacing-irregular":
        warnings = [warning for result in geometry_orderings for warning in result.warnings
                    if warning.code == "irregular_spacing"]
        if not warnings:
            raise AssertionError("spacing irregularity injection was not detected")
        return _injected_summary(root, generated, injection,
                                 "irregular physical slice spacing detected",
                                 details={"warning_count": len(warnings)})
    if injection == "orientation-conflict":
        if not any(item["consistency"] == "inconsistent" for item in orientation_results) or not any(
                any(w.code == "inconsistent_orientation" for w in result.warnings)
                for result in geometry_orderings):
            raise AssertionError("orientation conflict injection was not detected")
        if not any(item["resolved"] == "AMBIGUOUS" for item in laterality_results):
            raise AssertionError("laterality conflict injection was not detected")
        return _injected_summary(root, generated, injection,
                                 "orientation and laterality conflicts detected")
    if not geometry_ordering_proven or not instance_number_trap_proven:
        raise AssertionError("selected slices are not ordered by physical geometry")
    if any(item["consistency"] == "inconsistent" for item in orientation_results):
        raise AssertionError("happy path has inconsistent slice orientation")
    if not all(plane_counts.values()) or not all(laterality_counts.values()):
        raise AssertionError("synthetic geometry/laterality did not cover all canonical cases")

    preprocessing24 = asdict(selector24.config)
    preprocessing16 = asdict(selector16.config)
    preprocessing32 = asdict(selector32.config)
    selection_ids = {
        "physical_span_24": digest([result.to_dict() for result in selections24]),
        "uniform_16": digest([result.to_dict() for result in selections16]),
        "physical_span_32": digest([result.to_dict() for result in selections32]),
    }
    preprocessing_ids = {"physical_span_24": digest(preprocessing24),
                         "uniform_16": digest(preprocessing16),
                         "physical_span_32": digest(preprocessing32)}
    dataset = DatasetVersion("synthetic-rsna", f"seed-{seed}", index.index_id,
        "synthetic-smoke-v1", tuple(TARGETS), preprocessing=preprocessing24,
        dataset_index_artifact_id=index.index_id, synthetic=True)
    dataset_uniform = DatasetVersion("synthetic-rsna", f"seed-{seed}", index.index_id,
        "synthetic-smoke-v1", tuple(TARGETS), preprocessing=preprocessing16,
        dataset_index_artifact_id=index.index_id, synthetic=True)
    dataset_32 = DatasetVersion("synthetic-rsna", f"seed-{seed}", index.index_id,
        "synthetic-smoke-v1", tuple(TARGETS), preprocessing=preprocessing32,
        dataset_index_artifact_id=index.index_id, synthetic=True)
    if dataset.source_manifest_sha256 != dataset_uniform.source_manifest_sha256 or dataset.source_manifest_sha256 != dataset_32.source_manifest_sha256:
        raise AssertionError("preprocessing changes altered source identity")
    if len({dataset.dataset_version_id, dataset_uniform.dataset_version_id,
            dataset_32.dataset_version_id}) != 3:
        raise AssertionError("preprocessing changes did not alter DatasetVersion identity")
    fold_plan = generate_fold_plan(index, n_folds=5, strategy="group", random_state=seed,
        labels=target_rows, dataset_version_id=dataset.dataset_version_id)
    assignments = dict(fold_plan.assignments)
    leakage_result = validate_leakage(index, assignments=assignments,
        policy=LeakagePolicy.STRICT, dataset_version_id=dataset.dataset_version_id)
    if injection == "patient-leakage":
        patient_studies = next(study for study in index.studies if study.patient_id)
        sibling = next(study for study in index.studies
                       if study.patient_id == patient_studies.patient_id and
                       study.study_instance_uid != patient_studies.study_instance_uid)
        assignments[sibling.study_instance_uid] = "injected-leakage-fold"
        leakage_result = validate_leakage(index, assignments=assignments,
            policy=LeakagePolicy.STRICT, dataset_version_id=dataset.dataset_version_id)
        try:
            require_valid_leakage_report(leakage_result)
        except LeakageValidationError:
            if not any(issue.type.value == "PATIENT_CROSS_FOLD" for issue in leakage_result.issues):
                raise AssertionError("leakage injection failed for an unexpected reason")
            return _injected_summary(root, generated, injection,
                "patient leakage blocked by LeakageGuard",
                details={"leakage_report_id": leakage_result.report_id,
                         "issue_types": sorted({issue.type.value for issue in leakage_result.issues})})
        raise AssertionError("patient leakage injection passed the leakage guard")
    require_valid_leakage_report(leakage_result)
    leakage = leakage_result.to_dict()
    leakage_counts = _leakage_counts(leakage_result)
    fold_ids = tuple(f"fold_{i}" for i in range(fold_plan.n_folds))
    other_plan = generate_fold_plan(index, n_folds=5, strategy="group", random_state=seed + 1,
        labels=target_rows, dataset_version_id=dataset.dataset_version_id)
    if fold_plan.fold_plan_id == other_plan.fold_plan_id:
        raise AssertionError("fold seed change did not change FoldPlan identity")

    index2 = root / "index"
    (root / "folds").mkdir(parents=True, exist_ok=True)
    _write_json(index2 / "target-schema.json", {"registry": "official TargetRegistry",
        "target_registry": TARGET_REGISTRY.to_dict(), "target_schema_id": target_schema_id,
        "label_source": "synthetic", "positive_values": [1], "negative_values": [0],
        "missing_is_negative": False})
    _write_json(root / "folds" / "assignments.json", assignments)
    save_fold_plan(fold_plan, root / "folds" / "fold-plan.json")
    _write_json(root / "folds" / "seed-change.json", {"same_dataset_version_id": dataset.dataset_version_id,
        "fold_plan_id": fold_plan.fold_plan_id, "other_seed_fold_plan_id": other_plan.fold_plan_id})
    _write_json(root / "reports" / "geometry.json", orientation_results)
    _write_json(root / "reports" / "geometry-ordering.json",
                [result.to_dict() for result in geometry_orderings])
    _write_json(root / "reports" / "laterality.json", laterality_results)
    _write_json(root / "reports" / "leakage.json", leakage)
    _write_json(root / "reports" / "cache.json", {"cold": cold.report.to_dict(), "warm": warm.report.to_dict()})
    _write_json(root / "reports" / "targets.json", {"study_count": len(target_rows),
        "label_identity": label_identity,
        "label_count": sum(value is not None for row in target_rows.values() for value in row.values()),
        "missing_count": sum(value is None for row in target_rows.values() for value in row.values()),
        "missing_is_negative": False})
    # Close and reload the persisted representations, then recompute their material decisions.
    reloaded = load_manifest(index2 / "manifest.json")
    reloaded_assignments = json.loads((root / "folds" / "assignments.json").read_text(encoding="utf-8"))
    reloaded_fold_plan = load_fold_plan(root / "folds" / "fold-plan.json",
        dataset_version_id=dataset.dataset_version_id, dataset=reloaded).to_dict()
    reloaded_target_schema = json.loads((index2 / "target-schema.json").read_text(encoding="utf-8"))
    reloaded_labels = _load_targets(raw / "targets.csv")
    reloaded_selections = []
    for study in reloaded.studies:
        for series in study.series:
            ordered = order_series_slices(series)
            reloaded_selections.append(selector24.select({"slices": ordered.slices,
                "series_instance_uid": series.series_instance_uid,
                "warnings": series.warnings}).to_dict())
    reloaded_leakage = validate_leakage(reloaded, assignments=reloaded_assignments,
        policy=LeakagePolicy.STRICT, dataset_version_id=dataset.dataset_version_id)
    if reloaded.index_id != index.index_id or digest(reloaded_assignments) != digest(assignments):
        raise AssertionError("manifest or fold assignment changed across serialization")
    if (reloaded_fold_plan["fold_plan_id"] != fold_plan.fold_plan_id or
            reloaded_fold_plan["n_folds"] != len(fold_ids) or
            reloaded_target_schema["target_schema_id"] != target_schema_id or
            reloaded_target_schema["target_registry"]["targets"] != TARGET_REGISTRY.to_dict()["targets"] or
            digest(reloaded_labels) != label_identity):
        raise AssertionError("fold plan, target order or labels changed across serialization")
    if (digest(reloaded_selections) != selection_ids["physical_span_24"] or
            reloaded_leakage.report_id != leakage_result.report_id):
        raise AssertionError("selection or leakage output changed across serialization")

    second_root = root / "location-check"
    shutil.copytree(raw, second_root / "raw")
    location_result = load_or_refresh(second_root / "raw", second_root / "index" / "index.sqlite3")
    if location_result.index.index_id != index.index_id:
        raise AssertionError("absolute output location contaminated material index identity")
    location_dataset = DatasetVersion("synthetic-rsna", f"seed-{seed}", location_result.index.index_id,
        "synthetic-smoke-v1", tuple(TARGETS), preprocessing=preprocessing24,
        dataset_index_artifact_id=location_result.index.index_id, synthetic=True)
    location_fold_plan = generate_fold_plan(location_result.index, n_folds=5, strategy="group",
        random_state=seed, labels=target_rows, dataset_version_id=location_dataset.dataset_version_id)
    location_selections = []
    for study in location_result.index.studies:
        for series in study.series:
            ordered = order_series_slices(series)
            location_selections.append(selector24.select({"slices": ordered.slices,
                "series_instance_uid": series.series_instance_uid,
                "warnings": series.warnings}).to_dict())
    location_selection_id = digest(location_selections)
    if (location_dataset.dataset_version_id != dataset.dataset_version_id or
            location_fold_plan.fold_plan_id != fold_plan.fold_plan_id or
            location_selection_id != selection_ids["physical_span_24"] or
            digest(_load_targets(second_root / "raw" / "targets.csv")) != label_identity):
        raise AssertionError("output directory changed a material pipeline identity")
    shutil.rmtree(second_root)
    stats = index.statistics
    summary = {"status": "READY", "synthetic": True, "seed": seed,
        "dataset_version_id": dataset.dataset_version_id, "source_identity": index.index_id,
        "fold_plan_id": fold_plan.fold_plan_id,
        "preprocessing_id": preprocessing_ids["physical_span_24"],
        "preprocessing_ids": preprocessing_ids, "selection_ids": selection_ids,
        "target_schema_id": target_schema_id, "target_order": list(TARGETS), "leakage": "PASS",
        "leakage_counts": leakage_counts, "leakage_report_id": leakage_result.report_id,
        "label_identity": label_identity,
        "patients": len({study.patient_id for study in index.studies}),
        "studies": stats["n_studies"], "series": stats["n_series"], "slices": stats["n_slices"],
        "folds": len(fold_ids), "cache": {"cold": cold.report.mode, "warm": warm.report.mode,
            "warm_reparsed": warm.report.reparsed}, "geometry_ordering": "PASS",
        "planes": plane_counts, "laterality": laterality_counts,
        "labels": {"hard_values": [0, 1], "missing_is_negative": False,
                   "missing_count": sum(value is None for row in target_rows.values() for value in row.values())},
        "tests": {"cold_warm_identity": "PASS", "preprocessing_identity": "PASS", "fold_seed_identity": "PASS",
                  "serialization_reload": "PASS", "output_location": "PASS",
                  "instance_number_trap": "PASS"}}
    summary["duration_seconds"] = round(time.perf_counter() - started, 3)

    # Persist READY only after every cache, serialization, leakage, and location gate passed.
    prepared_path = root / "prepared" / "prepared-dataset.json"
    pending_path = prepared_path.with_name("prepared-dataset.json.pending")
    prepared = {"status": "READY", "readiness_scope": "metadata-only prepared-data integrity",
        "synthetic": True, "model_trained": False, "clinical_validation": "NOT_PERFORMED",
        "kaggle_score": None, "production_ready": False,
        "dataset_index_id": index.index_id, "dataset_version": dataset.to_dict(),
        "preprocessing_id": preprocessing_ids["physical_span_24"],
        "target_registry_id": target_schema_id, "fold_plan_id": fold_plan.fold_plan_id,
        "fold_plan": fold_plan.to_dict(), "leakage_report_id": leakage_result.report_id,
        "leakage_report": leakage, "seed": seed,
        "config": {"synthetic_data": asdict(config), "preprocessing": preprocessing24,
                   "fold_strategy": "group", "n_folds": 5}}
    _write_json(pending_path, prepared)
    persisted = json.loads(pending_path.read_text(encoding="utf-8"))
    if (persisted != prepared or persisted["status"] != "READY" or
            persisted["dataset_index_id"] != index.index_id or
            persisted["target_registry_id"] != TARGET_REGISTRY_ID or
            persisted["fold_plan_id"] != fold_plan.fold_plan_id or
            persisted["leakage_report_id"] != leakage_result.report_id or
            persisted["config"]["synthetic_data"]["seed"] != seed):
        pending_path.unlink(missing_ok=True)
        raise AssertionError("prepared READY artifact failed persisted provenance validation")
    pending_path.replace(prepared_path)
    _write_json(root / "smoke-summary.json", summary)
    return summary


def _load_targets(path: Path) -> dict[str, dict[str, int | None]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        result = {}
        for row in rows:
            result[row["study_instance_uid"]] = {name: (None if row[name] == "" else int(row[name]))
                                                for name in TARGETS}
        return result


def _leakage_counts(report: Any) -> dict[str, int | str]:
    issues = report.counts["issues_by_type"]
    return {
        "patient_leakage": issues.get("PATIENT_CROSS_FOLD", 0),
        "study_leakage": issues.get("STUDY_CROSS_FOLD", 0),
        "series_leakage": issues.get("SERIES_CROSS_FOLD", 0),
        "slice_leakage": issues.get("SLICE_CROSS_FOLD", 0),
        "status": "PASS" if report.passed else "FAIL",
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _injected_summary(root: Path, generated: dict, injection: str, detected: str,
                      *, details: dict | None = None) -> dict:
    summary = {"status": "EXPECTED_FAILURE", "synthetic": True, "injection": injection,
               "detected": detected, **generated}
    if details:
        summary.update(details)
    _write_json(root / "smoke-summary.json", summary)
    return summary
