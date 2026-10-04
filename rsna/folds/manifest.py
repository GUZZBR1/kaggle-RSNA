"""Versioned JSON persistence with tamper and dataset mismatch detection."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Any

from ..identity import digest
from .models import SCHEMA_VERSION, FoldPlanManifest
from .validate import check_dataset_studies


def save_fold_plan(plan: FoldPlanManifest, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = load_fold_plan(target)
        if existing.locked and existing.fold_plan_id != plan.fold_plan_id:
            raise ValueError("cannot overwrite a locked FoldPlan")
    payload = json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return plan.fold_plan_id


def lock_fold_plan(path: str | Path) -> FoldPlanManifest:
    """Lock a saved plan by creating its canonical locked identity and persisting it."""
    plan = load_fold_plan(path)
    if plan.locked:
        return plan
    locked_plan = replace(plan, locked=True, fold_plan_id="")
    save_fold_plan(locked_plan, path)
    return locked_plan


def validate_fold_plan_dataset(manifest: FoldPlanManifest, dataset: Any, *, policy: str = "strict") -> None:
    """Validate a canonical plan against the current dataset using one binding check."""
    if not isinstance(manifest, FoldPlanManifest):
        raise TypeError("manifest must be a FoldPlanManifest")
    from .generate import _dataset_fingerprint, _records

    records = _records(dataset)
    current_ids = {row["study_id"] for row in records}
    missing = check_dataset_studies(manifest, current_ids, policy=policy)
    if missing and policy == "strict":
        raise ValueError(f"missing studies from DatasetVersion: {list(missing[:5])}")
    fingerprints = {row["study_id"]: digest(
        {"patient_id": row["patient_id"], "series_ids": row["series_ids"],
         "series_available": row["series_available"], "n_slices": row["n_slices"]})
        for row in records}
    changed = sorted(study for study in current_ids & set(manifest.study_fingerprints)
                     if fingerprints[study] != manifest.study_fingerprints[study])
    if changed:
        raise ValueError(f"study identity changed since FoldPlan generation: {changed[:5]}")
    if _dataset_fingerprint(dataset, records) != manifest.provenance.get("dataset_fingerprint"):
        raise ValueError("DatasetIndex identity mismatch; FoldPlan is bound to different dataset inputs")
    object.__setattr__(manifest, "application_warnings",
                       (f"expected studies missing from dataset: {list(missing)}",) if missing else ())


def load_fold_plan(path: str | Path, *, dataset_version_id: str | None = None,
                   dataset: Any = None, policy: str = "strict") -> FoldPlanManifest:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    raw = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                     parse_constant=lambda value: (_ for _ in ()).throw(
                         ValueError(f"invalid JSON constant: {value}")))
    allowed = {"schema_version", "dataset_version_id", "strategy", "n_folds", "random_state",
               "generator_version", "input_fingerprint", "provenance",
               "grouping_key", "assignments", "group_assignments", "study_groups",
               "study_fingerprints", "configuration", "statistics", "warnings",
               "locked", "fold_plan_id"}
    if not isinstance(raw, dict) or set(raw) != allowed:
        raise ValueError("invalid FoldPlan manifest fields")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported fold manifest schema version: {raw.get('schema_version')}")
    manifest = FoldPlanManifest(**raw)
    if dataset_version_id is not None and dataset_version_id != manifest.dataset_version_id:
        raise ValueError("DatasetVersion mismatch; FoldPlan is bound to a different dataset")
    if dataset is not None:
        validate_fold_plan_dataset(manifest, dataset, policy=policy)
    return manifest
