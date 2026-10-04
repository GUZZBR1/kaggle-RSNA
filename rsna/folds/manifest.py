"""Versioned JSON persistence with tamper and dataset mismatch detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..identity import digest
from .models import FoldPlanManifest
from .validate import check_dataset_studies


def save_fold_plan(plan: FoldPlanManifest, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = load_fold_plan(target)
        if existing.locked and existing.fold_plan_id != plan.fold_plan_id:
            raise ValueError("cannot overwrite a locked FoldPlan")
    payload = json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    target.write_text(payload, encoding="utf-8")
    return plan.fold_plan_id


def load_fold_plan(path: str | Path, *, dataset_version_id: str | None = None,
                   dataset: Any = None, policy: str = "strict") -> FoldPlanManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {"schema_version", "dataset_version_id", "strategy", "n_folds", "random_state",
               "grouping_key", "assignments", "group_assignments", "study_groups",
               "study_fingerprints", "configuration", "statistics", "warnings",
               "locked", "fold_plan_id"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("invalid FoldPlan manifest fields")
    manifest = FoldPlanManifest(**raw)
    if dataset_version_id is not None and dataset_version_id != manifest.dataset_version_id:
        raise ValueError("DatasetVersion mismatch; FoldPlan is bound to a different dataset")
    if dataset is not None:
        from .generate import _records
        records = _records(dataset)
        current = {row["study_id"] for row in records}
        missing = check_dataset_studies(manifest, current, policy=policy)
        current_fingerprints = {row["study_id"]: digest(
            {"patient_id": row["patient_id"], "series_ids": row["series_ids"],
             "series_available": row["series_available"]}) for row in records}
        changed = sorted(study for study in current & set(manifest.study_fingerprints)
                         if current_fingerprints[study] != manifest.study_fingerprints[study])
        if changed:
            raise ValueError(f"study identity changed since FoldPlan generation: {changed[:5]}")
        object.__setattr__(manifest, "application_warnings",
                           (f"expected studies missing from dataset: {list(missing)}",) if missing else ())
    return manifest
