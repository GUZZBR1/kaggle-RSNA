"""Leakage invariants and application-time dataset checks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping


def validate_leakage(records: list[dict[str, Any]], assignments: Mapping[str, str]) -> None:
    patient_folds: dict[str, set[str]] = defaultdict(set)
    series_folds: dict[str, set[str]] = defaultdict(set)
    study_folds: dict[str, set[str]] = defaultdict(set)
    record_ids = [row["study_id"] for row in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate study records would create conflicting assignments")
    for row in records:
        study_id = row["study_id"]
        fold = assignments[study_id]
        study_folds[study_id].add(fold)
        patient_id = row.get("patient_id")
        if patient_id:
            patient_folds[str(patient_id)].add(fold)
        for series_id in row.get("series_ids", ()):
            if series_id:
                series_folds[str(series_id)].add(fold)
    for label, owners in (("patient", patient_folds), ("study", study_folds), ("series", series_folds)):
        leaks = sorted(key for key, folds in owners.items() if len(folds) > 1)
        if leaks:
            raise ValueError(f"{label} leakage detected across folds: {leaks[:5]}")
    if set(assignments) != {row["study_id"] for row in records}:
        raise ValueError("every input study must have exactly one assignment")


def check_dataset_studies(manifest: Any, study_ids: set[str], *, policy: str = "strict") -> tuple[str, ...]:
    if policy != "strict":
        raise ValueError("only strict application policy is supported; generate and lock a new plan to extend")
    planned = set(manifest.assignments)
    new = sorted(study_ids - planned)
    missing = tuple(sorted(planned - study_ids))
    if new:
        raise ValueError(f"unassigned studies in dataset: {new[:5]}; strict policy forbids silent assignment")
    return missing
