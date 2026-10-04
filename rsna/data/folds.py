"""Deterministic patient-grouped folds and a metadata-level leakage audit."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Mapping

from .models import DatasetIndex


class LeakageError(ValueError):
    """Raised when a patient or one of its studies spans validation folds."""


def make_fold_assignments(index: DatasetIndex, *, seed: int = 42, n_folds: int = 5) -> dict[str, str]:
    if n_folds < 2 or seed < 0:
        raise ValueError("n_folds must be at least two and seed must be nonnegative")
    studies = [study for study in index.studies if study.study_instance_uid and study.patient_id]
    patients = sorted({study.patient_id for study in studies})
    if len(patients) < n_folds:
        raise ValueError(f"{len(patients)} patient groups cannot fill {n_folds} folds")
    order = sorted(patients, key=lambda patient: hashlib.sha256(
        f"{seed}:{patient}".encode()).hexdigest())
    patient_fold = {patient: f"fold_{i % n_folds}" for i, patient in enumerate(order)}
    return {study.study_instance_uid: patient_fold[study.patient_id] for study in studies}


def validate_no_leakage(index: DatasetIndex, assignments: Mapping[str, str]) -> dict:
    patient_folds: dict[str, set[str]] = defaultdict(set)
    study_folds: dict[str, set[str]] = defaultdict(set)
    series_folds: dict[str, set[str]] = defaultdict(set)
    slice_folds: dict[str, set[str]] = defaultdict(set)
    for study in index.studies:
        fold = assignments.get(study.study_instance_uid or "")
        if fold is None:
            continue
        patient = study.patient_id or f"unknown:{study.study_id}"
        patient_folds[patient].add(fold)
        study_folds[study.study_id].add(fold)
        for series in study.series:
            series_folds[series.series_id].add(fold)
            for item in series.slices:
                slice_folds[item.slice_id].add(fold)
    repeated_patients = {key: sorted(value) for key, value in patient_folds.items() if len(value) > 1}
    repeated_studies = {key: sorted(value) for key, value in study_folds.items() if len(value) > 1}
    repeated_series = {key: sorted(value) for key, value in series_folds.items() if len(value) > 1}
    repeated_slices = {key: sorted(value) for key, value in slice_folds.items() if len(value) > 1}
    report = {"status": "FAIL" if any((repeated_patients, repeated_studies, repeated_series, repeated_slices)) else "PASS",
              "patient_leakage": len(repeated_patients), "study_leakage": len(repeated_studies),
              "series_leakage": len(repeated_series), "slice_leakage": len(repeated_slices),
              "patient_conflicts": repeated_patients}
    if report["status"] == "FAIL":
        raise LeakageError(f"patient/group leakage across folds: {report}")
    return report
