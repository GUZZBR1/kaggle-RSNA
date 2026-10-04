"""Deterministic patient-grouped folds and a metadata-level leakage audit."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Mapping

from .models import DatasetIndex


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
