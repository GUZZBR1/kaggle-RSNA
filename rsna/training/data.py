"""Leakage-gated tensor batches for study-level model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from ..contracts import DatasetVersion
from ..folds.models import FoldPlanManifest
from ..labels import LabelRecord
from ..leakage import LeakagePolicy, require_valid_leakage_report, validate_leakage
from ..targets import TARGETS


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("tensor training data requires the optional 'torch' dependency") from exc
    return torch


@dataclass(frozen=True)
class StudyTensorRecord:
    study_id: str
    patient_id: str
    inputs: Any
    labels: LabelRecord
    series_ids: tuple[str, ...] = ()
    sop_ids: tuple[str, ...] = ()
    file_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.study_id, str) or not self.study_id.strip():
            raise ValueError("study_id must be nonempty")
        if not isinstance(self.patient_id, str) or not self.patient_id.strip():
            raise ValueError("patient_id must be nonempty")
        if not isinstance(self.labels, LabelRecord):
            raise TypeError("labels must be a LabelRecord")
        if self.labels.study_id != self.study_id:
            raise ValueError("label study_id does not match StudyTensorRecord")
        if self.labels.patient_id is not None and self.labels.patient_id != self.patient_id:
            raise ValueError("label patient_id does not match StudyTensorRecord")
        for name in ("series_ids", "sop_ids", "file_hashes"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} must contain nonempty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, values)
        torch = _torch()
        if not isinstance(self.inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if self.inputs.ndim < 1:
            raise ValueError("inputs must have at least one dimension")


@dataclass(frozen=True)
class TrainingBatch:
    inputs: Any
    targets: Any
    mask: Any
    study_ids: tuple[str, ...]


class FoldTensorDataset:
    """Study tensors bound to a DatasetVersion and a validated fold manifest.

    ``offset`` in :meth:`iter_batches` is a number of records to skip in the
    selected deterministic order, which supports resuming a partially consumed
    epoch without mutating the global torch RNG.
    """

    def __init__(self, records: list[StudyTensorRecord] | tuple[StudyTensorRecord, ...],
                 dataset_version: DatasetVersion, fold_plan: FoldPlanManifest):
        if not isinstance(dataset_version, DatasetVersion):
            raise TypeError("dataset_version must be a DatasetVersion")
        if not isinstance(fold_plan, FoldPlanManifest):
            raise TypeError("fold_plan must be a FoldPlanManifest")
        if tuple(dataset_version.class_names) != tuple(TARGETS):
            raise ValueError("DatasetVersion class_names must exactly match TARGETS order")
        if dataset_version.dataset_version_id != fold_plan.dataset_version_id:
            raise ValueError("FoldPlan dataset identity does not match DatasetVersion")
        supplied_records = tuple(records)
        if any(not isinstance(record, StudyTensorRecord) for record in supplied_records):
            raise TypeError("records must contain only StudyTensorRecord values")
        self.records = tuple(sorted(supplied_records, key=lambda record: record.study_id))
        if len({record.study_id for record in self.records}) != len(self.records):
            raise ValueError("study IDs must be unique")
        expected = set(fold_plan.assignments)
        observed = {record.study_id for record in self.records}
        if observed != expected:
            missing, extra = sorted(expected - observed), sorted(observed - expected)
            raise ValueError(f"records must exactly cover FoldPlan studies (missing={missing}, extra={extra})")

        self.dataset_version = dataset_version
        self.fold_plan = fold_plan
        identity_records: list[dict[str, Any]] = []
        for record in self.records:
            fold_id = fold_plan.assignments[record.study_id]
            if fold_plan.grouping_key == "patient_id" and fold_plan.study_groups[record.study_id] != record.patient_id:
                raise ValueError(f"patient_id for study {record.study_id!r} does not match FoldPlan grouping")
            identity_records.append({"entity_type": "study", "study_id": record.study_id,
                                    "study_uid": record.study_id, "patient_id": record.patient_id,
                                    "fold_id": fold_id})
            for series_id in record.series_ids:
                identity_records.append({"entity_type": "series", "study_id": record.study_id,
                                         "study_uid": record.study_id, "series_id": series_id,
                                         "series_uid": series_id, "patient_id": record.patient_id,
                                         "fold_id": fold_id})
            for sop_id in record.sop_ids:
                row = {"entity_type": "slice", "study_id": record.study_id,
                       "study_uid": record.study_id, "slice_id": sop_id, "sop_uid": sop_id,
                       "patient_id": record.patient_id, "fold_id": fold_id}
                identity_records.append(row)
            for file_hash in record.file_hashes:
                row = {"entity_type": "slice", "study_id": record.study_id,
                       "study_uid": record.study_id, "patient_id": record.patient_id,
                       "fold_id": fold_id, "file_hash": file_hash}
                identity_records.append(row)
        self.leakage_records = tuple(identity_records)
        self.leakage_report = validate_leakage(
            identity_records, assignments=dict(fold_plan.assignments), policy=LeakagePolicy.STRICT,
            dataset_version=dataset_version)
        require_valid_leakage_report(self.leakage_report)

    def iter_batches(self, fold_id: str, split: str = "train", batch_size: int = 1,
                     seed: int = 0, epoch: int = 0, offset: int = 0) -> Iterator[TrainingBatch]:
        if fold_id not in self.fold_plan.group_assignments.values():
            raise ValueError(f"unknown fold_id: {fold_id}")
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        for name, value in (("batch_size", batch_size), ("seed", seed), ("epoch", epoch), ("offset", offset)):
            if type(value) is not int or value < (1 if name == "batch_size" else 0):
                raise ValueError(f"{name} must be a {'positive' if name == 'batch_size' else 'nonnegative'} integer")
        torch = _torch()
        selected = [record for record in self.records
                    if (self.fold_plan.assignments[record.study_id] != fold_id if split == "train"
                        else self.fold_plan.assignments[record.study_id] == fold_id)]
        if split == "train":
            generator = torch.Generator()
            generator.manual_seed((seed + epoch) % (2**63))
            order = torch.randperm(len(selected), generator=generator).tolist()
            selected = [selected[index] for index in order]
        for start in range(offset, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            inputs = torch.stack([record.inputs for record in batch])
            targets = torch.zeros((len(batch), len(TARGETS)), dtype=torch.float32)
            mask = torch.zeros((len(batch), len(TARGETS)), dtype=torch.bool)
            for row, record in enumerate(batch):
                for column, target in enumerate(TARGETS):
                    value = record.labels.values[target]
                    if value is not None:
                        targets[row, column] = float(value)
                        mask[row, column] = True
            yield TrainingBatch(inputs, targets, mask, tuple(record.study_id for record in batch))
