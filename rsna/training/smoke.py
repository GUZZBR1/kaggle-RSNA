"""Tiny deterministic end-to-end training fixture used by CLI and tests."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

from ..contracts import DatasetVersion, ModelCandidate, TrainingJob
from ..folds.generate import generate_fold_plan
from ..identity import digest
from ..labels import LabelRecord
from ..targets import TARGETS
from .data import FoldTensorDataset, StudyTensorRecord
from .engine import FoldData, TrainingEngine
from .telemetry import JsonlTelemetrySink


def make_smoke_job():
    import torch
    from torch.utils.data import TensorDataset

    dataset_version = DatasetVersion("training-smoke", "v1", "a" * 64, "synthetic-v1",
                                     TARGETS, synthetic=True)
    records = []
    fold_inputs = []
    for i in range(8):
        study_id, patient_id = f"study-{i:02}", f"patient-{i:02}"
        fold_inputs.append({"study_id": study_id, "patient_id": patient_id})
        values = {name: ((i + target_index) % 2 if (i + target_index) % 5 else None)
                  for target_index, name in enumerate(TARGETS)}
        labels = LabelRecord(study_id, values, "manual_review", allow_partial=True,
                             patient_id=patient_id)
        inputs = torch.tensor([float(i % 2), float(i) / 8, 1.0, float((i * 3) % 5)],
                              dtype=torch.float32)
        records.append(StudyTensorRecord(study_id, patient_id, inputs, labels))
    plan = generate_fold_plan(fold_inputs, n_folds=2, random_state=19,
                              dataset_version_id=dataset_version.dataset_version_id)
    dataset = FoldTensorDataset(records, dataset_version, plan)
    candidate = ModelCandidate("smoke-linear", {"kind": "tiny-linear", "input_features": 4},
                               {"optimizer": "AdamW"})
    fold_id = sorted(set(plan.assignments.values()))[0]
    config = {"epochs": 2, "batch_size": 2, "learning_rate": 0.02,
              "device": "cpu", "deterministic": True, "optimizer": "adamw"}
    experiment_id = digest({"name": "training-smoke", "dataset": dataset_version.dataset_version_id,
                            "fold_plan": plan.fold_plan_id, "candidate": candidate.model_candidate_id,
                            "configuration": config})
    job = TrainingJob(experiment_id, dataset_version.dataset_version_id, plan.fold_plan_id,
                      candidate.model_candidate_id, fold_id, 123,
                      {"cpus": 1, "gpus": 0}, config,
                      validated_leakage_report_id=dataset.leakage_report.report_id)

    by_study = {record.study_id: record for record in records}
    def fold_loader(training_job, train_ids, validation_ids):
        def tensor_dataset(ids):
            selected = [by_study[study_id] for study_id in ids]
            inputs = torch.stack([record.inputs for record in selected])
            targets = torch.tensor([[0.0 if record.labels.values[name] is None
                                     else float(record.labels.values[name]) for name in TARGETS]
                                    for record in selected], dtype=torch.float32)
            masks = torch.tensor([list(record.labels.mask) for record in selected], dtype=torch.bool)
            return TensorDataset(inputs, targets, masks)
        return FoldData(tensor_dataset(train_ids), tensor_dataset(validation_ids),
                        tuple(train_ids), tuple(validation_ids),
                        {"adapter": "FoldTensorDataset", "study_ids": list(by_study)})
    return dataset, candidate, job, fold_loader


def smoke_model_factory(job: TrainingJob):
    import torch
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, len(TARGETS)))


def run_training_smoke(output_dir: str | Path | None = None, *, resume_from: str | Path | None = None,
                       stop_after_optimizer_steps: int | None = None) -> dict[str, Any]:
    dataset, candidate, job, fold_loader = make_smoke_job()
    if output_dir is None:
        if resume_from is not None:
            raise ValueError("--output-dir is required when resuming a smoke checkpoint")
        with tempfile.TemporaryDirectory(prefix="rsna-train-smoke-") as temp:
            return _run(dataset, candidate, job, fold_loader, Path(temp),
                        resume_from=resume_from, stop_after_optimizer_steps=stop_after_optimizer_steps)
    return _run(dataset, candidate, job, fold_loader, Path(output_dir),
                resume_from=resume_from, stop_after_optimizer_steps=stop_after_optimizer_steps)


def build_engine(dataset, candidate, job, fold_loader, directory: Path):
    from .engine import TrainingEngine
    from ..targets import TARGETS
    import torch
    model_factory = smoke_model_factory
    return TrainingEngine(dataset_version=dataset.dataset_version,
        dataset_index=list(dataset.leakage_records), fold_plan=dataset.fold_plan,
        fold_loader=fold_loader, model_factory=model_factory, artifact_dir=directory / "checkpoints",
        event_sink=JsonlTelemetrySink(directory / "training.jsonl"))


def _run(dataset, candidate, job, fold_loader, directory: Path, *, resume_from=None,
         stop_after_optimizer_steps=None) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    result = build_engine(dataset, candidate, job, fold_loader, directory).run(
        job, resume_from=resume_from, stop_after_optimizer_steps=stop_after_optimizer_steps)
    status = "interrupted" if result.provenance.get("failure_state", {}).get("code") == "INTERRUPTED" else result.status
    return {"status": status, "training_job_id": result.training_job_id,
            "dataset_version_id": result.dataset_version_id, "fold_plan_id": result.fold_plan_id,
            "fold_id": result.fold_id, "leakage_report_id": result.provenance.get("validated_leakage_report_id"),
            "checkpoint_uri": result.checkpoint.artifact.uri if result.checkpoint else None,
            "metrics": dict(result.metrics), "failure": result.failure,
            "telemetry_events": len(JsonlTelemetrySink(directory / "training.jsonl").read_events()),
            "synthetic": True}
