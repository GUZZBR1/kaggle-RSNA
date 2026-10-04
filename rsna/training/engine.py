"""Model-agnostic, resumable PyTorch training for one canonical FoldPlan fold."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Protocol
from uuid import uuid4

from ..contracts import (ArtifactReference, CheckpointArtifact, DatasetVersion,
                         TrainingJob, TrainingResult)
from ..folds.models import FoldPlanManifest
from ..identity import digest, jsonable
from ..leakage import LeakagePolicy, require_valid_leakage_report, validate_leakage
from ..telemetry.events import TelemetryEvent
from .telemetry import utc_timestamp


class IndexedDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...


@dataclass(frozen=True)
class FoldData:
    """Unversioned adapter result; persisted lineage remains in existing contracts."""

    train: IndexedDataset
    validation: IndexedDataset
    train_study_ids: tuple[str, ...]
    validation_study_ids: tuple[str, ...]
    provenance: Mapping[str, Any]


class FoldLoader(Protocol):
    def __call__(self, job: TrainingJob, train_study_ids: tuple[str, ...],
                 validation_study_ids: tuple[str, ...]) -> FoldData: ...


class ModelFactory(Protocol):
    def __call__(self, job: TrainingJob) -> Any: ...


class TrainingEngine:
    """Train an injected model while binding every result to existing RSNA contracts.

    ``fold_loader`` must return map-style datasets and identify exactly which
    StudyInstanceUIDs it loaded. Checkpoints are written only at optimizer-step
    or epoch boundaries, so gradient accumulation state never needs to be guessed.
    """

    def __init__(self, *, dataset_version: DatasetVersion, dataset_index: Any,
                 fold_plan: FoldPlanManifest, fold_loader: FoldLoader,
                 model_factory: ModelFactory, artifact_dir: str | Path,
                 optimizer_factory: Callable[[Any, Mapping[str, Any]], Any] | None = None,
                 scheduler_factory: Callable[[Any, Mapping[str, Any]], Any] | None = None,
                 event_sink: Callable[[TelemetryEvent], None] | Any | None = None,
                 provider: str = "local-torch"):
        self.dataset_version = dataset_version
        self.dataset_index = dataset_index
        self.fold_plan = fold_plan
        self.fold_loader = fold_loader
        self.model_factory = model_factory
        self.artifact_dir = Path(artifact_dir)
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.event_sink = event_sink
        self.provider = provider

    def run(self, job: TrainingJob, *, resume_from: CheckpointArtifact | str | Path | None = None,
            stop_after_optimizer_steps: int | None = None,
            execution_id: str | None = None) -> TrainingResult:
        execution_id = execution_id or f"torch-{uuid4().hex}"
        provenance: dict[str, Any] = {"engine": "rsna.training.TrainingEngine",
                                      "training_job_id": job.training_job_id}
        latest_checkpoint: CheckpointArtifact | None = None
        phase = "validation"
        runtime: Mapping[str, Any] | None = None
        started = time.perf_counter()
        try:
            self._emit("training.started", job, {"execution_id": execution_id})
            torch, np = _dependencies()
            train_ids, validation_ids, leakage_report_id = self._validate_lineage(job)
            provenance.update({"fold_plan_id": self.fold_plan.fold_plan_id,
                               "dataset_version_id": self.dataset_version.dataset_version_id,
                               "validated_leakage_report_id": leakage_report_id,
                               "train_study_ids": list(train_ids),
                               "validation_study_ids": list(validation_ids),
                               "random_state": job.random_state})
            config = dict(job.configuration)
            _validate_config(config, torch, stop_after_optimizer_steps)
            device = _device(torch, config, job.resources)
            amp_enabled = bool(config.get("amp", False))
            amp_dtype = _amp_dtype(torch, config.get("amp_dtype", "float16"), device, amp_enabled)
            runtime = _runtime_state(torch, np)
            _seed_runtime(torch, np, job.random_state, bool(config.get("deterministic", True)))
            phase = "data"
            fold_data = self.fold_loader(job, train_ids, validation_ids)
            _validate_fold_data(fold_data, train_ids, validation_ids)
            if len(fold_data.train) < 1 or len(fold_data.validation) < 1:
                raise ValueError("training and validation datasets must both be non-empty")
            provenance["data"] = jsonable(fold_data.provenance)
            provenance["device"] = str(device)
            provenance["amp"] = amp_enabled
            provenance["amp_dtype"] = str(amp_dtype).replace("torch.", "") if amp_dtype else None
            provenance["deterministic"] = bool(config.get("deterministic", True))
            provenance["torch_version"] = str(torch.__version__)
            provenance["numpy_version"] = str(np.__version__)
            phase = "model"
            model = self.model_factory(job)
            if not isinstance(model, torch.nn.Module):
                raise TypeError("model_factory must return torch.nn.Module")
            model.to(device)
            optimizer = _make_optimizer(torch, model, config, self.optimizer_factory)
            scheduler = _make_scheduler(torch, optimizer, config, self.scheduler_factory)
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda"
                                          and amp_dtype == torch.float16)
            state = {"epoch": 0, "cursor": 0, "order": None, "global_step": 0,
                     "history": [], "train_loss_sum": 0.0, "train_label_count": 0}
            if resume_from is not None:
                phase = "resume"
                state, latest_checkpoint = self._load_checkpoint(
                    torch, np, resume_from, job, model, optimizer, scheduler, scaler, device)
            provenance["resume_from_checkpoint_id"] = latest_checkpoint.checkpoint_id if latest_checkpoint else None
            provenance["effective_batch_size"] = int(config["batch_size"]) * int(config.get("accumulation_steps", 1))
            phase = "training"
            max_epochs = int(config["epochs"])
            batch_size = int(config["batch_size"])
            accumulation_steps = int(config.get("accumulation_steps", 1))
            checkpoint_every = int(config.get("checkpoint_every_steps", 1))
            clip_norm = config.get("gradient_clip_norm")
            scheduler_interval = config.get("scheduler_interval", "epoch")
            epoch_timings: list[float] = list(state["history"])
            total_started = time.perf_counter()
            self._set_device_metrics(torch, device)

            while int(state["epoch"]) < max_epochs:
                epoch = int(state["epoch"])
                if state["order"] is None:
                    generator = torch.Generator(device="cpu").manual_seed(_epoch_seed(job.random_state, epoch))
                    state["order"] = torch.randperm(len(fold_data.train), generator=generator).tolist()
                    state["cursor"] = 0
                order = list(state["order"])
                if sorted(order) != list(range(len(fold_data.train))):
                    raise ValueError("checkpoint sample order does not match the current training dataset")
                if not 0 <= int(state["cursor"]) <= len(order):
                    raise ValueError("checkpoint batch cursor is outside the current training dataset")
                epoch_started = time.perf_counter()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                pending_batches = 0
                pending_labels = 0
                epoch_loss_sum = 0.0
                epoch_label_count = 0
                batch_starts = list(range(int(state["cursor"]), len(order), batch_size))
                for batch_number, start in enumerate(batch_starts):
                    indices = order[start:start + batch_size]
                    batch = _collate(torch, [fold_data.train[index] for index in indices])
                    inputs, targets, mask = _unpack_batch(
                        torch, batch, device, target_count=len(self.dataset_version.class_names))
                    with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                        enabled=amp_enabled):
                        logits = _logits(model, inputs)
                        loss_sum, label_count = _masked_bce(torch, logits, targets, mask)
                    if not bool(torch.isfinite(loss_sum).all()):
                        raise FloatingPointError("non-finite masked BCE loss")
                    if label_count:
                        scaler.scale(loss_sum).backward()
                        pending_labels += label_count
                        epoch_label_count += label_count
                        epoch_loss_sum += float(loss_sum.detach().float().cpu().item())
                        state["train_label_count"] += label_count
                        state["train_loss_sum"] += float(loss_sum.detach().float().cpu().item())
                    pending_batches += 1
                    state["cursor"] = start + len(indices)
                    flush = pending_batches >= accumulation_steps or batch_number == len(batch_starts) - 1
                    if not flush:
                        continue
                    if pending_labels:
                        scaler.unscale_(optimizer)
                        _normalize_gradients(model, pending_labels)
                        if clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        state["global_step"] = int(state["global_step"]) + 1
                        if scheduler is not None and scheduler_interval == "step":
                            scheduler.step()
                    pending_batches = 0
                    pending_labels = 0
                    if (state["global_step"] and int(state["global_step"]) % checkpoint_every == 0):
                        latest_checkpoint = self._save_checkpoint(
                            torch, np, job, model, optimizer, scheduler, scaler, state, device)
                        self._emit("training.checkpoint_saved", job,
                                   {"checkpoint_id": latest_checkpoint.checkpoint_id,
                                    "global_step": state["global_step"]})
                    if (stop_after_optimizer_steps is not None and
                            int(state["global_step"]) >= stop_after_optimizer_steps):
                        latest_checkpoint = self._save_checkpoint(
                            torch, np, job, model, optimizer, scheduler, scaler, state, device)
                        raise _Interrupted(latest_checkpoint)

                if epoch_label_count == 0 and int(state["epoch"]) == 0 and int(state["global_step"]) == 0:
                    raise ValueError("training fold contains no observed labels")
                phase = "validation"
                validation_loss, validation_count = _evaluate(
                    torch, model, fold_data.validation, batch_size, device, amp_enabled, amp_dtype,
                    target_count=len(self.dataset_version.class_names))
                epoch_seconds = time.perf_counter() - epoch_started
                if scheduler is not None and scheduler_interval == "epoch":
                    scheduler.step()
                epoch_timings.append(epoch_seconds)
                train_loss = epoch_loss_sum / epoch_label_count if epoch_label_count else None
                epoch_record = {"epoch": epoch + 1, "train_loss": train_loss,
                                "validation_loss": validation_loss,
                                "train_observed_labels": epoch_label_count,
                                "validation_observed_labels": validation_count,
                                "seconds": epoch_seconds}
                state["history"] = epoch_timings
                state["epoch"] = epoch + 1
                state["cursor"] = 0
                state["order"] = None
                self._emit("training.epoch_finished", job, epoch_record)
                latest_checkpoint = self._save_checkpoint(
                    torch, np, job, model, optimizer, scheduler, scaler, state, device)
                phase = "training"

            wall_seconds = max(time.perf_counter() - total_started, 1e-9)
            validation_metrics = _evaluate(
                torch, model, fold_data.validation, batch_size, device, amp_enabled, amp_dtype,
                target_count=len(self.dataset_version.class_names))
            validation_loss, validation_count = validation_metrics
            metrics = {"train_loss": (float(state["train_loss_sum"]) /
                                      max(int(state["train_label_count"]), 1)),
                       "global_step": float(state["global_step"]),
                       "validation_observed_labels": float(validation_count),
                       "examples_per_second": float(len(fold_data.train) * max_epochs / wall_seconds),
                       "training_seconds": float(wall_seconds)}
            if validation_loss is not None:
                metrics["validation_loss"] = float(validation_loss)
            peak_memory = self._peak_memory(torch, device)
            provenance.update({"epoch_timings_seconds": epoch_timings,
                               "peak_memory_bytes": peak_memory,
                               "checkpoint_uri": latest_checkpoint.artifact.uri,
                               "checkpoint_sha256": latest_checkpoint.artifact.sha256,
                               "optimizer": str(config.get("optimizer", "adamw")),
                               "scheduler": str(config.get("scheduler", "none")),
                               "accumulation_steps": accumulation_steps,
                               "gradient_clip_norm": clip_norm})
            self._emit("training.succeeded", job, {"global_step": state["global_step"],
                                                     "metrics": metrics})
            _restore_runtime(torch, np, runtime)
            return TrainingResult(job.training_job_id, job.experiment_id,
                job.dataset_version_id, job.fold_plan_id, job.model_candidate_id,
                job.fold_id, "succeeded", metrics, self.provider, execution_id,
                latest_checkpoint, provenance)
        except _Interrupted as interrupted:
            latest_checkpoint = interrupted.checkpoint
            provenance["failure_state"] = {"code": "INTERRUPTED", "phase": phase,
                                           "recoverable": True,
                                           "checkpoint_id": latest_checkpoint.checkpoint_id}
            failure = "TrainingInterrupted: stopped at requested optimizer-step boundary"
            try:
                self._emit("training.interrupted", job,
                           {"checkpoint_id": latest_checkpoint.checkpoint_id})
            except Exception:
                pass
            if runtime is not None:
                _restore_runtime(torch, np, runtime)
            return self._failed_result(job, execution_id, failure, provenance, latest_checkpoint)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            provenance["failure_state"] = {"code": _failure_code(exc), "phase": phase,
                                           "exception_type": type(exc).__name__,
                                           "message": str(exc), "recoverable": latest_checkpoint is not None}
            try:
                self._emit("training.failed", job, provenance["failure_state"])
            except Exception:
                pass
            if runtime is not None:
                _restore_runtime(torch, np, runtime)
            return self._failed_result(job, execution_id, failure, provenance, latest_checkpoint)

    def _validate_lineage(self, job: TrainingJob) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        if job.dataset_version_id != self.dataset_version.dataset_version_id:
            raise ValueError("TrainingJob DatasetVersion does not match the supplied DatasetVersion")
        if job.fold_plan_id != self.fold_plan.fold_plan_id:
            raise ValueError("TrainingJob FoldPlan does not match the supplied canonical FoldPlan")
        if self.fold_plan.dataset_version_id != self.dataset_version.dataset_version_id:
            raise ValueError("FoldPlan DatasetVersion does not match the supplied DatasetVersion")
        if job.fold_id not in self.fold_plan.group_assignments.values():
            raise ValueError(f"TrainingJob fold {job.fold_id!r} is absent from FoldPlan")
        report = validate_leakage(self.dataset_index, assignments=self.fold_plan.assignments,
            policy=LeakagePolicy.STRICT, dataset_version_id=self.dataset_version.dataset_version_id,
            dataset_version=self.dataset_version)
        report_id = require_valid_leakage_report(report)
        if (job.validated_leakage_report_id is not None and
                job.validated_leakage_report_id != report_id):
            raise ValueError("TrainingJob validated_leakage_report_id does not match current LeakageGuard report")
        validation_ids = tuple(sorted(study for study, fold in self.fold_plan.assignments.items()
                                      if fold == job.fold_id))
        train_ids = tuple(sorted(study for study, fold in self.fold_plan.assignments.items()
                                 if fold != job.fold_id))
        if not train_ids or not validation_ids:
            raise ValueError("FoldPlan must provide non-empty train and validation studies")
        return train_ids, validation_ids, report_id

    def _save_checkpoint(self, torch: Any, np: Any, job: TrainingJob, model: Any,
                         optimizer: Any, scheduler: Any, scaler: Any,
                         state: Mapping[str, Any], device: Any) -> CheckpointArtifact:
        payload = {"format_version": 1, "training_job_id": job.training_job_id,
                   "dataset_version_id": job.dataset_version_id,
                   "fold_plan_id": job.fold_plan_id, "model_candidate_id": job.model_candidate_id,
                   "fold_id": job.fold_id, "configuration_sha256": digest(job.configuration),
                   "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                   "scheduler_state": scheduler.state_dict() if scheduler else None,
                   "scaler_state": scaler.state_dict(), "progress": dict(state),
                   "rng_state": _capture_rng(torch, np)}
        folder = self.artifact_dir / job.training_job_id
        folder.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=folder)
        os.close(handle)
        try:
            torch.save(payload, temporary)
            sha = _file_sha256(temporary)
            target = folder / f"{sha}.pt"
            if target.exists():
                os.unlink(temporary)
            else:
                os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        artifact = ArtifactReference(sha, "application/vnd.pytorch.checkpoint", str(target.resolve()), sha,
                                     {"training_job_id": job.training_job_id, "format_version": 1})
        return CheckpointArtifact(artifact, job.model_candidate_id, job.dataset_version_id,
                                  job.fold_plan_id, job.fold_id, job.training_job_id,
                                  "pytorch-state-v1")

    def _load_checkpoint(self, torch: Any, np: Any, source: CheckpointArtifact | str | Path,
                         job: TrainingJob, model: Any, optimizer: Any, scheduler: Any,
                         scaler: Any, device: Any) -> tuple[dict[str, Any], CheckpointArtifact]:
        checkpoint = source if isinstance(source, CheckpointArtifact) else None
        path = Path(checkpoint.artifact.uri if checkpoint else source)
        if checkpoint is not None:
            expected = (job.training_job_id, job.dataset_version_id, job.fold_plan_id,
                        job.model_candidate_id, job.fold_id)
            actual = (checkpoint.training_job_id, checkpoint.dataset_version_id,
                      checkpoint.fold_plan_id, checkpoint.model_candidate_id, checkpoint.fold_id)
            if actual != expected:
                raise ValueError("checkpoint lineage does not match TrainingJob")
            if _file_sha256(path) != checkpoint.artifact.sha256:
                raise ValueError("checkpoint artifact content hash mismatch")
        payload = torch.load(path, map_location=device, weights_only=True)
        for key, expected in (("training_job_id", job.training_job_id),
                              ("dataset_version_id", job.dataset_version_id),
                              ("fold_plan_id", job.fold_plan_id),
                              ("model_candidate_id", job.model_candidate_id),
                              ("fold_id", job.fold_id),
                              ("configuration_sha256", digest(job.configuration))):
            if payload.get(key) != expected:
                raise ValueError(f"checkpoint {key} does not match TrainingJob")
        if payload.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format version")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None:
            if payload.get("scheduler_state") is None:
                raise ValueError("checkpoint is missing scheduler state")
            scheduler.load_state_dict(payload["scheduler_state"])
        elif payload.get("scheduler_state") is not None:
            raise ValueError("checkpoint has scheduler state but current job does not")
        scaler.load_state_dict(payload.get("scaler_state", {}))
        _restore_rng(torch, np, payload["rng_state"])
        if checkpoint is None:
            sha = _file_sha256(path)
            reference = ArtifactReference(sha, "application/vnd.pytorch.checkpoint", str(path.resolve()), sha,
                                          {"training_job_id": job.training_job_id, "format_version": 1})
            checkpoint = CheckpointArtifact(reference, job.model_candidate_id, job.dataset_version_id,
                job.fold_plan_id, job.fold_id, job.training_job_id, "pytorch-state-v1")
        return dict(payload["progress"]), checkpoint

    def _failed_result(self, job: TrainingJob, execution_id: str, failure: str,
                       provenance: Mapping[str, Any], checkpoint: CheckpointArtifact | None) -> TrainingResult:
        return TrainingResult(job.training_job_id, job.experiment_id, job.dataset_version_id,
            job.fold_plan_id, job.model_candidate_id, job.fold_id, "failed", {}, self.provider,
            execution_id, checkpoint, provenance, failure)

    def _emit(self, name: str, job: TrainingJob, attributes: Mapping[str, Any]) -> None:
        if self.event_sink is None:
            return
        event = TelemetryEvent(name, utc_timestamp(), {"training_job_id": job.training_job_id,
            "experiment_id": job.experiment_id, "dataset_version_id": job.dataset_version_id,
            "fold_plan_id": job.fold_plan_id, "model_candidate_id": job.model_candidate_id,
            "fold_id": job.fold_id, **jsonable(attributes)})
        if callable(self.event_sink):
            self.event_sink(event)
        else:
            self.event_sink.emit(event)

    @staticmethod
    def _set_device_metrics(torch: Any, device: Any) -> None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    @staticmethod
    def _peak_memory(torch: Any, device: Any) -> int | None:
        if device.type == "cuda":
            return int(torch.cuda.max_memory_allocated(device))
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(peak * (1 if os.name == "nt" else 1024))
        except (ImportError, AttributeError, OSError):
            return None


def _dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch training requires the optional 'training' dependencies") from exc
    return torch, np


def _validate_config(config: dict[str, Any], torch: Any, stop_after: int | None) -> None:
    defaults = {"epochs": 1, "batch_size": 8, "learning_rate": 1e-3,
                "accumulation_steps": 1, "checkpoint_every_steps": 1,
                "optimizer": "adamw", "scheduler": "none", "scheduler_interval": "epoch",
                "device": "auto", "amp": False, "deterministic": True}
    for key, value in defaults.items():
        config.setdefault(key, value)
    for key in ("epochs", "batch_size", "accumulation_steps", "checkpoint_every_steps"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate",):
        if isinstance(config[key], bool) or not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be a finite positive number")
    if config.get("gradient_clip_norm") is not None:
        value = config["gradient_clip_norm"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("gradient_clip_norm must be a finite positive number")
    if config["optimizer"] not in {"adam", "adamw", "sgd"}:
        raise ValueError("optimizer must be adam, adamw, or sgd")
    if config["scheduler"] not in {"none", "step", "cosine"}:
        raise ValueError("scheduler must be none, step, or cosine")
    if config["scheduler_interval"] not in {"epoch", "step"}:
        raise ValueError("scheduler_interval must be epoch or step")
    if config["device"] not in {"auto", "cpu", "cuda", "cuda:0"}:
        raise ValueError("device must be auto, cpu, cuda, or cuda:0")
    if type(config["amp"]) is not bool or type(config["deterministic"]) is not bool:
        raise ValueError("amp and deterministic must be booleans")
    if config["amp"] and not torch.cuda.is_available() and config["device"] != "cpu":
        raise ValueError("AMP was requested but CUDA is unavailable")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("stop_after_optimizer_steps must be a positive integer")
    if int(config.get("num_workers", 0)) != 0:
        raise ValueError("resumable training currently requires num_workers=0")


def _device(torch: Any, config: Mapping[str, Any], resources: Mapping[str, Any]) -> Any:
    requested = str(config["device"])
    if requested == "auto":
        requested = "cuda:0" if float(resources.get("gpus", 0)) > 0 and torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return torch.device(requested)


def _amp_dtype(torch: Any, name: str, device: Any, enabled: bool) -> Any:
    if not enabled:
        return None
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in dtypes:
        raise ValueError("amp_dtype must be float16 or bfloat16")
    if device.type == "cpu" and name != "bfloat16":
        raise ValueError("CPU AMP supports only bfloat16")
    return dtypes[name]


def _make_optimizer(torch: Any, model: Any, config: Mapping[str, Any], factory: Any) -> Any:
    if factory is not None:
        return factory(model, config)
    name = config["optimizer"]
    options = dict(config.get("optimizer_options", {}))
    options.setdefault("lr", float(config["learning_rate"]))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **options)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **options)
    return torch.optim.AdamW(model.parameters(), **options)


def _make_scheduler(torch: Any, optimizer: Any, config: Mapping[str, Any], factory: Any) -> Any:
    if factory is not None:
        return factory(optimizer, config)
    name = config["scheduler"]
    options = dict(config.get("scheduler_options", {}))
    if name == "step":
        options.setdefault("step_size", 1)
        options.setdefault("gamma", 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, **options)
    if name == "cosine":
        options.setdefault("T_max", int(config["epochs"]))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **options)
    return None


def _validate_fold_data(data: FoldData, expected_train: tuple[str, ...],
                        expected_validation: tuple[str, ...]) -> None:
    if not isinstance(data, FoldData):
        raise TypeError("fold_loader must return FoldData")
    train_ids = tuple(sorted(data.train_study_ids))
    validation_ids = tuple(sorted(data.validation_study_ids))
    if train_ids != expected_train or validation_ids != expected_validation:
        raise ValueError("fold_loader study IDs do not match the canonical FoldPlan boundary")
    if set(train_ids) & set(validation_ids):
        raise ValueError("training and validation studies overlap")


def _collate(torch: Any, samples: Sequence[Any]) -> Any:
    return torch.utils.data.default_collate(samples)


def _unpack_batch(torch: Any, batch: Any, device: Any, *, target_count: int) -> tuple[Any, Any, Any]:
    if isinstance(batch, Mapping):
        if "inputs" not in batch or "targets" not in batch or "mask" not in batch:
            raise ValueError("mapping batches require inputs, targets, and mask")
        inputs, targets, mask = batch["inputs"], batch["targets"], batch["mask"]
    elif isinstance(batch, (tuple, list)) and len(batch) == 3:
        inputs, targets, mask = batch
    else:
        raise ValueError("each batch must be (inputs, targets, mask) or a matching mapping")
    inputs = _to_device(torch, inputs, device)
    targets = torch.as_tensor(targets, device=device, dtype=torch.float32)
    mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
    if targets.shape != mask.shape or targets.ndim != 2:
        raise ValueError("targets and mask must share shape [batch, targets]")
    if targets.shape[1] != target_count:
        raise ValueError(f"batch target dimension {targets.shape[1]} does not match DatasetVersion ({target_count})")
    observed = targets[mask]
    if observed.numel() and (not bool(torch.isfinite(observed).all()) or
                             bool(((observed < 0) | (observed > 1)).any())):
        raise ValueError("observed targets must be finite probabilities in [0, 1]")
    return inputs, targets, mask


def _to_device(torch: Any, value: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _to_device(torch, item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(torch, item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(torch, item, device) for item in value]
    return torch.as_tensor(value, device=device)


def _logits(model: Any, inputs: Any) -> Any:
    output = model(inputs)
    if isinstance(output, Mapping):
        if "logits" not in output:
            raise ValueError("model mapping output must include logits")
        output = output["logits"]
    if not hasattr(output, "shape") or output.ndim != 2:
        raise ValueError("model must return logits with shape [batch, targets]")
    return output.float()


def _masked_bce(torch: Any, logits: Any, targets: Any, mask: Any) -> tuple[Any, int]:
    if logits.shape != targets.shape:
        raise ValueError("model logits and targets must have identical shapes")
    count = int(mask.sum().item())
    per_label = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (per_label * mask).sum(), count


def _normalize_gradients(model: Any, observed_labels: int) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(observed_labels)


def _evaluate(torch: Any, model: Any, dataset: IndexedDataset, batch_size: int,
              device: Any, amp_enabled: bool, amp_dtype: Any, *, target_count: int) -> tuple[float | None, int]:
    previous = model.training
    model.eval()
    total_loss = 0.0
    total_labels = 0
    try:
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                batch = _collate(torch, [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))])
                inputs, targets, mask = _unpack_batch(torch, batch, device, target_count=target_count)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits = _logits(model, inputs)
                    loss, count = _masked_bce(torch, logits, targets, mask)
                if count:
                    if not bool(torch.isfinite(loss).all()):
                        raise FloatingPointError("non-finite validation BCE loss")
                    total_loss += float(loss.float().cpu().item())
                    total_labels += count
    finally:
        model.train(previous)
    return (total_loss / total_labels if total_labels else None), total_labels


def _runtime_state(torch: Any, np: Any) -> dict[str, Any]:
    state = _capture_rng(torch, np)
    state.update({"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "cudnn_deterministic": getattr(torch.backends.cudnn, "deterministic", False),
                  "cudnn_benchmark": getattr(torch.backends.cudnn, "benchmark", False)})
    return state


def _restore_runtime(torch: Any, np: Any, state: Mapping[str, Any]) -> None:
    _restore_rng(torch, np, state)
    torch.use_deterministic_algorithms(state["deterministic_algorithms"])
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = state["cudnn_benchmark"]


def _seed_runtime(torch: Any, np: Any, seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def _capture_rng(torch: Any, np: Any) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"algorithm": numpy_state[0],
                     "keys": torch.as_tensor(numpy_state[1].astype("int64")),
                     "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                     "cached_gaussian": float(numpy_state[4])},
            "torch_cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(torch: Any, np: Any, state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], numpy_state["keys"].cpu().numpy().astype("uint32"),
                         int(numpy_state["position"]), int(numpy_state["has_gauss"]),
                         float(numpy_state["cached_gaussian"])))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("cuda") is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG state does not match available devices")
        torch.cuda.set_rng_state_all(state["cuda"])


def _epoch_seed(seed: int, epoch: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{epoch}".encode()).digest()[:8], "big") % (2**63 - 1)


def _file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _failure_code(exc: Exception) -> str:
    return {"ValueError": "INVALID_CONFIGURATION_OR_LINEAGE",
            "TypeError": "INVALID_ADAPTER_OR_MODEL",
            "FloatingPointError": "NONFINITE_TRAINING_STATE",
            "RuntimeError": "RUNTIME_FAILURE"}.get(type(exc).__name__, "TRAINING_FAILURE")


class _Interrupted(Exception):
    def __init__(self, checkpoint: CheckpointArtifact):
        self.checkpoint = checkpoint
