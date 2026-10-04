"""Five-fold CNN-224 runs that reuse the canonical training and OOF engines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time
import tomllib
from typing import Any, Mapping, Sequence

from ..artifacts import JsonArtifactStore
from ..contracts import (DatasetVersion, ExperimentSpec, ModelCandidate,
                         PredictionArtifact, TrainingJob)
from ..data.manifest import load_manifest
from ..data.selection import SliceSelectionConfig
from ..evaluation.oof import OOF_PREDICTIONS_SCHEMA_VERSION, evaluate_oof
from ..folds.manifest import load_fold_plan
from ..identity import digest
from ..labels import LabelRecord
from ..leakage import LeakagePolicy, require_valid_leakage_report, validate_leakage
from ..targets import TARGETS, TARGET_REGISTRY_ID
from ..training.data import FoldTensorDataset, StudyTensorRecord
from ..training.engine import FoldData, TrainingEngine
from ..training.telemetry import JsonlTelemetrySink
from .model import build_cnn224_model


def _sha256(path: Path) -> str:
    digest_value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(base: Path, value: str) -> Path:
    result = Path(value).expanduser()
    return result if result.is_absolute() else base / result


def _load_series_manifest(path: Path) -> dict[str, str]:
    payload = _json(path)
    if isinstance(payload, dict) and set(payload) == {"studies"}:
        payload = payload["studies"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError("selected_series must be a nonempty StudyInstanceUID-to-SeriesInstanceUID object")
    if any(not isinstance(uid, str) or not uid.strip() for uid in payload):
        raise ValueError("selected_series contains an empty StudyInstanceUID")
    if any(not isinstance(series_uid, str) or not series_uid.strip()
           for series_uid in payload.values()):
        raise ValueError("selected_series contains an empty SeriesInstanceUID")
    return dict(payload)


def _load_labels(path: Path) -> tuple[LabelRecord, ...]:
    payload = _json(path)
    if isinstance(payload, dict):
        payload = payload.get("labels")
    if not isinstance(payload, list):
        raise ValueError("labels must be a list or an object containing a labels list")
    labels = tuple(LabelRecord.from_dict(row) for row in payload)
    if any(label.label_type != "hard" for label in labels):
        raise ValueError("CNN-224 baseline requires hard labels; soft labels are unsupported")
    if len({label.study_id for label in labels}) != len(labels):
        raise ValueError("labels contain duplicate StudyInstanceUID rows")
    return labels


def _tensor_rows(records: Sequence[StudyTensorRecord], ids: Sequence[str]):
    import torch

    by_id = {record.study_id: record for record in records}
    rows = [by_id[uid] for uid in ids]
    inputs = torch.stack([record.inputs for record in rows])
    targets = torch.zeros((len(rows), len(TARGETS)), dtype=torch.float32)
    mask = torch.zeros((len(rows), len(TARGETS)), dtype=torch.bool)
    for row_index, record in enumerate(rows):
        for target_index, target in enumerate(TARGETS):
            value = record.labels.values[target]
            if value is not None:
                targets[row_index, target_index] = float(value)
                mask[row_index, target_index] = True
    return torch.utils.data.TensorDataset(inputs, targets, mask)


def _build_model(model_configuration: Mapping[str, Any], pretrained_path: Path | None = None):
    import torch

    builder_keys = {"variant", "slice_count", "input_size", "channels", "pooling",
                    "pretrained", "output_count"}
    model = build_cnn224_model({key: model_configuration[key]
                                for key in builder_keys if key in model_configuration})
    if bool(model_configuration.get("pretrained", False)):
        if pretrained_path is None or not pretrained_path.is_file():
            raise FileNotFoundError("pretrained=true requires an existing local pretrained_weights file")
        payload = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "model_state" in payload:
            payload = payload["model_state"]
        if not isinstance(payload, dict):
            raise ValueError("pretrained_weights must contain a state dict or model_state mapping")
        model.load_state_dict(payload, strict=True)
    return model


def _run_pipeline(*, dataset_version: DatasetVersion, dataset_index: Any,
                  fold_plan: Any, labels: Sequence[LabelRecord],
                  records: Sequence[StudyTensorRecord], model_configuration: Mapping[str, Any],
                  training_configuration: Mapping[str, Any], output_dir: Path,
                  synthetic: bool, resource_cpus: int, resource_gpus: int,
                  seed: int, pretrained_weights_path: Path | None = None) -> dict[str, Any]:
    import torch

    if tuple(dataset_version.class_names) != TARGETS:
        raise ValueError("DatasetVersion target order must exactly match TARGETS")
    if dataset_version.synthetic is not synthetic:
        raise ValueError("baseline synthetic flag must match DatasetVersion")
    if fold_plan.dataset_version_id != dataset_version.dataset_version_id:
        raise ValueError("FoldPlan DatasetVersion mismatch")
    expected_fold_ids = tuple(f"fold_{index}" for index in range(fold_plan.n_folds))
    if set(fold_plan.assignments.values()) != set(expected_fold_ids):
        raise ValueError("FoldPlan must assign studies to every declared fold")
    if len(expected_fold_ids) != (2 if synthetic else 5):
        raise ValueError("CNN-224 real runs require the canonical five-fold FoldPlan")
    label_by_uid = {item.study_id: item for item in labels}
    if set(label_by_uid) != set(fold_plan.assignments):
        raise ValueError("label StudyInstanceUID coverage does not exactly match the FoldPlan")
    for uid, label in label_by_uid.items():
        if label.patient_id and uid in fold_plan.study_groups and fold_plan.grouping_key == "patient_id":
            if fold_plan.study_groups[uid] != label.patient_id:
                raise ValueError(f"PatientID mismatch for StudyInstanceUID {uid!r}")

    tensor_dataset = FoldTensorDataset(records, dataset_version, fold_plan)
    leakage = validate_leakage(dataset_index, assignments=fold_plan.assignments,
        policy=LeakagePolicy.STRICT, dataset_version_id=dataset_version.dataset_version_id,
        dataset_version=dataset_version)
    leakage_report_id = require_valid_leakage_report(leakage)

    candidate_configuration = dict(model_configuration)
    candidate_configuration.update({
        "target_order": list(TARGETS),
        "target_registry_id": TARGET_REGISTRY_ID,
    })
    implementation_parts = {
        name: _sha256(Path(__file__).with_name(name))
        for name in ("model.py", "data.py", "runner.py")
    }
    candidate = ModelCandidate(
        "cnn224-slice-encoder", candidate_configuration,
        dict(training_configuration), implementation_sha256=digest(implementation_parts),
    )
    experiment = ExperimentSpec(
        name="cnn224-baseline",
        dataset_version_id=dataset_version.dataset_version_id,
        fold_plan_id=fold_plan.fold_plan_id,
        model_candidate_ids=(candidate.model_candidate_id,),
        fold_ids=expected_fold_ids,
        class_names=TARGETS,
        resources={"cpus": resource_cpus, "gpus": resource_gpus},
        training_configuration=dict(training_configuration),
        evaluation_policy={"primary_metric": "macro_auc", "engine": "rsna.evaluation.oof"},
        synthetic=synthetic,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_store = JsonArtifactStore(output_dir / "predictions")
    fold_reports: list[dict[str, Any]] = []
    prediction_artifacts: list[PredictionArtifact] = []
    total_started = time.perf_counter()

    for fold_index, fold_id in enumerate(expected_fold_ids):
        train_ids = tuple(sorted(uid for uid, assigned in fold_plan.assignments.items()
                                 if assigned != fold_id))
        validation_ids = tuple(sorted(uid for uid, assigned in fold_plan.assignments.items()
                                      if assigned == fold_id))
        if not train_ids or not validation_ids:
            raise ValueError(f"FoldPlan fold {fold_id!r} has an empty training or validation split")

        fold_training = dict(training_configuration)
        fold_seed = seed + fold_index
        resources = {"cpus": resource_cpus, "gpus": resource_gpus}
        job = TrainingJob(
            experiment.experiment_id, dataset_version.dataset_version_id,
            fold_plan.fold_plan_id, candidate.model_candidate_id, fold_id,
            fold_seed, resources, fold_training,
            validated_leakage_report_id=leakage_report_id,
        )

        def fold_loader(_job, train_study_ids, validation_study_ids):
            return FoldData(
                _tensor_rows(records, train_study_ids),
                _tensor_rows(records, validation_study_ids),
                tuple(train_study_ids), tuple(validation_study_ids),
                {"adapter": "FoldTensorDataset", "fold_tensor_dataset_id": digest({
                    "study_ids": [record.study_id for record in tensor_dataset.records],
                    "fold_plan_id": fold_plan.fold_plan_id,
                })},
            )

        checkpoint_dir = output_dir / "checkpoints"
        telemetry_path = output_dir / "telemetry" / f"{fold_id}.jsonl"
        engine = TrainingEngine(
            dataset_version=dataset_version, dataset_index=dataset_index,
            fold_plan=fold_plan, fold_loader=fold_loader,
            model_factory=lambda _job: _build_model(candidate.model_configuration,
                                                      pretrained_weights_path),
            artifact_dir=checkpoint_dir,
            event_sink=JsonlTelemetrySink(telemetry_path),
        )
        train_started = time.perf_counter()
        result = engine.run(job)
        training_seconds = time.perf_counter() - train_started
        if result.status != "succeeded" or result.checkpoint is None:
            raise RuntimeError(f"training failed for {fold_id}: {result.failure}")

        checkpoint_path = Path(result.checkpoint.artifact.uri)
        if _sha256(checkpoint_path) != result.checkpoint.artifact.sha256:
            raise ValueError(f"checkpoint integrity failure for {fold_id}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for name, expected in (("training_job_id", job.training_job_id),
                               ("dataset_version_id", dataset_version.dataset_version_id),
                               ("fold_plan_id", fold_plan.fold_plan_id),
                               ("model_candidate_id", candidate.model_candidate_id),
                               ("fold_id", fold_id)):
            if payload.get(name) != expected:
                raise ValueError(f"checkpoint {name} mismatch for {fold_id}")
        model = _build_model(candidate.model_configuration, pretrained_weights_path)
        model.load_state_dict(payload["model_state"])
        requested_device = str(training_configuration.get("device", "auto"))
        use_cuda = (requested_device.startswith("cuda") or
                    (requested_device == "auto" and resource_gpus > 0 and torch.cuda.is_available()))
        device = torch.device("cuda:0" if use_cuda else "cpu")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested for inference but unavailable")
        model.to(device)
        model.eval()
        validation_records = {record.study_id: record for record in records}
        inference_started = time.perf_counter()
        prediction_rows: list[dict[str, Any]] = []
        batch_size = int(training_configuration.get("batch_size", 8))
        with torch.inference_mode():
            for start in range(0, len(validation_ids), batch_size):
                batch_ids = validation_ids[start:start + batch_size]
                inputs = torch.stack([validation_records[uid].inputs for uid in batch_ids]).to(device)
                probabilities = torch.sigmoid(model(inputs)).cpu()
                if not bool(torch.isfinite(probabilities).all()) or bool(
                        ((probabilities < 0) | (probabilities > 1)).any()):
                    raise ValueError(f"invalid prediction probabilities for {fold_id}")
                for uid, scores in zip(batch_ids, probabilities.tolist(), strict=True):
                    prediction_rows.append({"StudyInstanceUID": uid, "scores": scores})
        inference_seconds = max(time.perf_counter() - inference_started, 1e-12)
        if [row["StudyInstanceUID"] for row in prediction_rows] != list(validation_ids):
            raise ValueError(f"validation prediction coverage/order mismatch for {fold_id}")
        prediction_ref = prediction_store.put_json({
            "schema_version": OOF_PREDICTIONS_SCHEMA_VERSION,
            "class_names": list(TARGETS), "rows": prediction_rows,
        })
        prediction_artifacts.append(PredictionArtifact(
            prediction_ref, "oof", candidate.model_candidate_id,
            dataset_version.dataset_version_id, result.checkpoint.checkpoint_id,
            TARGETS, len(prediction_rows), fold_plan.fold_plan_id, fold_id,
            synthetic=synthetic, target_schema_version=dataset_version.target_schema_version,
            provenance="cnn224_validation_inference",
        ))
        global_steps = int(result.metrics.get("global_step", 0))
        fold_reports.append({
            "fold_id": fold_id,
            "batch_size": int(training_configuration.get("batch_size", 8)),
            "configured_epochs": int(training_configuration.get("epochs", 1)),
            "training_job_id": job.training_job_id,
            "checkpoint_id": result.checkpoint.checkpoint_id,
            "prediction_artifact_id": prediction_artifacts[-1].prediction_artifact_id,
            "train_studies": len(train_ids), "validation_studies": len(validation_ids),
            "epochs_executed": len(result.provenance.get("epoch_timings_seconds", ())),
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "samples_per_second": len(validation_ids) / inference_seconds,
            "steps_per_second": global_steps / training_seconds if training_seconds else 0.0,
            "global_steps": global_steps,
            "peak_vram_bytes": result.provenance.get("peak_memory_bytes")
                if str(training_configuration.get("device", "auto")).startswith("cuda") else None,
        })

    evaluation_store = JsonArtifactStore(output_dir / "evaluation")
    oof = evaluate_oof(experiment, fold_plan, labels, prediction_artifacts, evaluation_store)
    elapsed = time.perf_counter() - total_started
    return {
        "status": "completed", "synthetic": synthetic,
        "dataset_version_id": dataset_version.dataset_version_id,
        "dataset_index_id": getattr(dataset_index, "index_id", None),
        "fold_plan_id": fold_plan.fold_plan_id,
        "model_candidate": candidate.to_dict(),
        "experiment": experiment.to_dict(),
        "folds": fold_reports,
        "runtime": {"total_seconds": elapsed,
                    "training_seconds": sum(item["training_seconds"] for item in fold_reports),
                    "inference_seconds": sum(item["inference_seconds"] for item in fold_reports)},
        "oof": oof,
    }


def run_cnn224(config_path: str | Path) -> dict[str, Any]:
    """Run all canonical folds using a TOML config and explicit series curation."""
    run_started = time.perf_counter()
    config_file = Path(config_path).expanduser().resolve()
    config = tomllib.loads(config_file.read_text(encoding="utf-8"))
    paths = config["inputs"]
    base = config_file.parent
    dataset_version = DatasetVersion.from_dict(_json(_resolve(base, paths["dataset_version"])))
    dataset_index = load_manifest(_resolve(base, paths["dataset_index"]))
    if (dataset_version.dataset_index_artifact_id is not None
            and dataset_version.dataset_index_artifact_id != dataset_index.index_id):
        raise ValueError("DatasetIndex identity does not match DatasetVersion")
    fold_plan = load_fold_plan(_resolve(base, paths["fold_plan"]),
                               dataset_version_id=dataset_version.dataset_version_id,
                               dataset=dataset_index, policy="strict")
    labels = _load_labels(_resolve(base, paths["labels"]))
    selected_series = _load_series_manifest(_resolve(base, paths["selected_series"]))
    from .data import load_study_tensors

    model = config["model"]
    training = dict(config["training"])
    training.setdefault("seed", fold_plan.random_state)
    slice_count = int(model.get("slice_count", 24))
    input_size = int(model.get("input_size", 224))
    if slice_count != 24 or input_size != 224:
        raise ValueError("real CNN-224 runs require input_size=224 and slice_count=24")
    if set(selected_series) != set(fold_plan.assignments):
        raise ValueError("selected_series must contain exactly every FoldPlan StudyInstanceUID")
    selected_series_identity = digest(selected_series)
    preprocessing_spec = {
        "image_decoder": "pydicom.pixel_array-v1",
        "slice_selection": SliceSelectionConfig(
            strategy="uniform", count=slice_count,
            short_series_policy="strict",
        ).to_preprocessing_spec(),
        "series_manifest_sha256": selected_series_identity,
        "normalization": "per-study-percentile-1-99-minmax-v1",
        "resize": {"size": input_size, "mode": "bilinear", "align_corners": False},
        "channels": 1,
    }
    model_configuration = {
        "architecture": "cnn224_slice_encoder_v1",
        "variant": "cnn224",
        "input_size": input_size,
        "slice_count": slice_count,
        "pooling": model.get("pooling", "mean"),
        "channels": list(model.get("channels", [16, 32, 64])),
        "channel_policy": "single_channel_grayscale",
        "head": "linear_12_logits",
        "pretrained": bool(model.get("pretrained", False)),
        "preprocessing": preprocessing_spec,
    }
    pretrained_weights_path = None
    if model_configuration["pretrained"]:
        weights_path = _resolve(base, model["pretrained_weights"])
        if not weights_path.is_file():
            raise FileNotFoundError("pretrained=true requires a local pretrained_weights file")
        model_configuration["pretrained_weights_sha256"] = _sha256(weights_path)
        pretrained_weights_path = weights_path
    output_dir = _resolve(base, config.get("output", {}).get("output_dir", "artifacts/cnn224"))
    preprocessing_started = time.perf_counter()
    tensor_cache_identity = digest({
        "dataset_version_id": dataset_version.dataset_version_id,
        "dataset_index_id": dataset_index.index_id,
        "series_manifest_sha256": selected_series_identity,
        "preprocessing": preprocessing_spec,
        "data_adapter_sha256": _sha256(Path(__file__).with_name("data.py")),
    })
    tensor_store_path = output_dir / "preprocessing" / f"{tensor_cache_identity}.npy"
    records = load_study_tensors(
        dataset_index, _resolve(base, paths["dataset_root"]), selected_series,
        labels, slice_count=slice_count, input_size=input_size,
        tensor_store_path=tensor_store_path,
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    result = _run_pipeline(dataset_version=dataset_version, dataset_index=dataset_index,
        fold_plan=fold_plan, labels=labels, records=records,
        model_configuration=model_configuration, training_configuration=training,
        output_dir=output_dir, synthetic=False,
        resource_cpus=int(config.get("resources", {}).get("cpus", 4)),
        resource_gpus=int(config.get("resources", {}).get("gpus", 1)),
        seed=int(training.get("seed", fold_plan.random_state)),
        pretrained_weights_path=pretrained_weights_path)
    result["runtime"]["data_preprocessing_seconds"] = preprocessing_seconds
    result["runtime"]["total_seconds"] = time.perf_counter() - run_started
    result["preprocessed_tensor_store"] = str(tensor_store_path)
    return result


def run_cnn224_smoke(output_dir: str | Path | None = None) -> dict[str, Any]:
    """CPU-only synthetic 2-fold smoke through model, engine, checkpoint, OOF and AUC."""
    import torch

    from ..contracts import DatasetVersion
    from ..folds.generate import generate_fold_plan

    dataset_version = DatasetVersion(
        "cnn224-tiny-smoke", "1", "a" * 64, "cnn224-tiny-smoke-v1", TARGETS,
        preprocessing={"cnn224": {"input_size": 24, "slice_count": 2,
            "normalization": "synthetic-unit-values", "channels": 1}}, synthetic=True,
    )
    fold_inputs = []
    labels: list[LabelRecord] = []
    records: list[StudyTensorRecord] = []
    generator = torch.Generator(device="cpu").manual_seed(734)
    for index in range(8):
        uid, patient_id = f"smoke-study-{index:02}", f"smoke-patient-{index:02}"
        fold_inputs.append({"study_id": uid, "patient_id": patient_id})
        values = {target: (None if index == target_index % 8 else index % 2)
                  for target_index, target in enumerate(TARGETS)}
        label = LabelRecord(uid, values, "manual_review", allow_partial=True,
                            patient_id=patient_id)
        labels.append(label)
        inputs = torch.rand((2, 1, 24, 24), generator=generator)
        records.append(StudyTensorRecord(uid, patient_id, inputs, label,
                                         series_ids=(f"series-{index:02}",),
                                         sop_ids=(f"slice-{index:02}-0", f"slice-{index:02}-1")))
    fold_plan = generate_fold_plan(fold_inputs, n_folds=2, random_state=27,
                                   dataset_version_id=dataset_version.dataset_version_id)
    tensor_dataset = FoldTensorDataset(records, dataset_version, fold_plan)
    model_configuration = {
        "architecture": "cnn224_slice_encoder_v1", "variant": "tiny", "input_size": 24,
        "slice_count": 2, "pooling": "mean", "channels": [2, 4],
        "channel_policy": "single_channel_grayscale", "head": "linear_12_logits",
        "pretrained": False,
        "preprocessing": dict(dataset_version.preprocessing)["cnn224"],
    }
    training = {"epochs": 1, "batch_size": 2, "learning_rate": 0.001,
                "optimizer": "adamw", "optimizer_options": {"weight_decay": 0.0001},
                "scheduler": "none", "device": "cpu", "amp": False,
                "deterministic": True, "num_workers": 0}
    destination = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="rsna-cnn224-smoke-"))
    result = _run_pipeline(
        dataset_version=dataset_version, dataset_index=tensor_dataset.leakage_records,
        fold_plan=fold_plan, labels=labels, records=records,
        model_configuration=model_configuration, training_configuration=training,
        output_dir=destination, synthetic=True, resource_cpus=1, resource_gpus=0,
        seed=27,
    )
    if len(result["folds"]) != 2 or len(result["oof"]["evaluations"]) != 1:
        raise RuntimeError("tiny CNN-224 smoke did not execute the complete OOF path")
    if any(row["validation_studies"] != 4 for row in result["folds"]):
        raise RuntimeError("tiny CNN-224 smoke did not validate each held-out fold")
    result["status"] = "succeeded"
    return result
