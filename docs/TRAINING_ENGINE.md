# Reproducible training engine

The optional `training` extra adds NumPy and PyTorch. Install it with `pip install -e '.[training]'`; metadata-only data commands keep working without it.

## Contracts and data boundary

`TrainingEngine` trains one `TrainingJob` against a canonical `DatasetVersion` and `FoldPlanManifest`. Before loading a model or tensors, it runs the existing strict LeakageGuard, derives the training and validation study IDs from the plan, and checks the job's optional `validated_leakage_report_id`. The `FoldTensorDataset` adapter additionally requires exact study coverage, canonical `TARGETS` order, patient/group agreement, and explicit label masks. Unknown labels are excluded from BCE loss and metrics.

No architecture is bundled. A `FoldLoader` receives the job and the exact train/validation study IDs and returns `FoldData` containing indexable datasets, those same IDs, and serializable provenance. Each sample yields `(inputs, targets, mask)`. A `ModelFactory` receives the job and returns a `torch.nn.Module` that emits `[batch, len(TARGETS)]` logits. The `train run` CLI factory receives both the job and its `ModelCandidate`, so it can build the declared architecture. Optimizer and scheduler factories can be injected; built-in choices are Adam, AdamW, SGD, no scheduler, StepLR, and cosine annealing.

## Configuration

Put effective training settings in `TrainingJob.configuration`, since that mapping is included in job identity and checkpoint compatibility:

```json
{
  "epochs": 5,
  "batch_size": 8,
  "accumulation_steps": 2,
  "learning_rate": 0.0001,
  "optimizer": "adamw",
  "scheduler": "cosine",
  "scheduler_interval": "epoch",
  "device": "auto",
  "amp": true,
  "amp_dtype": "float16",
  "deterministic": true,
  "checkpoint_every_steps": 10,
  "num_workers": 0
}
```

`device=auto` uses CUDA only when the job requests a GPU and CUDA is available. AMP supports CUDA float16/bfloat16 and CPU bfloat16. Resumable loading currently requires `num_workers=0`; this keeps data iteration state explicit. Deterministic algorithms are enabled by default and unsupported nondeterministic operations fail explicitly.

## API

```python
engine = TrainingEngine(
    dataset_version=dataset_version,
    dataset_index=index,
    fold_plan=fold_plan,
    fold_loader=fold_loader,
    model_factory=model_factory,
    artifact_dir="artifacts/training",
    event_sink=JsonlTelemetrySink("artifacts/training/events.jsonl"),
)
result = engine.run(job)
```

The engine returns the existing `TrainingResult` and `CheckpointArtifact` contracts. Checkpoints are written atomically, content addressed by SHA-256, and include model, optimizer, scheduler, scaler, fold order/cursor, lineage/configuration identity, cumulative elapsed time, per-batch step timings, and Python/NumPy/Torch RNG states. Pass the returned artifact to `run(job, resume_from=checkpoint)` to resume; the artifact hash and job/dataset/fold/model lineage are verified before state is loaded. Pass `stop_after_optimizer_steps=N` to exercise a recoverable stop at an optimizer boundary; the result is marked failed with `failure_state.recoverable=true` and includes the persisted checkpoint. Successful result metrics report total examples/second, mean batch-step time, and elapsed training seconds; provenance includes every batch-step and epoch timing plus peak memory when available.

`TelemetryEvent` sinks receive start, epoch, checkpoint, interruption, success, and failure events. `JsonlTelemetrySink` appends canonical JSONL records. The cloud helper `build_cloud_training_spec` produces a declarative job envelope with lineage, resources, input/output URIs, environment, command, and optional resume URI; it intentionally does not submit to a cloud service.

## CLI

`python -m rsna train smoke` runs a tiny synthetic CPU job and prints JSON. It creates temporary checkpoint and telemetry files unless `--output-dir DIR` is supplied. With an output directory, `--stop-after-optimizer-steps 1` prints an interruption result and checkpoint; rerun with the same `--output-dir DIR --resume-from CHECKPOINT` to finish.

`python -m rsna train run --job job.json` runs a declared job with importable factories. The JSON document contains `job` and `candidate` contract data, `dataset_factory` and `model_factory` values in `module:callable` form, and optional `dataset_configuration`, `artifact_dir`, `telemetry_path`, `resume_from`, and `stop_after_optimizer_steps`. The dataset factory is called with `(job, dataset_configuration)` and returns a mapping with `dataset_version`, `dataset_index`, `fold_plan`, and `fold_loader`. The model factory is called with `(job, candidate)` and returns a `torch.nn.Module`. Both are application code and must be importable in the current environment.

Cloud launchers can consume `build_cloud_training_spec`; this package does not provision infrastructure, upload data, or decide an image preprocessing pipeline.
