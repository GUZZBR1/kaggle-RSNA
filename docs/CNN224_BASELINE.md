# CNN-224 baseline

The baseline trains one study-level model across the existing canonical five-fold `FoldPlan`, reloads each fold checkpoint for validation-only inference, writes canonical `PredictionArtifact`s, then delegates OOF aggregation and AUC to `rsna.evaluation.oof`. It does not create replacement folds or a parallel training loop.

## Model and input policy

The current training extra provides PyTorch but not `torchvision`; this implementation therefore uses a compact `torch.nn` shared 2D slice encoder instead of depending on EfficientNet or ResNet packages. Each grayscale slice passes through the same three convolution blocks. Feature vectors are mean pooled across slices and a linear head returns 12 logits. Optional `pooling = "mean_max"` concatenates mean and max pooled features. The input contract is `[batch, 24, 1, 224, 224]` for real runs. The channel policy is explicitly one grayscale channel.

The real run requires `selected-series.json`, a mapping from every `StudyInstanceUID` in the FoldPlan to exactly one `SeriesInstanceUID`. This prevents mixing distinct MRI sequences within a study. The existing `SliceSelector` orders the chosen series and samples 24 evenly spaced slices; series with fewer than 24 slices fail. Pixel decoding uses `pydicom.pixel_array`. Intensities are clipped to the pooled 1st and 99th percentiles within each study, scaled to `[0,1]`, then resized by bilinear interpolation to 224×224. The series map hash and preprocessing choices enter the `ModelCandidate` identity; filesystem paths do not.

`pretrained` defaults to `false`, so smoke and real runs never download weights. If `pretrained = true`, `pretrained_weights` must name a local state dict compatible with this CNN architecture. Its SHA-256 enters candidate identity. No ImageNet model weights are bundled: they are not available in the current PyTorch-only environment.

Missing labels remain masked and are handled by the Training Engine's masked BCE-with-logits loss. Soft labels are rejected. Inference applies sigmoid once and checks finite probabilities in `[0,1]`. Validation predictions are keyed by `StudyInstanceUID`; the OOF engine verifies fold ownership and complete exactly-once coverage.

## Configuration and commands

Copy [configs/models/cnn224-baseline.toml](../configs/models/cnn224-baseline.toml), update its paths to the existing serialized DatasetVersion, DatasetIndex, five-fold plan, labels, raw DICOM root, and explicit series map, then run:

```text
python -m rsna baseline cnn224 run --config configs/models/cnn224-baseline.toml --json
```

The runner checks that the DatasetIndex identity matches the DatasetVersion when it is bound, validates the supplied FoldPlan against the index, runs LeakageGuard before training, and fails if any input or fold is missing or mismatched. It uses the default target registry order and the existing `TrainingEngine`, which owns deterministic seeds, masked BCE, AMP, accumulation, clipping, checkpointing, telemetry, and step timings. It records training/inference time, throughput, steps per second, and peak CUDA memory when training on CUDA. Output artifacts include fold checkpoints, telemetry, per-fold OOF prediction artifacts, and the OOF Evaluation Engine's aggregate, target table, comparison report, and `Evaluation` contract.

The tiny CPU end-to-end smoke uses the same model interface and the existing Training Engine, generates two synthetic folds, reloads both checkpoints, emits OOF artifacts and validates all 12 AUCs without a GPU or RSNA files:

```text
python -m rsna baseline cnn224 smoke --json
```

## Limits

This smoke validates pipeline wiring and synthetic metrics; it does not establish a real competition score. No real five-fold GPU run was performed in this environment. Compressed DICOM transfer syntaxes may require a pydicom pixel decoder plugin in the runtime environment; decoding failures stop the run with the affected indexed slice path.
