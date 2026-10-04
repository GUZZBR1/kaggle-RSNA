"""Small shared-slice CNNs for the 224-pixel RSNA baseline.

Each study is represented as ``[slice_count, 1, height, width]`` and a batch as
``[batch, slice_count, 1, height, width]``. The same encoder is applied to
every slice before deterministic pooling across slices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised only without optional torch
    raise RuntimeError("baseline CNN requires the optional 'torch' dependency") from exc


@dataclass(frozen=True)
class CNN224Config:
    """Architecture settings shared by the full and tiny variants."""

    variant: Literal["cnn224", "tiny"] = "cnn224"
    slice_count: int = 24
    input_size: int = 224
    channels: tuple[int, ...] = (16, 32, 64)
    pooling: Literal["mean", "mean_max"] = "mean"
    pretrained: bool = False
    output_count: int = 12

    def __post_init__(self) -> None:
        if self.variant not in {"cnn224", "tiny"}:
            raise ValueError("variant must be 'cnn224' or 'tiny'")
        if type(self.slice_count) is not int or self.slice_count < 1:
            raise ValueError("slice_count must be a positive integer")
        if type(self.input_size) is not int or self.input_size < 16:
            raise ValueError("input_size must be an integer of at least 16")
        if not self.channels or any(type(channel) is not int or channel < 1 for channel in self.channels):
            raise ValueError("channels must contain positive integers")
        if self.pooling not in {"mean", "mean_max"}:
            raise ValueError("pooling must be 'mean' or 'mean_max'")
        if type(self.pretrained) is not bool:
            raise ValueError("pretrained must be a boolean")
        if type(self.output_count) is not int or self.output_count != 12:
            raise ValueError("output_count must be 12 for the canonical RSNA targets")

    @classmethod
    def tiny(cls, **overrides: object) -> "CNN224Config":
        """Construct a compact model using the same input/output contract."""
        values: dict[str, object] = {"variant": "tiny", "channels": (4, 8)}
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]


class _SliceEncoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.extend((
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))
            in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.layers = nn.Sequential(*layers)
        self.feature_count = channels[-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images).flatten(1)


class CNN224Model(nn.Module):
    """A shared 2D slice encoder with study-level slice pooling and 12 logits."""

    def __init__(self, config: CNN224Config | None = None):
        super().__init__()
        self.config = config or CNN224Config()
        self.encoder = _SliceEncoder(self.config.channels)
        multiplier = 2 if self.config.pooling == "mean_max" else 1
        self.classifier = nn.Linear(self.encoder.feature_count * multiplier,
                                    self.config.output_count)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim != 5:
            raise ValueError("inputs must have shape [batch, slices, 1, height, width]")
        batch, slices, channels, height, width = inputs.shape
        if slices != self.config.slice_count:
            raise ValueError(f"expected {self.config.slice_count} slices, got {slices}")
        if channels != 1:
            raise ValueError("CNN224 accepts exactly one grayscale channel")
        if height != self.config.input_size or width != self.config.input_size:
            raise ValueError(f"expected {self.config.input_size}x{self.config.input_size} slices")
        features = self.encoder(inputs.reshape(batch * slices, channels, height, width))
        features = features.reshape(batch, slices, -1)
        mean_features = features.mean(dim=1)
        if self.config.pooling == "mean_max":
            features = torch.cat((mean_features, features.amax(dim=1)), dim=1)
        else:
            features = mean_features
        return self.classifier(features)


def build_model(config: CNN224Config | None = None) -> CNN224Model:
    """Build either the standard CNN-224 or its smaller ``tiny`` variant."""
    return CNN224Model(config)


def build_cnn224_model(configuration: Mapping[str, Any]) -> nn.Module:
    """Build a model from the training adapter's JSON-like configuration."""
    if not isinstance(configuration, Mapping):
        raise TypeError("configuration must be a mapping")
    values = dict(configuration)
    if "target_count" in values:
        if "output_count" in values and values["output_count"] != values["target_count"]:
            raise ValueError("target_count and output_count disagree")
        values["output_count"] = values.pop("target_count")
    allowed = set(CNN224Config.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unsupported CNN-224 configuration keys: {sorted(unknown)}")
    if isinstance(values.get("channels"), list):
        values["channels"] = tuple(values["channels"])
    config = CNN224Config(**values)
    return CNN224Model(config)
