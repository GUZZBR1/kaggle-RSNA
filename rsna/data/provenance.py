"""Shared configuration and JSON-safe provenance primitives for DICOM audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

Confidence = str
OrientationConsistency = str


@dataclass(frozen=True)
class OrientationConfig:
    plane_tolerance_deg: float = 15.0
    orientation_consistency_tolerance_deg: float = 1.0
    minor_variation_deg: float = 0.25
    laterality_text_fallback: bool = True
    normalization_mode: str = "preserve_native"
    conflict_policy: str = "ambiguous"

    def __post_init__(self) -> None:
        for name in ("plane_tolerance_deg", "orientation_consistency_tolerance_deg", "minor_variation_deg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 90:
                raise ValueError(f"{name} must be between 0 and 90 degrees")
        if self.normalization_mode not in {"preserve_native", "left_canonical"}:
            raise ValueError("normalization_mode must be preserve_native or left_canonical")
        if self.conflict_policy not in {"ambiguous", "unknown"}:
            raise ValueError("conflict_policy must be ambiguous or unknown")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


@dataclass(frozen=True)
class OrientationProvenance:
    original_orientation: Any
    resolved_plane: str
    orientation: Any
    original_laterality: Any
    laterality: Any
    normalization_plan: Any
    confidence: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


def build_orientation_provenance(series: Any, config: OrientationConfig | None = None) -> OrientationProvenance:
    """Return one JSON-ready audit record without reading or changing pixel data."""
    from .orientation import describe_series_orientation
    from .laterality import resolve_series_laterality
    from .normalization import build_normalization_plan

    config = config or OrientationConfig()
    slices = list(series) if isinstance(series, (list, tuple)) else [series]
    orientation = describe_series_orientation(series, config)
    laterality = resolve_series_laterality(series, config)
    plan = build_normalization_plan(series, orientation=orientation, laterality=laterality, config=config)
    orientation_raw = [_raw_fields(item, ("ImageOrientationPatient",)) for item in slices]
    laterality_raw = [_raw_fields(item, ("Laterality", "ImageLaterality", "SeriesDescription",
                                           "ProtocolName", "StudyDescription")) for item in slices]
    warnings = tuple(orientation.warnings) + tuple(laterality.warnings) + tuple(plan.warnings)
    confidence = "unknown" if laterality.confidence == "unknown" else plan.confidence
    return OrientationProvenance(orientation_raw, orientation.plane, orientation.to_dict(),
                                 laterality_raw, laterality.to_dict(), plan.to_dict(), confidence, warnings)


def _raw_fields(item: Any, names: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {}
    return {name: item[name] for name in names if name in item}
