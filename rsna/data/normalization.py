"""Declarative normalization plans; this module never operates on image pixels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .provenance import OrientationConfig, to_jsonable


@dataclass(frozen=True)
class NormalizationPlan:
    mode: str
    target_laterality: str
    source_laterality: str
    source_plane: str
    requires_flip: bool
    flip_axis: str | None
    requires_rotation: bool
    rotation_degrees: float | None
    confidence: str
    evidence: tuple[Any, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


def _field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def build_normalization_plan(series: Any, *, orientation: Any = None, laterality: Any = None,
                             config: OrientationConfig | None = None) -> NormalizationPlan:
    """Describe a future transform from evidence; never decode or modify pixels."""
    from .orientation import describe_series_orientation
    from .laterality import resolve_series_laterality

    config = config or OrientationConfig()
    orientation = orientation or describe_series_orientation(series, config)
    laterality = laterality or resolve_series_laterality(series, config)
    source_side = _field(laterality, "resolved", "UNKNOWN")
    plane = _field(orientation, "plane", "unknown")
    if _field(laterality, "confidence") == "unknown" or _field(orientation, "confidence") == "unknown":
        confidence = "unknown"
    elif _field(laterality, "confidence") == "high" and _field(orientation, "confidence") == "high":
        confidence = "high"
    else:
        confidence = "low"
    warnings = list(_field(laterality, "warnings", ()) or ()) + list(_field(orientation, "warnings", ()) or ())
    evidence = list(_field(laterality, "evidence", ()) or ()) + list(_field(orientation, "evidence", ()) or ())
    if config.normalization_mode == "preserve_native":
        return NormalizationPlan(config.normalization_mode, "NATIVE", source_side, plane, False, None,
                                 False, None, confidence, tuple(evidence), tuple(warnings))
    if source_side not in {"LEFT", "RIGHT"}:
        warnings.append("canonical normalization not planned because laterality is not uniquely left or right")
        return NormalizationPlan(config.normalization_mode, "LEFT", source_side, plane, False, None,
                                 False, None, "unknown", tuple(evidence), tuple(warnings))
    flip = source_side == "RIGHT"
    if plane == "oblique":
        warnings.append("oblique geometry needs a separate resampling decision; no rotation is planned")
    if flip:
        # Patient left-right direction may map to row, column, or slice ordering;
        # without pixel geometry it is unsafe to name a pixel-array axis.
        warnings.append("a left-right reflection is planned; pixel axis is unresolved without full geometry")
    return NormalizationPlan(config.normalization_mode, "LEFT", source_side, plane, flip,
                             "unresolved" if flip else None, False, None,
                             confidence if not flip or _field(orientation, "normal") is not None else "low",
                             tuple(evidence), tuple(warnings))
