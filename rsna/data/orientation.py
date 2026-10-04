"""Auditable DICOM orientation and anatomical-plane descriptions.

DICOM ImageOrientationPatient stores direction cosines for the image rows and
columns in the patient coordinate system (x=L, y=P, z=H for BIPED patients).
Their cross product is the slice normal. Plane classification uses the absolute
normal components, so reversing slice order does not change the plane label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from .provenance import Confidence, OrientationConsistency, OrientationConfig


@dataclass(frozen=True)
class OrientationDescriptor:
    row_cosines: tuple[float, float, float] | None
    column_cosines: tuple[float, float, float] | None
    normal: tuple[float, float, float] | None
    plane: str
    obliquity_degrees: float | None
    confidence: str
    consistency: str = "unknown"
    max_slice_deviation_degrees: float | None = None
    warnings: tuple[str, ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numbers(value: Any) -> tuple[float, ...] | None:
    if isinstance(value, str):
        parts = value.replace(",", "\\").split("\\")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = value
    else:
        return None
    try:
        values = tuple(float(part) for part in parts)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in values):
        return None
    return values


def _unit(vector: Sequence[float], *, tolerance: float = 1e-4) -> tuple[float, float, float] | None:
    length = math.sqrt(sum(item * item for item in vector))
    if length == 0 or abs(length - 1.0) > tolerance:
        return None
    return tuple(item / length for item in vector)  # type: ignore[return-value]


def _valid_orientation(value: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    numbers = _numbers(value)
    if numbers is None or len(numbers) != 6:
        return None
    row, column = _unit(numbers[:3]), _unit(numbers[3:])
    if row is None or column is None or abs(sum(a*b for a, b in zip(row, column))) > 1e-4:
        return None
    normal_raw = (row[1]*column[2]-row[2]*column[1],
                  row[2]*column[0]-row[0]*column[2],
                  row[0]*column[1]-row[1]*column[0])
    normal = _unit(normal_raw)
    return (row, column) if normal is not None else None


def _plane(normal: tuple[float, float, float], tolerance_deg: float) -> tuple[str, float]:
    magnitudes = tuple(abs(value) for value in normal)
    axis = max(range(3), key=magnitudes.__getitem__)
    obliquity = math.degrees(math.acos(max(-1.0, min(1.0, magnitudes[axis]))))
    if obliquity > tolerance_deg:
        return "oblique", obliquity
    return ("sagittal", "coronal", "axial")[axis], obliquity


def assess_orientation_consistency(orientations: Sequence[Any], *, tolerance_deg: float = 1.0,
                                   minor_variation_deg: float = 0.25) -> tuple[str, float | None, tuple[str, ...]]:
    if (isinstance(tolerance_deg, bool) or not isinstance(tolerance_deg, (int, float))
            or isinstance(minor_variation_deg, bool) or not isinstance(minor_variation_deg, (int, float))
            or not math.isfinite(tolerance_deg) or not math.isfinite(minor_variation_deg)
            or tolerance_deg < 0 or minor_variation_deg < 0 or minor_variation_deg > tolerance_deg):
        raise ValueError("consistency tolerances must satisfy 0 <= minor_variation_deg <= tolerance_deg")
    parsed = [_valid_orientation(value) for value in orientations]
    valid = [item for item in parsed if item is not None]
    if not valid:
        return "unknown", None, ("missing or invalid ImageOrientationPatient",)
    if len(valid) != len(parsed):
        return "inconsistent", None, ("some slice orientations are missing or invalid",)
    reference = valid[0]
    reference_normal = (reference[0][1]*reference[1][2]-reference[0][2]*reference[1][1],
                        reference[0][2]*reference[1][0]-reference[0][0]*reference[1][2],
                        reference[0][0]*reference[1][1]-reference[0][1]*reference[1][0])
    deviations = []
    for row, col in valid[1:]:
        normal = (row[1]*col[2]-row[2]*col[1], row[2]*col[0]-row[0]*col[2], row[0]*col[1]-row[1]*col[0])
        # Compare both in-plane axes as well as the signed normal. Normal-only
        # comparison would miss slices whose pixel grid rotates within the plane.
        for reference_axis, current_axis in zip((*reference, reference_normal), (row, col, normal)):
            cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(reference_axis, current_axis))))
            deviations.append(math.degrees(math.acos(cosine)))
    maximum = max(deviations, default=0.0)
    if maximum > tolerance_deg:
        return "inconsistent", maximum, ("slice orientation deviation exceeds tolerance",)
    if maximum > minor_variation_deg:
        return "minor_variation", maximum, ()
    return "consistent", maximum, ()


def describe_series_orientation(series: Mapping[str, Any] | Sequence[Mapping[str, Any]],
                                config: OrientationConfig | None = None) -> OrientationDescriptor:
    config = config or OrientationConfig()
    slices = list(series) if isinstance(series, Sequence) and not isinstance(series, (str, bytes, Mapping)) else [series]
    values = [item.get("ImageOrientationPatient") if isinstance(item, Mapping) else None for item in slices]
    warnings: list[str] = []
    first = None
    parsed = None
    for value in values:
        if value is not None and (candidate := _valid_orientation(value)) is not None:
            first, parsed = value, candidate
            break
    if first is None:
        if any(value is not None for value in values):
            return OrientationDescriptor(None, None, None, "unknown", None, "unknown",
                warnings=("all ImageOrientationPatient values are malformed or non-orthogonal",))
        return OrientationDescriptor(None, None, None, "unknown", None, "unknown", warnings=("missing ImageOrientationPatient",))
    assert parsed is not None
    row, column = parsed
    raw = _numbers(first)
    assert raw is not None
    normal = _unit((row[1]*column[2]-row[2]*column[1], row[2]*column[0]-row[0]*column[2], row[0]*column[1]-row[1]*column[0]))
    assert normal is not None
    plane, obliquity = _plane(normal, config.plane_tolerance_deg)
    consistency, maximum, consistency_warnings = assess_orientation_consistency(values,
        tolerance_deg=config.orientation_consistency_tolerance_deg, minor_variation_deg=config.minor_variation_deg)
    warnings.extend(consistency_warnings)
    confidence = "high" if consistency in {"consistent", "minor_variation"} else "medium"
    return OrientationDescriptor(row, column, normal, plane, obliquity, confidence,
        consistency, maximum, tuple(warnings), ({"source": "DICOM.ImageOrientationPatient", "value": list(raw)},))
