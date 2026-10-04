"""DICOM patient-space geometry and deterministic MRI slice ordering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from statistics import median, pstdev
from typing import Any, Iterable, Mapping, Sequence


Vector3 = tuple[float, float, float]


def _numbers(value: Any, length: int, field_name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        value = value.split("\\")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must contain {length} numeric values") from exc
    if len(values) != length or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{field_name} must contain {length} finite numeric values")
    return values


def parse_position(value: Any) -> Vector3:
    """Parse DICOM ImagePositionPatient (DS strings and pydicom values included)."""
    return _numbers(value, 3, "ImagePositionPatient")  # type: ignore[return-value]


def parse_orientation(value: Any) -> tuple[Vector3, Vector3]:
    """Return row and column direction cosines after validating their geometry."""
    values = _numbers(value, 6, "ImageOrientationPatient")
    row, column = values[:3], values[3:]
    row_norm, column_norm = _norm(row), _norm(column)
    if not 0.9 <= row_norm <= 1.1 or not 0.9 <= column_norm <= 1.1:
        raise ValueError("ImageOrientationPatient direction cosine norm is outside [0.9, 1.1]")
    row, column = _scale(row, 1 / row_norm), _scale(column, 1 / column_norm)
    if abs(_dot(row, column)) > 0.1:
        raise ValueError("ImageOrientationPatient row and column directions are not orthogonal")
    return row, column


def cross_product(left: Sequence[float], right: Sequence[float]) -> Vector3:
    if len(left) != 3 or len(right) != 3:
        raise ValueError("cross product inputs must have three values")
    left = _numbers(left, 3, "left vector")
    right = _numbers(right, 3, "right vector")
    result = (left[1] * right[2] - left[2] * right[1],
              left[2] * right[0] - left[0] * right[2],
              left[0] * right[1] - left[1] * right[0])
    magnitude = _norm(result)
    if magnitude <= 1e-8:
        raise ValueError("orientation cross product is degenerate")
    return _scale(result, 1 / magnitude)


def project_position(position: Any, normal: Any) -> float:
    """Project a patient-space position onto a supplied normal vector."""
    point = parse_position(position)
    direction = _numbers(normal, 3, "normal vector")
    magnitude = _norm(direction)
    if magnitude <= 1e-8:
        raise ValueError("normal vector is degenerate")
    return _dot(point, _scale(direction, 1 / magnitude))


@dataclass(frozen=True)
class GeometryConfig:
    orientation_tolerance_deg: float = 1.0
    position_tolerance_mm: float = 1e-3
    duplicate_position_tolerance_mm: float = 1e-3
    spacing_outlier_factor: float = 2.0
    strict_geometry: bool = False
    duplicate_policy: str = "warn"

    def __post_init__(self) -> None:
        if not 0 <= self.orientation_tolerance_deg < 90:
            raise ValueError("orientation_tolerance_deg must be in [0, 90)")
        for name in ("position_tolerance_mm", "duplicate_position_tolerance_mm", "spacing_outlier_factor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.duplicate_policy not in {"warn", "keep_all", "deduplicate_exact", "strict"}:
            raise ValueError("unsupported duplicate_policy")


@dataclass(frozen=True)
class GeometryWarning:
    code: str
    message: str
    slice_ids: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message,
                "slice_ids": list(self.slice_ids), "details": dict(self.details)}


@dataclass(frozen=True)
class SeriesGeometry:
    normal_vector: Vector3 | None
    ordering_method: str
    n_slices: int
    position_min: float | None
    position_max: float | None
    physical_span: float | None
    slice_coordinates: tuple[tuple[str, float], ...]
    median_spacing: float | None
    min_spacing: float | None
    max_spacing: float | None
    spacing_std: float | None
    orientation_consistency: Mapping[str, Any]
    duplicate_positions: tuple[float, ...]
    spacing_metadata: Mapping[str, Any]
    warnings: tuple[GeometryWarning, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"normal_vector": list(self.normal_vector) if self.normal_vector else None,
                "ordering_method": self.ordering_method, "n_slices": self.n_slices,
                "position_min": self.position_min, "position_max": self.position_max,
                "physical_span": self.physical_span,
                "slice_coordinates": [[slice_id, value] for slice_id, value in self.slice_coordinates],
                "median_spacing": self.median_spacing,
                "min_spacing": self.min_spacing, "max_spacing": self.max_spacing,
                "spacing_std": self.spacing_std,
                "orientation_consistency": dict(self.orientation_consistency),
                "duplicate_positions": list(self.duplicate_positions),
                "spacing_metadata": dict(self.spacing_metadata),
                "warnings": [warning.to_dict() for warning in self.warnings]}


@dataclass(frozen=True)
class OrderingResult:
    slices: tuple[Any, ...]
    method: str
    confidence: str
    reversed: bool
    warnings: tuple[GeometryWarning, ...]
    diagnostics: SeriesGeometry
    series_instance_uid: str | None = None
    projected_positions_mm: tuple[float | None, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"slice_ids": [_slice_id(item) for item in self.slices],
                "projected_positions_mm": list(self.projected_positions_mm),
                "method": self.method, "confidence": self.confidence,
                "reversed": self.reversed, "series_instance_uid": self.series_instance_uid,
                "warnings": [warning.to_dict() for warning in self.warnings],
                "diagnostics": self.diagnostics.to_dict()}


def order_series_slices(series: Any, config: GeometryConfig | None = None) -> OrderingResult:
    """Order a SeriesRecord or iterable of SliceRecords by ascending physical coordinate.

    All slices use one metadata tier. Partial coordinate sets are never mixed with
    lower-tier keys, which would compare quantities without a shared scale.
    """
    config = config or GeometryConfig()
    if hasattr(series, "slices"):
        slices = tuple(series.slices)
        series_uid = getattr(series, "series_instance_uid", None)
    else:
        slices = tuple(series)
        series_uid = None
    warnings: list[GeometryWarning] = []
    if not slices:
        empty = SeriesGeometry(None, "stable_fallback", 0, None, None, None, (), None,
                               None, None, None, {"consistent": None, "max_deviation_deg": None}, (),
                               {"slice_thickness_mm": [], "spacing_between_slices_mm": [],
                                "pixel_spacing_mm": []}, ())
        return OrderingResult((), "stable_fallback", "low", False, (), empty, series_uid)

    metadata = [_metadata(item) for item in slices]
    spacing_metadata: dict[str, list[Any]] = {
        "slice_thickness_mm": [], "spacing_between_slices_mm": [], "pixel_spacing_mm": []}
    positions: list[Vector3 | None] = []
    orientations: list[tuple[Vector3, Vector3] | None] = []
    ids = [_slice_id(item) for item in slices]
    for item_id, values in zip(ids, metadata):
        for tag, output_key in (("SliceThickness", "slice_thickness_mm"),
                                ("SpacingBetweenSlices", "spacing_between_slices_mm")):
            raw = values.get(tag)
            measure = _finite_scalar(raw)
            invalid_measure = measure is None or (tag == "SliceThickness" and measure <= 0) or (
                tag == "SpacingBetweenSlices" and measure == 0)
            if raw is not None and invalid_measure:
                warnings.append(GeometryWarning("invalid_spacing_metadata",
                    f"{tag} must be a finite nonzero measurement (and positive for SliceThickness)",
                    (item_id,), {"value": str(raw)}))
            elif tag == "SpacingBetweenSlices" and measure is not None and measure < 0:
                warnings.append(GeometryWarning("signed_spacing_between_slices",
                    "negative SpacingBetweenSlices is retained; its meaning may be IOD-specific",
                    (item_id,), {"value_mm": measure}))
            spacing_metadata[output_key].append(measure)
        raw_pixel_spacing = values.get("PixelSpacing")
        if raw_pixel_spacing is None:
            spacing_metadata["pixel_spacing_mm"].append(None)
        else:
            try:
                pixel_spacing = _numbers(raw_pixel_spacing, 2, "PixelSpacing")
                rows = _finite_scalar(values.get("Rows"))
                columns = _finite_scalar(values.get("Columns"))
                invalid_pixel_spacing = (pixel_spacing[0] < 0 or pixel_spacing[1] < 0 or
                    (pixel_spacing[0] == 0 and rows != 1) or
                    (pixel_spacing[1] == 0 and columns != 1))
                spacing_metadata["pixel_spacing_mm"].append(list(pixel_spacing))
                if invalid_pixel_spacing:
                    warnings.append(GeometryWarning("invalid_spacing_metadata",
                        "PixelSpacing values must be positive except for a single row or column",
                        (item_id,), {"values_mm": list(pixel_spacing), "rows": rows,
                                     "columns": columns}))
            except ValueError as exc:
                spacing_metadata["pixel_spacing_mm"].append(None)
                warnings.append(GeometryWarning("invalid_spacing_metadata", str(exc), (item_id,),
                                                {"value": str(raw_pixel_spacing)}))
        try:
            positions.append(parse_position(values.get("ImagePositionPatient"))
                             if values.get("ImagePositionPatient") is not None else None)
        except ValueError as exc:
            positions.append(None)
            warnings.append(GeometryWarning("invalid_position", str(exc), (item_id,)))
        try:
            orientations.append(parse_orientation(values.get("ImageOrientationPatient"))
                                if values.get("ImageOrientationPatient") is not None else None)
        except ValueError as exc:
            orientations.append(None)
            warnings.append(GeometryWarning("invalid_orientation", str(exc), (item_id,)))

    uids_by_value: dict[str, list[str]] = defaultdict(list)
    for item_id, values in zip(ids, metadata):
        uid = values.get("SOPInstanceUID")
        if uid is not None and str(uid).strip():
            uids_by_value[str(uid).strip()].append(item_id)
    duplicate_uids = {uid: occurrences for uid, occurrences in uids_by_value.items()
                      if len(occurrences) > 1}
    if duplicate_uids:
        duplicate_ids = tuple(item_id for occurrences in duplicate_uids.values()
                              for item_id in occurrences)
        if config.duplicate_policy == "strict" or config.strict_geometry:
            raise ValueError("duplicate SOPInstanceUID values in strict mode")
        if config.duplicate_policy != "keep_all":
            warnings.append(GeometryWarning("duplicate_sop_instance_uid",
                "multiple slice records share a SOPInstanceUID",
                duplicate_ids, {"uids": duplicate_uids}))

    normals: list[Vector3 | None] = []
    for item_id, orientation in zip(ids, orientations):
        try:
            normals.append(cross_product(*orientation) if orientation else None)
        except ValueError as exc:
            normals.append(None)
            warnings.append(GeometryWarning("degenerate_orientation", str(exc), (item_id,)))
    valid_normals = [normal for normal in normals if normal is not None]
    valid_orientations = [orientation for orientation in orientations if orientation is not None]
    ref_normal = valid_normals[0] if valid_normals else None
    ref_orientation = valid_orientations[0] if valid_orientations else None
    deviations = [max(_angle_degrees(ref_orientation[0], orientation[0]),
                       _angle_degrees(ref_orientation[1], orientation[1]))
                  for orientation in valid_orientations] if ref_orientation else []
    max_deviation = max(deviations, default=None)
    orientation_consistent = (max_deviation is None or max_deviation <= config.orientation_tolerance_deg)
    if not orientation_consistent:
        warnings.append(GeometryWarning("inconsistent_orientation",
            "slice orientations exceed the configured angular tolerance",
            tuple(ids[i] for i, orientation in enumerate(orientations) if orientation is not None and
                  max(_angle_degrees(ref_orientation[0], orientation[0]),
                      _angle_degrees(ref_orientation[1], orientation[1])) > config.orientation_tolerance_deg),
            {"max_deviation_deg": max_deviation,
             "tolerance_deg": config.orientation_tolerance_deg}))
    if config.strict_geometry and any(w.code in {"invalid_position", "invalid_orientation",
            "invalid_spacing_metadata", "degenerate_orientation", "inconsistent_orientation"}
            for w in warnings):
        raise ValueError("invalid or inconsistent DICOM geometry in strict mode")

    stable_ties = [_metadata_tie_key(values) for values in metadata]
    keys: list[Any]
    coords: list[float] | None = None
    if all(p is not None for p in positions) and all(n is not None for n in normals) and orientation_consistent:
        normal = _mean_normal(valid_normals)
        coords = [project_position(p, normal) for p in positions if p is not None]
        method, confidence = "geometry", "high"
        keys = list(zip(coords, ids, stable_ties))
    elif all(p is not None for p in positions) and ref_normal is not None and orientation_consistent:
        normal = _mean_normal(valid_normals)
        coords = [project_position(p, normal) for p in positions if p is not None]
        method, confidence = "position_inferred_normal", "medium"
        warnings.append(GeometryWarning("inferred_normal", "normal inferred from orientations available elsewhere in the series"))
        keys = list(zip(coords, ids, stable_ties))
    else:
        normal = ref_normal
        keys = []
        method = ""
        confidence = "low"
        if not all(p is not None for p in positions):
            warnings.append(GeometryWarning("geometry_incomplete", "not all slices have valid image positions"))
        location_values = [_finite_scalar(item.get("SliceLocation")) for item in metadata]
        if all(v is not None for v in location_values):
            spread = max(location_values) - min(location_values)  # type: ignore[arg-type]
            if spread > config.position_tolerance_mm:
                keys = list(zip(location_values, ids, stable_ties))
                method, confidence = "slice_location", "medium"
                warnings.append(GeometryWarning("fallback_slice_location", "ordering uses SliceLocation because full geometry was unavailable"))
        if not keys:
            instance_values = [_finite_scalar(item.get("InstanceNumber")) for item in metadata]
            if all(v is not None for v in instance_values):
                keys = list(zip(instance_values, ids, stable_ties))
                method, confidence = "instance_number", "low"
                warnings.append(GeometryWarning("fallback_instance_number", "ordering uses InstanceNumber because spatial coordinates were unavailable"))
            else:
                stable_values = [_stable_identifier(item, values) for item, values in zip(slices, metadata)]
                keys = list(zip(stable_values, ids, stable_ties))
                method, confidence = "stable_fallback", "low"
                warnings.append(GeometryWarning("fallback_stable_identifier", "ordering uses SOPInstanceUID or relative path"))

    order = sorted(range(len(slices)), key=lambda i: keys[i])
    ordered_indices = list(order)
    ordered = tuple(slices[i] for i in ordered_indices)
    sorted_coords = sorted(coords) if coords is not None else None
    duplicate_positions: list[float] = []
    zero_spacing_intervals: list[float] = []
    spacings: list[float] = []
    duplicate_slice_ids: list[str] = []
    if sorted_coords is not None:
        for left, right in zip(sorted_coords, sorted_coords[1:]):
            delta = right - left
            if delta > config.duplicate_position_tolerance_mm:
                spacings.append(delta)
            else:
                zero_spacing_intervals.append(delta)
        coordinate_order = sorted(range(len(coords)), key=lambda i: (coords[i], ids[i]))
        group: list[int] = []
        group_anchor: float | None = None
        for index in coordinate_order:
            coordinate = coords[index]
            if group_anchor is None or coordinate - group_anchor <= config.duplicate_position_tolerance_mm:
                if group_anchor is None:
                    group_anchor = coordinate
                group.append(index)
            else:
                if len(group) > 1:
                    duplicate_positions.append(sum(coords[i] for i in group) / len(group))
                    duplicate_slice_ids.extend(ids[i] for i in group)
                group, group_anchor = [index], coordinate
        if len(group) > 1:
            duplicate_positions.append(sum(coords[i] for i in group) / len(group))
            duplicate_slice_ids.extend(ids[i] for i in group)
        if duplicate_positions:
            policy = config.duplicate_policy
            duplicate_warning = GeometryWarning("duplicate_positions", "multiple slices occupy the same physical position",
                tuple(duplicate_slice_ids),
                {"positions_mm": duplicate_positions, "policy": policy})
            if policy == "strict" or config.strict_geometry:
                raise ValueError("duplicate slice positions in strict mode")
            if policy != "keep_all":
                warnings.append(duplicate_warning)
            if zero_spacing_intervals and policy != "keep_all":
                warnings.append(GeometryWarning("zero_spacing",
                    "adjacent projected slice coordinates are coincident within tolerance",
                    tuple(duplicate_slice_ids), {"intervals_mm": zero_spacing_intervals,
                                                  "tolerance_mm": config.duplicate_position_tolerance_mm}))
            if policy == "deduplicate_exact":
                keep = []
                last = None
                for idx in order:
                    value = coords[idx]
                    if last is None or abs(value - last) > config.duplicate_position_tolerance_mm:
                        keep.append(idx)
                        last = value
                ordered_indices = keep
                ordered = tuple(slices[i] for i in ordered_indices)
    if len(spacings) > 1:
        typical = median(spacings)
        irregular = min(spacings) < typical / config.spacing_outlier_factor or max(spacings) > typical * config.spacing_outlier_factor
        if irregular:
            warnings.append(GeometryWarning("irregular_spacing", "inter-slice spacing varies beyond the configured factor",
                                            details={"spacing_mm": spacings, "median_mm": typical,
                                                     "factor": config.spacing_outlier_factor}))
    pos_min = min(sorted_coords) if sorted_coords else None
    pos_max = max(sorted_coords) if sorted_coords else None
    geometry_warnings = tuple(warnings)
    diag = SeriesGeometry(normal, method, len(ordered), pos_min, pos_max,
        pos_max - pos_min if pos_min is not None and pos_max is not None else None,
        tuple(sorted(zip(ids, coords))) if coords is not None else (),
        median(spacings) if spacings else None, min(spacings) if spacings else None,
        max(spacings) if spacings else None, pstdev(spacings) if spacings else None,
        {"consistent": orientation_consistent if valid_normals else None,
         "max_deviation_deg": max_deviation, "tolerance_deg": config.orientation_tolerance_deg},
        tuple(duplicate_positions), spacing_metadata, geometry_warnings)
    reversed_input = len(order) > 1 and order == list(range(len(slices) - 1, -1, -1))
    projected_positions = tuple(coords[i] if coords is not None else None
                                 for i in ordered_indices)
    return OrderingResult(ordered, method, confidence, reversed_input,
                          geometry_warnings, diag, series_uid, projected_positions)


def _metadata(item: Any) -> Mapping[str, Any]:
    if isinstance(item, Mapping):
        return item.get("metadata", item)
    return getattr(item, "metadata", {})


def _slice_id(item: Any) -> str:
    metadata = _metadata(item)
    if isinstance(item, Mapping):
        return str(item.get("slice_id") or item.get("relative_path") or
                   metadata.get("relative_path") or metadata.get("SOPInstanceUID") or "")
    return str(getattr(item, "slice_id", "") or getattr(item, "relative_path", "") or
               metadata.get("relative_path") or metadata.get("SOPInstanceUID") or "")


def _stable_identifier(item: Any, metadata: Mapping[str, Any]) -> str:
    if isinstance(item, Mapping):
        return str(item.get("relative_path") or metadata.get("relative_path") or
                   metadata.get("SOPInstanceUID") or _slice_id(item))
    return str(getattr(item, "relative_path", "") or metadata.get("relative_path") or
               metadata.get("SOPInstanceUID") or _slice_id(item))


def _metadata_tie_key(metadata: Mapping[str, Any]) -> str:
    """Stable final tie-breaker when upstream identifiers are duplicated."""
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)


def _finite_scalar(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return scalar if math.isfinite(scalar) else None


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _scale(value: Sequence[float], factor: float) -> Vector3:
    return tuple(item * factor for item in value)  # type: ignore[return-value]


def _angle_degrees(left: Vector3, right: Vector3) -> float:
    # A normal's sign is meaningful: opposing normals imply reversed planes.
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(left, right)))))


def _mean_normal(normals: Iterable[Vector3]) -> Vector3:
    values = tuple(normals)
    total = tuple(sum(n[i] for n in values) for i in range(3))
    magnitude = _norm(total)
    if magnitude <= 1e-8:
        raise ValueError("series normals cancel and cannot define a stable direction")
    return _scale(total, 1 / magnitude)
