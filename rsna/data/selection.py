"""Deterministic selection of MRI slice metadata references."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..identity import digest
from .geometry import GeometryConfig, OrderingResult, order_series_slices

ORDERING_ASSUMPTION = "geometry.order_series_slices"
GEOMETRY_REQUIREMENT = "valid_projected_coordinates_for_physical_span; otherwise_explicit_fallback"


@dataclass(frozen=True)
class SliceSelectionConfig:
    strategy: str = "uniform"
    count: int = 24
    short_series_policy: str = "keep_all"
    allow_fallback: bool = True
    geometry_config: GeometryConfig = field(default_factory=GeometryConfig)

    def __post_init__(self) -> None:
        if self.strategy not in {"uniform", "center", "physical_span"}:
            raise ValueError("strategy must be uniform, center, or physical_span")
        if type(self.count) is not int or self.count < 1:
            raise ValueError("count must be a positive integer")
        if self.short_series_policy not in {"keep_all", "repeat_nearest", "pad_reference", "strict"}:
            raise ValueError("unsupported short_series_policy")
        if type(self.allow_fallback) is not bool:
            raise ValueError("allow_fallback must be a boolean")
        if not isinstance(self.geometry_config, GeometryConfig):
            raise ValueError("geometry_config must be a GeometryConfig")

    def to_preprocessing_spec(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible identity payload for DatasetVersion.preprocessing."""
        selection = {
            "strategy": self.strategy,
            "count": self.count,
            "short_series_policy": self.short_series_policy,
            "ordering_assumption": ORDERING_ASSUMPTION,
            "geometry_config": _geometry_config_dict(self.geometry_config),
            "geometry_requirement": GEOMETRY_REQUIREMENT if self.strategy == "physical_span"
                else "canonical_geometry_ordering_when_available",
        }
        if self.strategy == "physical_span":
            selection.update(allow_fallback=self.allow_fallback,
                fallback_policy="uniform" if self.allow_fallback else "error")
        return {"slice_selection": selection}


@dataclass(frozen=True)
class SliceSelectionResult:
    selected: tuple[Any, ...]
    requested_count: int
    actual_count: int
    strategy: str
    fallback: str | None
    source_count: int
    selected_indices: tuple[int | None, ...]
    selected_positions_mm: tuple[float | None, ...]
    coverage_fraction: float
    physical_span_source: float | None
    physical_span_selected: float | None
    first_selected_position: float | None
    last_selected_position: float | None
    warnings: tuple[str, ...] = ()
    series_uid: str | None = None
    ordering_assumption: str = ORDERING_ASSUMPTION
    parameters: Mapping[str, Any] = field(default_factory=dict)
    configuration_id: str = ""
    geometry_ordering_method: str | None = None
    geometry_confidence: str | None = None
    ordered_count: int = 0
    study_instance_uid: str | None = None
    series_id: str | None = None
    study_id: str | None = None
    coverage_basis: str = "ordered_index"

    def to_dict(self) -> dict[str, Any]:
        selected_references = [_selection_reference(item, self.study_instance_uid,
            self.study_id, self.series_uid, self.series_id) for item in self.selected]
        return {"study_instance_uid": self.study_instance_uid, "study_id": self.study_id,
                "series_uid": self.series_uid, "series_id": self.series_id,
                "requested_count": self.requested_count, "actual_count": self.actual_count,
                "selected_count": sum(item is not None for item in self.selected),
                "source_count": self.source_count, "original_count": self.source_count,
                "ordered_count": self.ordered_count, "selected_indices": list(self.selected_indices),
                "selected_positions_mm": list(self.selected_positions_mm),
                "selected_sop_instance_uids": [_sop_uid(item) for item in self.selected],
                "selected_references": selected_references,
                "fallback": self.fallback, "coverage_fraction": self.coverage_fraction,
                "coverage_basis": self.coverage_basis,
                "physical_span_source": self.physical_span_source,
                "physical_span_selected": self.physical_span_selected,
                "first_selected_position": self.first_selected_position,
                "last_selected_position": self.last_selected_position,
                "warnings": list(self.warnings), "ordering_assumption": self.ordering_assumption,
                "parameters": dict(self.parameters), "configuration_id": self.configuration_id,
                "geometry_ordering_method": self.geometry_ordering_method,
                "geometry_confidence": self.geometry_confidence}


class SliceSelector:
    def __init__(self, config: SliceSelectionConfig | Mapping[str, Any]):
        self.config = config if isinstance(config, SliceSelectionConfig) else SliceSelectionConfig(**config)

    def select(self, series: Any, *, study: Any | None = None) -> SliceSelectionResult:
        raw_slices = (tuple(series.get("slices", ())) if isinstance(series, Mapping)
                      else tuple(getattr(series, "slices", series)) if not isinstance(series, OrderingResult)
                      else ())
        geometry_input = series.get("slices", ()) if isinstance(series, Mapping) else series
        ordering = (series if isinstance(series, OrderingResult) else
                    order_series_slices(geometry_input, config=self.config.geometry_config))
        if ordering.config != self.config.geometry_config:
            raise ValueError("precomputed OrderingResult GeometryConfig does not match selection config")
        ordered = ordering.slices
        ordered_count = len(ordered)
        original_count = ordering.source_slice_count
        source_indices = ordering.source_indices
        if len(source_indices) != ordered_count:
            source_indices = tuple(range(ordered_count))
            original_count = max(original_count, ordered_count)
        positions = ordering.ordered_positions_mm
        if len(positions) != ordered_count:
            positions = (None,) * ordered_count
        complete_positions = ordered_count > 0 and all(position is not None for position in positions)
        series_uid = ordering.series_instance_uid
        series_id = ordering.series_id
        study_uid = ordering.study_instance_uid
        study_id = _identity_value(study, "study_id")
        study_uid = _identity_value(study, "study_instance_uid") or study_uid
        if isinstance(series, Mapping):
            series_uid = series.get("series_instance_uid", series_uid)
            series_id = series.get("series_id", series_id)
            study_uid = series.get("study_instance_uid", study_uid)
            study_id = series.get("study_id", study_id)
        elif not isinstance(series, OrderingResult):
            series_uid = getattr(series, "series_instance_uid", series_uid)
            series_id = getattr(series, "series_id", series_id)
            study_uid = getattr(series, "study_instance_uid", study_uid)
        warnings = _collect_warnings(study, ())
        warnings.extend(_collect_warnings(None if isinstance(series, OrderingResult) else series, raw_slices))
        warnings.extend(warning.message for warning in ordering.warnings)
        config_spec = self.config.to_preprocessing_spec()
        config_id = digest(config_spec)
        requested = self.config.count
        if ordered_count < requested and self.config.short_series_policy == "strict":
            raise ValueError(f"series has {ordered_count} ordered slices, fewer than requested {requested}")
        strategy = self.config.strategy
        fallback = None
        if strategy == "physical_span" and not complete_positions:
            if not self.config.allow_fallback:
                raise ValueError("physical_span requires valid projected physical coordinates for every ordered slice")
            missing = ordered_count - sum(position is not None for position in positions)
            warnings.append(f"physical_span unavailable ({missing}/{ordered_count} positions missing); used uniform")
            strategy, fallback = "uniform", "uniform"
        target = requested if ordered_count >= requested or self.config.short_series_policy == "repeat_nearest" else ordered_count
        if strategy == "uniform":
            chosen = _uniform_indices(ordered_count, target)
        elif strategy == "center":
            start = max(0, (ordered_count - target) // 2)
            chosen = list(range(start, start + target)) if target <= ordered_count else _repeat_indices(ordered_count, target)
        elif ordered_count:
            chosen = _physical_indices(positions, target)
        else:
            chosen = []
        pad_count = (requested - len(chosen) if ordered_count < requested
                     and self.config.short_series_policy == "pad_reference" else 0)
        parameters = dict(config_spec["slice_selection"])
        if self.config.strategy == "center" and chosen:
            parameters.update(start_index=chosen[0], end_index=chosen[-1] + 1,
                              center_index=(chosen[0] + chosen[-1]) / 2)
        selected = tuple(ordered[i] for i in chosen) + (None,) * pad_count
        selected_indices = tuple(source_indices[i] for i in chosen) + (None,) * pad_count
        selected_positions = tuple(positions[i] for i in chosen) + (None,) * pad_count
        physical_positions = [position for position in positions if position is not None]
        source_span = ordering.diagnostics.physical_span if complete_positions else None
        selected_physical = ([position for position in selected_positions if position is not None]
                             if complete_positions else [])
        selected_span = (max(selected_physical) - min(selected_physical)
                         if selected_physical else None)
        if source_span is not None and selected_span is not None and source_span > 0:
            coverage, coverage_basis = min(1.0, selected_span / source_span), "physical_span_mm"
        elif ordered_count > 1:
            coverage = ((max(chosen) - min(chosen)) / (ordered_count - 1) if chosen else 0.0)
            coverage_basis = "ordered_index"
        else:
            coverage, coverage_basis = (1.0 if ordered_count else 0.0), "ordered_index"
        first, last = ((selected_physical[0], selected_physical[-1])
                       if selected_physical else (None, None))
        if ordered_count < requested and self.config.short_series_policy == "keep_all":
            warnings.append(f"short series: kept all {ordered_count} ordered slices (requested {requested})")
        if ordered_count == 0 and self.config.short_series_policy == "repeat_nearest":
            warnings.append("empty series: no real slice can be repeated")
        if pad_count:
            warnings.append(f"short series: padded {pad_count} empty references (requested {requested})")
        return SliceSelectionResult(selected=selected, requested_count=requested, actual_count=len(selected),
            strategy=strategy, fallback=fallback, source_count=original_count,
            selected_indices=selected_indices, selected_positions_mm=selected_positions,
            coverage_fraction=coverage, physical_span_source=source_span,
            physical_span_selected=selected_span, first_selected_position=first,
            last_selected_position=last,
            warnings=tuple(warnings), series_uid=series_uid,
            parameters=parameters, configuration_id=config_id,
            geometry_ordering_method=ordering.method, geometry_confidence=ordering.confidence,
            ordered_count=ordered_count, study_instance_uid=study_uid,
            series_id=series_id, study_id=study_id,
            coverage_basis=coverage_basis)


def select_slices(slices: Sequence[Any], *, strategy: str = "uniform", count: int = 24,
                  short_series_policy: str = "keep_all", **kwargs: Any) -> SliceSelectionResult:
    return SliceSelector(SliceSelectionConfig(strategy, count, short_series_policy, **kwargs)).select(slices)


def _uniform_indices(size: int, count: int) -> list[int]:
    if count <= 0 or size <= 0:
        return []
    if count > size:
        return _repeat_indices(size, count)
    if count == 1:
        return [(size - 1) // 2]
    # Integer arithmetic avoids cumulative rounding drift and always includes both ends.
    return [(i * (size - 1) * 2 + (count - 1)) // (2 * (count - 1)) for i in range(count)]


def _repeat_indices(size: int, count: int) -> list[int]:
    if count <= size:
        return _uniform_indices(size, count)
    if size == 1:
        return [0] * count
    if count == 1:
        return [(size - 1) // 2]
    return [(i * (size - 1) * 2 + (count - 1)) // (2 * (count - 1))
            for i in range(count)]


def _physical_indices(positions: Sequence[float | None], count: int) -> list[int]:
    """Sample canonical, already ordered Geometry coordinates without reprojection."""
    size = len(positions)
    if count <= 0 or size == 0:
        return []
    numeric = tuple(float(position) for position in positions)
    low, high = numeric[0], numeric[-1]
    if count > size:
        return _repeat_indices(size, count)
    if count == 1:
        midpoint = (low + high) / 2
        return [min(range(size), key=lambda index: (abs(numeric[index] - midpoint), index))]
    chosen = [0]
    cursor = 0
    for slot in range(1, count):
        if slot == count - 1:
            chosen.append(size - 1)
            continue
        target = low + (high - low) * slot / (count - 1)
        lower, upper = chosen[-1] + 1, size - (count - slot)
        cursor = max(cursor, lower)
        while cursor < upper and abs(numeric[cursor + 1] - target) < abs(numeric[cursor] - target):
            cursor += 1
        chosen.append(cursor)
    return chosen


def _metadata(item: Any) -> Mapping[str, Any]:
    value = getattr(item, "metadata", item)
    if isinstance(value, Mapping) and isinstance(value.get("metadata"), Mapping):
        value = value["metadata"]
    return value if isinstance(value, Mapping) else {}


def _collect_warnings(series: Any, slices: Sequence[Any]) -> list[str]:
    series_warnings = (series.get("warnings", ()) if isinstance(series, Mapping)
                       else getattr(series, "warnings", ()))
    warnings = list(series_warnings or ())
    for item in slices:
        item_warnings = (item.get("warnings", ()) if isinstance(item, Mapping)
                         else getattr(item, "warnings", ()))
        metadata_warnings = _metadata(item).get("metadata_warnings", ())
        warnings.extend(item_warnings or ())
        warnings.extend(metadata_warnings or ())
    return warnings


def _sop_uid(item: Any) -> str | None:
    return _metadata(item).get("SOPInstanceUID")


def _selection_reference(item: Any, study_uid: str | None, study_id: str | None,
                         series_uid: str | None, series_id: str | None) -> dict[str, Any] | None:
    if item is None:
        return {"reference_type": "padding_reference", "padding": True,
                "study_instance_uid": study_uid, "study_id": study_id,
                "series_instance_uid": series_uid, "series_id": series_id,
                "sop_instance_uid": None, "slice_id": None,
                "relative_path": None, "source_entity_ids": []}
    metadata = _metadata(item)
    path = item.get("relative_path") if isinstance(item, Mapping) else getattr(item, "relative_path", None)
    slice_id = item.get("slice_id") if isinstance(item, Mapping) else getattr(item, "slice_id", None)
    source_ids = [identity for identity in (study_id, study_uid, series_id, series_uid, slice_id)
                  if identity is not None]
    return {"reference_type": "slice_reference", "padding": False,
            "study_instance_uid": study_uid, "study_id": study_id,
            "series_instance_uid": series_uid, "series_id": series_id,
            "sop_instance_uid": metadata.get("SOPInstanceUID"),
            "slice_id": slice_id, "relative_path": path,
            "source_entity_ids": source_ids}


def _identity_value(value: Any, name: str) -> str | None:
    item = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
    return str(item) if item is not None else None


def _geometry_config_dict(config: GeometryConfig) -> dict[str, Any]:
    return {"orientation_tolerance_deg": config.orientation_tolerance_deg,
            "position_tolerance_mm": config.position_tolerance_mm,
            "duplicate_position_tolerance_mm": config.duplicate_position_tolerance_mm,
            "spacing_outlier_factor": config.spacing_outlier_factor,
            "strict_geometry": config.strict_geometry,
            "duplicate_policy": config.duplicate_policy}
