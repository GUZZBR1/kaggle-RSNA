"""Deterministic selection of MRI slice metadata references."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

ORDERING_ASSUMPTION = "explicit_mm_or_projected_ipp; tie_break_sop_path; else_preserve_input_order"


@dataclass(frozen=True)
class SliceSelectionConfig:
    strategy: str = "uniform"
    count: int = 24
    short_series_policy: str = "keep_all"
    physical_position_tolerance_mm: float = 1e-3
    allow_fallback: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in {"uniform", "center", "physical_span"}:
            raise ValueError("strategy must be uniform, center, or physical_span")
        if type(self.count) is not int or self.count < 1:
            raise ValueError("count must be a positive integer")
        if self.short_series_policy not in {"keep_all", "repeat_nearest", "pad_reference", "strict"}:
            raise ValueError("unsupported short_series_policy")
        if (isinstance(self.physical_position_tolerance_mm, bool)
                or not isinstance(self.physical_position_tolerance_mm, (int, float))
                or not math.isfinite(self.physical_position_tolerance_mm)
                or self.physical_position_tolerance_mm < 0):
            raise ValueError("physical_position_tolerance_mm must be finite and nonnegative")

    def to_preprocessing_spec(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible identity payload for DatasetVersion.preprocessing."""
        return {"slice_selection": {
            "strategy": self.strategy,
            "count": self.count,
            "short_series_policy": self.short_series_policy,
            "physical_position_tolerance_mm": self.physical_position_tolerance_mm,
            "allow_fallback": self.allow_fallback,
            "fallback_policy": ("uniform" if self.allow_fallback else "error")
                if self.strategy == "physical_span" else "not_applicable",
            "ordering_assumption": ORDERING_ASSUMPTION,
        }}


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
    mean_selected_spacing: float | None
    warnings: tuple[str, ...] = ()
    series_uid: str | None = None
    ordering_assumption: str = ORDERING_ASSUMPTION
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"series_uid": self.series_uid, "strategy": self.strategy,
                "requested_count": self.requested_count, "actual_count": self.actual_count,
                "source_count": self.source_count, "selected_indices": list(self.selected_indices),
                "selected_positions_mm": list(self.selected_positions_mm),
                "selected_sop_instance_uids": [_sop_uid(item) for item in self.selected],
                "selected_references": [_selection_reference(item) for item in self.selected],
                "fallback": self.fallback, "coverage_fraction": self.coverage_fraction,
                "physical_span_source": self.physical_span_source,
                "physical_span_selected": self.physical_span_selected,
                "first_selected_position": self.first_selected_position,
                "last_selected_position": self.last_selected_position,
                "mean_selected_spacing": self.mean_selected_spacing,
                "warnings": list(self.warnings), "ordering_assumption": self.ordering_assumption,
                "parameters": dict(self.parameters)}


class SliceSelector:
    def __init__(self, config: SliceSelectionConfig | Mapping[str, Any]):
        self.config = config if isinstance(config, SliceSelectionConfig) else SliceSelectionConfig(**config)

    def select(self, series: Any) -> SliceSelectionResult:
        slices = tuple(series.get("slices", series) if isinstance(series, Mapping)
                       else getattr(series, "slices", series))
        uid = (series.get("series_instance_uid") if isinstance(series, Mapping)
               else getattr(series, "series_instance_uid", None))
        count = len(slices)
        positions = tuple(_physical_position(item) for item in slices)
        warnings = _collect_warnings(series, slices)
        if count == 0:
            return SliceSelectionResult((), self.config.count, 0, self.config.strategy,
                None, 0, (), (), 0.0, None, None, None, None, None, tuple(warnings), uid,
                parameters=self.config.to_preprocessing_spec()["slice_selection"])

        # Position-backed ordering is stable for shuffled inputs; ties retain identity order.
        complete_positions = all(value is not None for value in positions)
        if complete_positions:
            order = sorted(range(count), key=lambda i: (positions[i], _stable_key(slices[i])))
        else:
            order = list(range(count))
            if self.config.strategy == "physical_span":
                missing = sum(value is None for value in positions)
                if not self.config.allow_fallback:
                    raise ValueError("physical_span requires a physical position for every slice")
                warnings.append(f"physical_span unavailable ({missing}/{count} positions missing); used uniform")

        ordered = [slices[i] for i in order]
        ordered_positions = [positions[i] for i in order]
        strategy = self.config.strategy
        fallback = None
        if strategy == "physical_span" and not complete_positions:
            strategy, fallback = "uniform", "uniform"

        wanted = self.config.count
        if count < wanted and self.config.short_series_policy == "strict":
            raise ValueError(f"series has {count} slices, fewer than requested {wanted}")
        target = wanted if count >= wanted or self.config.short_series_policy == "repeat_nearest" else count
        if strategy == "uniform":
            chosen = _uniform_indices(count, target)
        elif strategy == "center":
            start = max(0, (count - target) // 2)
            chosen = list(range(start, start + target)) if target <= count else _repeat_indices(count, target)
        else:
            chosen = _physical_indices(ordered_positions, target,
                                       float(self.config.physical_position_tolerance_mm))

        pad_count = (wanted - len(chosen) if count < wanted
                     and self.config.short_series_policy == "pad_reference" else 0)

        parameters = self.config.to_preprocessing_spec()["slice_selection"]
        if self.config.strategy == "center":
            parameters.update(start_index=chosen[0], end_index=chosen[-1] + 1,
                              center_index=(chosen[0] + chosen[-1]) / 2)

        selected = tuple(ordered[i] for i in chosen) + (None,) * pad_count
        selected_indices = tuple(order[i] for i in chosen) + (None,) * pad_count
        selected_positions = tuple(ordered_positions[i] for i in chosen) + (None,) * pad_count
        finite_positions = list(ordered_positions) if complete_positions else []
        source_span = max(finite_positions) - min(finite_positions) if len(finite_positions) > 1 else 0.0 if finite_positions else None
        chosen_finite = list(selected_positions) if complete_positions else []
        selected_span = max(chosen_finite) - min(chosen_finite) if len(chosen_finite) > 1 else 0.0 if chosen_finite else None
        if source_span is not None and source_span > 0 and selected_span is not None:
            coverage = min(1.0, selected_span / source_span)
        elif count > 1:
            observed_indices = [i for i in selected_indices if i is not None]
            coverage = ((max(observed_indices) - min(observed_indices)) / (count - 1)
                        if observed_indices else 0.0)
        else:
            coverage = 1.0 / count
        first, last = (chosen_finite[0], chosen_finite[-1]) if chosen_finite else (None, None)
        diffs = [abs(b - a) for a, b in zip(chosen_finite, chosen_finite[1:])]
        mean_spacing = sum(diffs) / len(diffs) if diffs else None
        if any(b - a <= self.config.physical_position_tolerance_mm
               for a, b in zip(finite_positions, finite_positions[1:])):
            warnings.append("duplicate physical positions are present; source slices were preserved")
        if count < wanted and self.config.short_series_policy == "keep_all":
            warnings.append(f"short series: kept all {count} slices (requested {wanted})")
        if pad_count:
            warnings.append(f"short series: padded {pad_count} empty references (requested {wanted})")
        return SliceSelectionResult(selected, wanted, len(selected), strategy, fallback, count,
            selected_indices, selected_positions, coverage, source_span, selected_span,
            first, last, mean_spacing, tuple(warnings), uid,
            parameters=parameters)


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


def _physical_indices(positions: Sequence[float | None], count: int,
                      tolerance_mm: float) -> list[int]:
    size = len(positions)
    groups = _position_groups(positions, tolerance_mm)
    representatives = [group[0] for group in groups]
    if representatives[-1] != size - 1:
        representatives.append(size - 1)
    low, high = positions[0], positions[-1]
    if count > size:
        candidates = representatives
        result = []
        cursor = 0
        for slot in range(count):
            if slot == count - 1:
                cursor = len(candidates) - 1
            else:
                target = low + (high - low) * slot / (count - 1)
                while (cursor + 1 < len(candidates)
                       and abs(positions[candidates[cursor + 1]] - target)
                       < abs(positions[candidates[cursor]] - target)):
                    cursor += 1
            result.append(candidates[cursor])
        return result
    if count == 1:
        return [min(range(size), key=lambda i: (abs(positions[i] - (positions[0] + positions[-1]) / 2), i))]
    candidates = representatives
    if len(candidates) < count:
        candidates = list(candidates)
        present = set(candidates)
        extras = deque((indices, 0) for group in groups
                       if (indices := [index for index in group if index not in present]))
        while len(candidates) < count and extras:
            group_extras, offset = extras.popleft()
            candidates.append(group_extras[offset])
            if offset + 1 < len(group_extras):
                extras.append((group_extras, offset + 1))
        candidates.sort()
    candidate_slots = []
    for slot in range(count):
        if slot == 0:
            candidate_slot = 0
        elif slot == count - 1:
            candidate_slot = len(candidates) - 1
        else:
            target = low + (high - low) * slot / (count - 1)
            lower = candidate_slots[-1] + 1
            upper = len(candidates) - (count - slot)
            candidate_slot = lower
            while (candidate_slot < upper
                   and abs(positions[candidates[candidate_slot + 1]] - target)
                   < abs(positions[candidates[candidate_slot]] - target)):
                candidate_slot += 1
        candidate_slots.append(candidate_slot)
    return [candidates[j] for j in candidate_slots]


def _position_groups(positions: Sequence[float | None], tolerance_mm: float) -> list[list[int]]:
    groups: list[list[int]] = []
    for index, position in enumerate(positions):
        if not groups or position - positions[groups[-1][0]] > tolerance_mm:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def _physical_position(item: Any) -> float | None:
    metadata = _metadata(item)
    direct = metadata.get("physical_position_mm", metadata.get("position_mm"))
    if direct is not None:
        try:
            value = float(direct)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None
    position = metadata.get("ImagePositionPatient")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        return None
    try:
        xyz = [float(v) for v in position]
        if not all(math.isfinite(v) for v in xyz):
            return None
        orientation = metadata.get("ImageOrientationPatient")
        if not isinstance(orientation, (list, tuple)) or len(orientation) != 6:
            return None
        row, col = [float(v) for v in orientation[:3]], [float(v) for v in orientation[3:]]
        if not all(math.isfinite(v) for v in (*row, *col)):
            return None
        normal = (row[1]*col[2]-row[2]*col[1], row[2]*col[0]-row[0]*col[2], row[0]*col[1]-row[1]*col[0])
        norm = math.sqrt(sum(v*v for v in normal))
        if norm == 0 or not math.isfinite(norm):
            return None
        return sum(a*b for a, b in zip(xyz, normal)) / norm
    except (TypeError, ValueError, OverflowError):
        return None


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


def _stable_key(item: Any) -> tuple[str, str, str, str]:
    metadata = _metadata(item)
    path = item.get("relative_path", "") if isinstance(item, Mapping) else getattr(item, "relative_path", "")
    return (str(metadata.get("SOPInstanceUID") or ""), str(path or ""),
            str(getattr(item, "slice_id", "") or ""),
            repr(sorted(metadata.items(), key=lambda entry: str(entry[0]))))


def _sop_uid(item: Any) -> str | None:
    return _metadata(item).get("SOPInstanceUID")


def _selection_reference(item: Any) -> dict[str, str | None] | None:
    if item is None:
        return None
    metadata = _metadata(item)
    path = item.get("relative_path") if isinstance(item, Mapping) else getattr(item, "relative_path", None)
    slice_id = getattr(item, "slice_id", None)
    return {"sop_instance_uid": metadata.get("SOPInstanceUID"),
            "slice_id": slice_id, "relative_path": path}
