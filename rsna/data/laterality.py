"""Conservative, auditable laterality resolution for DICOM series and studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Sequence

from .provenance import OrientationConfig

_LEFT = {"L", "LT", "LEFT"}
_RIGHT = {"R", "RT", "RIGHT"}
_BILATERAL = {"B", "BILAT", "BILATERAL"}
_TOKEN = re.compile(r"(?<![A-Z0-9])(LEFT|RIGHT|LT|RT|BILATERAL|BILAT|L|R|B)(?![A-Z0-9])", re.I)


@dataclass(frozen=True)
class LateralityEvidence:
    source: str
    value: str
    resolved: str
    weight: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class LateralityResolution:
    resolved: str
    confidence: str
    evidence: tuple[LateralityEvidence, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"resolved": self.resolved, "confidence": self.confidence,
                "evidence": [item.to_dict() for item in self.evidence], "warnings": list(self.warnings)}


def _get(data: Mapping[str, Any], key: str) -> Any:
    value = data.get(key)
    return value if value is not None else data.get(key.lower())


def _parse(value: Any, *, text: bool) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if not text:
        return "LEFT" if normalized in _LEFT else "RIGHT" if normalized in _RIGHT else "BILATERAL" if normalized in _BILATERAL else None
    matches = {"LEFT" if token.upper() in _LEFT else "RIGHT" if token.upper() in _RIGHT else "BILATERAL"
               for token in _TOKEN.findall(normalized)}
    return next(iter(matches)) if len(matches) == 1 else "AMBIGUOUS" if matches else None


def parse_laterality_text(value: Any) -> str | None:
    """Parse standalone laterality terms; ordinary words such as 'bright' do not match."""
    return _parse(value, text=True)


def resolve_series_laterality(series: Mapping[str, Any] | Sequence[Mapping[str, Any]],
                               config: OrientationConfig | None = None) -> LateralityResolution:
    config = config or OrientationConfig()
    slices = list(series) if isinstance(series, Sequence) and not isinstance(series, (str, bytes, Mapping)) else [series]
    structured: list[LateralityEvidence] = []
    textual: list[LateralityEvidence] = []
    warnings: list[str] = []
    for index, item in enumerate(slices):
        if not isinstance(item, Mapping):
            warnings.append(f"slice {index} metadata is not an object")
            continue
        for key in ("Laterality", "ImageLaterality"):
            raw = _get(item, key)
            if raw is None or raw == "":
                continue
            parsed = _parse(raw, text=False)
            if parsed is None:
                warnings.append(f"malformed {key}: {raw!r}")
            else:
                structured.append(LateralityEvidence(f"DICOM.{key}", str(raw), parsed, "primary"))
        if config.laterality_text_fallback:
            for key in ("SeriesDescription", "ProtocolName", "StudyDescription"):
                raw = _get(item, key)
                parsed = _parse(raw, text=True)
                if parsed:
                    textual.append(LateralityEvidence(key, str(raw), parsed, "supporting"))
    strong = {entry.resolved for entry in structured}
    all_values = strong | {entry.resolved for entry in textual}
    if "AMBIGUOUS" in all_values:
        warnings.append("text metadata contains contradictory laterality terms")
        return LateralityResolution("AMBIGUOUS", "unknown", tuple(structured + textual), tuple(warnings))
    if len(strong) > 1 or (not strong and len(all_values) > 1) or (strong and any(v not in strong for v in {e.resolved for e in textual})):
        warnings.append("conflicting laterality evidence")
        resolved = "AMBIGUOUS" if config.conflict_policy == "ambiguous" else "UNKNOWN"
        return LateralityResolution(resolved, "unknown", tuple(structured + textual), tuple(warnings))
    if strong:
        return LateralityResolution(next(iter(strong)), "high", tuple(structured + textual), tuple(warnings))
    if textual:
        return LateralityResolution(next(iter(all_values)), "low", tuple(textual), tuple(warnings))
    return LateralityResolution("UNKNOWN", "unknown", (), tuple(warnings))


def resolve_study_laterality(series_list: Sequence[Mapping[str, Any] | Sequence[Mapping[str, Any]]],
                              config: OrientationConfig | None = None) -> LateralityResolution:
    results = [resolve_series_laterality(item, config) for item in series_list]
    known = {result.resolved for result in results if result.resolved in {"LEFT", "RIGHT", "BILATERAL"}}
    evidence = tuple(entry for result in results for entry in result.evidence)
    warnings = tuple(warning for result in results for warning in result.warnings)
    unresolved_conflict = any("conflict" in warning.lower() or "contradictory" in warning.lower()
                              for result in results for warning in result.warnings)
    if unresolved_conflict or any(result.resolved == "AMBIGUOUS" for result in results) or len(known) > 1:
        return LateralityResolution("AMBIGUOUS", "unknown", evidence, warnings + ("study contains conflicting series laterality",))
    if not known:
        return LateralityResolution("UNKNOWN", "unknown", evidence, warnings)
    confidence = "high" if all(result.resolved in known and result.confidence == "high" for result in results) else "low"
    return LateralityResolution(next(iter(known)), confidence, evidence, warnings)
