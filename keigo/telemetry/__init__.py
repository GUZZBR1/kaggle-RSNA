"""Structured, provider-neutral execution telemetry."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TelemetryEvent:
    name: str
    attributes: dict[str, Any]
    timestamp: str

    @classmethod
    def create(cls, name: str, **attributes: Any) -> "TelemetryEvent":
        return cls(name, attributes, datetime.now(timezone.utc).isoformat())
