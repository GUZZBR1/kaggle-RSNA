from dataclasses import dataclass, field
from typing import Any, Mapping

from ..identity import freeze_json


@dataclass(frozen=True)
class TelemetryEvent:
    name: str
    timestamp: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name.strip() or not self.timestamp.strip():
            raise ValueError("telemetry event name and timestamp are required")
        object.__setattr__(self, "attributes", freeze_json(self.attributes))
