"""Append-only JSONL persistence for provider-neutral telemetry events."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock

from ..identity import canonical_json
from ..telemetry.events import TelemetryEvent


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with an explicit ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_events(path: str | Path) -> list[TelemetryEvent]:
    """Load and validate all telemetry events from a JSONL file."""
    events = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("event must be a JSON object")
                events.append(TelemetryEvent(**payload))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid telemetry event on line {line_number}: {exc}") from exc
    return events


class JsonlTelemetrySink:
    """Callable event sink that appends canonical JSON records, one per line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def emit(self, event: TelemetryEvent) -> None:
        if not isinstance(event, TelemetryEvent):
            raise TypeError("event must be a TelemetryEvent")
        record = {"name": event.name, "timestamp": event.timestamp,
                  "attributes": event.attributes}
        line = canonical_json(record) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)

    __call__ = emit

    def read_events(self) -> list[TelemetryEvent]:
        return load_events(self.path) if self.path.exists() else []
