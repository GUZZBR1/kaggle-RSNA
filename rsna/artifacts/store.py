"""Small content-addressed artifact store with manifest verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..contracts import ArtifactReference


class JsonArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def serialize(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"

    def put_json(self, value: Any, *, media_type: str = "application/json",
                 manifest: dict[str, Any] | None = None) -> ArtifactReference:
        payload = self.serialize(value).encode("utf-8")
        sha = hashlib.sha256(payload).hexdigest()
        path = self.root / f"{sha}.json"
        if not path.exists():
            path.write_bytes(payload)
        return ArtifactReference(sha, media_type, str(path.resolve()), sha, manifest or {})

    @staticmethod
    def load_json(reference: ArtifactReference) -> Any:
        path = Path(reference.uri)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != reference.sha256:
            raise ValueError("artifact content hash mismatch")
        return json.loads(payload)
