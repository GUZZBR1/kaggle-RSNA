"""Versioned JSON manifest persistence for a dataset index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import DatasetIndex, index_from_dict


def save_manifest(index: DatasetIndex, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(index.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: str | Path) -> DatasetIndex:
    return index_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
