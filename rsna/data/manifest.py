"""Versioned JSON manifest persistence for a dataset index."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from .models import DatasetIndex, index_from_dict


def save_manifest(index: DatasetIndex, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(index.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temp = Path(name)
    try:
        with temp.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: str | Path) -> DatasetIndex:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return index_from_dict(json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {value}"))))
