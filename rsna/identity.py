"""Canonical JSON and stable SHA-256 helpers shared by all contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def freeze_json(value: Any) -> Any:
    from types import MappingProxyType

    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value {type(value).__name__} is not JSON-compatible")


def canonical_json(value: Any) -> str:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
