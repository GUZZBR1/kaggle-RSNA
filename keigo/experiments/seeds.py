"""Read-only seed classification checks; cloud admission needs a transactional ledger."""

from __future__ import annotations

import json
from pathlib import Path

SPLITS = ("development", "validation", "holdout")


class SeedRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._registry = json.loads(self.path.read_text(encoding="utf-8"))
        if self._registry.get("schema_version") != 1:
            raise ValueError("unsupported seed registry schema")
        splits = self._registry.get("seed_splits", {})
        if set(splits) != set(SPLITS):
            raise ValueError("seed registry must define development, validation and holdout")
        seen: set[int] = set()
        for split, seeds in splits.items():
            if any(type(seed) is not int or seed < 0 for seed in seeds):
                raise ValueError(f"{split} seeds must be nonnegative integers")
            if len(seeds) != len(set(seeds)) or seen.intersection(seeds):
                raise ValueError("seed registry contains duplicates or overlapping splits")
            seen.update(seeds)

    def validate(self, seeds: tuple[int, ...], split: str) -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown seed split: {split}")
        available = set(self._registry["seed_splits"][split])
        missing = sorted(set(seeds) - available)
        if missing:
            raise ValueError(f"unregistered {split} seeds: {missing}")
