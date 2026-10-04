"""Training interfaces; GPU workloads can be added without changing orchestration."""

from typing import Any, Protocol


class Trainer(Protocol):
    def train(self, dataset_uri: str, configuration: dict[str, Any]) -> dict[str, Any]: ...
