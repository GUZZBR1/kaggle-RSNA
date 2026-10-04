"""Agent entrypoint contracts. Strategy implementations live in candidate artifacts."""

from typing import Any, Mapping, Protocol


class Agent(Protocol):
    def __call__(self, observation: Mapping[str, Any]) -> Mapping[str, Any]: ...
