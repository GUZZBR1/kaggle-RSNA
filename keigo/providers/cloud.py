"""Adapter for a cloud job API; vendor SDKs stay outside the core package."""

from __future__ import annotations

from typing import Protocol

from ..contracts import SimulationJob, SimulationResult


class CloudTransport(Protocol):
    def submit(self, job: dict) -> str: ...

    def get_result(self, execution_id: str) -> dict: ...


class CloudProvider:
    """Cloud provider backed by an injected transport implementation.

    The platform-specific transport owns credentials, networking, polling and result
    decoding. No provider or account is selected until one is configured.
    """

    def __init__(self, transport: CloudTransport, name: str = "cloud") -> None:
        self._transport = transport
        self.name = name

    def submit_simulation(self, job: SimulationJob) -> str:
        return self._transport.submit(job.to_dict())

    def get_result(self, execution_id: str) -> SimulationResult:
        payload = self._transport.get_result(execution_id)
        return SimulationResult(**payload)
