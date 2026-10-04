"""Provider contract: orchestration addresses jobs, providers handle execution details."""

from __future__ import annotations

from typing import Protocol

from ..contracts import SimulationJob, SimulationResult


class SimulationProvider(Protocol):
    name: str

    def submit_simulation(self, job: SimulationJob) -> str:
        """Submit a simulation and return its provider-specific execution ID."""
        ...

    def get_result(self, execution_id: str) -> SimulationResult:
        """Wait for a submitted simulation and return its versioned result."""
        ...
