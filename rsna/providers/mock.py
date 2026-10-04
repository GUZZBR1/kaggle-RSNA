"""Explicitly synthetic provider for contract and CLI smoke tests."""

from __future__ import annotations

from ..contracts import TrainingJob
from ..identity import digest
from .local import LocalProvider


class MockProvider(LocalProvider):
    def __init__(self):
        super().__init__(self._mock_train)

    @staticmethod
    def _mock_train(job: TrainingJob):
        checksum = digest({"mock_checkpoint_for": job.training_job_id})
        return {"checkpoint_uri": f"mock://checkpoints/{checksum}",
                "checkpoint_sha256": checksum, "format": "synthetic-placeholder",
                "metrics": {"synthetic_smoke": 1.0},
                "provenance": {"synthetic": True}}
