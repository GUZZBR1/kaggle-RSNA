"""Provider protocol shared by local and remote training backends."""

from __future__ import annotations

from typing import Protocol

from ..contracts import TrainingJob, TrainingResult


class TrainingProvider(Protocol):
    def submit_training(self, job: TrainingJob) -> str: ...
    def get_training_result(self, execution_id: str) -> TrainingResult: ...
