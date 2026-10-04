"""Extension point for future training implementations; no model is bundled."""

from typing import Protocol

from ..contracts import TrainingJob


class Trainer(Protocol):
    def __call__(self, job: TrainingJob) -> dict: ...
