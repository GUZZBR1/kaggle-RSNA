"""Provider-neutral, lineage-checked model training extension points."""

from typing import Protocol

from ..contracts import TrainingJob
from .engine import FoldData, FoldLoader, ModelFactory, TrainingEngine


class Trainer(Protocol):
    def __call__(self, job: TrainingJob) -> dict: ...


__all__ = ["FoldData", "FoldLoader", "ModelFactory", "Trainer", "TrainingEngine"]
