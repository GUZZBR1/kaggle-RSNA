"""Future-facing interface for materializing a Kaggle submission artifact."""

from __future__ import annotations

from typing import Protocol

from .contracts import PredictionArtifact, SubmissionArtifact


class SubmissionBuilder(Protocol):
    def build(self, predictions: list[PredictionArtifact]) -> SubmissionArtifact: ...
