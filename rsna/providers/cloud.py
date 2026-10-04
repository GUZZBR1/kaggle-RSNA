"""Cloud execution boundary; concrete infrastructure is supplied by the caller."""

from __future__ import annotations

from typing import Callable

from ..contracts import TrainingJob, TrainingResult


class CloudProvider:
    def __init__(self, submit: Callable[[TrainingJob], str],
                 fetch_result: Callable[[str], TrainingResult]):
        self._submit = submit
        self._fetch_result = fetch_result

    def submit_training(self, job: TrainingJob) -> str:
        return self._submit(job)

    def get_training_result(self, execution_id: str) -> TrainingResult:
        return self._fetch_result(execution_id)
