"""Metadata-only dataset inspection and validation services."""

from .query import (DatasetIndex, DatasetReadError, EntityNotFoundError,
                    inspect_manifest, inspect_series, inspect_slice, inspect_study,
                    load_dataset_index, sample_entities)
from .summary import build_dataset_stats, build_dataset_summary
from .validate import ValidationReport, validate_artifact, validate_dataset

__all__ = [
    "DatasetIndex", "DatasetReadError", "EntityNotFoundError", "ValidationReport",
    "build_dataset_stats", "build_dataset_summary", "inspect_manifest", "inspect_series",
    "inspect_slice", "inspect_study", "load_dataset_index", "sample_entities",
    "validate_artifact", "validate_dataset",
]
