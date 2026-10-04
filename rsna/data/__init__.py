"""DICOM metadata indexing and auditable orientation contracts."""

from .dicom import read_dicom_metadata
from .index import discover_dataset
from .laterality import (LateralityEvidence, LateralityResolution, parse_laterality_text,
                         resolve_series_laterality, resolve_study_laterality)
from .manifest import load_manifest, save_manifest
from .models import DatasetIndex, SeriesRecord, SliceRecord, StudyRecord
from .normalization import NormalizationPlan, build_normalization_plan
from .orientation import (OrientationDescriptor, assess_orientation_consistency,
                          describe_series_orientation)
from .provenance import OrientationConfig, OrientationProvenance, build_orientation_provenance

__all__ = ["DatasetIndex", "LateralityEvidence", "LateralityResolution", "NormalizationPlan",
           "OrientationConfig", "OrientationDescriptor", "OrientationProvenance", "SeriesRecord",
           "SliceRecord", "StudyRecord", "assess_orientation_consistency", "build_normalization_plan",
           "build_orientation_provenance", "describe_series_orientation", "discover_dataset",
           "load_manifest", "parse_laterality_text", "read_dicom_metadata", "resolve_series_laterality",
           "resolve_study_laterality", "save_manifest"]
