"""DICOM metadata, orientation, and deterministic slice selection APIs."""

from .laterality import (LateralityEvidence, LateralityResolution, parse_laterality_text,
                         resolve_series_laterality, resolve_study_laterality)
from .normalization import NormalizationPlan, build_normalization_plan
from .orientation import (OrientationDescriptor, assess_orientation_consistency,
                          describe_series_orientation)
from .provenance import OrientationConfig, OrientationProvenance, build_orientation_provenance
from .selection import SliceSelectionConfig, SliceSelectionResult, SliceSelector, select_slices

__all__ = ["LateralityEvidence", "LateralityResolution", "NormalizationPlan",
           "OrientationConfig", "OrientationDescriptor", "OrientationProvenance",
           "SliceSelectionConfig", "SliceSelectionResult", "SliceSelector",
           "assess_orientation_consistency", "build_normalization_plan",
           "build_orientation_provenance", "describe_series_orientation", "parse_laterality_text",
           "resolve_series_laterality", "resolve_study_laterality", "select_slices"]
