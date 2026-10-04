"""DICOM metadata indexing and auditable orientation contracts."""

from .laterality import (LateralityEvidence, LateralityResolution, parse_laterality_text,
                         resolve_series_laterality, resolve_study_laterality)
from .normalization import NormalizationPlan, build_normalization_plan
from .orientation import (OrientationDescriptor, assess_orientation_consistency,
                          describe_series_orientation)
from .provenance import OrientationConfig, OrientationProvenance, build_orientation_provenance

__all__ = ["LateralityEvidence", "LateralityResolution", "NormalizationPlan",
           "OrientationConfig", "OrientationDescriptor", "OrientationProvenance",
           "assess_orientation_consistency", "build_normalization_plan",
           "build_orientation_provenance", "describe_series_orientation", "parse_laterality_text",
           "resolve_series_laterality", "resolve_study_laterality"]
