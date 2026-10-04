"""RSNA MRI indexing, geometry, and persistent cache contracts."""

from .cache import INDEX_SCHEMA_VERSION, IndexResult, RefreshReport, load_or_refresh
from .dicom import read_dicom_metadata
from .geometry import (GeometryConfig, GeometryWarning, OrderingResult, SeriesGeometry,
                       order_series_slices)
from .index import discover_dataset
from .laterality import (LateralityEvidence, LateralityResolution, parse_laterality_text,
                         resolve_series_laterality, resolve_study_laterality)
from .manifest import load_manifest, save_manifest
from .models import DatasetIndex, SeriesRecord, SliceRecord, StudyRecord
from .normalization import NormalizationPlan, build_normalization_plan
from .orientation import (OrientationDescriptor, assess_orientation_consistency,
                          describe_series_orientation)
from .provenance import OrientationConfig, OrientationProvenance, build_orientation_provenance
from .selection import SliceSelectionConfig, SliceSelectionResult, SliceSelector, select_slices

__all__ = ["INDEX_SCHEMA_VERSION", "IndexResult", "RefreshReport", "load_or_refresh", "DatasetIndex",
           "LateralityEvidence", "LateralityResolution", "NormalizationPlan",
           "OrientationConfig", "OrientationDescriptor", "OrientationProvenance", "SeriesRecord",
           "SliceRecord", "StudyRecord", "SliceSelectionConfig", "SliceSelectionResult", "SliceSelector",
           "GeometryConfig", "OrderingResult", "SeriesGeometry", "GeometryWarning",
           "assess_orientation_consistency", "build_normalization_plan",
           "build_orientation_provenance", "describe_series_orientation", "discover_dataset",
           "load_manifest", "parse_laterality_text", "read_dicom_metadata", "resolve_series_laterality",
           "resolve_study_laterality", "save_manifest", "select_slices", "order_series_slices"]
