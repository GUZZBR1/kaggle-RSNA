"""Read selected DICOM header attributes without decoding pixel data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pydicom


TAGS = ("SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID", "PatientID",
        "InstanceNumber", "ImagePositionPatient", "ImageOrientationPatient", "SliceThickness",
        "SpacingBetweenSlices", "PixelSpacing", "Rows", "Columns", "Modality",
        "SeriesDescription", "ProtocolName", "Laterality", "PatientPosition")


def _value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) or value.__class__.__name__ == "MultiValue":
        return [_value(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return str(value)


def read_dicom_metadata(path: str | Path) -> dict[str, Any]:
    """Read only selected header tags; never access the pixel array."""
    dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=TAGS)
    metadata = {tag: _value(getattr(dataset, tag, None)) for tag in TAGS}
    metadata["metadata_warnings"] = [f"missing {tag}" for tag in
        ("SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID") if not metadata[tag]]
    return metadata
