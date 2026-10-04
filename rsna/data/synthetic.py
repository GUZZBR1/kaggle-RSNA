"""Small deterministic RSNA-like DICOM fixtures for pipeline integration checks."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path
import random
from typing import Any

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage
from ..targets import TARGETS as OFFICIAL_TARGETS


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 42
    patients: int = 8
    studies_per_patient: int = 2
    series_per_study: int = 2
    min_slices: int = 12
    max_slices: int = 16
    missing_label_rate: float = 0.05
    instance_number_trap: bool = True
    injection: str | None = None

    def __post_init__(self) -> None:
        if self.seed < 0 or self.patients < 5 or self.studies_per_patient < 1:
            raise ValueError("seed must be nonnegative and the dataset needs at least five patients")
        if self.series_per_study < 1 or self.min_slices < 2 or self.max_slices < self.min_slices:
            raise ValueError("invalid series or slice counts")
        if not 0 <= self.missing_label_rate <= 1:
            raise ValueError("missing_label_rate must be between 0 and 1")


TARGETS = OFFICIAL_TARGETS


def _uid(seed: int, key: str) -> str:
    raw = hashlib.sha256(f"rsna-synthetic:{seed}:{key}".encode()).digest()[:16]
    return f"2.25.{int.from_bytes(raw, 'big')}"


def generate_synthetic_dataset(config: SyntheticConfig, output_dir: str | Path) -> dict[str, Any]:
    """Write DICOM metadata and a synthetic hard-label CSV; no clinical data or pixels."""
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"synthetic output directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    rows: list[dict[str, Any]] = []
    planes = ("sagittal", "coronal", "axial")
    for patient_number in range(config.patients):
        patient = f"SYN-P{patient_number:03d}"
        for study_number in range(config.studies_per_patient):
            study_key = f"p{patient_number}/st{study_number}"
            study_uid = _uid(config.seed, study_key)
            labels: dict[str, int | str] = {}
            for target_number, target in enumerate(TARGETS):
                label = int((patient_number + study_number + target_number + config.seed) % 2)
                force_missing = (patient_number == 0 and study_number == 0 and target_number == 0
                                 and config.missing_label_rate > 0)
                if force_missing or rng.random() < config.missing_label_rate:
                    labels[target] = ""
                else:
                    labels[target] = label
            rows.append({"study_instance_uid": study_uid, "patient_id": patient, **labels})
            for series_number in range(config.series_per_study):
                series_key = f"{study_key}/se{series_number}"
                series_uid = _uid(config.seed, series_key)
                plane = planes[(patient_number + study_number + series_number) % len(planes)]
                orientation = {
                    "sagittal": [0, 1, 0, 0, 0, 1],
                    "coronal": [1, 0, 0, 0, 0, 1],
                    "axial": [1, 0, 0, 0, 1, 0],
                }[plane]
                row, column = orientation[:3], orientation[3:]
                normal = (row[1] * column[2] - row[2] * column[1],
                          row[2] * column[0] - row[0] * column[2],
                          row[0] * column[1] - row[1] * column[0])
                count = rng.randint(config.min_slices, config.max_slices)
                if (config.injection == "spacing-irregular" and patient_number == 0 and
                        study_number == 0 and series_number == 0):
                    count = max(4, count)
                side = "LEFT" if (patient_number + series_number) % 2 == 0 else "RIGHT"
                pending = []
                for physical_index in range(count):
                    sop_uid = _uid(config.seed, f"{series_key}/sl{physical_index}")
                    if config.injection == "duplicate-sop" and patient_number == 0 and study_number == 0 and series_number == 0 and physical_index == 1:
                        sop_uid = _uid(config.seed, f"{series_key}/sl0")
                    spacing = physical_index * 3.0
                    if (config.injection == "spacing-irregular" and patient_number == 0 and
                            study_number == 0 and series_number == 0 and physical_index == count - 1):
                        spacing += 3.3
                    position = [value * spacing for value in normal]
                    missing_position = config.injection == "missing-position" and patient_number == 0 and study_number == 0 and series_number == 0
                    bad_orientation = config.injection == "orientation-conflict" and physical_index == count - 1 and patient_number == 0 and study_number == 0 and series_number == 0
                    item = {
                        "sop_uid": sop_uid, "study_uid": study_uid, "series_uid": series_uid,
                        "patient": patient, "series_key": series_key, "plane": plane,
                        "path_token": hashlib.sha256(f"{series_key}:{physical_index}".encode()).hexdigest()[:12],
                        "orientation": ([1, 0, 0, 0, 1, 0] if bad_orientation else orientation),
                        "position": None if missing_position else position,
                        "instance_number": count - physical_index if config.instance_number_trap else physical_index + 1,
                        "missing_pixel_spacing": (config.injection == "missing-metadata" and
                            patient_number == 0 and study_number == 0 and series_number == 0 and physical_index == 0),
                        "side": ("RIGHT" if physical_index == count - 1 else side)
                            if config.injection == "orientation-conflict" and patient_number == 0 and study_number == 0 and series_number == 0
                            else side,
                        "pixel_data": bytes([physical_index % 251]) * (16 * 16),
                    }
                    pending.append(item)
                # Reverse file creation order to avoid leaking anatomy through directory order.
                for item in reversed(pending):
                    _write_dicom(root, item)
    with (root / "targets.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("study_instance_uid", "patient_id", *TARGETS))
        writer.writeheader()
        writer.writerows(rows)
    return {"patients": config.patients, "studies": len(rows),
            "series": len(rows) * config.series_per_study,
            "target_names": list(TARGETS), "synthetic": True}


def _write_dicom(root: Path, item: dict[str, Any]) -> None:
    # Keep stable relative paths even when InstanceNumber is deliberately wrong.
    relative = Path("raw") / item["series_key"] / f"slice_{item['path_token']}.dcm"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = item["sop_uid"]
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = _uid(0, "synthetic-implementation")
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = item["sop_uid"]
    ds.StudyInstanceUID = item["study_uid"]
    ds.SeriesInstanceUID = item["series_uid"]
    ds.PatientID = item["patient"]
    ds.Modality = "MR"
    ds.SeriesDescription = item["plane"].title()
    ds.Laterality = item["side"][0]
    ds.ImageLaterality = item["side"][0]
    ds.InstanceNumber = item["instance_number"]
    if item["position"] is not None:
        ds.ImagePositionPatient = [str(value) for value in item["position"]]
    ds.ImageOrientationPatient = [str(value) for value in item["orientation"]]
    if not item["missing_pixel_spacing"]:
        ds.PixelSpacing = ["0.4", "0.4"]
    ds.SliceThickness = "3.0"
    ds.SpacingBetweenSlices = "3.0"
    ds.Rows = 16
    ds.Columns = 16
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = item["pixel_data"]
    ds.save_as(path, enforce_file_format=True)
