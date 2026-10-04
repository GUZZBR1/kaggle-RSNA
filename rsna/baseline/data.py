"""Deterministic DICOM-to-tensor adapter for the CNN-224 baseline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..data.models import DatasetIndex, SliceRecord
from ..data.selection import SliceSelectionConfig, SliceSelector
from ..labels import LabelRecord
from ..targets import TARGETS
from ..training.data import StudyTensorRecord


def _torch():
    try:
        import torch
        from torch.nn import functional as F
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("CNN-224 data loading requires the optional 'torch' dependency") from exc
    return torch, F


def _decode_dicom_pixels(path: Path) -> Any:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - pydicom is a project dependency
        raise RuntimeError("DICOM pixel decoding requires pydicom") from exc
    return pydicom.dcmread(str(path)).pixel_array


def load_study_tensors(
    dataset_index: DatasetIndex,
    raw_root: str | Path,
    selected_series_by_study: Mapping[str, str],
    labels: Mapping[str, LabelRecord] | Sequence[LabelRecord],
    *,
    slice_count: int = 24,
    input_size: int = 224,
    pixel_decoder: Callable[[Path], Any] | None = None,
    tensor_store_path: str | Path | None = None,
) -> tuple[StudyTensorRecord, ...]:
    """Decode indexed studies into `[slices, 1, height, width]` tensors.

    ``selected_series_by_study`` must explicitly map every indexed StudyInstanceUID
    to exactly one SeriesInstanceUID. Labels are keyed by StudyInstanceUID (the
    ``LabelRecord.study_id`` semantic key). Selection uses canonical geometry ordering
    followed by exactly ``slice_count`` uniformly spaced slices; short series fail.

    Pixel values are clipped at the pooled per-study 1st and 99th percentiles and
    scaled to `[0, 1]`, then bilinearly resized to the configured square. ``pixel_decoder``
    is an optional test/smoke seam receiving each resolved DICOM path. If
    ``tensor_store_path`` is set, outputs are written study-by-study to a float32
    `.npy` memory map and returned tensors view that disk-backed storage.
    """
    torch, F = _torch()
    if not isinstance(dataset_index, DatasetIndex):
        raise TypeError("dataset_index must be a DatasetIndex")
    if type(slice_count) is not int or slice_count < 1:
        raise ValueError("slice_count must be a positive integer")
    if type(input_size) is not int or input_size < 1:
        raise ValueError("input_size must be a positive integer")
    if not isinstance(selected_series_by_study, Mapping):
        raise TypeError("selected_series_by_study must map StudyInstanceUID to SeriesInstanceUID")
    studies = {study.study_instance_uid: study for study in dataset_index.studies}
    if any(not uid for uid in studies) or len(studies) != len(dataset_index.studies):
        raise ValueError("every indexed study must have a unique StudyInstanceUID")
    expected_uids = set(studies)
    if any(not isinstance(uid, str) or not uid.strip() for uid in selected_series_by_study):
        raise ValueError("series mapping study UIDs must be nonempty strings")
    if set(selected_series_by_study) != expected_uids:
        missing, extra = sorted(expected_uids - set(selected_series_by_study)), sorted(set(selected_series_by_study) - expected_uids)
        raise ValueError(f"series mapping must exactly cover indexed studies (missing={missing}, extra={extra})")
    normalized_series: dict[str, str] = {}
    for study_uid, series_uid in selected_series_by_study.items():
        if not isinstance(study_uid, str) or not study_uid.strip():
            raise ValueError("series mapping study UIDs must be nonempty strings")
        if not isinstance(series_uid, str) or not series_uid.strip():
            raise ValueError(f"selected series UID for study {study_uid!r} must be nonempty")
        normalized_series[study_uid] = series_uid

    if isinstance(labels, Mapping):
        label_records = dict(labels)
        if any(not isinstance(uid, str) or not uid.strip() for uid in label_records):
            raise ValueError("label mapping keys must be nonempty StudyInstanceUID values")
        if any(not isinstance(record, LabelRecord) for record in label_records.values()):
            raise TypeError("labels mapping values must be LabelRecord values")
        if any(uid != record.study_id for uid, record in label_records.items()):
            raise ValueError("label mapping keys must match LabelRecord.study_id")
    elif isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        label_records = {}
        for record in labels:
            if not isinstance(record, LabelRecord):
                raise TypeError("labels sequence must contain only LabelRecord values")
            if record.study_id in label_records:
                raise ValueError(f"duplicate label record for study {record.study_id!r}")
            label_records[record.study_id] = record
    else:
        raise TypeError("labels must be a UID mapping or a sequence of LabelRecord values")
    if set(label_records) != expected_uids:
        missing, extra = sorted(expected_uids - set(label_records)), sorted(set(label_records) - expected_uids)
        raise ValueError(f"labels must exactly cover indexed studies (missing={missing}, extra={extra})")

    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    tensor_store = None
    if tensor_store_path is not None:
        if not expected_uids:
            raise ValueError("cannot create a tensor store for an empty dataset")
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - torch normally supplies numpy
            raise RuntimeError("disk-backed tensor storage requires numpy") from exc
        store_path = Path(tensor_store_path).expanduser().resolve()
        store_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_store = np.lib.format.open_memmap(
            store_path, mode="w+", dtype=np.float32,
            shape=(len(expected_uids), slice_count, 1, input_size, input_size),
        )
    decoder = _decode_dicom_pixels if pixel_decoder is None else pixel_decoder
    selector = SliceSelector(SliceSelectionConfig(
        strategy="uniform", count=slice_count, short_series_policy="strict"))
    records: list[StudyTensorRecord] = []
    for record_index, study_uid in enumerate(sorted(expected_uids)):
        study = studies[study_uid]
        if not study.patient_id:
            raise ValueError(f"study {study_uid!r} has no PatientID")
        series_uid = normalized_series[study_uid]
        matches = [series for series in study.series if series.series_instance_uid == series_uid]
        if len(matches) != 1:
            raise ValueError(f"study {study_uid!r} must contain exactly one selected series {series_uid!r}")
        series = matches[0]
        if series.study_instance_uid != study_uid:
            raise ValueError(f"selected series {series_uid!r} has mismatched StudyInstanceUID")
        if not series.slices:
            raise ValueError(f"selected series {series_uid!r} has no slices")
        _validate_indexed_slices(series.slices, study_uid, series_uid)
        selected = selector.select(series, study=study)
        if selected.actual_count != slice_count or len(selected.selected) != slice_count:
            raise ValueError(f"study {study_uid!r} did not yield exactly {slice_count} selected slices")
        selected_slices = tuple(selected.selected)
        if any(not isinstance(item, SliceRecord) for item in selected_slices):
            raise ValueError(f"study {study_uid!r} selection contains missing or invalid slice references")
        selected_sops = tuple(str(item.metadata["SOPInstanceUID"]) for item in selected_slices)
        if len(set(selected_sops)) != slice_count:
            raise ValueError(f"study {study_uid!r} selection contains duplicate SOPInstanceUID values")

        arrays = []
        expected_shape = None
        for item in selected_slices:
            path = (root / item.relative_path).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"slice path escapes raw_root: {item.relative_path!r}")
            if not path.is_file():
                raise FileNotFoundError(path)
            try:
                pixels = torch.as_tensor(decoder(path))
            except Exception as exc:
                raise ValueError(f"could not decode DICOM pixels at {item.relative_path!r}") from exc
            if pixels.ndim != 2:
                raise ValueError(f"DICOM pixels must be a 2D grayscale array at {item.relative_path!r}")
            if pixels.is_complex():
                raise ValueError(f"DICOM pixels must be real-valued at {item.relative_path!r}")
            if pixels.shape[0] < 1 or pixels.shape[1] < 1:
                raise ValueError(f"DICOM pixel array is empty at {item.relative_path!r}")
            if expected_shape is None:
                expected_shape = tuple(pixels.shape)
            elif tuple(pixels.shape) != expected_shape:
                raise ValueError(f"DICOM slice shape mismatch in study {study_uid!r}")
            pixels = pixels.to(dtype=torch.float32)
            if not bool(torch.isfinite(pixels).all()):
                raise ValueError(f"DICOM pixels contain non-finite values at {item.relative_path!r}")
            arrays.append(pixels)

        stack = torch.stack(arrays, dim=0)
        low, high = torch.quantile(stack.flatten(), torch.tensor([0.01, 0.99], dtype=stack.dtype)).unbind()
        if not bool(torch.isfinite(low) & torch.isfinite(high)):
            raise ValueError(f"study {study_uid!r} has no finite pixel intensity range")
        if float(high) <= float(low):
            normalized = torch.zeros_like(stack)
        else:
            normalized = (stack.clamp(min=low, max=high) - low) / (high - low)
        resized = F.interpolate(normalized.unsqueeze(1), size=(input_size, input_size),
                                mode="bilinear", align_corners=False)
        if tuple(resized.shape) != (slice_count, 1, input_size, input_size):
            raise ValueError(f"preprocessed study {study_uid!r} has an invalid tensor shape")
        if not bool(torch.isfinite(resized).all()):
            raise ValueError(f"preprocessed study {study_uid!r} contains non-finite values")
        if tensor_store is not None:
            tensor_store[record_index] = resized.numpy()
            inputs = torch.from_numpy(tensor_store[record_index])
        else:
            inputs = resized.contiguous()
        labels_for_study = label_records[study_uid]
        if labels_for_study.patient_id is not None and labels_for_study.patient_id != study.patient_id:
            raise ValueError(f"label PatientID does not match indexed study {study_uid!r}")
        records.append(StudyTensorRecord(
            study_id=study_uid,
            patient_id=study.patient_id,
            inputs=inputs,
            labels=labels_for_study,
            series_ids=(series_uid,),
            sop_ids=selected_sops,
        ))
    if tensor_store is not None:
        tensor_store.flush()
    return tuple(records)


def _validate_indexed_slices(slices: Sequence[SliceRecord], study_uid: str, series_uid: str) -> None:
    seen_sop: set[str] = set()
    seen_path: set[str] = set()
    for item in slices:
        if not isinstance(item, SliceRecord):
            raise ValueError(f"series {series_uid!r} contains an invalid slice record")
        if not isinstance(item.relative_path, str) or not item.relative_path.strip():
            raise ValueError(f"series {series_uid!r} contains a slice without a path")
        if item.relative_path in seen_path:
            raise ValueError(f"series {series_uid!r} contains duplicate slice paths")
        seen_path.add(item.relative_path)
        metadata = item.metadata
        sop_uid = metadata.get("SOPInstanceUID")
        if not isinstance(sop_uid, str) or not sop_uid.strip():
            raise ValueError(f"series {series_uid!r} contains a slice without SOPInstanceUID")
        if sop_uid in seen_sop:
            raise ValueError(f"series {series_uid!r} contains duplicate SOPInstanceUID {sop_uid!r}")
        seen_sop.add(sop_uid)
        if metadata.get("StudyInstanceUID") != study_uid:
            raise ValueError(f"slice {sop_uid!r} has mismatched StudyInstanceUID")
        if metadata.get("SeriesInstanceUID") != series_uid:
            raise ValueError(f"slice {sop_uid!r} has mismatched SeriesInstanceUID")
