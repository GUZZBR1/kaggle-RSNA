"""Filesystem discovery and deterministic MRI dataset indexing."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from ..identity import digest
from .dicom import read_dicom_metadata
from .models import DatasetIndex, SeriesRecord, SliceRecord, StudyRecord

DISCOVERY_VERSION = "1"
_NON_IMAGE_SUFFIXES = {".csv", ".tsv", ".json", ".txt", ".md", ".yaml", ".yml", ".toml"}


def discover_dataset(root: str | Path, *, on_invalid: str = "warn") -> DatasetIndex:
    """Index DICOM headers beneath root. Invalid imaging candidates are isolated.

    on_invalid is strict (raise), warn (record warning), or skip-invalid (count and omit warning).
    Known tabular/document files are discovered separately and never parsed as DICOM.
    """
    if on_invalid not in {"strict", "warn", "skip-invalid"}:
        raise ValueError("on_invalid must be strict, warn, or skip-invalid")
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)

    files = sorted((p for p in root_path.rglob("*") if p.is_file()),
                   key=lambda p: p.relative_to(root_path).as_posix())
    slices: list[SliceRecord] = []
    index_warnings: list[str] = []
    invalid_files = 0
    metadata_files: list[str] = []
    for path in files:
        relative = path.relative_to(root_path).as_posix()
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            metadata_files.append(relative)
            continue
        if suffix in _NON_IMAGE_SUFFIXES:
            continue
        if suffix not in {".dcm", ""}:
            continue
        try:
            metadata = read_dicom_metadata(path)
        except Exception as exc:
            invalid_files += 1
            # Include only the relative location and error class so transient local
            # paths in OS exception messages cannot leak into deterministic manifests.
            message = f"invalid DICOM {relative}: {type(exc).__name__}"
            if on_invalid == "strict":
                raise ValueError(message) from exc
            if on_invalid == "warn":
                index_warnings.append(message)
            continue
        slices.append(SliceRecord(relative, path.stat().st_size, metadata,
                                  tuple(metadata["metadata_warnings"])))

    return build_index(root_path, files, slices, index_warnings, invalid_files, metadata_files)


def build_index(root_path: Path, files: list[Path], slices: list[SliceRecord],
                index_warnings: list[str], invalid_files: int,
                metadata_files: list[str]) -> DatasetIndex:
    """Build canonical relationships/statistics from already parsed cached metadata."""
    # Group using DICOM identifiers when present, with relative directories as a stable fallback.
    by_study: dict[str, list[SliceRecord]] = defaultdict(list)
    for item in slices:
        study_uid = item.metadata["StudyInstanceUID"]
        study_key = f"uid:{study_uid}" if study_uid else f"path:{Path(item.relative_path).parent.parent.as_posix()}"
        by_study[study_key].append(item)

    studies: list[StudyRecord] = []
    all_sops: dict[str, list[str]] = defaultdict(list)
    series_owner: dict[str, set[str]] = defaultdict(set)
    for items in by_study.values():
        study_uid = _single(items, "StudyInstanceUID")
        patient_ids = sorted({str(i.metadata["PatientID"]) for i in items if i.metadata["PatientID"]})
        study_warnings = []
        if len(patient_ids) > 1:
            study_warnings.append(f"conflicting PatientID values in study {study_uid or items[0].relative_path}: {patient_ids}")
        by_series: dict[str, list[SliceRecord]] = defaultdict(list)
        for item in items:
            uid = item.metadata["SeriesInstanceUID"]
            key = f"uid:{uid}" if uid else f"path:{Path(item.relative_path).parent.as_posix()}"
            by_series[key].append(item)
            sop = item.metadata["SOPInstanceUID"]
            if sop:
                all_sops[str(sop)].append(item.relative_path)
            if uid:
                series_owner[str(uid)].add(str(item.metadata["StudyInstanceUID"] or ""))
        series_records = []
        for group in by_series.values():
            series_uid = _single(group, "SeriesInstanceUID")
            series_study_uid = _single(group, "StudyInstanceUID")
            warnings = []
            observed_studies = sorted({str(i.metadata["StudyInstanceUID"]) for i in group if i.metadata["StudyInstanceUID"]})
            if len(observed_studies) > 1:
                warnings.append(f"conflicting StudyInstanceUID values in series {series_uid}: {observed_studies}")
            series_records.append(SeriesRecord(series_uid, series_study_uid, tuple(group), tuple(warnings)))
        studies.append(StudyRecord(study_uid, patient_ids[0] if len(patient_ids) == 1 else None,
                                   tuple(series_records), tuple(study_warnings)))

    duplicate_sops = {uid: sorted(paths) for uid, paths in all_sops.items() if len(paths) > 1}
    for uid, paths in sorted(duplicate_sops.items()):
        index_warnings.append(f"duplicate SOPInstanceUID {uid}: {paths}")
    for uid, owners in sorted(series_owner.items()):
        owners.discard("")
        if len(owners) > 1:
            index_warnings.append(f"SeriesInstanceUID {uid} appears in multiple studies: {sorted(owners)}")
    metadata_files.sort()
    statistics = {"n_studies": len(studies), "n_series": sum(len(s.series) for s in studies),
        "n_slices": len(slices), "invalid_files": invalid_files,
        "missing_uid_count": sum(sum(not i.metadata[tag] for tag in
            ("SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID")) for i in slices),
        "duplicate_sop_uid_count": len(duplicate_sops), "metadata_files": len(metadata_files)}
    # Root identity deliberately excludes absolute paths and filesystem timestamps.
    root_identity = digest({"relative_paths": [p.relative_to(root_path).as_posix() for p in files]})
    return DatasetIndex(root_identity, DISCOVERY_VERSION, tuple(studies), tuple(sorted(index_warnings)),
                        statistics, tuple(metadata_files))


def _single(items: list[SliceRecord], field: str) -> str | None:
    values = sorted({str(item.metadata[field]) for item in items if item.metadata[field]})
    return values[0] if len(values) == 1 else None
