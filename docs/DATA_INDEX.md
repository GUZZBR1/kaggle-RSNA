# RSNA dataset metadata index

The `rsna.data` package provides the first filesystem and MRI metadata layer. It is
independent of experiment execution and does not assign labels, folds, or model inputs.

```text
filesystem
   ↓
DICOM metadata reader (pydicom, stop_before_pixels)
   ↓
SliceRecord
   ↓
SeriesRecord
   ↓
StudyRecord
   ↓
DatasetIndex
   ↓
versioned JSON manifest
   ↓
DatasetVersion (source manifest hash + optional index artifact ID)
```

Run `python -m rsna data-index --input ROOT --output DIRECTORY`. The recursive discovery
uses DICOM Study/Series UIDs as grouping keys and stable relative-directory fallbacks when
UIDs are missing. CSV/TSV files are listed in `metadata_files`; other common document formats
are ignored. Files with `.dcm` or no extension are read as DICOM candidates. Invalid files
are isolated by default (`--on-invalid warn`); `strict` raises at the first invalid candidate
and `skip-invalid` counts and omits it without a warning. DICOM errors are local to the file.

Only selected metadata tags are requested using `pydicom.dcmread(..., stop_before_pixels=True)`.
No pixel arrays are loaded. Geometry fields are preserved, but slices are not anatomically
sorted. The index detects duplicate SOP UIDs, series UIDs associated with multiple studies,
conflicting patient IDs within one study, and missing core UIDs. It does not reconcile values.

The manifest contains `manifest_schema_version`, discovery version, relative source paths,
studies, series, slices, warnings, statistics, metadata-table paths, and a deterministic
index ID. `save_manifest` returns the SHA-256 of the exact saved bytes; `load_manifest`
reconstructs the immutable records without scanning the source tree. Root paths, timestamps,
hostnames, and execution order do not enter identities. Relative paths and file sizes do, so
moving the dataset root intact preserves IDs while changing relative layout changes IDs.

To bind the manifest into experiments, set `DatasetVersion.source_manifest_sha256` to the
manifest byte hash and optionally `dataset_index_artifact_id` to the manifest artifact hash.
The new optional field is excluded from legacy IDs when omitted, preserving existing
`DatasetVersion` identities. No class labels are inferred by indexing.

Known limits: symlinks are followed by `Path.rglob` behavior of the host Python version;
compressed DICOM, nonstandard extensionless binary discovery, pixel validation, anatomical
ordering, label/CSV joins, and parallel scanning are outside this layer.
