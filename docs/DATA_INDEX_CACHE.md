# Persistent dataset index cache

`python -m rsna data-index --input /data/rsna --output artifacts/dataset-index`
creates `index.sqlite3` as operational storage and an inspectable
`manifest.json` export of the canonical `DatasetIndex`. The SQLite cache stores only relative
paths, file state, parse status, and selected DICOM header metadata. It never
stores pixel arrays, images, tensors, embeddings, or model features.

The canonical index is the existing deterministic `DatasetIndex`; its schema
version is `INDEX_SCHEMA_VERSION` in `rsna.data.cache`. Its identity includes
metadata and relative paths, not the absolute root, timestamps, timing metrics,
or storage location. Optional `--dataset-version-id` binds the cache to a
`DatasetVersion` identity.

## Execution paths

- **Cold:** enumerate files, parse candidate DICOM headers, construct canonical
  relationships, and atomically replace the SQLite cache.
- **Warm:** enumerate and stat files, compare relative path, size, and
  nanosecond mtime with cached state, validate cache hashes and identities, and
  load metadata without opening DICOMs.
- **Changed:** reuse unchanged per-file records, parse only new or changed DICOM
  candidates, drop removed paths, rebuild relationships/counts, then atomically
  replace the cache.

The source fingerprint hashes sorted `(relative_path, size, mtime_ns, kind)`
tuples. It deliberately avoids reading every file body on warm runs. A content
change that preserves both size and nanosecond mtime cannot be detected by this
fast strategy; use `--rebuild` when external tooling may preserve both values.
Renames are treated as removal plus addition. Candidate suffixes are `.dcm` or
no suffix; known document extensions and other suffixes are ignored, while CSV
and TSV files are tracked as metadata inputs.

## CLI controls

- `--refresh` forces an incremental check.
- `--rebuild` reparses all candidate DICOMs.
- `--validate-only` validates cache and source without writing.
- `--cache-policy strict|rebuild` rejects invalid/missing caches or rebuilds
  and reports why. The default is `rebuild`.
- `--on-invalid strict|warn|skip-invalid` controls malformed DICOM handling.

SQLite holds per-file operational cache state; `DatasetIndex` and its JSON
manifest remain the canonical semantic representation. Transactions plus a
single-file atomic replacement prevent readers from seeing a partial cache. The
format stays portable and inspectable with standard tools.
