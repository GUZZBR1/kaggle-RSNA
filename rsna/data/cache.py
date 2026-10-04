"""Persistent, metadata-only dataset index cache with incremental refresh."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Any

from ..identity import digest
from .dicom import read_dicom_metadata
from .index import DISCOVERY_VERSION, _NON_IMAGE_SUFFIXES, build_index
from .models import DatasetIndex, SliceRecord

INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RefreshReport:
    mode: str
    added: int
    modified: int
    removed: int
    reused: int
    reparsed: int
    warnings: tuple[str, ...] = ()
    scan_seconds: float = 0.0
    parse_seconds: float = 0.0
    load_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "added": self.added, "modified": self.modified,
                "removed": self.removed, "reused": self.reused, "reparsed": self.reparsed,
                "warnings": list(self.warnings), "scan_seconds": self.scan_seconds,
                "parse_seconds": self.parse_seconds, "load_seconds": self.load_seconds}


@dataclass(frozen=True)
class IndexResult:
    index: DatasetIndex
    report: RefreshReport
    source_fingerprint: str


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _cache_rows(path: Path, dataset_version_id: str | None) -> tuple[dict[str, tuple], dict]:
    try:
        with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
            manifest = dict(db.execute("select key, value from manifest"))
            rows = {r[0]: (r[1], r[2], r[3], r[4], r[5]) for r in db.execute(
                "select path,size,mtime_ns,kind,status,record from files")}
        if int(manifest["schema_version"]) != INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported index schema version")
        required = {"schema_version", "dataset_version_id", "file_count", "source_fingerprint",
                    "index_id", "payload_sha256", "record_count"}
        if not required.issubset(manifest):
            raise ValueError("cache manifest is missing required fields")
        if manifest["dataset_version_id"] != (dataset_version_id or ""):
            raise ValueError("DatasetVersion binding mismatch")
        payload = _json([[p, *rows[p]] for p in sorted(rows)]).encode()
        if hashlib.sha256(payload).hexdigest() != manifest["payload_sha256"]:
            raise ValueError("index payload hash mismatch")
        if len(rows) != int(manifest["file_count"]):
            raise ValueError("index record count mismatch")
        return rows, manifest
    except (sqlite3.Error, OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid dataset index cache: {exc}") from exc


def load_or_refresh(root: str | Path, cache_path: str | Path, *,
                    dataset_version_id: str | None = None, refresh: bool = False,
                    rebuild: bool = False, validate_only: bool = False,
                    cache_policy: str = "rebuild", on_invalid: str = "warn") -> IndexResult:
    """Load a valid cache or parse only added/changed DICOM candidates.

    The source fingerprint uses relative path, size and nanosecond mtime. It never
    hashes entire DICOM bodies during a warm check; headers are parsed only when
    this stat fingerprint changes for a candidate.
    """
    if cache_policy not in {"strict", "rebuild"}:
        raise ValueError("cache_policy must be strict or rebuild")
    if on_invalid not in {"strict", "warn", "skip-invalid"}:
        raise ValueError("invalid on_invalid policy")
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)
    target = Path(cache_path)
    target_absolute = target.expanduser().resolve()
    try:
        target_absolute.relative_to(root_path)
    except ValueError:
        pass
    else:
        raise ValueError("index cache must be stored outside the raw dataset root")
    target = target_absolute
    t0 = time.perf_counter()
    files = sorted((p for p in root_path.rglob("*") if p.is_file()),
                   key=lambda p: p.relative_to(root_path).as_posix())
    states = {}
    for path in files:
        rel = path.relative_to(root_path).as_posix()
        st = path.stat()
        suffix = path.suffix.lower()
        kind = "metadata" if suffix in {".csv", ".tsv"} else (
            "ignore" if suffix in _NON_IMAGE_SUFFIXES or suffix not in {".dcm", ""} else "dicom")
        states[rel] = (st.st_size, st.st_mtime_ns, kind)
    scan_seconds = time.perf_counter() - t0
    source_fingerprint = digest([[p, *states[p]] for p in sorted(states)])
    old_rows: dict[str, tuple] = {}
    cached_index: DatasetIndex | None = None
    cache_error = None
    if target.exists() and not rebuild:
        try:
            old_rows, manifest = _cache_rows(target, dataset_version_id)
            old_files = [root_path / rel for rel in sorted(old_rows)]
            cached_index = _load_index(old_rows, root_path, old_files,
                                      on_invalid=manifest.get("on_invalid", "warn"))
            if (cached_index.index_id != manifest["index_id"] or
                    cached_index.statistics["n_slices"] != int(manifest["record_count"])):
                raise ValueError("cached index identity or record count mismatch")
        except ValueError as exc:
            cache_error = str(exc)
            if cache_policy == "strict" or validate_only:
                raise
    elif not target.exists() and cache_policy == "strict" and not rebuild:
        raise FileNotFoundError(f"dataset index cache does not exist: {target}")
    if validate_only:
        if not target.exists():
            raise FileNotFoundError(f"dataset index cache does not exist: {target}")
        old_states = {p: tuple(row[:3]) for p, row in old_rows.items()}
        if old_states != states:
            raise ValueError("source fingerprint mismatch")
        if manifest.get("on_invalid", "warn") != on_invalid:
            raise ValueError("cache invalid-file policy mismatch")
        assert cached_index is not None
        return IndexResult(cached_index,
            RefreshReport("validate-only", 0, 0, 0, len(states), 0,
                          scan_seconds=scan_seconds), source_fingerprint)

    added_paths = set(states) - set(old_rows)
    removed_paths = set(old_rows) - set(states)
    modified_paths = {p for p in set(states) & set(old_rows)
                      if tuple(states[p]) != tuple(old_rows[p][:3])}
    changed = added_paths | modified_paths
    parse_start = time.perf_counter()
    warnings: list[str] = []
    records = dict(old_rows)
    for rel in removed_paths:
        records.pop(rel, None)
    for rel in changed:
        size, mtime_ns, kind = states[rel]
        if kind != "dicom":
            records[rel] = (size, mtime_ns, kind, "ignored", None)
            continue
        try:
            metadata = read_dicom_metadata(root_path / rel)
            record = SliceRecord(rel, size, metadata, tuple(metadata["metadata_warnings"]))
            records[rel] = (size, mtime_ns, kind, "ok", _json(record.to_dict()))
        except Exception as exc:
            message = f"invalid DICOM {rel}: {type(exc).__name__}: {exc}"
            if on_invalid == "strict":
                raise ValueError(message) from exc
            warnings.append(message)
            records[rel] = (size, mtime_ns, kind, "invalid", _json({"warning": message}))
    parse_seconds = time.perf_counter() - parse_start
    policy_changed = bool(old_rows and manifest.get("on_invalid", "warn") != on_invalid)
    if cache_error or rebuild:
        mode = "rebuild"
    elif not target.exists():
        mode = "cold-build"
    elif not refresh and not changed and not removed_paths and not policy_changed:
        mode = "warm-load"
    else:
        mode = "incremental-refresh"
    # Bind current file state, including ignored and tabular files.
    normalized_rows = {p: (states[p][0], states[p][1], states[p][2],
                           records.get(p, (0, 0, "", "ignored", None))[3],
                           records.get(p, (0, 0, "", "ignored", None))[4]) for p in states}
    index = (cached_index if mode == "warm-load" else
             _load_index(normalized_rows, root_path, files, on_invalid=on_invalid))
    assert index is not None
    if mode == "warm-load" and (index.index_id != manifest["index_id"] or
                                  index.statistics["n_slices"] != int(manifest["record_count"]) or
                                  manifest["source_fingerprint"] != source_fingerprint):
        raise ValueError("cached index identity, source fingerprint or record count mismatch")
    report = RefreshReport(mode, len(added_paths), len(modified_paths), len(removed_paths),
        len(states) - len(changed), sum(states[p][2] == "dicom" for p in changed),
        tuple(warnings + ([cache_error] if cache_error else [])),
        scan_seconds, parse_seconds, time.perf_counter() - parse_start - parse_seconds)
    if mode != "warm-load":
        _write_cache(target, normalized_rows, dataset_version_id, source_fingerprint, index, on_invalid)
    return IndexResult(index, report, source_fingerprint)


def _load_index(rows: dict[str, tuple], root: Path, files: list[Path], *,
                on_invalid: str = "warn") -> DatasetIndex:
    slices = []
    warnings = []
    invalid = 0
    metadata = []
    for rel, row in sorted(rows.items()):
        size, _, kind, status, value = row
        if kind == "metadata":
            metadata.append(rel)
        if kind == "dicom" and status == "ok":
            slices.append(SliceRecord(**json.loads(value)))
        elif kind == "dicom" and status == "invalid":
            invalid += 1
            warning = json.loads(value)["warning"]
            if on_invalid == "strict":
                raise ValueError(f"invalid cached DICOM {rel}: {warning}")
            if on_invalid == "warn":
                warnings.append(warning)
    return build_index(root, files, slices, warnings, invalid, metadata)


def _write_cache(path: Path, rows: dict[str, tuple], dataset_version_id: str | None,
                 source_fingerprint: str, index: DatasetIndex, on_invalid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json([[p, *rows[p]] for p in sorted(rows)]).encode()
    manifest = {"schema_version": str(INDEX_SCHEMA_VERSION),
                "dataset_version_id": dataset_version_id or "", "on_invalid": on_invalid,
                "file_count": str(len(rows)),
                "source_fingerprint": source_fingerprint, "index_id": index.index_id,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "record_count": str(index.statistics["n_slices"])}
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with closing(sqlite3.connect(temp)) as db:
            db.execute("create table manifest (key text primary key, value text not null)")
            db.execute("create table files (path text primary key, size integer, mtime_ns integer, kind text, status text, record text)")
            db.executemany("insert into manifest values (?,?)", sorted(manifest.items()))
            db.executemany("insert into files values (?,?,?,?,?,?)",
                           [(p, *rows[p]) for p in sorted(rows)])
            db.commit()
        fd = os.open(temp, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
