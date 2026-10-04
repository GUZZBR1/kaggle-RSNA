"""Reusable structural and byte-integrity validation; never decodes DICOM pixels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlparse

from .. import contracts

VALIDATOR_VERSION = "1.0"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    entity: str | None = None


@dataclass
class ValidationReport:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    info: list[ValidationIssue] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    validator_version: str = VALIDATOR_VERSION
    level: str = "basic"
    warnings_as_errors: bool = False
    corrupted: bool = False

    @property
    def passed(self) -> bool:
        return not self.errors and not (self.warnings_as_errors and self.warnings)

    @property
    def exit_code(self) -> int:
        return 3 if self.corrupted else (0 if self.passed else 1)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, **asdict(self), "exit_code": self.exit_code}

    def error(self, code: str, message: str, entity: str | None = None) -> None:
        self.errors.append(ValidationIssue(code, message, entity))

    def warn(self, code: str, message: str, entity: str | None = None) -> None:
        self.warnings.append(ValidationIssue(code, message, entity))


def _report(level: str, warnings_as_errors: bool) -> ValidationReport:
    if level not in {"basic", "full"}:
        raise ValueError("validation level must be basic or full")
    return ValidationReport(level=level, warnings_as_errors=warnings_as_errors)


def _read_json(path: Path, report: ValidationReport) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ValueError(f"invalid JSON constant: {value}")))
    except (OSError, UnicodeError, ValueError) as exc:
        report.error("UNREADABLE_ARTIFACT", f"Cannot read JSON manifest {path}: {exc}")
        report.corrupted = True
        return None


def _version(value: Mapping[str, Any], report: ValidationReport, entity: str) -> None:
    version = value.get("schema_version")
    if version is None:
        report.warn("MISSING_SCHEMA_VERSION", "Schema version is not declared", entity)
    elif type(version) is not int or version != 1:
        report.error("UNSUPPORTED_SCHEMA_VERSION", "schema_version must be integer 1", entity)


def _contract(value: Mapping[str, Any], cls: type, report: ValidationReport,
              entity: str) -> None:
    """Reuse the foundation contracts as the authoritative schema and ID checks."""
    try:
        data = dict(value)
        filename = re.sub(r"(?<!^)(?=[A-Z])", "-", cls.__name__).lower()
        if cls is contracts.ArtifactReference:
            filename = "artifact"
        schema_path = Path(__file__).resolve().parents[2] / "schemas" / f"{filename}.schema.json"
        if schema_path.is_file():
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            for key in schema.get("required", []):
                if key not in data:
                    raise ValueError(f"required field missing: {key}")
            for key, definition in schema.get("properties", {}).items():
                if key in data and "type" in definition:
                    kinds = definition["type"]
                    kinds = kinds if isinstance(kinds, list) else [kinds]
                    if not any(_schema_type(data[key], kind) for kind in kinds):
                        raise ValueError(f"{key} must have type {' or '.join(kinds)}")
                if key in data and "pattern" in definition and isinstance(data[key], str):
                    if re.search(definition["pattern"], data[key]) is None:
                        raise ValueError(f"{key} does not match its declared schema pattern")
        for key in ("artifact", "checkpoint"):
            nested = data.get(key)
            if nested is not None and isinstance(nested, Mapping):
                if key == "artifact":
                    data[key] = contracts.ArtifactReference(**nested)
                else:
                    checkpoint = dict(nested)
                    checkpoint["artifact"] = contracts.ArtifactReference(**checkpoint["artifact"])
                    data[key] = contracts.CheckpointArtifact(**checkpoint)
        # Python's bool is an int; the serialized schema requires an integer.
        if "schema_version" not in data or type(data["schema_version"]) is not int:
            raise ValueError("schema_version is required and must be an integer")
        identity = next((key for key in data if key == "artifact_id" or
                         key == "dataset_version_id" and cls is contracts.DatasetVersion), None)
        if identity is not None and not _is_sha(data[identity]):
            raise ValueError(f"{identity} must be a lowercase SHA-256 digest")
        cls(**data)
    except (TypeError, ValueError, KeyError, AttributeError, OSError) as exc:
        report.error("SCHEMA_INVALID", f"{cls.__name__}: {exc}", entity)


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _schema_type(value: Any, kind: str) -> bool:
    return {"object": lambda: isinstance(value, Mapping),
            "array": lambda: isinstance(value, (list, tuple)),
            "string": lambda: isinstance(value, str),
            "integer": lambda: type(value) is int,
            "number": lambda: type(value) in (int, float),
            "boolean": lambda: type(value) is bool,
            "null": lambda: value is None}[kind]()


def _reference_path(value: Any, base: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    # A Windows drive prefix is a local path, not a network scheme.
    if parsed.scheme and not (len(parsed.scheme) == 1 and value[1:3] in {":/", ":\\"}):
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _check_file(reference: Mapping[str, Any], base: Path, report: ValidationReport,
                entity: str) -> None:
    location = reference.get("path", reference.get("relative_path", reference.get("uri")))
    if location is None:
        return
    if not isinstance(location, str) or not location:
        report.error("INVALID_REFERENCE", "Reference path must be a nonempty string", entity)
        return
    try:
        path = _reference_path(location, base)
    except ValueError as exc:
        report.error("INVALID_REFERENCE", str(exc), entity)
        return
    if path is None:
        report.warn("REFERENCE_UNVERIFIED", f"Cannot verify non-local reference: {location}", entity)
        return
    report.statistics["referenced_files_checked"] = report.statistics.get("referenced_files_checked", 0) + 1
    try:
        if not path.is_file():
            report.error("MISSING_REFERENCED_FILE", f"Referenced file does not exist: {path}", entity)
            return
        expected = reference.get("sha256", reference.get("payload_sha256", reference.get(
            "hash", reference.get("source_manifest_sha256"))))
        if expected is not None and not _is_sha(expected):
            report.error("INVALID_HASH", "Expected hash must be a lowercase SHA-256 digest", entity)
            return
        if expected is not None:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            report.statistics["hashes_checked"] = report.statistics.get("hashes_checked", 0) + 1
            if hasher.hexdigest() != expected:
                report.error("HASH_MISMATCH", f"SHA-256 mismatch for {path}", entity)
        elif report.level == "full":
            report.warn("MISSING_HASH", f"No hash declared for {path}", entity)
    except (OSError, ValueError) as exc:
        report.error("UNREADABLE_REFERENCE", f"Cannot read {path}: {exc}", entity)
        report.corrupted = True


def _references(value: Any, base: Path, report: ValidationReport, entity: str = "manifest",
                strings_are_paths: bool = False) -> None:
    if isinstance(value, Mapping):
        if any(key in value for key in ("path", "relative_path", "uri")):
            _check_file(value, base, report, entity)
        for key, item in value.items():
            if isinstance(item, (Mapping, list, tuple)):
                _references(item, base, report, f"{entity}.{key}",
                            strings_are_paths=key in {"files", "referenced_files"})
            elif key == "referenced_files" and item is not None:
                report.error("SCHEMA_INVALID", "referenced_files must be a list or mapping", entity)
    elif isinstance(value, (list, tuple)):
        for number, item in enumerate(value):
            if isinstance(item, str) and strings_are_paths:
                _check_file({"path": item}, base, report, f"{entity}[{number}]")
            else:
                _references(item, base, report, f"{entity}[{number}]")


def validate_dataset(index: Any, *, level: str = "basic", dataset_root: str | Path | None = None,
                     warnings_as_errors: bool = False) -> ValidationReport:
    """Validate a DatasetIndex, manifest mapping, or JSON manifest path.

    Basic checks schema, identifiers, relationships and DatasetVersion bindings.
    Full additionally streams referenced bytes for SHA-256; it never reads pixels.
    """
    from .query import ensure_index, load_dataset_index

    report = _report(level, warnings_as_errors)
    source = None
    original = index if isinstance(index, Mapping) else None
    if isinstance(index, (str, Path)):
        source = Path(index)
        if source.is_dir():
            source /= "manifest.json"
        parsed = _read_json(source, report)
        if report.corrupted:
            return report
        if not isinstance(parsed, Mapping):
            report.error("SCHEMA_INVALID", "Dataset manifest must be a JSON object")
            return report
        original = parsed
        if not any(name in parsed for name in ("studies", "series", "slices")):
            reference = parsed.get("index_path", parsed.get("payload_path"))
            if reference is None and isinstance(parsed.get("payload"), str):
                reference = parsed["payload"]
            if isinstance(reference, str):
                referenced_path = _reference_path(reference, source.parent)
                if referenced_path is None:
                    report.error("SCHEMA_INVALID", "Metadata index must reference a local file")
                    return report
                _read_json(referenced_path, report)
                if report.corrupted:
                    return report
        try:
            index = load_dataset_index(source)
        except (ValueError, TypeError, OSError) as exc:
            report.error("SCHEMA_INVALID", str(exc))
            return report
    else:
        try:
            index = ensure_index(index)
        except (ValueError, TypeError, AttributeError) as exc:
            report.error("SCHEMA_INVALID", str(exc))
            return report
        source = index.source
    metadata = index.metadata
    _version(metadata, report, "dataset")
    if original is not None:
        if original.get("schema_version") is not None and original.get("schema_version") != metadata.get("schema_version"):
            _version(original, report, "manifest")
        original_id = original.get("dataset_version_id")
        if original_id is not None and metadata.get("dataset_version_id") is not None:
            if original_id != metadata["dataset_version_id"]:
                report.error("DATASET_VERSION_MISMATCH", "Metadata index differs from outer manifest DatasetVersion")
    present = index.entities_present
    if not present:
        report.error("MISSING_ENTITY_TABLES", "Manifest contains no studies, series or slices tables")
    tables = {name: getattr(index, name) for name in ("studies", "series", "slices")}
    uid_fields = {"studies": "study_uid", "series": "series_uid", "slices": "sop_uid"}
    known: dict[str, dict[str, Mapping[str, Any]]] = {}
    for name, rows in tables.items():
        if name not in present:
            continue
        report.statistics[name] = len(rows)
        ids = known[name] = {}
        required = {"studies": ("study_uid",), "series": ("series_uid", "study_uid"),
                    "slices": ("sop_uid", "series_uid", "study_uid")}[name]
        for number, row in enumerate(rows):
            entity = f"{name}[{number}]"
            for key in required:
                if not isinstance(row.get(key), str) or not row[key].strip():
                    report.error("MISSING_REQUIRED_FIELD", f"{key} must be a nonempty string", entity)
            uid = row.get(uid_fields[name])
            if isinstance(uid, str) and uid.strip():
                if uid in ids:
                    report.error("DUPLICATE_UID", f"Duplicate {uid_fields[name]}: {uid}", entity)
                else:
                    ids[uid] = row
            if name == "slices":
                for key in ("position", "orientation", "spacing"):
                    if row.get(key) is None:
                        report.warn(f"MISSING_{key.upper()}", f"Slice {uid or number} has no {key}", entity)
            for key in ("sha256", "payload_sha256", "hash"):
                if row.get(key) is not None and not _is_sha(row[key]):
                    report.error("INVALID_HASH", f"{key} must be a lowercase SHA-256 digest", entity)
            for key, length in (("position", 3), ("orientation", 6), ("spacing", 2)):
                vector = row.get(key)
                if vector is not None and (not isinstance(vector, (list, tuple)) or len(vector) != length
                        or any(type(number) not in (int, float) or not math.isfinite(number) for number in vector)):
                    report.error("INVALID_METADATA", f"{key} must contain {length} finite numbers", entity)
            for key in ("rows", "columns", "n_slices"):
                if row.get(key) is not None and (type(row[key]) is not int or row[key] < 0
                        or key in {"rows", "columns"} and row[key] == 0):
                    report.error("INVALID_METADATA", f"{key} must be a valid nonnegative integer", entity)
            warnings = row.get("warnings", []) or []
            if not isinstance(warnings, (list, tuple)):
                report.error("SCHEMA_INVALID", "warnings must be a list", entity)
                warnings = []
            for warning in warnings:
                if isinstance(warning, Mapping):
                    report.warn(str(warning.get("code", "DATA_WARNING")),
                                str(warning.get("message", warning)), entity)
                else:
                    report.warn("DATA_WARNING", str(warning), entity)
    for name in ("series", "slices"):
        for number, row in enumerate(tables[name]):
            entity = f"{name}[{number}]"
            study_uid = row.get("study_uid")
            study = known.get("studies", {}).get(study_uid) if isinstance(study_uid, str) else None
            if "studies" in present and study is None:
                report.error("ORPHAN_SERIES" if name == "series" else "ORPHAN_SLICE",
                             f"Unknown parent study: {row.get('study_uid')}", entity)
            if study and row.get("patient_id") is not None and study.get("patient_id") is not None:
                if row["patient_id"] != study["patient_id"]:
                    report.error("PATIENT_MISMATCH", "Patient differs from the parent study", entity)
            if name == "slices" and "series" in present:
                series_uid = row.get("series_uid")
                series = known.get("series", {}).get(series_uid) if isinstance(series_uid, str) else None
                if series is None:
                    report.error("ORPHAN_SLICE", f"Unknown parent series: {row.get('series_uid')}", entity)
                elif series.get("study_uid") != row.get("study_uid"):
                    report.error("STUDY_MISMATCH", "Slice study differs from the parent series study", entity)
    if "slices" in present:
        counts: dict[str, int] = {}
        for row in tables["slices"]:
            uid = row.get("series_uid")
            if isinstance(uid, str):
                counts[uid] = counts.get(uid, 0) + 1
        for row in tables["series"]:
            if type(row.get("n_slices")) is int and isinstance(row.get("series_uid"), str):
                if row["n_slices"] != counts.get(row["series_uid"], 0):
                    report.error("SLICE_COUNT_MISMATCH", "Declared slice count differs from indexed slices", row["series_uid"])
    # Even manifests containing only one table must not assign one Study UID to two patients.
    patients: dict[str, Any] = {}
    for name, rows in tables.items():
        for number, row in enumerate(rows):
            uid, patient = row.get("study_uid"), row.get("patient_id")
            if isinstance(uid, str) and patient is not None:
                if uid in patients and patients[uid] != patient:
                    report.error("PATIENT_MISMATCH", f"Study {uid} has inconsistent patients", f"{name}[{number}]")
                patients[uid] = patient
    version = metadata.get("dataset_version")
    bound_id = metadata.get("dataset_version_id")
    if isinstance(version, Mapping):
        _contract(version, contracts.DatasetVersion, report, "dataset_version")
        if bound_id is not None and bound_id != version.get("dataset_version_id"):
            report.error("DATASET_VERSION_MISMATCH", "DatasetVersion differs from manifest binding")
        bound_id = version.get("dataset_version_id")
    elif version is not None and not isinstance(version, str):
        report.error("SCHEMA_INVALID", "dataset_version must be an object or a version string")
    if bound_id is None:
        report.warn("MISSING_DATASET_VERSION", "DatasetVersion binding is not declared")
    elif not _is_sha(bound_id):
        report.error("INVALID_DATASET_VERSION_ID", "dataset_version_id must be a lowercase SHA-256 digest")
    for name, rows in tables.items():
        for number, row in enumerate(rows):
            if row.get("dataset_version_id") is not None and bound_id is not None:
                if row["dataset_version_id"] != bound_id:
                    report.error("DATASET_VERSION_MISMATCH", "Entity DatasetVersion differs from manifest", f"{name}[{number}]")
    if level == "full":
        root = dataset_root or metadata.get("dataset_root") or metadata.get("root")
        base = Path(root) if root is not None else (source.parent if source else Path.cwd())
        if not base.is_absolute() and source:
            base = source.parent / base
        # Check normalized records so aliases such as relative_path are available.
        for name, rows in tables.items():
            for number, row in enumerate(rows):
                _check_file(row, base, report, f"{name}[{number}]")
        for key in ("files", "referenced_files", "artifacts"):
            if key in metadata:
                _references(metadata[key], base, report, key, strings_are_paths=True)
        for key in ("index_path", "payload_path"):
            if metadata.get(key) is not None:
                _check_file({"path": metadata[key], "sha256": metadata.get(
                    "payload_sha256", metadata.get("index_sha256", metadata.get("sha256")))},
                    source.parent if source else base, report, key)
        if isinstance(version, Mapping) and version.get("uri"):
            _check_file({"uri": version["uri"], "sha256": version.get("source_manifest_sha256")},
                        source.parent if source else base, report, "dataset_version.source_manifest")
    else:
        report.info.append(ValidationIssue("HASH_CHECKS_SKIPPED", "Use full validation to verify referenced file bytes"))
    return report


def validate_artifact(path: str | Path, *, level: str = "full",
                      warnings_as_errors: bool = False) -> ValidationReport:
    """Validate a local JSON artifact, reference manifest, or artifact directory."""
    report = _report(level, warnings_as_errors)
    source = Path(path)
    if source.is_dir():
        source /= "manifest.json"
    value = _read_json(source, report)
    if report.corrupted:
        return report
    if not isinstance(value, Mapping):
        report.error("SCHEMA_INVALID", "Artifact manifest must be a JSON object")
        return report
    _version(value, report, "artifact")
    identity_contracts = (("artifact_id", contracts.ArtifactReference),
                          ("dataset_version_id", contracts.DatasetVersion),
                          ("fold_plan_id", contracts.FoldPlan),
                          ("model_candidate_id", contracts.ModelCandidate),
                          ("experiment_id", contracts.ExperimentSpec),
                          ("training_job_id", contracts.TrainingJob),
                          ("checkpoint_id", contracts.CheckpointArtifact),
                          ("training_result_id", contracts.TrainingResult),
                          ("prediction_artifact_id", contracts.PredictionArtifact),
                          ("evaluation_id", contracts.Evaluation),
                          ("submission_artifact_id", contracts.SubmissionArtifact))
    # More specific contracts also contain their upstream IDs; prioritize them.
    selected = None
    is_dataset = any(key in value for key in ("studies", "series", "slices", "index_path")) or any(
        isinstance(value.get(key), Mapping) and any(table in value[key] for table in ("studies", "series", "slices"))
        for key in ("index", "data", "payload"))
    if is_dataset:
        dataset_report = validate_dataset(source, level=level, warnings_as_errors=warnings_as_errors)
        report.errors.extend(dataset_report.errors)
        report.warnings.extend(dataset_report.warnings)
        report.info.extend(dataset_report.info)
        report.statistics.update(dataset_report.statistics)
        report.corrupted = dataset_report.corrupted
    else:
        for key, cls in reversed(identity_contracts):
            if key in value:
                selected = cls
                break
    if selected is contracts.ArtifactReference and "uri" not in value and any(
            key in value for key in ("payload_path", "path")):
        selected = None
        if not _is_sha(value.get("artifact_id")):
            report.error("INVALID_ARTIFACT_ID", "artifact_id must be a lowercase SHA-256 digest")
        expected = value.get("payload_sha256", value.get("sha256"))
        if not _is_sha(expected):
            report.error("INVALID_HASH", "Payload SHA-256 must be declared")
        elif value.get("artifact_id") != expected:
            report.error("ARTIFACT_ID_MISMATCH", "artifact_id must equal the payload SHA-256")
        if level == "full":
            _check_file({"path": value.get("payload_path", value.get("path")), "sha256": expected},
                        source.parent, report, "artifact.payload")
    elif selected:
        _contract(value, selected, report, "artifact")
    elif not is_dataset:
        report.warn("SCHEMA_UNAVAILABLE", "No foundation contract matches this artifact; only available structural checks apply")
    if "artifact" in value and isinstance(value["artifact"], Mapping):
        _contract(value["artifact"], contracts.ArtifactReference, report, "artifact.artifact")
    # Content-addressed store payloads have their byte digest as the file stem.
    if _is_sha(source.stem) and level == "full":
        _check_file({"path": str(source.resolve()), "sha256": source.stem}, source.parent,
                    report, "artifact.payload")
    if level == "full" and not is_dataset:
        _references(value, source.parent, report)
    elif level == "basic" and not is_dataset:
        report.info.append(ValidationIssue("HASH_CHECKS_SKIPPED", "Use full validation to verify payload and reference bytes"))
    return report
