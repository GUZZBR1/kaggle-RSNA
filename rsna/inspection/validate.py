"""Reusable structural and byte-integrity validation; never decodes DICOM pixels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import json
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
        report.warn("UNVERIFIED", f"Cannot verify non-local reference: {location}", entity)
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
    """Validate canonical Issue 1 DatasetIndex records without decoding pixels."""
    from ..data import DatasetIndex
    from ..data.models import index_from_dict

    report = _report(level, warnings_as_errors)
    source: Path | None = None
    try:
        if isinstance(index, (str, Path)):
            source = Path(index)
            if source.is_dir():
                source /= "manifest.json"
            # Parse raw JSON separately so malformed schema versions and duplicate keys are diagnosed.
            raw = _read_json(source, report)
            if report.corrupted:
                return report
            if not isinstance(raw, Mapping) or type(raw.get("manifest_schema_version")) is not int or raw.get("manifest_schema_version") != 1:
                report.error("SCHEMA_INVALID", "manifest_schema_version must be integer 1")
                return report
            index = index_from_dict(raw)
        if not isinstance(index, DatasetIndex):
            raise TypeError("expected canonical rsna.data.DatasetIndex or its manifest path")
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        report.error("SCHEMA_INVALID", str(exc))
        return report

    report.statistics.update(dict(index.statistics))
    report.statistics.update({"studies": len(index.studies),
                              "series": sum(len(study.series) for study in index.studies),
                              "slices": sum(series.n_slices for study in index.studies for series in study.series)})
    if not index.studies and index.statistics.get("n_studies") is None:
        report.warn("UNKNOWN_COUNTS", "Study count is unavailable")
    sop_uids: set[str] = set()
    for si, study in enumerate(index.studies):
        entity = f"studies[{si}]"
        if not study.study_instance_uid:
            report.error("MISSING_REQUIRED_FIELD", "StudyInstanceUID is required", entity)
        if not study.patient_id:
            report.warn("MISSING_PATIENT_ID", "PatientID is unavailable", entity)
        if any("conflicting PatientID" in warning for warning in study.warnings):
            report.error("CONFLICTING_PATIENT_ID", "Study contains conflicting PatientID values", entity)
        for warning in study.warnings:
            report.warn("DATA_WARNING", warning, entity)
        for ri, series in enumerate(study.series):
            series_entity = f"{entity}.series[{ri}]"
            if not series.series_instance_uid:
                report.error("MISSING_REQUIRED_FIELD", "SeriesInstanceUID is required", series_entity)
            if series.study_instance_uid != study.study_instance_uid:
                report.error("ORPHAN_SERIES", "Series StudyInstanceUID does not match its Study parent", series_entity)
            for warning in series.warnings:
                report.warn("DATA_WARNING", warning, series_entity)
            for di, item in enumerate(series.slices):
                slice_entity = f"{series_entity}.slices[{di}]"
                metadata = item.metadata
                sop = metadata.get("SOPInstanceUID")
                if not sop:
                    report.error("MISSING_REQUIRED_FIELD", "SOPInstanceUID is required", slice_entity)
                elif sop in sop_uids:
                    report.error("DUPLICATE_SOP_UID", f"Duplicate SOPInstanceUID: {sop}", slice_entity)
                else:
                    sop_uids.add(sop)
                if metadata.get("StudyInstanceUID") != study.study_instance_uid:
                    report.error("ORPHAN_SLICE", "Slice StudyInstanceUID does not match its Study parent", slice_entity)
                if metadata.get("SeriesInstanceUID") != series.series_instance_uid:
                    report.error("ORPHAN_SLICE", "Slice SeriesInstanceUID does not match its Series parent", slice_entity)
                for optional in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns"):
                    if metadata.get(optional) is None:
                        report.warn(f"MISSING_{optional.upper()}", f"Optional metadata {optional} is unavailable", slice_entity)
                for warning in item.warnings:
                    report.warn("DATA_WARNING", warning, slice_entity)
                if level == "full":
                    if dataset_root is None:
                        report.warn("UNVERIFIED", f"Cannot verify source file without dataset_root: {item.relative_path}", slice_entity)
                    else:
                        path = Path(dataset_root) / item.relative_path
                        report.statistics["referenced_files_checked"] = report.statistics.get("referenced_files_checked", 0) + 1
                        try:
                            if not path.is_file():
                                report.error("MISSING_REFERENCED_FILE", f"Referenced DICOM file does not exist: {path}", slice_entity)
                        except OSError as exc:
                            report.error("UNREADABLE_REFERENCE", str(exc), slice_entity)
                            report.corrupted = True
    duplicate_count = index.statistics.get("duplicate_sop_uid_count")
    if duplicate_count is not None and duplicate_count > 0 and not any(item.code == "DUPLICATE_SOP_UID" for item in report.errors):
        report.error("DUPLICATE_SOP_UID", f"Index reports {duplicate_count} duplicate SOPInstanceUID values")
    if not index.warnings and not any(study.warnings for study in index.studies):
        report.info.append(ValidationIssue("NO_INDEX_WARNINGS", "Dataset index has no discovery warnings"))
    else:
        for warning in index.warnings:
            report.warn("INDEX_WARNING", warning)
    report.warn("MISSING_DATASET_VERSION", "DatasetVersion binding is unavailable in this manifest")
    if level == "basic":
        report.info.append(ValidationIssue("HASH_CHECKS_SKIPPED", "Source file checks require full validation and dataset_root"))
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
        _version(value, report, "artifact")
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
