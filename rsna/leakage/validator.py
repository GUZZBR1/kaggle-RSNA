"""Dataset split and fold leakage checks. Runtime is linear in record count."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import DatasetVersion, FoldPlan
from ..identity import canonical_json, digest
from .duplicates import groups, records_of, value
from .types import (DEFAULT_SEVERITIES, LeakageIssue, LeakageIssueType, LeakagePolicy,
                    LeakageReport)


class LeakageValidationError(RuntimeError):
    """Raised by an explicit gate when a strict LeakageReport contains errors."""

    def __init__(self, report: LeakageReport):
        self.report = report
        super().__init__(f"leakage validation failed with {report.n_errors} critical issue(s)")


def require_valid_leakage_report(report: LeakageReport) -> str:
    """Return the validated report ID or block the caller from proceeding."""
    if not report.passed:
        raise LeakageValidationError(report)
    return report.report_id


def validate_split(train_records: Sequence[Mapping[str, Any]],
                   validation_records: Sequence[Mapping[str, Any]], *,
                   test_records: Sequence[Mapping[str, Any]] = (),
                   holdout_records: Sequence[Mapping[str, Any]] = (),
                   dataset_version_id: str | None = None,
                   policy: LeakagePolicy | str = LeakagePolicy.STRICT,
                   severity_overrides: Mapping[str, str] | None = None) -> LeakageReport:
    """Validate explicit train/validation/test/holdout collections without FoldPlan."""
    records: list[Mapping[str, Any]] = []
    for split_name, split_records in (("train", train_records), ("validation", validation_records),
                                      ("test", test_records), ("holdout", holdout_records)):
        for record in split_records:
            records.append({**record, "fold_id": split_name})
    return validate_leakage(records, policy=policy, dataset_version_id=dataset_version_id,
                            severity_overrides=severity_overrides)


def validate_leakage(dataset_index: Any, fold_plan: FoldPlan | Mapping[str, Any] | None = None, *,
                     assignments: Mapping[str, str] | None = None,
                     policy: LeakagePolicy | str = LeakagePolicy.STRICT,
                     dataset_version_id: str | None = None,
                     dataset_version: DatasetVersion | Mapping[str, Any] | None = None,
                     severity_overrides: Mapping[str, str] | None = None) -> LeakageReport:
    """Validate identity, relationships, provenance and split assignment.

    Records are mappings with patient_id/study_uid/series_uid/sop_uid/file_hash/path,
    fold_id (or split/partition), source_entity_ids and parent_artifact_ids as available.
    A path is recorded only as evidence and never used as entity identity.
    """
    policy = LeakagePolicy(policy)
    records = records_of(dataset_index)
    issues: list[LeakageIssue] = []
    overrides = {str(k): v for k, v in (severity_overrides or {}).items()}

    def emit(kind: LeakageIssueType, entity: str, *, folds: Sequence[str] = (),
             related: Sequence[str] = (), message: str = "", evidence: Mapping[str, Any] | None = None,
             severity: str | None = None) -> None:
        severity = severity or overrides.get(kind.value, DEFAULT_SEVERITIES.get(kind, "warning"))
        issues.append(LeakageIssue(kind, severity, str(entity), tuple(folds), tuple(related), message,
                                   evidence or {}))

    index_version = dataset_version_id
    if isinstance(dataset_index, Mapping):
        index_version = index_version or dataset_index.get("dataset_version_id")
        if dataset_version is None:
            dataset_version = dataset_index.get("dataset_version")
    else:
        index_version = index_version or getattr(dataset_index, "dataset_version_id", None)
    raw_dataset_version = dataset_version
    if isinstance(raw_dataset_version, Mapping):
        raw_dataset_version = DatasetVersion(**raw_dataset_version)
    if raw_dataset_version is not None and not isinstance(raw_dataset_version, DatasetVersion):
        raise TypeError("dataset_version must be a DatasetVersion or its JSON mapping")
    if raw_dataset_version is not None:
        declared_version_id = raw_dataset_version.dataset_version_id
        if index_version and index_version != declared_version_id:
            emit(LeakageIssueType.DATASET_VERSION_MISMATCH, declared_version_id,
                 related=(str(index_version),), message="DatasetIndex wrapper names a different DatasetVersion",
                 evidence={"dataset_version_id": index_version,
                           "provided_dataset_version_id": declared_version_id})
        index_version = declared_version_id
        index_artifact_id = None
        if isinstance(dataset_index, Mapping):
            index_artifact_id = dataset_index.get("dataset_index_artifact_id") or dataset_index.get("artifact_id")
        else:
            index_artifact_id = getattr(dataset_index, "dataset_index_artifact_id", None)
        if (raw_dataset_version.dataset_index_artifact_id is not None and index_artifact_id is not None
                and raw_dataset_version.dataset_index_artifact_id != index_artifact_id):
            emit(LeakageIssueType.DATASET_VERSION_MISMATCH, declared_version_id,
                 related=(str(index_artifact_id),), message="DatasetVersion references a different DatasetIndex artifact",
                 evidence={"expected_dataset_index_artifact_id": raw_dataset_version.dataset_index_artifact_id,
                           "dataset_index_artifact_id": index_artifact_id})
    raw_plan = fold_plan
    if isinstance(raw_plan, Mapping) and "fold_plan" in raw_plan:
        assignments = assignments or raw_plan.get("assignments")
        raw_plan = raw_plan["fold_plan"]
    if isinstance(raw_plan, Mapping):
        assignments = assignments or raw_plan.get("assignments")
        plan_values = {k: v for k, v in raw_plan.items() if k in FoldPlan.__dataclass_fields__}
        try:
            raw_plan = FoldPlan(**plan_values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid FoldPlan: {exc}") from exc
    if raw_plan is not None and not isinstance(raw_plan, FoldPlan):
        raise TypeError("fold_plan must be a FoldPlan or its JSON mapping")
    plan = raw_plan
    if plan is not None:
        if index_version and plan.dataset_version_id != index_version:
            emit(LeakageIssueType.DATASET_VERSION_MISMATCH, plan.dataset_version_id,
                 related=(index_version,), message="FoldPlan is bound to a different dataset version",
                 evidence={"dataset_version_id": index_version, "fold_plan_dataset_version_id": plan.dataset_version_id})
        elif index_version is None:
            index_version = plan.dataset_version_id

    if assignments:
        assignment_map = {str(k): str(v) for k, v in assignments.items()}
        used_assignment_keys: set[str] = set()
        for index, record in enumerate(records):
            current = value(record, "fold_id")
            explicit_group = record.get("group_id")
            if explicit_group is not None and str(explicit_group) not in assignment_map:
                emit(LeakageIssueType.UNKNOWN_GROUP_ID, str(explicit_group),
                     message="Record references a group absent from the assignment manifest")
            keys = [str(value(record, key)) for key in ("sop_uid", "series_uid", "study_uid", "patient_id",
                                                         "group_id", "id", "entity_id")
                    if value(record, key) is not None]
            matched_keys = {key for key in keys if key in assignment_map}
            used_assignment_keys.update(matched_keys)
            known = {assignment_map[key] for key in matched_keys}
            if len(known) > 1:
                emit(LeakageIssueType.UNKNOWN_GROUP_ID, keys[0] if keys else "<record>",
                     folds=sorted(known), message="Related assignment keys resolve to different folds")
            elif known and current is not None and str(current) not in known:
                emit(LeakageIssueType.UNKNOWN_GROUP_ID, keys[0] if keys else "<record>",
                     folds=(str(current), *sorted(known)),
                     message="Record fold conflicts with the assignment manifest")
            elif known:
                records[index] = {**record, "fold_id": next(iter(known))}
            elif current is None:
                emit(LeakageIssueType.UNKNOWN_GROUP_ID, keys[0] if keys else "<record>",
                     message="No fold assignment is available for this record")
        for unknown in sorted(set(assignment_map) - used_assignment_keys):
            emit(LeakageIssueType.UNKNOWN_GROUP_ID, unknown,
                 folds=(assignment_map[unknown],),
                 message="Assignment manifest contains an identity absent from the dataset")

    known_folds = set(plan.fold_ids) if plan else set()
    for record in records:
        fold = value(record, "fold_id")
        if fold is None:
            emit(LeakageIssueType.UNKNOWN_GROUP_ID, _entity_id(record),
                 message="Record has no fold or split assignment")
        elif plan and str(fold) not in known_folds:
            emit(LeakageIssueType.UNKNOWN_FOLD_ID, str(fold), message="Assignment is absent from FoldPlan.fold_ids")

    _identity_checks(records, emit)
    _relationship_checks(records, emit)
    _provenance_checks(records, emit)

    by_type = Counter(issue.type.value for issue in issues)
    errors = sum(issue.severity == "error" for issue in issues)
    warnings = sum(issue.severity == "warning" for issue in issues)
    checked = {
        "n_patients_checked": len(groups(records, "patient_id")),
        "n_studies_checked": len(groups(records, "study_uid")),
        "n_series_checked": len(groups(records, "series_uid")),
        "n_slices_checked": len(groups(records, "sop_uid")),
        "n_file_hashes_checked": len(groups(records, "file_hash")),
    }
    counts = {**checked, "n_issues": len(issues), "n_errors": errors, "n_warnings": warnings,
              "n_infos": sum(issue.severity == "info" for issue in issues),
              "issues_by_type": dict(sorted(by_type.items()))}
    material_sha = _input_material_sha256(records, assignment_map if assignments else {}, overrides)
    index_provenance = dataset_index
    wrapper_artifact_id = None
    if isinstance(dataset_index, Mapping):
        index_provenance = dataset_index.get("dataset_index", dataset_index.get("index", dataset_index))
        wrapper_artifact_id = dataset_index.get("dataset_index_artifact_id") or dataset_index.get("artifact_id")
    index_artifact_id = wrapper_artifact_id or (
        index_provenance.get("dataset_index_artifact_id", index_provenance.get("artifact_id"))
        if isinstance(index_provenance, Mapping) else getattr(index_provenance, "dataset_index_artifact_id", None))
    if index_artifact_id is None and raw_dataset_version is not None:
        index_artifact_id = raw_dataset_version.dataset_index_artifact_id
    provenance = {
        "dataset_index_id": (index_provenance.get("index_id") if isinstance(index_provenance, Mapping)
                             else getattr(index_provenance, "index_id", None)),
        "dataset_index_artifact_id": index_artifact_id,
        "assignment_manifest_sha256": plan.assignment_manifest_sha256 if plan else None,
        "assignment_source": "explicit" if assignments else (
            "records" if any(value(record, "fold_id") is not None for record in records) else "unassigned"),
        "n_source_entity_links": sum(len(_as_ids(record.get("source_entity_ids"))) for record in records),
        "n_parent_artifact_links": sum(len(_as_ids(record.get("parent_artifact_ids"))) for record in records),
        "n_label_provenance_records": sum(isinstance(record.get("label_provenance"), Mapping) for record in records),
    }
    return LeakageReport(errors == 0, tuple(issues), counts, checked,
                         plan.fold_plan_id if plan else None, index_version, policy,
                         input_material_sha256=material_sha, provenance=provenance)


def _input_material_sha256(records: list[Mapping[str, Any]], assignments: Mapping[str, str],
                           severity_overrides: Mapping[str, str]) -> str:
    fields = ("patient_id", "study_uid", "series_uid", "sop_uid", "file_hash", "fold_id",
              "entity_type", "group_id", "metadata_identity", "file_size", "source_entity_ids",
              "parent_artifact_ids", "label_provenance", "metadata")
    material_records = []
    for record in records:
        material_records.append({key: _without_locations(record[key]) for key in fields if key in record})
    material_records.sort(key=canonical_json)
    clean_assignments = {key: _without_locations(value) for key, value in assignments.items()}
    return digest({"records": material_records, "assignments": clean_assignments,
                   "severity_overrides": dict(sorted(severity_overrides.items()))})


def _without_locations(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _without_locations(item) for key, item in value.items()
                if str(key).lower() not in {"path", "paths", "relative_path", "absolute_path",
                                            "file_path", "uri", "uris", "host", "hostname",
                                            "timestamp", "created_at", "modified_at"}}
    if isinstance(value, (tuple, list)):
        return [_without_locations(item) for item in value if not _looks_like_absolute_path(item)]
    if isinstance(value, str) and _looks_like_absolute_path(value):
        return None
    return value


def _as_ids(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    return tuple(value) if isinstance(value, (tuple, list, set, frozenset)) else (value,)


def _looks_like_absolute_path(value: Any) -> bool:
    return isinstance(value, str) and (value.startswith(("/", "\\\\")) or "://" in value
                                       or (len(value) > 2 and value[1:3] == ":\\"))


def _entity_id(record: Mapping[str, Any]) -> str:
    for field in ("sop_uid", "series_uid", "study_uid", "patient_id", "id", "entity_id"):
        item = value(record, field)
        if item is not None:
            return str(item)
    return "<record>"


def _identity_checks(records: list[Mapping[str, Any]], emit: Any) -> None:
    groups_by_field = (("patient_id", LeakageIssueType.PATIENT_CROSS_FOLD),
                       ("study_uid", LeakageIssueType.STUDY_CROSS_FOLD),
                       ("series_uid", LeakageIssueType.SERIES_CROSS_FOLD),
                       ("sop_uid", LeakageIssueType.SLICE_CROSS_FOLD),
                       ("file_hash", LeakageIssueType.DUPLICATE_FILE_HASH))
    grouped = {field: groups(records, field) for field, _ in groups_by_field}
    for field, kind in groups_by_field:
        for identity, rows in grouped[field].items():
            folds = sorted({str(value(row, "fold_id")) for row in rows if value(row, "fold_id") is not None})
            paths = sorted({str(value(row, "path")) for row in rows if value(row, "path") is not None})
            if len(folds) > 1:
                emit(kind, identity, folds=folds,
                     message=f"{field} appears in multiple folds/splits", evidence={"paths": paths, "records": len(rows)},
                     severity="error" if field == "file_hash" else None)
            elif field == "file_hash" and len(rows) > 1:
                emit(kind, identity, folds=folds,
                     message="Duplicate file content detected within one fold", evidence={"paths": paths, "records": len(rows)})
            if field == "sop_uid" and len(rows) > 1:
                emit(LeakageIssueType.DUPLICATE_SOP_UID, identity, folds=folds,
                     message="Repeated sop_uid detected", evidence={"paths": paths, "records": len(rows)})

    for field, kind_dup, entity_name in (("series_uid", LeakageIssueType.DUPLICATE_SERIES_UID, "series"),
                                         ("study_uid", LeakageIssueType.DUPLICATE_STUDY_UID, "study")):
        for identity, rows in grouped[field].items():
            entity_rows = [row for row in rows if str(row.get("entity_type", row.get("kind", ""))).lower() == entity_name]
            paths = sorted({str(value(row, "path")) for row in entity_rows if value(row, "path") is not None})
            if len(entity_rows) > 1 and len(paths) > 1:
                emit(kind_dup, identity, folds=[str(value(row, "fold_id")) for row in entity_rows
                                               if value(row, "fold_id") is not None],
                     message=f"Repeated {field} has multiple explicit {entity_name} representations",
                     evidence={"paths": paths, "records": len(entity_rows)})

    for study, rows in grouped["study_uid"].items():
        patients = sorted({str(value(row, "patient_id")) for row in rows if value(row, "patient_id") is not None})
        if len(patients) > 1:
            emit(LeakageIssueType.CONFLICTING_PATIENT_ID, study, related=patients,
                 message="One StudyInstanceUID is associated with conflicting PatientIDs",
                 evidence={"patient_ids": patients})
    for series, rows in grouped["series_uid"].items():
        studies = sorted({str(value(row, "study_uid")) for row in rows if value(row, "study_uid") is not None})
        if len(studies) > 1:
            emit(LeakageIssueType.SERIES_STUDY_MISMATCH, series, related=studies,
                 message="One SeriesInstanceUID is associated with multiple StudyInstanceUIDs")
    for sop, rows in grouped["sop_uid"].items():
        series = sorted({str(value(row, "series_uid")) for row in rows if value(row, "series_uid") is not None})
        if len(series) > 1:
            emit(LeakageIssueType.SLICE_SERIES_MISMATCH, sop, related=series,
                 message="One SOPInstanceUID is associated with multiple SeriesInstanceUIDs")

    metadata_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        meta_key = row.get("metadata_identity")
        if meta_key:
            metadata_groups[str(meta_key)].append(row)
    for key, rows in metadata_groups.items():
        identities = sorted({_entity_id(row) for row in rows})
        folds = sorted({str(value(row, "fold_id")) for row in rows if value(row, "fold_id") is not None})
        if len(identities) > 1:
            emit(LeakageIssueType.NEAR_DUPLICATE_METADATA, key, folds=folds, related=identities,
                 message="Records share an explicitly supplied metadata identity",
                 evidence={"identities": identities})


def _relationship_checks(records: list[Mapping[str, Any]], emit: Any) -> None:
    series_rows = groups(records, "series_uid")
    study_by_series: dict[str, set[str]] = defaultdict(set)
    for record in records:
        series = value(record, "series_uid")
        study = value(record, "study_uid")
        if series is not None and study is not None:
            study_by_series[str(series)].add(str(study))
    reported_unbound_series: set[str] = set()
    for series, series_records in series_rows.items():
        explicit_unbound = any(
            str(record.get("entity_type", record.get("kind", ""))).lower() == "series"
            and value(record, "study_uid") is None for record in series_records)
        if series not in study_by_series or explicit_unbound:
            emit(LeakageIssueType.UNBOUND_SERIES, series,
                 message="Series has no StudyInstanceUID parent")
            reported_unbound_series.add(series)
    for record in records:
        kind = str(record.get("entity_type", record.get("kind", ""))).lower()
        if kind == "series" and value(record, "study_uid") is None:
            series = value(record, "series_uid")
            identity = str(series) if series is not None else _entity_id(record)
            if identity not in reported_unbound_series:
                emit(LeakageIssueType.UNBOUND_SERIES, identity,
                     message="Series has no StudyInstanceUID parent")
                reported_unbound_series.add(identity)
    for sop, slice_records in groups(records, "sop_uid").items():
        if any(value(record, "series_uid") is None for record in slice_records):
            emit(LeakageIssueType.UNBOUND_SLICE, sop, message="Slice has no SeriesInstanceUID parent")
    for record in records:
        kind = str(record.get("entity_type", record.get("kind", ""))).lower()
        if kind in {"slice", "image", "instance"} and value(record, "series_uid") is None:
            if value(record, "sop_uid") is None:
                emit(LeakageIssueType.UNBOUND_SLICE, _entity_id(record),
                     message="Slice has no SeriesInstanceUID parent")
def _provenance_checks(records: list[Mapping[str, Any]], emit: Any) -> None:
    lineage: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        for key in ("source_entity_ids", "parent_artifact_ids"):
            values = row.get(key, ())
            if isinstance(values, str):
                values = (values,)
            for identity in values or ():
                lineage[str(identity)].append(row)
        provenance = row.get("label_provenance")
        if isinstance(provenance, Mapping):
            source_fold = provenance.get("source_fold_id", provenance.get("source_split"))
            target_fold = value(row, "fold_id")
            if source_fold is not None and target_fold is not None and str(source_fold) != str(target_fold):
                emit(LeakageIssueType.LABEL_PROVENANCE_LEAKAGE, _entity_id(row),
                     folds=(str(source_fold), str(target_fold)),
                     message="Label provenance cites data from a different split",
                     evidence={"source_fold": str(source_fold)})
    for identity, rows in lineage.items():
        folds = sorted({str(value(row, "fold_id")) for row in rows if value(row, "fold_id") is not None})
        if len(folds) > 1:
            emit(LeakageIssueType.PROVENANCE_CROSS_SPLIT, identity, folds=folds,
                 message="Derived records from one source span multiple splits")
