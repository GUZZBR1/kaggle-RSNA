"""Single entry point for reproducible, metadata-only RSNA data preparation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import csv
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
import tomllib
from typing import Any, Callable, Mapping, Sequence

from .artifacts.store import JsonArtifactStore
from .contracts import DatasetVersion, FoldPlan
from .data.geometry import GeometryConfig, order_series_slices
from .data.index import build_index, discover_dataset
from .data.cache import load_or_refresh
from .data.models import DatasetIndex, SliceRecord, index_from_dict
from .data.provenance import OrientationConfig, build_orientation_provenance
from .data.selection import SliceSelectionConfig, SliceSelector
from .targets import Target, TargetRegistry, TARGET_REGISTRY, validate_targets
from .labels import LabelRecord
from .folds import generate_fold_plan, load_fold_plan, save_fold_plan
from .identity import digest, freeze_json, jsonable
from .leakage import LeakageReport, validate_leakage
from .leakage.validator import LeakageValidationError, require_valid_leakage_report


STAGES = ("DISCOVER", "INDEX", "GEOMETRY", "ORIENTATION", "SELECT", "LABELS",
          "FOLDS", "LEAKAGE", "FINALIZE")
STAGE_STATUSES = frozenset({"PASS", "FAILED", "SKIPPED", "UNAVAILABLE"})
REQUIRED_STAGES = frozenset({"DISCOVER", "INDEX", "LABELS", "FOLDS", "LEAKAGE", "FINALIZE"})
OPTIONAL_STAGES = frozenset(set(STAGES) - REQUIRED_STAGES)


@dataclass(frozen=True)
class PreparationConfig:
    source_root: str | Path
    output_dir: str | Path
    mode: str = "real"
    name: str = "RSNA"
    version: str = "1"
    target_names: tuple[str, ...] | None = None
    orientation_mode: str = "preserve_native"
    orientation_plane_tolerance_deg: float = 15.0
    orientation_consistency_tolerance_deg: float = 1.0
    slice_strategy: str = "physical_span"
    slice_count: int = 24
    strict_geometry: bool = False
    n_folds: int = 5
    fold_seed: int = 42
    fold_strategy: str = "group"
    fold_assignment_overrides: Mapping[str, str] = field(default_factory=dict)
    leakage_policy: str = "strict"
    allow_synthetic_leakage_bypass: bool = False
    reuse_valid_artifacts: bool = True
    index_cache: bool = True
    index_invalid_policy: str = "strict"

    def __post_init__(self) -> None:
        if self.mode not in {"real", "synthetic"}:
            raise ValueError("mode must be real or synthetic")
        OrientationConfig(plane_tolerance_deg=self.orientation_plane_tolerance_deg,
            orientation_consistency_tolerance_deg=self.orientation_consistency_tolerance_deg,
            normalization_mode=self.orientation_mode)
        if self.orientation_mode not in {"preserve_native", "left_canonical"}:
            raise ValueError("unsupported orientation mode")
        if self.fold_strategy not in {"group", "multilabel_group_stratified"}:
            raise ValueError("unsupported fold strategy")
        if self.leakage_policy not in {"strict", "audit", "report"}:
            raise ValueError("leakage_policy must be strict or audit")
        if self.index_invalid_policy not in {"strict", "warn", "skip-invalid"}:
            raise ValueError("unsupported index invalid-file policy")
        if self.allow_synthetic_leakage_bypass and self.mode != "synthetic":
            raise ValueError("leakage bypass is permitted only in synthetic mode")
        for name in ("reuse_valid_artifacts", "index_cache", "strict_geometry", "allow_synthetic_leakage_bypass"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if not str(self.source_root).strip() or not str(self.output_dir).strip():
            raise ValueError("source_root and output_dir are required")
        if not isinstance(self.fold_assignment_overrides, Mapping):
            raise ValueError("fold_assignment_overrides must be an object")
        object.__setattr__(self, "fold_assignment_overrides", freeze_json(self.fold_assignment_overrides))
        if self.target_names is not None:
            object.__setattr__(self, "target_names", tuple(self.target_names))

    @classmethod
    def from_toml(cls, path: str | Path) -> "PreparationConfig":
        config_path = Path(path)
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        dataset = data.get("data", {})
        index = data.get("index", {})
        orientation = data.get("orientation", {})
        selection = data.get("slice_selection", {})
        folds = data.get("folds", {})
        leakage = data.get("leakage", {})
        source_root = _resolve(config_path.parent, dataset.get("root", "./data"))
        output_dir = _resolve(config_path.parent, dataset.get("output_dir", "./artifacts/data"))
        return cls(source_root=source_root, output_dir=output_dir,
            mode=dataset.get("mode", "real"), name=dataset.get("name", "RSNA"),
            version=str(dataset.get("version", "1")),
            target_names=tuple(dataset["target_names"]) if "target_names" in dataset else None,
            orientation_mode=orientation.get("mode", "preserve_native"),
            orientation_plane_tolerance_deg=orientation.get("plane_tolerance_deg", 15.0),
            orientation_consistency_tolerance_deg=orientation.get("consistency_tolerance_deg", 1.0),
            slice_strategy=selection.get("strategy", "physical_span"),
            slice_count=selection.get("count", 24), strict_geometry=data.get("geometry", {}).get("strict", False),
            n_folds=folds.get("n_folds", 5), fold_seed=folds.get("seed", 42),
            fold_strategy=folds.get("strategy", "group"),
            fold_assignment_overrides=folds.get("assignments", {}),
            leakage_policy=leakage.get("policy", "strict"),
            allow_synthetic_leakage_bypass=leakage.get("allow_synthetic_bypass", False),
            reuse_valid_artifacts=index.get("reuse_valid_artifacts", True),
            index_cache=index.get("cache", True), index_invalid_policy=index.get("on_invalid", "strict"))


@dataclass(frozen=True)
class StageRecord:
    stage: str
    input_id: str
    output_id: str | None
    status: str
    duration_seconds: float
    reused: bool = False
    records_processed: int = 0
    warnings: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in STAGE_STATUSES:
            raise ValueError(f"unsupported preparation stage status: {self.status}")
        if self.status in {"SKIPPED", "UNAVAILABLE"} and not (self.reason and self.reason.strip()):
            raise ValueError(f"{self.status} preparation stages require a reason")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "warnings": list(self.warnings)}


@dataclass(frozen=True)
class PreparedDataset:
    status: str
    synthetic: bool
    dataset_version: DatasetVersion
    dataset_index: DatasetIndex
    fold_plan: FoldPlan
    fold_assignments: Mapping[str, str]
    target_registry: TargetRegistry
    leakage_report: LeakageReport
    preprocessing_spec: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    statistics: Mapping[str, Any]
    stages: tuple[StageRecord, ...]
    preparation_manifest_uri: str
    schema_version: int = 1
    labels: Mapping[str, Any] = field(default_factory=dict)
    leakage_bypassed: bool = False

    def __post_init__(self) -> None:
        if self.status not in {"READY", "SYNTHETIC"}:
            raise ValueError("PreparedDataset status must be READY or SYNTHETIC")
        if self.synthetic != (self.status == "SYNTHETIC"):
            raise ValueError("PreparedDataset status does not match synthetic flag")
        if self.leakage_bypassed and not self.synthetic:
            raise ValueError("leakage bypass requires explicit synthetic mode")
        if not self.leakage_report.passed and not self.leakage_bypassed:
            raise ValueError("PreparedDataset requires a passing leakage report")
        allowed_bypass = self.preprocessing_spec.get("leakage", {}).get("allow_synthetic_bypass", False)
        if self.leakage_bypassed != (self.synthetic and allowed_bypass and not self.leakage_report.passed):
            raise ValueError("leakage bypass does not match explicit material configuration")
        if self.fold_plan.dataset_version_id != self.dataset_version.dataset_version_id:
            raise ValueError("FoldPlan is not bound to DatasetVersion")
        if not self.fold_plan.assignment_manifest_sha256:
            raise ValueError("PreparedDataset requires a FoldPlan assignment manifest")
        if self.dataset_version.preprocessing.get("dataset_index_id") != self.dataset_index.index_id:
            raise ValueError("DatasetVersion does not reference this DatasetIndex")
        if self.dataset_version.preprocessing.get("target_registry_id") != self.target_registry.registry_id:
            raise ValueError("DatasetVersion does not reference this target schema")
        if self.dataset_version.synthetic != self.synthetic:
            raise ValueError("DatasetVersion synthetic provenance mismatch")
        if self.fold_plan.synthetic != self.synthetic:
            raise ValueError("FoldPlan synthetic flag does not match PreparedDataset")
        if self.dataset_version.preprocessing_version != self.preprocessing_spec.get("preprocessing_id"):
            raise ValueError("DatasetVersion preprocessing version does not match preprocessing spec")
        material = dict(self.preprocessing_spec)
        preprocessing_id = material.pop("preprocessing_id", None)
        if preprocessing_id != digest(material):
            raise ValueError("preprocessing spec ID does not match material configuration")
        if not self.fold_assignments or set(self.fold_assignments) != {
                study.study_id for study in self.dataset_index.studies}:
            raise ValueError("fold assignments must cover every indexed study exactly once")
        assignment_id = self.dataset_version.preprocessing["folds"]["assignment_manifest_sha256"]
        if self.fold_plan.assignment_manifest_sha256 != assignment_id:
            raise ValueError("FoldPlan assignment manifest does not match DatasetVersion")
        if self.schema_version != 1:
            raise ValueError("unsupported PreparedDataset schema version")
        stage_names = tuple(stage.stage for stage in self.stages)
        if stage_names != STAGES:
            raise ValueError("PreparedDataset must record every preparation stage in order")
        for stage in self.stages:
            if stage.status == "FAILED":
                raise ValueError(f"stage {stage.stage} failed before finalization")
            if stage.status in {"SKIPPED", "UNAVAILABLE"} and stage.stage not in OPTIONAL_STAGES:
                raise ValueError(f"required stage {stage.stage} cannot be {stage.status}")
            if stage.stage in REQUIRED_STAGES and stage.status != "PASS":
                raise ValueError(f"required stage {stage.stage} must pass before finalization")
        required_artifacts = {"dataset_index", "source_content", "target_registry", "fold_plan", "fold_assignments", "leakage_report",
                              "orientation_provenance", "selected_slices", "labels"}
        if not required_artifacts <= set(self.artifacts):
            raise ValueError(f"PreparedDataset is missing artifact references: {sorted(required_artifacts - set(self.artifacts))}")
        if digest(dict(self.dataset_version.preprocessing)) != digest(dict(self.preprocessing_spec)):
            raise ValueError("DatasetVersion and PreparedDataset preprocessing specs differ")
        if self.dataset_version.dataset_index_artifact_id != self.artifacts["dataset_index"]["artifact_id"]:
            raise ValueError("DatasetVersion index artifact binding mismatch")
        if self.fold_plan.assignment_manifest_sha256 != digest(_assignment_list(self.dataset_index, self.fold_assignments)):
            raise ValueError("FoldPlan assignments do not match PreparedDataset")
        if self.leakage_report.dataset_version_id != self.dataset_version.dataset_version_id:
            raise ValueError("LeakageReport DatasetVersion binding mismatch")
        if self.leakage_report.fold_plan_id != self.fold_plan.fold_plan_id:
            raise ValueError("LeakageReport FoldPlan binding mismatch")
        if self.preprocessing_spec.get("labels_sha256") != digest(self.labels):
            raise ValueError("labels do not match preprocessing identity")
        _validate_labels(self.dataset_index, self.target_registry, self.labels, self.synthetic)
        for name in ("fold_assignments", "preprocessing_spec", "artifacts", "statistics", "labels"):
            object.__setattr__(self, name, freeze_json(getattr(self, name)))

    def to_dict(self, *, include_telemetry: bool = True) -> dict[str, Any]:
        result = {"schema_version": self.schema_version, "status": self.status,
            "synthetic": self.synthetic, "dataset_version": self.dataset_version.to_dict(),
            "dataset_index": self.dataset_index.to_dict(), "fold_plan": self.fold_plan.to_dict(),
            "fold_assignments": jsonable(self.fold_assignments),
            "target_registry": self.target_registry.to_dict(),
            "leakage_report": self.leakage_report.to_dict(),
            "preprocessing_spec": jsonable(self.preprocessing_spec),
            "artifacts": jsonable(self.artifacts), "statistics": jsonable(self.statistics),
            "preparation_manifest_uri": self.preparation_manifest_uri,
            "lineage": self.lineage, "labels": jsonable(self.labels),
            "leakage_bypassed": self.leakage_bypassed}
        if include_telemetry:
            result["stages"] = [stage.to_dict() for stage in self.stages]
        return result

    @property
    def lineage(self) -> dict[str, str]:
        return {
            "source_manifest_sha256": self.dataset_version.source_manifest_sha256,
            "dataset_version_id": self.dataset_version.dataset_version_id,
            "dataset_index_id": self.dataset_index.index_id,
            "preprocessing_id": self.preprocessing_spec["preprocessing_id"],
            "target_schema_id": self.preprocessing_spec["target_schema_id"],
            "target_registry_id": self.target_registry.registry_id,
            "fold_plan_id": self.fold_plan.fold_plan_id,
            "assignment_manifest_sha256": self.fold_plan.assignment_manifest_sha256 or "",
            "leakage_report_id": self.leakage_report.report_id,
        }


class PreparationError(RuntimeError):
    def __init__(self, stage: str, status: str, message: str, stages: Sequence[StageRecord]):
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.status = status
        self.message = message
        self.stages = tuple(stages)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "failed_stage": self.stage, "error": self.message,
                "stages": [stage.to_dict() for stage in self.stages]}


class _Stages:
    def __init__(self) -> None:
        self.records: list[StageRecord] = []

    def run(self, name: str, input_id: str, action, *,
            unavailable_reason: Callable[[Any], str | None] | None = None):
        started = time.perf_counter()
        try:
            value, output_id, warnings, reused, count = action()
        except Exception as exc:
            self.records.append(StageRecord(name, input_id, None, "FAILED",
                time.perf_counter() - started, warnings=(str(exc),)))
            status = "INVALID" if isinstance(exc, (ValueError, FileNotFoundError, LeakageValidationError)) else "FAILED"
            raise PreparationError(name, status, str(exc), self.records) from exc
        reason = unavailable_reason(value) if unavailable_reason is not None else None
        status = "UNAVAILABLE" if reason else "PASS"
        self.records.append(StageRecord(name, input_id, output_id, status,
            time.perf_counter() - started, reused, count, tuple(warnings), reason))
        return value


def prepare_dataset(config: PreparationConfig) -> PreparedDataset:
    """Prepare one coherent metadata lineage; failed runs never publish a final manifest."""
    if not isinstance(config, PreparationConfig):
        raise TypeError("config must be a PreparationConfig")
    root = Path(config.source_root).expanduser().resolve()
    output = Path(config.output_dir).expanduser().resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("artifact output directory must be outside the raw dataset root")
    stages = _Stages()
    source = stages.run("DISCOVER", digest({"source": "configured"}),
        lambda: _discover(config, root))
    index, cache_mode, source_content = stages.run("INDEX", digest(source or {"kind": "dicom"}),
        lambda: _index(config, source, root, output))
    if not index.studies:
        last = stages.records[-1]
        stages.records[-1] = StageRecord("INDEX", last.input_id, None, "FAILED",
            last.duration_seconds, warnings=("dataset contains no indexed studies",))
        raise PreparationError("INDEX", "INVALID", "dataset contains no indexed studies", stages.records)
    ordered = stages.run("GEOMETRY", index.index_id,
        lambda: _geometry(index, strict=config.strict_geometry),
        unavailable_reason=_geometry_unavailable_reason)
    orientation_config = OrientationConfig(normalization_mode=config.orientation_mode,
        plane_tolerance_deg=config.orientation_plane_tolerance_deg,
        orientation_consistency_tolerance_deg=config.orientation_consistency_tolerance_deg)
    provenance = stages.run("ORIENTATION", digest(_geometry_material(ordered)),
        lambda: _orientation(ordered, orientation_config),
        unavailable_reason=_orientation_unavailable_reason)
    selection_input = digest({"geometry": _geometry_material(ordered),
        "strategy": config.slice_strategy, "count": config.slice_count})
    selected = stages.run("SELECT", selection_input,
        lambda: _select(ordered, SliceSelectionConfig(strategy=config.slice_strategy, count=config.slice_count)),
        unavailable_reason=_selection_unavailable_reason)
    selection_config = SliceSelectionConfig(strategy=config.slice_strategy, count=config.slice_count)
    registry, labels = stages.run("LABELS", index.index_id,
        lambda: _labels(config, root, source, index))
    assignments, assignment_sha, fold_ids, fold_manifest_id = stages.run("FOLDS",
        digest({"index": index.index_id, "seed": config.fold_seed, "n_folds": config.n_folds,
                "strategy": config.fold_strategy, "overrides": config.fold_assignment_overrides,
                "labels": digest(labels)}),
        lambda: _folds(index, config, output, registry, labels))

    store = JsonArtifactStore(output / "artifacts")
    # The source index reference is content-addressed before binding DatasetVersion.
    index_ref = store.put_json(index.to_dict(), manifest={"kind": "dataset_index", "index_id": index.index_id})
    source_sha = digest({"dataset_index_id": index.index_id,
                         "source_content_sha256": digest(source_content),
                         "labels_sha256": digest(labels),
                         "synthetic": config.mode == "synthetic"})
    target_schema_id = digest({"schema_version": registry.schema_version, "names": registry.names})
    material = {"schema_version": 1, "source_manifest_sha256": source_sha,
        "dataset_index_id": index.index_id, "target_schema_id": target_schema_id,
        "target_registry_id": registry.registry_id, "labels_sha256": digest(labels),
        "orientation": orientation_config.to_dict(),
        "orientation_provenance_sha256": digest(provenance),
        **selection_config.to_preprocessing_spec(),
        "geometry": {"strict": config.strict_geometry,
                     "methods": {key: value["ordering"].method for key, value in ordered.items()}},
        "folds": {"strategy": config.fold_strategy, "n_folds": config.n_folds,
                  "seed": config.fold_seed, "generator_manifest_id": fold_manifest_id,
                  "assignment_manifest_sha256": assignment_sha},
        "selected_slice_ids_sha256": digest(selected),
        "leakage": {"policy": "audit" if config.leakage_policy == "report" else config.leakage_policy,
                    "allow_synthetic_bypass": config.allow_synthetic_leakage_bypass}}
    preprocessing = {**material, "preprocessing_id": digest(material)}
    synthetic = config.mode == "synthetic"
    dataset = DatasetVersion(config.name, config.version, source_sha, preprocessing["preprocessing_id"],
        registry.names, preprocessing=preprocessing, dataset_index_artifact_id=index_ref.artifact_id,
        synthetic=synthetic)
    fold_plan = FoldPlan(dataset.dataset_version_id, config.fold_strategy, fold_ids, config.fold_seed,
        configuration={"n_folds": config.n_folds, "grouping": "patient",
                       "generator_manifest_id": fold_manifest_id},
        assignment_manifest_sha256=assignment_sha, synthetic=synthetic)
    leakage = stages.run("LEAKAGE", assignment_sha,
        lambda: _leakage(index, assignments, dataset, fold_plan, config))
    statistics = _statistics(index, selected, assignments)
    statistics["index_cache_mode"] = cache_mode
    statistics["fold_cache_mode"] = "warm_load" if stages.records[6].reused else "cold_build"
    final_path = output / "prepared" / f"{dataset.dataset_version_id}.json"
    result = stages.run("FINALIZE", dataset.dataset_version_id,
        lambda: _finalize(index, source_content, registry, labels, assignments, fold_plan, leakage,
            provenance, selected, dataset, preprocessing, statistics, config, final_path,
            tuple(stages.records), store, index_ref))
    result = PreparedDataset(**{**result.__dict__, "stages": tuple(stages.records)})
    # Publish only after every contract and artifact has passed final validation.
    try:
        _atomic_json(final_path, result.to_dict())
    except Exception as exc:
        last = stages.records[-1]
        stages.records[-1] = StageRecord("FINALIZE", last.input_id, None, "FAILED",
            last.duration_seconds, warnings=(str(exc),))
        raise PreparationError("FINALIZE", "FAILED", str(exc), stages.records) from exc
    return result


def load_prepared_dataset(path: str | Path) -> PreparedDataset:
    """Reload contracts and verify every artifact, binding and leakage decision."""
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported PreparedDataset schema version")
    dataset = DatasetVersion.from_dict(payload["dataset_version"])
    plan = FoldPlan(**payload["fold_plan"])
    index = index_from_dict(payload["dataset_index"])
    registry = TargetRegistry.from_dict(payload["target_registry"])
    report = LeakageReport.from_dict(payload["leakage_report"])
    result = PreparedDataset(payload["status"], payload["synthetic"], dataset, index, plan,
        payload["fold_assignments"], registry, report, payload["preprocessing_spec"],
        payload["artifacts"], payload["statistics"],
        tuple(StageRecord(**record) for record in payload.get("stages", ())),
        payload["preparation_manifest_uri"], payload["schema_version"],
        payload.get("labels", {}), payload.get("leakage_bypassed", False))
    if payload.get("lineage") != result.lineage:
        raise ValueError("saved lineage does not match component contracts")
    stored_manifest_path = Path(result.preparation_manifest_uri)
    if not stored_manifest_path.is_absolute():
        stored_manifest_path = manifest_path.parent / stored_manifest_path
    if stored_manifest_path.resolve() != manifest_path:
        raise ValueError("manifest URI does not match loaded path")
    if set(result.fold_assignments.values()) - set(plan.fold_ids):
        raise ValueError("saved fold assignments reference an unknown fold")
    artifacts = {name: JsonArtifactStore.load_json(_artifact_reference({
                     **ref, "uri": str(_resolve_artifact_uri(ref["uri"], manifest_path.parent))}))
                 for name, ref in result.artifacts.items()}
    expected = {"dataset_index": index.to_dict(), "target_registry": registry.to_dict(),
        "fold_plan": plan.to_dict(),
        "fold_assignments": _assignment_list(index, result.fold_assignments),
        "leakage_report": report.to_dict(), "labels": dict(result.labels)}
    for name, value in expected.items():
        if artifacts[name] != value:
            raise ValueError(f"saved {name} artifact does not match preparation manifest")
    for name, identity in (("selected_slices", "selected_slice_ids_sha256"),
                           ("orientation_provenance", "orientation_provenance_sha256")):
        if digest(artifacts[name]) != result.preprocessing_spec[identity]:
            raise ValueError(f"saved {name} artifact does not match preprocessing identity")
    source_sha = digest({"dataset_index_id": index.index_id,
                         "source_content_sha256": digest(artifacts["source_content"]),
                         "labels_sha256": digest(result.labels),
                         "synthetic": result.synthetic})
    if source_sha != dataset.source_manifest_sha256:
        raise ValueError("source lineage digest does not match indexed metadata and labels")
    target_schema = digest({"schema_version": registry.schema_version, "names": registry.names})
    if target_schema != result.preprocessing_spec["target_schema_id"]:
        raise ValueError("saved target schema identity mismatch")
    recomputed = validate_leakage(_leakage_input(index, result.labels), plan,
        assignments=result.fold_assignments, dataset_version=dataset, policy=report.policy)
    if recomputed.report_id != report.report_id:
        raise ValueError("saved leakage report does not match indexed data and assignments")
    if not result.leakage_bypassed:
        require_valid_leakage_report(report)
    return result


def _discover(config: PreparationConfig, root: Path):
    if not root.is_dir():
        raise NotADirectoryError(root)
    path = root / "source_manifest.json"
    source = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    synthetic = config.mode == "synthetic"
    if source is None and synthetic:
        raise ValueError("synthetic mode requires an explicit synthetic source_manifest.json")
    if source is not None:
        if source.get("schema_version") != 1:
            raise ValueError("unsupported source manifest schema version")
        if type(source.get("synthetic")) is not bool or source["synthetic"] != synthetic:
            raise ValueError("source manifest synthetic flag must match explicit preparation mode")
        if not isinstance(source.get("studies"), list) or not source["studies"]:
            raise ValueError("source manifest must contain studies")
    return source, digest(source or {"kind": "dicom"}), (), False, len(source["studies"]) if source else 0


def _index(config: PreparationConfig, source: Mapping[str, Any] | None, root: Path, output: Path):
    if source is None:
        if config.index_cache:
            cached = load_or_refresh(root, output / "cache" / "dataset-index.sqlite",
                rebuild=not config.reuse_valid_artifacts, on_invalid=config.index_invalid_policy)
            mode = cached.report.mode.replace("-", "_")
            source_content = _source_content_manifest(root, cached.index, output, synthetic=False)
            return (cached.index, mode, source_content), cached.index.index_id, (
                f"index cache mode: {mode}", *cached.report.warnings), mode == "warm_load", cached.index.statistics["n_slices"]
        index = discover_dataset(root, on_invalid=config.index_invalid_policy)
        source_content = _source_content_manifest(root, index, output, synthetic=False)
        return (index, "cold_build", source_content), index.index_id, index.warnings, False, index.statistics["n_slices"]

    # This bridge accepts already supplied metadata while reusing canonical models
    # and the main index builder. It never defines competing Study/Series/Slice types.
    # Preserve the declared hierarchy before build_index groups slices by UIDs.
    for study in source["studies"]:
        study_uid = study.get("study_instance_uid") or study.get("study_id")
        for series in study["series"]:
            series_uid = series.get("series_instance_uid") or series.get("series_id")
            declared_study_uid = series.get("study_instance_uid")
            if declared_study_uid is not None and declared_study_uid != study_uid:
                raise ValueError(f"series {series_uid} declares a different parent StudyInstanceUID")
            for item in series["slices"]:
                metadata = item.get("metadata", {})
                if metadata.get("StudyInstanceUID", study_uid) != study_uid:
                    raise ValueError(f"slice {item.get('slice_id')} declares a different parent StudyInstanceUID")
                if metadata.get("SeriesInstanceUID", series_uid) != series_uid:
                    raise ValueError(f"slice {item.get('slice_id')} declares a different parent SeriesInstanceUID")

    signature = _manifest_signature(root, source, config.mode == "synthetic")
    cache_path = output / "cache" / f"manifest-index-{signature}.json"
    warnings = []
    if config.index_cache and config.reuse_valid_artifacts and cache_path.exists():
        try:
            index = index_from_dict(json.loads(cache_path.read_text(encoding="utf-8")))
            source_content = _source_content_manifest(root, index, output, synthetic=config.mode == "synthetic")
            return (index, "warm_load", source_content), index.index_id, ("index cache mode: warm_load",), True, index.statistics["n_slices"]
        except (OSError, ValueError, KeyError, TypeError):
            warnings.append("corrupt dataset index cache rejected")
    slices, paths = [], []
    seen_paths = set()
    for study in source["studies"]:
        study_uid = study.get("study_instance_uid") or study.get("study_id")
        patient = study.get("patient_id")
        if not isinstance(study_uid, str) or not study_uid.strip():
            raise ValueError("source study identity is required")
        for series in study["series"]:
            series_uid = series.get("series_instance_uid") or series.get("series_id")
            if not isinstance(series_uid, str) or not series_uid.strip():
                raise ValueError("source series identity is required")
            for item in series["slices"]:
                relative = _relative_path(item.get("relative_path", item.get("path", "")))
                if relative in seen_paths:
                    raise ValueError("source manifest contains duplicate relative paths")
                seen_paths.add(relative)
                metadata = dict(item.get("metadata", {}))
                metadata.setdefault("StudyInstanceUID", study_uid)
                metadata.setdefault("SeriesInstanceUID", series_uid)
                metadata.setdefault("SOPInstanceUID", item.get("slice_id"))
                metadata.setdefault("PatientID", patient)
                metadata.setdefault("metadata_warnings", [])
                file_path = root / relative
                size = item.get("file_size", 0)
                if config.mode == "real":
                    file_path = _source_file(root, relative)
                    size = file_path.stat().st_size
                    actual_sha = _file_sha256(file_path)
                    if item.get("content_sha256") is not None and item["content_sha256"] != actual_sha:
                        raise ValueError(f"source content checksum mismatch: {relative}")
                    metadata["file_hash"] = actual_sha
                slices.append(SliceRecord(relative, size, metadata))
                paths.append(root / relative)
    index = build_index(root, sorted(paths), slices, [], 0, [])
    source_content = _source_content_manifest(root, index, output, synthetic=config.mode == "synthetic")
    if config.index_cache:
        _atomic_json(cache_path, index.to_dict())
    warnings.append("index cache mode: cold_build")
    return (index, "cold_build", source_content), index.index_id, tuple(warnings), False, len(slices)


def _source_content_manifest(root: Path, index: DatasetIndex, output: Path, *, synthetic: bool) -> dict[str, Any]:
    """Bind raw source bytes to DatasetVersion while reusing stat-validated checksums.

    The canonical index cache deliberately uses size/mtime for its warm fast path.
    This sidecar follows that cache contract and hashes only files whose stat tuple
    changed since the previous preparation.
    """
    current: dict[str, tuple[int, int, Path]] = {}
    missing_synthetic: list[str] = []
    for study in index.studies:
        for series in study.series:
            for item in series.slices:
                candidate = root / _relative_path(item.relative_path)
                if synthetic and not candidate.is_file():
                    missing_synthetic.append(item.relative_path)
                    continue
                path = _source_file(root, item.relative_path)
                stat = path.stat()
                current[item.relative_path] = (stat.st_size, stat.st_mtime_ns, path)
    cache_path = output / "cache" / f"source-content-{index.root_identity}-{'synthetic' if synthetic else 'real'}.json"
    cached: dict[str, Any] = {}
    if cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            entries = payload["files"]
            valid_entries = (isinstance(entries, list) and all(
                isinstance(item, Mapping) and isinstance(item.get("relative_path"), str)
                and type(item.get("size")) is int and type(item.get("mtime_ns")) is int
                and isinstance(item.get("sha256"), str) and len(item["sha256"]) == 64
                and all(char in "0123456789abcdef" for char in item["sha256"])
                for item in entries))
            paths = [item["relative_path"] for item in entries] if valid_entries else []
            if (payload["schema_version"] == 1 and valid_entries and len(paths) == len(set(paths))
                    and payload["files_sha256"] == digest(entries)):
                cached = {item["relative_path"]: item for item in entries}
        except (OSError, ValueError, KeyError, TypeError):
            cached = {}
    files = []
    for relative in sorted(current):
        size, mtime_ns, path = current[relative]
        previous = cached.get(relative, {})
        checksum = previous.get("sha256") if (previous.get("size") == size
            and previous.get("mtime_ns") == mtime_ns) else None
        if not isinstance(checksum, str) or len(checksum) != 64:
            checksum = _file_sha256(path)
        files.append({"relative_path": relative, "size": size, "mtime_ns": mtime_ns,
                      "sha256": checksum})
    content = {"schema_version": 1, "synthetic": synthetic,
               "missing_synthetic_paths": sorted(missing_synthetic), "files": files,
               "files_sha256": digest(files)}
    if cached != {item["relative_path"]: item for item in files} or not cache_path.is_file():
        _atomic_json(cache_path, content)
    return {"schema_version": 1, "synthetic": synthetic,
            "missing_synthetic_paths": sorted(missing_synthetic),
            "files": [{"relative_path": item["relative_path"], "sha256": item["sha256"]}
                      for item in files],
            "files_sha256": digest([{ "relative_path": item["relative_path"], "sha256": item["sha256"]}
                                    for item in files])}


def _manifest_signature(root: Path, source: Mapping[str, Any], synthetic: bool) -> str:
    files = []
    if not synthetic:
        for study in source["studies"]:
            for series in study["series"]:
                for item in series["slices"]:
                    relative = _relative_path(item.get("relative_path", item.get("path", "")))
                    path = _source_file(root, relative)
                    stat = path.stat()
                    files.append([relative, stat.st_size, stat.st_mtime_ns])
    # Input array ordering is immaterial; canonical index output remains the authority.
    def canonical(value, field_name=None):
        if isinstance(value, Mapping):
            return {key: canonical(item, key) for key, item in value.items()}
        if isinstance(value, list):
            items = [canonical(item) for item in value]
            return sorted(items, key=lambda item: json.dumps(item, sort_keys=True)) if field_name in {
                "studies", "series", "slices"} else items
        return value
    return digest({"source": canonical(source), "files": sorted(files)})


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("source path must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("source path must be a normalized relative POSIX path")
    return path.as_posix()


def _source_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"source path escapes dataset root: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"source file does not exist: {relative}")
    return path


def _file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _geometry(index: DatasetIndex, *, strict: bool):
    ordered, warnings = {}, []
    for study in index.studies:
        for series in study.series:
            result = order_series_slices(series, GeometryConfig(strict_geometry=strict))
            if strict and result.method not in {"geometry", "position_inferred_normal"}:
                raise ValueError("strict geometry requires complete physical positions and orientation")
            ordered[series.series_id] = {"study_id": study.study_id, "ordering": result}
            warnings.extend(warning.message for warning in result.warnings)
    return ordered, digest(_geometry_material(ordered)), tuple(sorted(set(warnings))), False, len(ordered)


def _geometry_material(ordered):
    return {key: {"study_id": value["study_id"], "ordering": value["ordering"].to_dict()}
            for key, value in ordered.items()}


def _geometry_unavailable_reason(ordered):
    fallback = sorted(series_id for series_id, value in ordered.items()
                      if value["ordering"].method not in {"geometry", "position_inferred_normal"})
    if fallback:
        return (f"Physical geometry unavailable for {len(fallback)} of {len(ordered)} series; "
                "ordering used fallback metadata")
    return None


def _orientation(ordered, config: OrientationConfig):
    records, warnings = [], []
    for series_id, value in ordered.items():
        provenance = build_orientation_provenance([item.metadata for item in value["ordering"].slices], config)
        records.append({"study_id": value["study_id"], "series_id": series_id,
                        "provenance": provenance.to_dict()})
        warnings.extend(provenance.warnings)
    return records, digest(records), tuple(sorted(set(warnings))), False, len(records)


def _orientation_unavailable_reason(records):
    unavailable = []
    for record in records:
        provenance = record["provenance"]
        orientation = provenance["orientation"]
        laterality = provenance["laterality"]
        if orientation["confidence"] == "unknown" or laterality["confidence"] == "unknown":
            unavailable.append(record["series_id"])
    if unavailable:
        return (f"Orientation or laterality evidence unavailable for "
                f"{len(unavailable)} of {len(records)} series")
    return None


def _select(ordered, config: SliceSelectionConfig):
    records, warnings = [], []
    for series_id, value in ordered.items():
        result = SliceSelector(config).select(value["ordering"])
        warnings.extend(result.warnings)
        if result.fallback:
            warnings.append(f"physical_span fallback used: {result.fallback}")
        material = result.to_dict()
        material.update(study_id=value["study_id"], series_id=series_id,
                        selected_slice_ids=[item.slice_id for item in result.selected if item is not None])
        records.append(material)
    return records, digest(records), tuple(sorted(set(warnings))), False, len(records)


def _selection_unavailable_reason(records):
    fallback = sum(bool(record.get("fallback")) for record in records)
    if fallback:
        return (f"Requested slice-selection capability unavailable for {fallback} of "
                f"{len(records)} series; recorded fallback selection was used")
    return None


def _labels(config: PreparationConfig, root: Path, source, index: DatasetIndex):
    synthetic = config.mode == "synthetic"
    registry_path = root / "target_registry.json"
    declared = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else None
    if declared is not None and "targets" in declared:
        registry = TargetRegistry.from_dict(declared)
    else:
        names = tuple(declared.get("target_names", ())) if declared else config.target_names
        if names is None:
            names = TARGET_REGISTRY.names
        validate_targets(names, allow_synthetic=synthetic)
        registry = TARGET_REGISTRY if tuple(names) == TARGET_REGISTRY.names else TargetRegistry(tuple(Target(name) for name in names))
        if declared and (declared.get("synthetic", False) != synthetic or (not synthetic and not declared.get("official", False))):
            raise ValueError("target registry provenance does not match preparation mode")
    if not synthetic:
        TARGET_REGISTRY.validate_target_order(registry.names)
        if registry.registry_id != TARGET_REGISTRY.registry_id:
            raise ValueError("real preparation requires the canonical official TargetRegistry")
    if config.target_names is not None:
        registry.validate_target_order(config.target_names)

    labels = {}
    if source is not None:
        for study in source["studies"]:
            uid = study.get("study_instance_uid") or study.get("study_id")
            record = index.study_by_uid(uid)
            if record is None:
                raise ValueError("source label study does not exist in dataset index")
            values = study.get("targets", study.get("labels", {}))
            if record.study_id in labels:
                raise ValueError("duplicate label study identity")
            labels[record.study_id] = {"values": values, "provenance": study.get("label_provenance",
                "manual_review" if synthetic else "official_gold"), "patient_id": study.get("patient_id")}
    else:
        path = next((root / name for name in ("train.csv", "labels.csv") if (root / name).is_file()), None)
        if path is None:
            raise FileNotFoundError("real preparation requires train.csv or labels.csv")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            label_columns = tuple(name for name in reader.fieldnames or () if name in registry.names)
            registry.validate_target_order(label_columns)
            for row in reader:
                uid = row.get("StudyInstanceUID", row.get("study_id"))
                study = index.study_by_uid(uid)
                if study is None:
                    raise ValueError(f"label references an unknown StudyInstanceUID: {uid}")
                if study.study_id in labels:
                    raise ValueError("duplicate label study identity")
                values = {name: int(row[name]) for name in registry.names}
                labels[study.study_id] = {"values": values, "provenance": row.get("label_provenance") or "official_gold",
                                         "patient_id": row.get("PatientID") or study.patient_id}
    _validate_labels(index, registry, labels, synthetic)
    return (registry, labels), digest({"registry": registry.to_dict(), "labels": labels}), (), False, len(labels)


def _validate_labels(index: DatasetIndex, registry: TargetRegistry, labels: Mapping[str, Any], synthetic: bool):
    if set(labels) != {study.study_id for study in index.studies}:
        raise ValueError("labels must cover every indexed study exactly once")
    for study in index.studies:
        record = labels[study.study_id]
        values = record["values"]
        if tuple(values) != registry.names:
            # JSON object order is irrelevant; target column ordering is fixed by the registry.
            if set(values) != set(registry.names):
                raise ValueError("label targets must match the target schema exactly")
        if record.get("patient_id") != study.patient_id:
            raise ValueError("label PatientID does not match indexed study")
        if synthetic and registry.names != TARGET_REGISTRY.names:
            if any(type(value) not in {int, float} or value not in (0, 1) for value in values.values()):
                raise ValueError("synthetic hard labels must be binary 0 or 1")
        else:
            LabelRecord(study.study_id, values, record["provenance"],
                        patient_id=study.patient_id, allow_partial=False)


def _folds(index: DatasetIndex, config: PreparationConfig, output: Path,
           registry: TargetRegistry, labels: Mapping[str, Any]):
    key = digest({"dataset_index_id": index.index_id, "n_folds": config.n_folds,
        "seed": config.fold_seed, "strategy": config.fold_strategy,
        "overrides": config.fold_assignment_overrides, "registry_id": registry.registry_id,
        "labels": digest(labels)})
    cache_dataset_id = key
    path = output / "cache" / f"fold-plan-{key}.json"
    warnings = []
    uid_aliases = {study.study_instance_uid: study.study_id for study in index.studies
                   if study.study_instance_uid}
    studies = [{"study_instance_uid": study.study_instance_uid,
                "patient_id": study.patient_id,
                "series_ids": [series.series_instance_uid or series.series_id for series in study.series],
                "warnings": list(study.warnings)} for study in index.studies]
    fold_labels = None
    if config.fold_strategy == "multilabel_group_stratified":
        if registry.names != TARGET_REGISTRY.names:
            raise ValueError("multilabel_group_stratified requires the official target registry")
        fold_labels = {study.study_instance_uid: labels[study.study_id]["values"]
                       for study in index.studies if study.study_instance_uid
                       and study.study_id in labels}
    manifest = None
    if config.reuse_valid_artifacts and path.exists():
        try:
            manifest = load_fold_plan(path, dataset_version_id=cache_dataset_id, dataset=studies)
        except (OSError, ValueError, KeyError, TypeError):
            warnings.append("invalid FoldPlan cache rejected")
    reused = manifest is not None
    if manifest is None:
        manifest = generate_fold_plan(studies, n_folds=config.n_folds,
            strategy=config.fold_strategy, random_state=config.fold_seed, labels=fold_labels,
            dataset_version_id=cache_dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            save_fold_plan(manifest, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    assignments = {uid_aliases.get(study_uid, study_uid): fold_id
                   for study_uid, fold_id in manifest.assignments.items()}
    resolved_overrides = {}
    for key_name, fold_id in config.fold_assignment_overrides.items():
        study_id = uid_aliases.get(key_name, key_name)
        if study_id in resolved_overrides and resolved_overrides[study_id] != fold_id:
            raise ValueError("conflicting fold overrides reference the same study")
        resolved_overrides[study_id] = fold_id
    if set(resolved_overrides) - set(assignments):
        raise ValueError("fold assignment overrides reference unknown studies")
    fold_ids = tuple(f"fold_{number}" for number in range(config.n_folds))
    if set(resolved_overrides.values()) - set(fold_ids):
        raise ValueError("fold assignment override references unknown fold")
    assignments.update(resolved_overrides)
    sha = digest(_assignment_list(index, assignments))
    return (assignments, sha, fold_ids, manifest.fold_plan_id), sha, tuple(warnings), reused, len(assignments)


def _leakage_input(index: DatasetIndex, labels: Mapping[str, Any]):
    data = index.to_dict()
    for study in data["studies"]:
        label = labels.get(study["study_id"], {})
        provenance = label.get("provenance")
        if isinstance(provenance, Mapping):
            study["label_provenance"] = provenance
    return data


def _leakage(index: DatasetIndex, assignments, dataset, plan, config):
    policy = "audit" if config.leakage_policy == "report" else config.leakage_policy
    # Reuse the canonical validator; retain a failed report when synthetic bypass is explicit.
    report = validate_leakage(index, plan, assignments=assignments, dataset_version=dataset, policy=policy)
    if not (config.mode == "synthetic" and config.allow_synthetic_leakage_bypass):
        require_valid_leakage_report(report)
    warnings = tuple(issue.message for issue in report.issues)
    if not report.passed:
        warnings += ("explicit synthetic leakage bypass used; dataset is SYNTHETIC",)
    return report, report.report_id, warnings, False, sum(report.checked_entities.values())


def _assignment_list(index: DatasetIndex, assignments: Mapping[str, str]):
    patients = {study.study_id: study.patient_id for study in index.studies}
    return [{"study_id": key, "patient_id": patients[key], "fold_id": assignments[key]}
            for key in sorted(assignments)]


def _statistics(index, selected, assignments):
    return {"patients": len({study.patient_id for study in index.studies}),
        "studies": len(index.studies), "series": sum(len(study.series) for study in index.studies),
        "slices": sum(len(series.slices) for study in index.studies for series in study.series),
        "selected_slices": sum(len(item["selected_slice_ids"]) for item in selected),
        "labeled_studies": len(index.studies), "assigned_studies": len(assignments)}


def _finalize(index, source_content, registry, labels, assignments, plan, leakage, provenance, selected, dataset,
              preprocessing, statistics, config, path, prior_stages, store, index_ref):
    values = {"source_content": source_content, "target_registry": registry.to_dict(),
        "fold_plan": plan.to_dict(), "labels": labels,
        "fold_assignments": _assignment_list(index, assignments), "leakage_report": leakage.to_dict(),
        "orientation_provenance": provenance, "selected_slices": selected}
    refs = {"dataset_index": index_ref.to_dict()}
    refs.update({name: store.put_json(value, manifest={"kind": name}).to_dict() for name, value in values.items()})
    synthetic = config.mode == "synthetic"
    stages = (*prior_stages, StageRecord("FINALIZE", dataset.dataset_version_id,
        dataset.dataset_version_id, "PASS", 0.0, records_processed=1))
    result = PreparedDataset("SYNTHETIC" if synthetic else "READY", synthetic, dataset, index,
        plan, assignments, registry, leakage, preprocessing, refs, statistics, stages,
        str(path), labels=labels, leakage_bypassed=synthetic and config.allow_synthetic_leakage_bypass and not leakage.passed)
    return result, dataset.dataset_version_id, (), False, 1


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.unlink(handle.name)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _artifact_reference(value: Mapping[str, Any]):
    from .contracts import ArtifactReference
    return ArtifactReference(**dict(value))


def _resolve_artifact_uri(uri: str, base: Path) -> Path:
    path = Path(uri)
    return path.resolve() if path.is_absolute() else (base / path).resolve()
