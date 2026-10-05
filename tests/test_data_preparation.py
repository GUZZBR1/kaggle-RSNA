"""Synthetic end-to-end checks for the public data preparation API."""
from contextlib import redirect_stdout
from dataclasses import replace
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom import dcmread
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

from rsna.cli import main
from rsna.targets import Target, TargetRegistry, TARGET_REGISTRY
from rsna.preparation import (
    PreparationConfig, PreparationError, STAGES, StageRecord, load_prepared_dataset, prepare_dataset,
)


TARGET_NAMES = tuple(f"class_{index:02d}" for index in range(12))


def synthetic_manifest():
    """Four independent patient groups, each with two studies and two series."""
    studies = []
    for patient in range(4):
        for visit in range(2):
            study_id = f"study_{patient}_{visit}"
            series = []
            for sequence in range(2):
                series_id = f"{study_id}_series_{sequence}"
                slices = []
                for position in reversed(range(36)):
                    slice_id = f"{series_id}_slice_{position:02d}"
                    slices.append({
                        "slice_id": slice_id,
                        "path": f"{study_id}/{series_id}/{position:02d}.dcm",
                        "metadata": {
                            "StudyInstanceUID": study_id,
                            "SeriesInstanceUID": series_id,
                            "SOPInstanceUID": slice_id,
                            "PatientID": f"patient_{patient}",
                            "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
                            "ImagePositionPatient": [0, 0, position * 2.5],
                            "InstanceNumber": position + 1,
                            "Laterality": "L",
                            "Rows": 32, "Columns": 32,
                        },
                    })
                series.append({"series_id": series_id, "slices": slices})
            studies.append({
                "study_id": study_id, "patient_id": f"patient_{patient}",
                "targets": {name: (patient + visit + index) % 2
                            for index, name in enumerate(TARGET_NAMES)},
                "series": series,
            })
    return {"schema_version": 1, "synthetic": True, "studies": studies}


def prepared_example_config(root: Path) -> PreparationConfig:
    source = root / "source"
    source.mkdir(parents=True)
    names = TARGET_REGISTRY.names
    studies = []
    for patient in range(2):
        study_id = f"synthetic_study_{patient + 1:02d}"
        series_id = f"synthetic_series_{patient + 1:02d}"
        slices = []
        for position in range(2):
            metadata = {
                "StudyInstanceUID": study_id,
                "SeriesInstanceUID": series_id,
                "SOPInstanceUID": f"synthetic_slice_{patient + 1:02d}_{position + 1:02d}",
                "PatientID": f"synthetic_patient_{patient + 1:02d}",
                "InstanceNumber": position + 1,
            }
            if patient == 0:
                metadata.update({
                    "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
                    "ImagePositionPatient": [0, 0, position * 2.5],
                    "Laterality": "L",
                })
            slices.append({
                "slice_id": metadata["SOPInstanceUID"],
                "path": f"{study_id}/{series_id}/{position + 1:02d}.dcm",
                "metadata": metadata,
            })
        studies.append({
            "study_id": study_id,
            "patient_id": f"synthetic_patient_{patient + 1:02d}",
            "targets": {name: (patient + index) % 2 for index, name in enumerate(names)},
            "series": [{"series_id": series_id, "slices": slices}],
        })
    (source / "source_manifest.json").write_text(json.dumps({
        "schema_version": 1, "synthetic": True, "studies": studies,
    }), encoding="utf-8")
    return PreparationConfig(source, root / "output", mode="synthetic",
        target_names=names, n_folds=2, slice_count=2)


class DataPreparationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"
        self.manifest = synthetic_manifest()
        self.write_manifest()
        self.config = PreparationConfig(
            self.source, self.output, mode="synthetic", target_names=TARGET_NAMES,
            n_folds=2, fold_seed=42,
        )

    def write_manifest(self):
        (self.source / "source_manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8")

    def write_registry(self, *, names=TARGET_NAMES, synthetic=True, official=False):
        registry = (TARGET_REGISTRY if tuple(names) == TARGET_REGISTRY.names else
                    TargetRegistry(tuple(Target(name) for name in names)))
        (self.source / "target_registry.json").write_text(
            json.dumps(registry.to_dict()), encoding="utf-8")

    def create_real_slice_files(self):
        for study in self.manifest["studies"]:
            for series in study["series"]:
                for item in series["slices"]:
                    path = self.source / item["path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(item["slice_id"].encode())

    def artifact(self, prepared, name):
        return json.loads(Path(prepared.artifacts[name]["uri"]).read_text(encoding="utf-8"))

    def material_ids(self, prepared):
        return (
            prepared.dataset_version.dataset_version_id,
            prepared.dataset_index.index_id,
            prepared.fold_plan.fold_plan_id,
            prepared.target_registry.registry_id,
            prepared.preprocessing_spec["preprocessing_id"],
        )

    def assert_failed(self, config, stage):
        with self.assertRaises(PreparationError) as context:
            prepare_dataset(config)
        failure = context.exception
        self.assertEqual(stage, failure.stage)
        self.assertIn(failure.status, {"INVALID", "FAILED"})
        self.assertEqual("FAILED", failure.stages[-1].status)
        self.assertEqual(stage, failure.stages[-1].stage)
        self.assertFalse(list(self.output.glob("prepared/*.json")))
        self.assertFalse(list(self.output.rglob("*.tmp")))
        return failure

    def test_synthetic_pipeline_produces_complete_bound_lineage(self):
        prepared = prepare_dataset(self.config)
        self.assertEqual("SYNTHETIC", prepared.status)
        self.assertTrue(prepared.synthetic)
        self.assertTrue(prepared.leakage_report.passed)
        self.assertFalse(prepared.leakage_bypassed)
        self.assertEqual(TARGET_NAMES, prepared.target_registry.names)
        self.assertEqual(TARGET_NAMES, prepared.dataset_version.class_names)
        self.assertEqual({"patients": 4, "studies": 8, "series": 16, "slices": 576,
                          "selected_slices": 384, "labeled_studies": 8,
                          "assigned_studies": 8},
                         {key: prepared.statistics[key] for key in ("patients", "studies", "series",
                            "slices", "selected_slices", "labeled_studies", "assigned_studies")})
        self.assertEqual(STAGES, tuple(stage.stage for stage in prepared.stages))
        for stage in prepared.stages:
            self.assertEqual("PASS", stage.status)
            self.assertGreaterEqual(stage.duration_seconds, 0)
            self.assertTrue(stage.input_id)
            self.assertTrue(stage.output_id)
        self.assertEqual(prepared.dataset_version.dataset_version_id,
                         prepared.fold_plan.dataset_version_id)
        self.assertEqual(prepared.dataset_version.source_manifest_sha256,
                         prepared.preprocessing_spec["source_manifest_sha256"])
        self.assertEqual(prepared.dataset_index.index_id,
                         prepared.preprocessing_spec["dataset_index_id"])
        for patient in range(4):
            first = prepared.dataset_index.study_by_uid(f"study_{patient}_0")
            second = prepared.dataset_index.study_by_uid(f"study_{patient}_1")
            self.assertEqual(prepared.fold_assignments[first.study_id],
                             prepared.fold_assignments[second.study_id])
        selections = self.artifact(prepared, "selected_slices")
        self.assertEqual(16, len(selections))
        for series in selections:
            self.assertEqual(24, len(series["selected_slice_ids"]))
            selected_paths = {item.slice_id: item.relative_path for study in prepared.dataset_index.studies
                              for collection in study.series for item in collection.slices}
            self.assertTrue(selected_paths[series["selected_slice_ids"][0]].endswith("/00.dcm"))
            self.assertTrue(selected_paths[series["selected_slice_ids"][-1]].endswith("/35.dcm"))
        self.assertTrue(Path(prepared.preparation_manifest_uri).is_file())

    def test_same_inputs_are_idempotent_except_telemetry(self):
        first = prepare_dataset(self.config)
        second = prepare_dataset(self.config)
        self.assertEqual(self.material_ids(first), self.material_ids(second))
        self.assertEqual(first.preprocessing_spec, second.preprocessing_spec)
        self.assertEqual(first.lineage, second.lineage)
        self.assertEqual({key: value["sha256"] for key, value in first.artifacts.items()
                          if key != "leakage_report"},
                         {key: value["sha256"] for key, value in second.artifacts.items()
                          if key != "leakage_report"})
        self.assertEqual(first.leakage_report.report_id, second.leakage_report.report_id)

    def test_slice_count_24_to_32_changes_preprocessing_and_selected_content(self):
        first = prepare_dataset(self.config)
        second = prepare_dataset(replace(self.config, slice_count=32))
        self.assertNotEqual(first.preprocessing_spec["preprocessing_id"],
                            second.preprocessing_spec["preprocessing_id"])
        self.assertNotEqual(first.dataset_version.dataset_version_id,
                            second.dataset_version.dataset_version_id)
        self.assertEqual(first.dataset_index.index_id, second.dataset_index.index_id)
        self.assertEqual(512, second.statistics["selected_slices"])

    def test_material_orientation_strategy_and_source_changes_affect_identity(self):
        baseline = prepare_dataset(self.config)
        for changes in ({"orientation_mode": "left_canonical"}, {"slice_strategy": "uniform"}):
            with self.subTest(changes=changes):
                changed = prepare_dataset(replace(self.config, **changes))
                self.assertNotEqual(baseline.preprocessing_spec["preprocessing_id"],
                                    changed.preprocessing_spec["preprocessing_id"])
        labels = self.manifest["studies"][0]["targets"]
        labels[TARGET_NAMES[0]] = 1 - labels[TARGET_NAMES[0]]
        self.write_manifest()
        changed = prepare_dataset(self.config)
        self.assertNotEqual(baseline.dataset_version.source_manifest_sha256,
                            changed.dataset_version.source_manifest_sha256)
        self.assertNotEqual(baseline.dataset_version.dataset_version_id,
                            changed.dataset_version.dataset_version_id)

    def test_target_schema_order_changes_material_identity_when_consistent(self):
        first = prepare_dataset(self.config)
        second = prepare_dataset(replace(self.config, target_names=tuple(reversed(TARGET_NAMES))))
        self.assertNotEqual(first.preprocessing_spec["target_schema_id"],
                            second.preprocessing_spec["target_schema_id"])
        self.assertNotEqual(first.preprocessing_spec["preprocessing_id"],
                            second.preprocessing_spec["preprocessing_id"])
        self.assertEqual(tuple(reversed(TARGET_NAMES)), second.dataset_version.class_names)

    def test_declared_slice_hierarchy_mismatch_is_rejected_at_index(self):
        self.manifest["studies"][0]["series"][0]["slices"][0]["metadata"]["StudyInstanceUID"] = "different-study"
        self.write_manifest()
        failure = self.assert_failed(self.config, "INDEX")
        self.assertIn("StudyInstanceUID", failure.message)

    def test_target_count_mismatch_is_rejected(self):
        self.assert_failed(replace(self.config, target_names=TARGET_NAMES[:-1]), "LABELS")
    def test_fold_seed_changes_fold_plan_identity(self):
        first = prepare_dataset(self.config)
        second = prepare_dataset(replace(self.config, fold_seed=43))
        self.assertNotEqual(first.fold_plan.fold_plan_id, second.fold_plan.fold_plan_id)
        self.assertEqual(first.dataset_index.index_id, second.dataset_index.index_id)
        self.assertEqual(43, second.fold_plan.random_state)
        self.assertNotEqual(first.preprocessing_spec["preprocessing_id"],
                            second.preprocessing_spec["preprocessing_id"])

    def test_output_directory_is_not_material(self):
        first = prepare_dataset(self.config)
        second = prepare_dataset(replace(self.config, output_dir=self.root / "elsewhere"))
        self.assertEqual(self.material_ids(first), self.material_ids(second))
        self.assertEqual(first.fold_assignments, second.fold_assignments)
        self.assertEqual(first.preprocessing_spec, second.preprocessing_spec)
        self.assertEqual({key: value["sha256"] for key, value in first.artifacts.items()
                          if key != "leakage_report"},
                         {key: value["sha256"] for key, value in second.artifacts.items()
                          if key != "leakage_report"})
        self.assertEqual(first.leakage_report.report_id, second.leakage_report.report_id)
        self.assertNotEqual(first.preparation_manifest_uri, second.preparation_manifest_uri)

    def test_input_order_does_not_change_material_results(self):
        first = prepare_dataset(self.config)
        self.manifest["studies"].reverse()
        for study in self.manifest["studies"]:
            study["series"].reverse()
            for series in study["series"]:
                series["slices"].reverse()
        self.write_manifest()
        second = prepare_dataset(self.config)
        self.assertEqual(self.material_ids(first), self.material_ids(second))
        self.assertEqual(self.artifact(first, "selected_slices"),
                         self.artifact(second, "selected_slices"))

    def test_patient_leakage_is_a_gate_even_with_report_policy(self):
        for policy in ("strict", "report"):
            with self.subTest(policy=policy):
                failure = self.assert_failed(replace(self.config, leakage_policy=policy,
                    fold_assignment_overrides={"study_0_0": "fold_0", "study_0_1": "fold_1"}),
                    "LEAKAGE")
                self.assertTrue(any("leakage" in warning.lower()
                                    for warning in failure.stages[-1].warnings))

    def test_leakage_bypass_requires_explicit_synthetic_permission(self):
        prepared = prepare_dataset(replace(self.config,
            fold_assignment_overrides={"study_0_0": "fold_0", "study_0_1": "fold_1"},
            allow_synthetic_leakage_bypass=True))
        self.assertEqual("SYNTHETIC", prepared.status)
        self.assertTrue(prepared.leakage_bypassed)
        self.assertFalse(prepared.leakage_report.passed)
        self.assertTrue(prepared.leakage_report.issues)
        with self.assertRaises(ValueError):
            replace(self.config, mode="real", allow_synthetic_leakage_bypass=True)

    def test_dataset_mismatch_is_rejected_at_bound_result(self):
        prepared = prepare_dataset(self.config)
        mismatched_fold = replace(prepared.fold_plan,
                                  dataset_version_id="a" * 64, fold_plan_id="")
        with self.assertRaisesRegex(ValueError, "bound|dataset|Dataset"):
            replace(prepared, fold_plan=mismatched_fold)

    def test_manifest_mode_must_match_explicit_config(self):
        self.assert_failed(replace(self.config, mode="real"), "DISCOVER")

    def test_label_schema_missing_extra_and_nonbinary_fail(self):
        original = dict(self.manifest["studies"][0]["targets"])
        malformed = [
            {name: value for name, value in original.items() if name != TARGET_NAMES[0]},
            {**original, "unexpected": 0},
            {**original, TARGET_NAMES[0]: 2},
            {**original, TARGET_NAMES[0]: True},
        ]
        for labels in malformed:
            with self.subTest(labels=labels):
                self.manifest["studies"][0]["targets"] = labels
                self.write_manifest()
                self.assert_failed(self.config, "LABELS")

    def test_target_registry_order_is_validated(self):
        self.write_registry(names=tuple(reversed(TARGET_NAMES)))
        self.assert_failed(self.config, "LABELS")

    def test_warm_index_and_fold_caches_are_reused(self):
        first = prepare_dataset(self.config)
        caches = {path: path.stat().st_mtime_ns for path in (self.output / "cache").glob("*.json")}
        self.assertTrue(caches)
        second = prepare_dataset(self.config)
        for name in ("INDEX", "FOLDS"):
            self.assertFalse(next(stage for stage in first.stages if stage.stage == name).reused)
            self.assertTrue(next(stage for stage in second.stages if stage.stage == name).reused)
        self.assertEqual(caches, {path: path.stat().st_mtime_ns for path in caches})

    def test_corrupt_index_cache_is_detected_and_rebuilt(self):
        first = prepare_dataset(self.config)
        cache = next((self.output / "cache").glob("manifest-index-*.json"))
        cache.write_text("{broken", encoding="utf-8")
        second = prepare_dataset(self.config)
        stage = next(stage for stage in second.stages if stage.stage == "INDEX")
        self.assertFalse(stage.reused)
        self.assertTrue(any("cache" in warning and "corrupt" in warning for warning in stage.warnings))
        self.assertEqual(self.material_ids(first), self.material_ids(second))
        self.assertEqual(second.dataset_index.index_id,
                         json.loads(cache.read_text())["index_id"])

    def test_reuse_can_be_disabled_without_changing_identity(self):
        first = prepare_dataset(self.config)
        second = prepare_dataset(replace(self.config, reuse_valid_artifacts=False))
        self.assertEqual(self.material_ids(first), self.material_ids(second))
        self.assertFalse(any(stage.reused for stage in second.stages))

    def test_missing_geometry_records_ordering_and_selection_fallbacks(self):
        for item in self.manifest["studies"][0]["series"][0]["slices"]:
            item["metadata"].pop("ImagePositionPatient")
            item["metadata"].pop("ImageOrientationPatient")
            item["metadata"].pop("Laterality")
        self.write_manifest()
        prepared = prepare_dataset(self.config)
        stages = {stage.stage: stage for stage in prepared.stages}
        self.assertEqual("UNAVAILABLE", stages["GEOMETRY"].status)
        self.assertEqual("UNAVAILABLE", stages["ORIENTATION"].status)
        self.assertEqual("UNAVAILABLE", stages["SELECT"].status)
        for name in ("GEOMETRY", "ORIENTATION", "SELECT"):
            self.assertTrue(stages[name].reason)
        self.assertTrue(any("InstanceNumber" in value for value in stages["GEOMETRY"].warnings))
        self.assertTrue(any("physical_span" in value and "fallback" in value
                            for value in stages["SELECT"].warnings))
        selected = self.artifact(prepared, "selected_slices")[0]["selected_slice_ids"]
        paths = {item.slice_id: item.relative_path for study in prepared.dataset_index.studies
                 for series in study.series for item in series.slices}
        self.assertTrue(paths[selected[0]].endswith("/00.dcm"))
        self.assertTrue(paths[selected[-1]].endswith("/35.dcm"))
        restored = load_prepared_dataset(prepared.preparation_manifest_uri)
        self.assertEqual({name: stage.status for name, stage in stages.items()},
                         {stage.stage: stage.status for stage in restored.stages})

    def test_unavailable_stage_states_are_stable_and_required_stages_cannot_be_unavailable(self):
        for item in self.manifest["studies"][0]["series"][0]["slices"]:
            item["metadata"].pop("ImagePositionPatient")
        self.write_manifest()
        first = prepare_dataset(self.config)
        second = prepare_dataset(self.config)
        self.assertEqual(self.material_ids(first), self.material_ids(second))
        self.assertEqual(
            [(stage.stage, stage.status, stage.reason) for stage in first.stages],
            [(stage.stage, stage.status, stage.reason) for stage in second.stages],
        )
        with self.assertRaisesRegex(ValueError, "required stage LABELS"):
            replace(first, stages=tuple(
                StageRecord(stage.stage, stage.input_id, stage.output_id,
                    "UNAVAILABLE", stage.duration_seconds, stage.reused,
                    stage.records_processed, stage.warnings, "required input absent")
                if stage.stage == "LABELS" else stage
                for stage in first.stages))
        with self.assertRaisesRegex(ValueError, "stage GEOMETRY failed"):
            replace(first, stages=tuple(
                StageRecord(stage.stage, stage.input_id, stage.output_id,
                    "FAILED", stage.duration_seconds, stage.reused,
                    stage.records_processed, stage.warnings)
                if stage.stage == "GEOMETRY" else stage
                for stage in first.stages))
        with self.assertRaisesRegex(ValueError, "unsupported preparation stage status"):
            StageRecord("GEOMETRY", "a" * 64, None, "UNKNOWN", 0.0)

    def test_required_label_failure_stays_failed_and_does_not_create_prepared_output(self):
        self.manifest["studies"][0]["targets"] = {}
        self.write_manifest()
        failure = self.assert_failed(self.config, "LABELS")
        self.assertEqual("FAILED", failure.stages[-1].status)
        self.assertFalse(list(self.output.glob("prepared/*.json")))

    def test_physical_span_without_positions_records_fallback(self):
        for item in self.manifest["studies"][0]["series"][0]["slices"]:
            item["metadata"].pop("ImagePositionPatient")
        self.write_manifest()
        prepared = prepare_dataset(self.config)
        selection = next(stage for stage in prepared.stages if stage.stage == "SELECT")
        self.assertTrue(any("physical_span" in warning and "fallback" in warning
                            for warning in selection.warnings))

    def test_strict_geometry_failure_keeps_completed_index(self):
        self.manifest["studies"][0]["series"][0]["slices"][0]["metadata"].pop("ImagePositionPatient")
        self.write_manifest()
        self.assert_failed(replace(self.config, strict_geometry=True), "GEOMETRY")
        self.assertEqual(1, len(list((self.output / "cache").glob("manifest-index-*.json"))))

    def test_intermediate_selection_failure_retains_valid_cache_and_no_final(self):
        self.assert_failed(replace(self.config, slice_strategy="unsupported"), "SELECT")
        cache = next((self.output / "cache").glob("manifest-index-*.json"))
        cache_bytes = cache.read_bytes()
        successful = prepare_dataset(self.config)
        self.assertTrue(next(stage for stage in successful.stages if stage.stage == "INDEX").reused)
        self.assertEqual(cache_bytes, cache.read_bytes())

    def test_real_mode_still_blocks_patient_leakage(self):
        names = TARGET_REGISTRY.names
        for study in self.manifest["studies"]:
            study["targets"] = {name: study["targets"][previous]
                                for name, previous in zip(names, TARGET_NAMES)}
        self.manifest["synthetic"] = False
        self.write_manifest()
        self.write_registry(names=names, synthetic=False, official=True)
        self.create_real_slice_files()
        self.assert_failed(replace(self.config, mode="real", target_names=names,
            fold_assignment_overrides={"study_0_0": "fold_0", "study_0_1": "fold_1"}),
            "LEAKAGE")

    def test_previous_valid_final_artifact_survives_later_failed_run(self):
        completed = prepare_dataset(self.config)
        path = Path(completed.preparation_manifest_uri)
        contents = path.read_bytes()
        with self.assertRaises(PreparationError) as context:
            prepare_dataset(replace(self.config, slice_strategy="unsupported"))
        self.assertEqual("SELECT", context.exception.stage)
        self.assertEqual(contents, path.read_bytes())
        self.assertEqual([path], list(self.output.glob("prepared/*.json")))
        self.assertFalse(list(self.output.rglob("*.tmp")))
        load_prepared_dataset(path)
    def test_real_mode_rejects_placeholder_targets(self):
        self.manifest["synthetic"] = False
        self.write_manifest()
        self.write_registry(synthetic=False, official=True)
        self.create_real_slice_files()
        failure = self.assert_failed(replace(self.config, mode="real"), "LABELS")
        self.assertIn("target", failure.message)

    def test_dicom_discovery_official_labels_and_incremental_cache_integration(self):
        """Use fabricated DICOM headers to exercise the real discovery/cache APIs."""
        (self.source / "source_manifest.json").unlink()
        uids = []
        for patient in range(4):
            uid = f"1.2.826.0.1.3680043.10.999.{patient + 1}"
            uids.append(uid)
            for position in range(4):
                path = self.source / str(patient) / f"{position}.dcm"
                path.parent.mkdir(parents=True, exist_ok=True)
                meta = FileMetaDataset()
                meta.MediaStorageSOPClassUID = MRImageStorage
                meta.MediaStorageSOPInstanceUID = f"{uid}.1.{position + 1}"
                meta.TransferSyntaxUID = ExplicitVRLittleEndian
                dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
                for name, value in {"SOPClassUID": MRImageStorage,
                    "SOPInstanceUID": meta.MediaStorageSOPInstanceUID,
                    "StudyInstanceUID": uid, "SeriesInstanceUID": f"{uid}.1",
                    "PatientID": f"patient_{patient}", "InstanceNumber": position + 1,
                    "ImagePositionPatient": [patient * 10, 0, position * 2.5],
                    "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
                    "PixelSpacing": [0.5, 0.5], "Rows": 32, "Columns": 32,
                    "Modality": "MR", "Laterality": "L"}.items():
                    setattr(dataset, name, value)
                dataset.save_as(path, enforce_file_format=True)
        with (self.source / "train.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("StudyInstanceUID", *TARGET_REGISTRY.names))
            for patient, uid in enumerate(uids):
                writer.writerow((uid, *((patient + index) % 2 for index in range(12))))
        config = replace(self.config, mode="real", target_names=TARGET_REGISTRY.names,
                         slice_count=3)
        first = prepare_dataset(config)
        self.assertEqual("READY", first.status)
        self.assertFalse(first.synthetic)
        self.assertEqual(TARGET_REGISTRY.registry_id, first.target_registry.registry_id)
        self.assertEqual("cold_build", first.statistics["index_cache_mode"])
        second = prepare_dataset(config)
        self.assertEqual(self.material_ids(first), self.material_ids(second))
        self.assertEqual("warm_load", second.statistics["index_cache_mode"])
        self.assertTrue(next(stage for stage in second.stages if stage.stage == "INDEX").reused)
        self.assertEqual(self.material_ids(second),
                         self.material_ids(load_prepared_dataset(second.preparation_manifest_uri)))
        changed_path = self.source / "0" / "3.dcm"
        changed = dcmread(changed_path)
        changed.ImagePositionPatient = [0, 0, 9.5]
        changed.save_as(changed_path, enforce_file_format=True)
        refreshed = prepare_dataset(config)
        self.assertEqual("incremental_refresh", refreshed.statistics["index_cache_mode"])
        self.assertFalse(next(stage for stage in refreshed.stages if stage.stage == "INDEX").reused)
        self.assertNotEqual(second.dataset_index.index_id, refreshed.dataset_index.index_id)
        self.assertNotEqual(second.dataset_version.dataset_version_id,
                            refreshed.dataset_version.dataset_version_id)

    def test_saved_preparation_can_be_reloaded(self):
        prepared = prepare_dataset(self.config)
        restored = load_prepared_dataset(prepared.preparation_manifest_uri)
        self.assertEqual(self.material_ids(prepared), self.material_ids(restored))
        self.assertEqual(prepared.to_dict(include_telemetry=False),
                         restored.to_dict(include_telemetry=False))

    def test_reload_rejects_corruption_of_every_referenced_artifact(self):
        prepared = prepare_dataset(self.config)
        for name, reference in prepared.artifacts.items():
            with self.subTest(artifact=name):
                path = Path(reference["uri"])
                original = path.read_bytes()
                try:
                    path.write_text("{}", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_prepared_dataset(prepared.preparation_manifest_uri)
                finally:
                    path.write_bytes(original)

    def test_reload_rejects_manifest_preprocessing_and_assignment_tampering(self):
        prepared = prepare_dataset(self.config)
        path = Path(prepared.preparation_manifest_uri)
        original = path.read_text(encoding="utf-8")
        for field in ("preprocessing_spec", "fold_assignments"):
            with self.subTest(field=field):
                payload = json.loads(original)
                if field == "preprocessing_spec":
                    payload[field]["slice_selection"]["count"] = 32
                else:
                    study_id = prepared.dataset_index.study_by_uid("study_0_0").study_id
                    current = payload[field][study_id]
                    payload[field][study_id] = "fold_1" if current == "fold_0" else "fold_0"
                path.write_text(json.dumps(payload), encoding="utf-8")
                try:
                    with self.assertRaises(ValueError):
                        load_prepared_dataset(path)
                finally:
                    path.write_text(original, encoding="utf-8")

    def write_cli_config(self):
        config = self.root / "prepare.toml"
        config.write_text(
            '[data]\nroot = "source"\noutput_dir = "output"\nmode = "synthetic"\n'
            + "target_names = " + json.dumps(list(TARGET_NAMES))
            + '\n[folds]\nn_folds = 2\nseed = 42\n'
            + '[slice_selection]\nstrategy = "physical_span"\ncount = 24\n', encoding="utf-8")
        return config

    def test_cli_human_output_reports_stages_and_synthetic_status(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["prepare-data", "--config", str(self.write_cli_config())])
        self.assertEqual(0, code)
        message = output.getvalue()
        for stage in STAGES:
            self.assertIn(f"{stage.title()}: PASS", message)
        self.assertIn("SYNTHETIC DATASET PREPARED", message)
        self.assertIn("DatasetVersion:", message)

    def test_cli_json_output_is_reloadable(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["prepare-data", "--config", str(self.write_cli_config()), "--json"])
        self.assertEqual(0, code)
        payload = json.loads(output.getvalue())
        self.assertEqual("SYNTHETIC", payload["status"])
        self.assertTrue(payload["leakage_report"]["passed"])
        restored = load_prepared_dataset(payload["preparation_manifest_uri"])
        self.assertEqual(payload["dataset_version"]["dataset_version_id"],
                         restored.dataset_version.dataset_version_id)

    def test_fresh_process_reloads_preserving_material_ids(self):
        prepared = prepare_dataset(self.config)
        script = (
            "import json,sys; from rsna.preparation import load_prepared_dataset; "
            "p=load_prepared_dataset(sys.argv[1]); "
            "print(json.dumps([p.dataset_version.dataset_version_id, "
            "p.dataset_index.index_id, p.fold_plan.fold_plan_id, "
            "p.target_registry.registry_id, p.preprocessing_spec['preprocessing_id']]))"
        )
        completed = subprocess.run([sys.executable, "-c", script, prepared.preparation_manifest_uri],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
            check=True, timeout=30)
        self.assertEqual(list(self.material_ids(prepared)), json.loads(completed.stdout))

    def test_versioned_synthetic_prepared_dataset_example_is_portable_and_current(self):
        repository = Path(__file__).resolve().parents[1]
        sample_path = repository / "examples" / "prepared-dataset.synthetic.json"
        schema_path = repository / "schemas" / "prepared-dataset.schema.json"
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        stage_schema = schema["properties"]["stages"]["items"]
        allowed_statuses = stage_schema["properties"]["status"]["enum"]
        self.assertEqual({"PASS", "FAILED", "SKIPPED", "UNAVAILABLE"}, set(allowed_statuses))
        self.assertEqual(1, payload["schema_version"])
        self.assertTrue(payload["synthetic"])
        self.assertEqual("SYNTHETIC", payload["status"])
        for stage in payload["stages"]:
            self.assertIn(stage["status"], allowed_statuses)
            if stage["status"] in {"SKIPPED", "UNAVAILABLE"}:
                self.assertTrue(stage["reason"])
        strings = []

        def collect_strings(value):
            if isinstance(value, str):
                strings.append(value)
            elif isinstance(value, dict):
                for nested in value.values():
                    collect_strings(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect_strings(nested)

        collect_strings(payload)
        for value in strings:
            self.assertIsNone(re.match(r"^(?:[A-Za-z]:[\\/]|/(?:home|tmp|Users)/)", value), value)
        self.assertTrue(all(not Path(reference["uri"]).is_absolute()
                            for reference in payload["artifacts"].values()))

        restored = load_prepared_dataset(sample_path)
        with tempfile.TemporaryDirectory() as tmp:
            generated = prepare_dataset(prepared_example_config(Path(tmp)))
        self.assertEqual(self.material_ids(generated), self.material_ids(restored))
        self.assertEqual(
            [(stage.stage, stage.status, stage.reason) for stage in generated.stages],
            [(stage.stage, stage.status, stage.reason) for stage in restored.stages],
        )
        try:
            from jsonschema import Draft202012Validator, ValidationError
            from referencing import Registry, Resource
        except ImportError:
            self.skipTest("JSON Schema validation runs in CI with the locked jsonschema tool")
        resources = []
        for path in (repository / "schemas").glob("*.schema.json"):
            resource_schema = json.loads(path.read_text(encoding="utf-8"))
            resource = Resource.from_contents(resource_schema)
            resources.append((path.name, resource))
            if resource_schema.get("$id"):
                resources.append((resource_schema["$id"], resource))
        registry = Registry().with_resources(resources)
        validator = Draft202012Validator(schema, registry=registry)
        validator.validate(payload)
        invalid_status = json.loads(json.dumps(payload))
        invalid_status["stages"][0]["status"] = "UNKNOWN"
        with self.assertRaises(ValidationError):
            validator.validate(invalid_status)
        missing_reason = json.loads(json.dumps(payload))
        unavailable = next(stage for stage in missing_reason["stages"]
                           if stage["status"] == "UNAVAILABLE")
        unavailable["reason"] = None
        with self.assertRaises(ValidationError):
            validator.validate(missing_reason)


if __name__ == "__main__":
    unittest.main()
