import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rsna.contracts import DatasetVersion, FoldPlan, TrainingJob
from rsna.data.models import DatasetIndex, SeriesRecord, SliceRecord, StudyRecord
from rsna.identity import digest
from rsna.leakage import (LeakageIssueType, LeakagePolicy, LeakageReport,
                          LeakageValidationError, require_valid_leakage_report,
                          validate_leakage, validate_split)


DATASET_ID = "a" * 64


def row(patient="p1", study="st1", series="se1", sop="sop1", fold="fold_0",
        path="/data/a.dcm", file_hash="1" * 64, **extra):
    return {"patient_id": patient, "study_uid": study, "series_uid": series,
            "sop_uid": sop, "fold_id": fold, "path": path, "file_hash": file_hash, **extra}


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.plan = FoldPlan(DATASET_ID, "patient_group", ("fold_0", "fold_1", "fold_2"), 7)

    def test_clean_dataset_passes(self):
        report = validate_leakage([row(sop="s1"), row(sop="s2", path="/data/b.dcm", file_hash="2" * 64)], self.plan,
                                  dataset_version_id=DATASET_ID)
        self.assertTrue(report.passed)
        self.assertEqual(2, report.counts["n_slices_checked"])

    def test_dataset_index_records_and_manifest_are_consumed_directly(self):
        slices = (SliceRecord("a.dcm", 12, {"PatientID": "p1", "StudyInstanceUID": "st1",
                    "SeriesInstanceUID": "se1", "SOPInstanceUID": "sop1"}),)
        series = SeriesRecord("se1", "st1", slices)
        study = StudyRecord("st1", "p1", (series,))
        index = DatasetIndex("root", "1", (study,), (), {"n_studies": 1})
        plan = FoldPlan(DATASET_ID, "patient_group", ("fold_0", "fold_1"), 7)
        direct = validate_leakage(index, plan, assignments={"st1": "fold_0"})
        manifest = validate_leakage(index.to_dict(), plan, assignments={"st1": "fold_0"})
        self.assertTrue(direct.passed)
        self.assertEqual(1, direct.counts["n_slices_checked"])
        self.assertEqual(direct.report_id, manifest.report_id)
        self.assertEqual(index.index_id, direct.to_dict()["provenance"]["dataset_index_id"])

    def test_same_patient_cross_fold_fails(self):
        r = validate_leakage([row(study="s1", sop="1"), row(study="s2", sop="2", fold="fold_1")], self.plan)
        self.assertIn(LeakageIssueType.PATIENT_CROSS_FOLD, {i.type for i in r.issues})
        self.assertFalse(r.passed)

    def test_same_study_cross_fold_fails(self):
        r = validate_leakage([row(sop="1"), row(sop="2", fold="fold_1")], self.plan)
        self.assertIn(LeakageIssueType.STUDY_CROSS_FOLD, {i.type for i in r.issues})

    def test_series_cross_fold_fails(self):
        r = validate_leakage([row(study="st1", sop="1"), row(study="st2", series="se1", sop="2", fold="fold_1")], self.plan)
        self.assertIn(LeakageIssueType.SERIES_CROSS_FOLD, {i.type for i in r.issues})

    def test_same_sop_uid_cross_fold_fails(self):
        r = validate_leakage([row(), row(fold="fold_1", path="/data/copy.dcm")], self.plan)
        self.assertIn(LeakageIssueType.SLICE_CROSS_FOLD, {i.type for i in r.issues})

    def test_same_hash_cross_fold_fails_with_different_uids(self):
        r = validate_leakage([row(), row(study="st2", series="se2", sop="sop2", fold="fold_1")], self.plan)
        self.assertIn(LeakageIssueType.DUPLICATE_FILE_HASH, {i.type for i in r.issues})
        self.assertFalse(r.passed)

    def test_same_hash_same_fold_is_audited_not_cross_fold(self):
        r = validate_leakage([row(), row(study="st2", series="se2", sop="sop2", path="/data/copy.dcm")], self.plan)
        issue = next(i for i in r.issues if i.type == LeakageIssueType.DUPLICATE_FILE_HASH)
        self.assertEqual("warning", issue.severity)
        self.assertTrue(r.passed)

    def test_conflicting_patient_for_study_fails(self):
        r = validate_leakage([row(patient="p1"), row(patient="p2", sop="sop2")], self.plan)
        self.assertIn(LeakageIssueType.CONFLICTING_PATIENT_ID, {i.type for i in r.issues})

    def test_unbound_series_fails(self):
        r = validate_leakage([{"entity_type": "series", "series_uid": "se1", "fold_id": "fold_0"}], self.plan)
        self.assertIn(LeakageIssueType.UNBOUND_SERIES, {i.type for i in r.issues})

    def test_anonymous_series_and_slice_still_report_missing_parents(self):
        records = [{"entity_type": "series", "fold_id": "fold_0"},
                   {"entity_type": "slice", "fold_id": "fold_0"}]
        report = validate_leakage(records, self.plan)
        kinds = {issue.type for issue in report.issues}
        self.assertIn(LeakageIssueType.UNBOUND_SERIES, kinds)
        self.assertIn(LeakageIssueType.UNBOUND_SLICE, kinds)

    def test_unbound_slice_fails(self):
        r = validate_leakage([{"entity_type": "slice", "sop_uid": "s1", "fold_id": "fold_0"}], self.plan)
        self.assertIn(LeakageIssueType.UNBOUND_SLICE, {i.type for i in r.issues})

    def test_missing_patient_uses_other_identities_without_inventing_one(self):
        a, b = row(patient=None), row(patient=None, sop="sop2", study="st2", series="se2", fold="fold_1")
        r = validate_leakage([a, b], self.plan)
        self.assertNotIn(LeakageIssueType.PATIENT_CROSS_FOLD, {i.type for i in r.issues})
        self.assertEqual(0, r.counts["n_patients_checked"])

    def test_duplicate_sop_uid_same_fold_is_audited(self):
        r = validate_leakage([row(), row(path="/data/copy.dcm")], self.plan)
        self.assertIn(LeakageIssueType.DUPLICATE_SOP_UID, {i.type for i in r.issues})
        self.assertTrue(r.passed)

    def test_series_with_multiple_study_parents_fails(self):
        r = validate_leakage([row(sop="s1"), row(study="st2", sop="s2")], self.plan)
        self.assertIn(LeakageIssueType.SERIES_STUDY_MISMATCH, {i.type for i in r.issues})

    def test_same_path_different_identity_is_not_leakage(self):
        r = validate_leakage([row(), row(patient="p2", study="st2", series="se2", sop="sop2", file_hash="2" * 64)], self.plan)
        self.assertNotIn(LeakageIssueType.SLICE_CROSS_FOLD, {i.type for i in r.issues})
        self.assertTrue(r.passed)

    def test_different_paths_same_hash_detects_overlap(self):
        r = validate_leakage([row(), row(study="st2", series="se2", sop="sop2", fold="fold_1", path="/val/b.dcm")], self.plan)
        self.assertIn(LeakageIssueType.DUPLICATE_FILE_HASH, {i.type for i in r.issues})

    def test_explicit_train_validation_without_foldplan(self):
        r = validate_split([row(fold=None)], [row(fold=None, path="/val/a.dcm")])
        self.assertIn(LeakageIssueType.PATIENT_CROSS_FOLD, {i.type for i in r.issues})

    def test_train_test_overlap(self):
        r = validate_split([row(fold=None)], [], test_records=[row(fold=None, path="/test/a.dcm")])
        self.assertIn(LeakageIssueType.STUDY_CROSS_FOLD, {i.type for i in r.issues})

    def test_wrong_dataset_binding_fails(self):
        r = validate_leakage([row()], self.plan, dataset_version_id="b" * 64)
        self.assertIn(LeakageIssueType.DATASET_VERSION_MISMATCH, {i.type for i in r.issues})

    def test_dataset_version_and_index_artifact_binding(self):
        index = {"records": [row()], "dataset_index_artifact_id": "f" * 64}
        version = DatasetVersion("dataset", "v1", "a" * 64, "prep-v1", ("normal",),
                                 dataset_index_artifact_id="f" * 64)
        plan = FoldPlan(version.dataset_version_id, "declared", ("fold_0",), 1)
        report = validate_leakage(index, plan, dataset_version=version)
        self.assertTrue(report.passed)
        self.assertEqual(version.dataset_version_id, report.dataset_version_id)
        self.assertEqual("f" * 64, report.to_dict()["provenance"]["dataset_index_artifact_id"])
        mismatch = validate_leakage({**index, "dataset_index_artifact_id": "e" * 64}, plan,
                                    dataset_version=version)
        self.assertIn(LeakageIssueType.DATASET_VERSION_MISMATCH, {i.type for i in mismatch.issues})

    def test_audit_returns_report_without_raising(self):
        r = validate_leakage([row(), row(fold="fold_1")], self.plan, policy=LeakagePolicy.AUDIT)
        self.assertFalse(r.passed)
        self.assertEqual("audit", r.policy.value)

    def test_strict_gate_blocks(self):
        r = validate_leakage([row(), row(fold="fold_1")], self.plan)
        with self.assertRaises(LeakageValidationError):
            require_valid_leakage_report(r)

    def test_order_does_not_change_material_report_id(self):
        rows = [row(), row(study="st2", series="se2", sop="sop2", fold="fold_1")]
        a = validate_leakage(rows, self.plan)
        b = validate_leakage(list(reversed(rows)), self.plan)
        self.assertEqual(a.report_id, b.report_id)

    def test_paths_do_not_change_material_report_id(self):
        first = validate_leakage([row(), row(fold="fold_1", path="/val/copy.dcm")], self.plan)
        second = validate_leakage([row(path="D:\\cache\\a.dcm"), row(fold="fold_1", path="D:\\val\\copy.dcm")], self.plan)
        self.assertEqual(first.report_id, second.report_id)

    def test_materially_different_clean_inputs_have_different_report_ids(self):
        first = validate_leakage([row()], self.plan)
        second = validate_leakage([row(patient="p2", study="st2", series="se2", sop="sop2")], self.plan)
        self.assertTrue(first.passed and second.passed)
        self.assertNotEqual(first.input_material_sha256, second.input_material_sha256)
        self.assertNotEqual(first.report_id, second.report_id)

    def test_fold_assignments_are_applied_and_checked(self):
        records = [row(fold=None)]
        valid = validate_leakage(records, self.plan, assignments={"st1": "fold_0"})
        self.assertTrue(valid.passed)
        bad = validate_leakage([row(fold="fold_1")], self.plan, assignments={"st1": "fold_0"})
        self.assertIn(LeakageIssueType.UNKNOWN_GROUP_ID, {i.type for i in bad.issues})

    def test_unknown_group_is_reported(self):
        r = validate_leakage([row(fold=None, group_id="unknown")], self.plan, assignments={"st1": "fold_0"})
        self.assertIn(LeakageIssueType.UNKNOWN_GROUP_ID, {i.type for i in r.issues})

    def test_derived_source_crossing_splits_fails(self):
        rows = [row(source_entity_ids=["raw-1"]), row(study="st2", series="se2", sop="sop2",
                                                        fold="fold_1", source_entity_ids=["raw-1"])]
        r = validate_leakage(rows, self.plan)
        self.assertIn(LeakageIssueType.PROVENANCE_CROSS_SPLIT, {i.type for i in r.issues})

    def test_label_provenance_crossing_splits_fails(self):
        r = validate_leakage([row(label_provenance={"source_fold_id": "fold_1"})], self.plan)
        self.assertIn(LeakageIssueType.LABEL_PROVENANCE_LEAKAGE, {i.type for i in r.issues})

    def test_metadata_identity_is_audited(self):
        r = validate_leakage([row(metadata_identity="meta-1"), row(study="st2", series="se2", sop="sop2",
                                                                     metadata_identity="meta-1")], self.plan)
        issue = next(i for i in r.issues if i.type == LeakageIssueType.NEAR_DUPLICATE_METADATA)
        self.assertEqual("warning", issue.severity)

    def test_report_round_trip_preserves_result(self):
        original = validate_leakage([row(), row(fold="fold_1")], self.plan)
        loaded = LeakageReport.from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(original.report_id, loaded.report_id)
        self.assertEqual(original.passed, loaded.passed)

    def test_report_and_issue_match_versioned_schemas(self):
        report = validate_leakage([row(), row(fold="fold_1")], self.plan)
        report_schema = json.loads(Path("schemas/leakage-report.schema.json").read_text())
        issue_schema = json.loads(Path("schemas/leakage-issue.schema.json").read_text())
        self.assertEqual(1, report_schema["properties"]["schema_version"]["const"])
        self.assertEqual(1, issue_schema["properties"]["schema_version"]["const"])
        payload = json.loads(json.dumps(report.to_dict()))
        self.assertEqual(set(report_schema["required"]), set(payload))
        for issue in payload["issues"]:
            self.assertEqual(set(issue_schema["required"]), set(issue))
            self.assertIn(issue["severity"], issue_schema["properties"]["severity"]["enum"])
            self.assertIn(issue["type"], issue_schema["properties"]["type"]["enum"])

    def test_optional_training_hook_preserves_legacy_job_identity(self):
        fields = {"experiment_id": "1" * 64, "dataset_version_id": "2" * 64,
                  "fold_plan_id": "3" * 64, "model_candidate_id": "4" * 64,
                  "fold_id": "fold_0", "random_state": 11,
                  "resources": {"cpus": 1, "gpus": 0}, "configuration": {"x": 1}}
        legacy_material = {"schema_version": 1, **fields}
        legacy = TrainingJob(**fields)
        bound = TrainingJob(**fields, validated_leakage_report_id="5" * 64)
        positional_legacy = TrainingJob(*fields.values(), 1)
        self.assertEqual(digest(legacy_material), legacy.training_job_id)
        self.assertNotEqual(legacy.training_job_id, bound.training_job_id)
        self.assertEqual("5" * 64, bound.to_dict()["validated_leakage_report_id"])
        self.assertEqual(1, positional_legacy.schema_version)

    def test_cli_exit_codes_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.json").write_text(json.dumps({"dataset_version_id": DATASET_ID, "records": [row()]}))
            (root / "plan.json").write_text(json.dumps(self.plan.to_dict()))
            out = root / "report.json"
            ok = subprocess.run([sys.executable, "-m", "rsna", "leakage-check", "--dataset-manifest", str(root / "data.json"),
                                 "--fold-plan", str(root / "plan.json"), "--output", str(out)], capture_output=True)
            self.assertEqual(0, ok.returncode, ok.stderr.decode())
            self.assertTrue(json.loads(out.read_text())["passed"])
            contaminated = {"dataset_version_id": DATASET_ID,
                            "records": [row(), row(fold="fold_1", path="/val/copy.dcm")]}
            (root / "data.json").write_text(json.dumps(contaminated))
            failed = subprocess.run([sys.executable, "-m", "rsna", "leakage-check", "--dataset-manifest", str(root / "data.json"),
                                     "--fold-plan", str(root / "plan.json"), "--output", str(out)], capture_output=True)
            self.assertEqual(1, failed.returncode)
            self.assertFalse(json.loads(out.read_text())["passed"])
            audit = subprocess.run([sys.executable, "-m", "rsna", "leakage-check", "--dataset-manifest", str(root / "data.json"),
                                    "--fold-plan", str(root / "plan.json"), "--output", str(out), "--policy", "audit"],
                                   capture_output=True)
            self.assertEqual(0, audit.returncode)
            self.assertFalse(json.loads(out.read_text())["passed"])

    def test_cli_accepts_saved_dataset_index_manifest(self):
        slices = (SliceRecord("a.dcm", 12, {"PatientID": "p1", "StudyInstanceUID": "st1",
                    "SeriesInstanceUID": "se1", "SOPInstanceUID": "sop1"}),)
        index = DatasetIndex("root", "1", (StudyRecord("st1", "p1",
                              (SeriesRecord("se1", "st1", slices),)),), (), {"n_studies": 1})
        plan = FoldPlan(DATASET_ID, "patient_group", ("fold_0",), 7)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_manifest = {**index.to_dict(), "dataset_version_id": DATASET_ID}
            (root / "data.json").write_text(json.dumps(dataset_manifest))
            (root / "plan.json").write_text(json.dumps({"fold_plan": plan.to_dict(),
                                                           "assignments": {"st1": "fold_0"}}))
            result = subprocess.run([sys.executable, "-m", "rsna", "leakage-check", "--dataset-manifest",
                                     str(root / "data.json"), "--fold-plan", str(root / "plan.json"),
                                     "--output", str(root / "report.json")], capture_output=True)
            self.assertEqual(0, result.returncode, result.stderr.decode())
            self.assertTrue(json.loads((root / "report.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
