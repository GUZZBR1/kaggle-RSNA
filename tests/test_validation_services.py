import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rsna.artifacts import JsonArtifactStore
from rsna.contracts import DatasetVersion
from rsna.inspection.validate import validate_artifact, validate_dataset


def manifest():
    return {"schema_version": 1, "dataset_version_id": "a" * 64,
            "studies": [{"study_uid": "study", "patient_id": "patient"}],
            "series": [{"series_uid": "series", "study_uid": "study", "n_slices": 1}],
            "slices": [{"sop_uid": "slice", "series_uid": "series", "study_uid": "study",
                        "position": [0, 0, 1], "orientation": [1, 0, 0, 0, 1, 0],
                        "spacing": [1, 1]}]}


class ValidationServicesTests(unittest.TestCase):
    def test_valid_dataset_and_serializable_report(self):
        report = validate_dataset(manifest())
        self.assertTrue(report.passed)
        self.assertEqual(0, report.exit_code)
        self.assertEqual(1, report.statistics["slices"])
        self.assertTrue(json.loads(json.dumps(report.to_dict()))["passed"])

    def test_duplicate_uid_fails_even_for_equal_records(self):
        data = manifest()
        data["slices"].append(copy.deepcopy(data["slices"][0]))
        report = validate_dataset(data)
        self.assertEqual(1, report.exit_code)
        self.assertIn("DUPLICATE_UID", {error.code for error in report.errors})

    def test_orphan_series_and_slice(self):
        data = manifest()
        data["series"][0]["study_uid"] = "unknown"
        data["slices"][0]["series_uid"] = "unknown"
        report = validate_dataset(data)
        self.assertTrue({"ORPHAN_SERIES", "ORPHAN_SLICE"}.issubset({x.code for x in report.errors}))

    def test_parent_patient_and_study_consistency(self):
        data = manifest()
        data["slices"][0].update(patient_id="other-patient", study_uid="other-study")
        data["studies"].append({"study_uid": "other-study", "patient_id": "patient"})
        report = validate_dataset(data)
        self.assertTrue({"PATIENT_MISMATCH", "STUDY_MISMATCH"}.issubset({x.code for x in report.errors}))

    def test_patient_consistency_without_study_table(self):
        data = {"series": [{"series_uid": "a", "study_uid": "study", "patient_id": "a"},
                           {"series_uid": "b", "study_uid": "study", "patient_id": "b"}]}
        self.assertIn("PATIENT_MISMATCH", {x.code for x in validate_dataset(data).errors})

    def test_absent_tables_are_not_counted_or_reported_as_orphans(self):
        data = {"schema_version": 1, "series": [{"series_uid": "a", "study_uid": "study"}]}
        report = validate_dataset(data)
        self.assertTrue(report.passed)
        self.assertEqual({"series": 1}, report.statistics)

    def test_warning_policy(self):
        data = manifest()
        del data["slices"][0]["position"]
        self.assertEqual(0, validate_dataset(data).exit_code)
        self.assertEqual(1, validate_dataset(data, warnings_as_errors=True).exit_code)

    def test_invalid_schemas_never_crash(self):
        for data in ([], {}, {"studies": "wrong"}, {"series": [5]},
                     {"schema_version": True, "studies": []},
                     {"series": [{"series_uid": [], "study_uid": {}}]},
                     {"slices": [{"sop_uid": "x", "study_uid": [], "series_uid": [], "warnings": 5}]}):
            with self.subTest(data=data):
                self.assertEqual(1, validate_dataset(data).exit_code)

    def test_invalid_metadata_and_hash_format(self):
        data = manifest()
        data["slices"][0].update(orientation=[1], sha256="x", rows=True)
        codes = {x.code for x in validate_dataset(data).errors}
        self.assertTrue({"INVALID_METADATA", "INVALID_HASH"}.issubset(codes))

    def test_dataset_version_contract_and_binding(self):
        version = DatasetVersion("test", "v1", "b" * 64, "prep", ("normal",))
        data = manifest()
        data["dataset_version"] = version.to_dict()
        self.assertIn("DATASET_VERSION_MISMATCH", {x.code for x in validate_dataset(data).errors})
        data["dataset_version_id"] = version.dataset_version_id
        self.assertTrue(validate_dataset(data).passed)
        data["slices"][0]["dataset_version_id"] = "c" * 64
        self.assertFalse(validate_dataset(data).passed)

    def test_declared_slice_count_consistency(self):
        data = manifest()
        data["series"][0]["n_slices"] = 2
        self.assertIn("SLICE_COUNT_MISMATCH", {x.code for x in validate_dataset(data).errors})

    def test_nested_dataset_version_cannot_override_binding(self):
        data = {"schema_version": 1, "dataset_version_id": "b" * 64, "index": manifest()}
        self.assertIn("DATASET_VERSION_MISMATCH", {x.code for x in validate_dataset(data).errors})
        data["schema_version"] = 2
        self.assertIn("UNSUPPORTED_SCHEMA_VERSION", {x.code for x in validate_dataset(data).errors})

    def test_source_manifest_hash_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text("{}", encoding="utf-8")
            sha = hashlib.sha256(source.read_bytes()).hexdigest()
            version = DatasetVersion("test", "v1", sha, "prep", ("normal",), uri=str(source))
            data = manifest()
            data.update(dataset_version=version.to_dict(), dataset_version_id=version.dataset_version_id)
            self.assertTrue(validate_dataset(data, level="full").passed)
            source.write_text('{"changed": true}', encoding="utf-8")
            self.assertIn("HASH_MISMATCH", {x.code for x in validate_dataset(data, level="full").errors})

    def test_missing_corrupt_and_invalid_json_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            self.assertEqual(3, validate_dataset(path).exit_code)
            for text in ("{", '{"schema_version": NaN}', "\ufffd",
                         '{"studies": [], "studies": []}'):
                path.write_text(text, encoding="utf-8")
                self.assertEqual(3, validate_dataset(path).exit_code)
            path.write_text("[]", encoding="utf-8")
            self.assertEqual(1, validate_dataset(path).exit_code)

    def test_full_hashes_references_without_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dicom = root / "invalid-but-opaque.dcm"
            dicom.write_bytes(b"opaque bytes without DICOM structure")
            data = manifest()
            data["slices"][0].update(path=dicom.name, sha256=hashlib.sha256(dicom.read_bytes()).hexdigest())
            self.assertTrue(validate_dataset(data, level="full", dataset_root=root).passed)
            dicom.write_bytes(b"changed")
            self.assertTrue(validate_dataset(data, level="basic", dataset_root=root).passed)
            report = validate_dataset(data, level="full", dataset_root=root)
            self.assertIn("HASH_MISMATCH", {x.code for x in report.errors})
            dicom.unlink()
            self.assertIn("MISSING_REFERENCED_FILE", {x.code for x in validate_dataset(data, level="full", dataset_root=root).errors})

    def test_external_index_corruption_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "manifest.json"
            index = root / "index.json"
            path.write_text(json.dumps({"schema_version": 1, "index_path": "index.json"}), encoding="utf-8")
            self.assertEqual(3, validate_dataset(path).exit_code)
            index.write_text("{", encoding="utf-8")
            self.assertEqual(3, validate_dataset(path).exit_code)
            index.write_text(json.dumps(manifest()), encoding="utf-8")
            path.write_text(json.dumps({"schema_version": 1, "index_path": "index.json", "index_sha256": "f" * 64}), encoding="utf-8")
            self.assertIn("HASH_MISMATCH", {x.code for x in validate_dataset(path, level="full").errors})

    def test_artifact_reference_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = JsonArtifactStore(root).put_json({"x": 1})
            path = root / "manifest.json"
            path.write_text(json.dumps(ref.to_dict()), encoding="utf-8")
            self.assertTrue(validate_artifact(path).passed)
            Path(ref.uri).write_text('{"x":2}', encoding="utf-8")
            self.assertIn("HASH_MISMATCH", {x.code for x in validate_artifact(path).errors})

    def test_content_addressed_payload_and_class_names_not_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = JsonArtifactStore(root).put_json({"class_names": ["normal", "abnormal"]})
            self.assertTrue(validate_artifact(ref.uri).passed)
            Path(ref.uri).write_text("{}", encoding="utf-8")
            self.assertIn("HASH_MISMATCH", {x.code for x in validate_artifact(ref.uri).errors})

    def test_generic_payload_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload.bin").write_bytes(b"payload")
            sha = hashlib.sha256(b"payload").hexdigest()
            path = root / "manifest.json"
            path.write_text(json.dumps({"schema_version": 1, "artifact_id": sha,
                                        "payload_path": "payload.bin", "payload_sha256": sha}), encoding="utf-8")
            self.assertTrue(validate_artifact(root).passed)
            (root / "payload.bin").write_bytes(b"wrong")
            self.assertFalse(validate_artifact(root).passed)

    def test_artifact_schema_corruption_and_missing_required_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("{", encoding="utf-8")
            self.assertEqual(3, validate_artifact(path).exit_code)
            path.write_text(json.dumps({"artifact_id": "a" * 64, "schema_version": 1}), encoding="utf-8")
            self.assertEqual(1, validate_artifact(path).exit_code)

    def test_referenced_file_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"schema_version": 1, "referenced_files": ["missing.json"]}), encoding="utf-8")
            self.assertIn("MISSING_REFERENCED_FILE", {x.code for x in validate_artifact(path).errors})

    def test_invalid_reference_types_and_paths_do_not_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for reference in ({"path": ["invalid"]}, {"path": "\x00invalid"}, {"path": "http://[invalid"}):
                path.write_text(json.dumps({"schema_version": 1, "files": [reference]}), encoding="utf-8")
                self.assertFalse(validate_artifact(path).passed)

    def test_invalid_level_is_api_usage_error(self):
        with self.assertRaises(ValueError):
            validate_dataset(manifest(), level="clinical")


if __name__ == "__main__":
    unittest.main()
