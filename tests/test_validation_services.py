import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rsna.artifacts import JsonArtifactStore
from rsna.data import DatasetIndex, SeriesRecord, SliceRecord, StudyRecord, load_manifest, save_manifest
from rsna.inspection.validate import validate_artifact, validate_dataset


def make_index(*, duplicate=False, orphan=False, conflict=False, missing_optional=False):
    metadata = {"SOPInstanceUID": "sop", "StudyInstanceUID": "study", "SeriesInstanceUID": "series",
                "InstanceNumber": 1, "ImagePositionPatient": [0, 0, 1],
                "ImageOrientationPatient": [1, 0, 0, 0, 1, 0], "PixelSpacing": [1, 1], "Rows": 10, "Columns": 10}
    if missing_optional:
        metadata.pop("ImagePositionPatient")
    first = SliceRecord("one.dcm", 10, metadata)
    slices = (first, SliceRecord("two.dcm", 10, dict(metadata))) if duplicate else (first,)
    if orphan:
        metadata["StudyInstanceUID"] = "missing-study"
        first = SliceRecord("one.dcm", 10, metadata)
        slices = (first,)
    series = SeriesRecord("series", "missing-study" if orphan else "study", slices)
    warnings = ("conflicting PatientID values",) if conflict else ()
    study = StudyRecord("study", "patient", (series,), warnings)
    stats = {"n_studies": 1, "n_series": 1, "n_slices": len(slices),
             "duplicate_sop_uid_count": int(duplicate), "invalid_files": 0}
    return DatasetIndex("root", "test", (study,), (), stats)


class ValidationServicesTests(unittest.TestCase):
    def test_valid_canonical_dataset_and_serializable_report(self):
        report = validate_dataset(make_index())
        self.assertTrue(report.passed, report.to_dict())
        self.assertEqual(0, report.exit_code)
        self.assertEqual(1, report.statistics["slices"])
        self.assertTrue(json.loads(json.dumps(report.to_dict()))["passed"])

    def test_duplicate_sop_uid_fails(self):
        report = validate_dataset(make_index(duplicate=True))
        self.assertEqual(1, report.exit_code)
        self.assertIn("DUPLICATE_SOP_UID", {error.code for error in report.errors})

    def test_orphan_series_and_slice_and_patient_conflict(self):
        report = validate_dataset(make_index(orphan=True, conflict=True))
        self.assertIn("ORPHAN_SERIES", {item.code for item in report.errors})
        self.assertIn("ORPHAN_SLICE", {item.code for item in report.errors})
        self.assertIn("CONFLICTING_PATIENT_ID", {item.code for item in report.errors})

    def test_optional_metadata_warns_but_does_not_fail(self):
        report = validate_dataset(make_index(missing_optional=True))
        self.assertTrue(report.passed)
        self.assertIn("MISSING_IMAGEPOSITIONPATIENT", {item.code for item in report.warnings})
        self.assertEqual(1, validate_dataset(make_index(missing_optional=True), warnings_as_errors=True).exit_code)

    def test_missing_corrupt_and_invalid_schema_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            self.assertEqual(3, validate_dataset(path).exit_code)
            path.write_text("{", encoding="utf-8")
            self.assertEqual(3, validate_dataset(path).exit_code)
            path.write_text(json.dumps({"manifest_schema_version": 99}), encoding="utf-8")
            self.assertEqual(1, validate_dataset(path).exit_code)

    def test_full_checks_referenced_source_files_without_pixel_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.dcm").write_bytes(b"opaque DICOM bytes")
            report = validate_dataset(make_index(), level="full", dataset_root=root)
            self.assertTrue(report.passed, report.to_dict())
            self.assertEqual(1, report.statistics["referenced_files_checked"])
            (root / "one.dcm").unlink()
            self.assertIn("MISSING_REFERENCED_FILE", {item.code for item in validate_dataset(
                make_index(), level="full", dataset_root=root).errors})

    def test_artifact_hash_and_schema_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = JsonArtifactStore(root).put_json({"x": 1})
            path = root / "manifest.json"
            path.write_text(json.dumps(ref.to_dict()), encoding="utf-8")
            self.assertTrue(validate_artifact(path).passed)
            Path(ref.uri).write_text('{"x":2}', encoding="utf-8")
            self.assertIn("HASH_MISMATCH", {x.code for x in validate_artifact(path).errors})
            path.write_text("{", encoding="utf-8")
            self.assertEqual(3, validate_artifact(path).exit_code)

    def test_generic_payload_manifest_references_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.bin"
            payload.write_bytes(b"payload")
            sha = hashlib.sha256(b"payload").hexdigest()
            path = root / "manifest.json"
            path.write_text(json.dumps({"schema_version": 1, "artifact_id": sha,
                "payload_path": "payload.bin", "payload_sha256": sha}), encoding="utf-8")
            self.assertTrue(validate_artifact(root).passed)
            payload.write_bytes(b"wrong")
            self.assertFalse(validate_artifact(root).passed)

    def test_saved_manifest_uses_one_canonical_validation_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            save_manifest(make_index(), path)
            dataset_report = validate_dataset(path)
            artifact_report = validate_artifact(path, level="basic")
            self.assertEqual(dataset_report.to_dict(), artifact_report.to_dict())
            self.assertNotIn("MISSING_SCHEMA_VERSION", {item.code for item in artifact_report.warnings})
            self.assertEqual(make_index(), load_manifest(path))

    def test_canonical_schema_rejects_malformed_nested_records_and_derived_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            mutations = (
                lambda value: value.update(manifest_schema_version=True),
                lambda value: value.update(studies={}),
                lambda value: value["studies"][0].update(series="flat"),
                lambda value: value["studies"][0]["series"][0].update(n_slices=99),
                lambda value: value["studies"][0]["series"][0].update(metadata={"Rows": 999}),
                lambda value: value["studies"][0]["series"][0]["slices"][0].update(file_size=True),
                lambda value: value.update(statistics={"n_slices": "one"}),
                lambda value: value["studies"][0].update(study_id=""),
                lambda value: value.update(warnings=[{"code": "flat-warning"}]),
            )
            for mutate in mutations:
                with self.subTest(mutation=mutate):
                    save_manifest(make_index(), path)
                    value = json.loads(path.read_text())
                    mutate(value)
                    path.write_text(json.dumps(value))
                    report = validate_dataset(path)
                    self.assertEqual(1, report.exit_code, report.to_dict())
                    self.assertIn("SCHEMA_INVALID", {item.code for item in report.errors})
                    with self.assertRaises(ValueError):
                        load_manifest(path)

    def test_duplicate_keys_and_nonfinite_json_are_unreadable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for text in ('{"manifest_schema_version":1,"manifest_schema_version":1}',
                         '{"value":NaN}'):
                path.write_text(text)
                self.assertEqual(3, validate_dataset(path).exit_code)
                with self.assertRaises(ValueError):
                    load_manifest(path)

    def test_invalid_level_is_api_usage_error(self):
        with self.assertRaises(ValueError):
            validate_dataset(make_index(), level="clinical")


if __name__ == "__main__":
    unittest.main()
