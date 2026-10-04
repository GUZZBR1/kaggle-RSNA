import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rsna.data import DatasetIndex, SeriesRecord, SliceRecord, StudyRecord, save_manifest

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "rsna", *map(str, args)], cwd=ROOT,
                          text=True, capture_output=True, check=False)


def make_index(*, duplicate=False, empty=False):
    if empty:
        return DatasetIndex("fixture", "test", (), (), {"n_studies": 0, "n_series": 0, "n_slices": 0,
                            "invalid_files": 0, "duplicate_sop_uid_count": 0})
    series = []
    for number, (uid, orient, count) in enumerate((
            ("series-1", (0, 1, 0, 0, 0, 1), 2), ("series-2", (1, 0, 0, 0, 1, 0), 1))):
        slices = []
        for position in range(count):
            sop = "slice-1-1" if duplicate and number == 1 else f"slice-{number + 1}-{position + 1}"
            slices.append(SliceRecord(f"{sop}.dcm", 10, {
                "SOPInstanceUID": sop, "StudyInstanceUID": "study-1", "SeriesInstanceUID": uid,
                "InstanceNumber": position + 1, "ImagePositionPatient": [0, 0, position],
                "ImageOrientationPatient": list(orient), "PixelSpacing": [0.5, 0.5], "Rows": 320, "Columns": 320,
                "Modality": "MR", "SeriesDescription": "T2", "Laterality": "R" if number == 0 else "L"}))
        series.append(SeriesRecord(uid, "study-1", tuple(slices)))
    study = StudyRecord("study-1", "patient-1", tuple(series))
    total = sum(item.n_slices for item in series)
    return DatasetIndex("fixture", "test", (study,), (), {"n_studies": 1, "n_series": 2, "n_slices": total,
                        "invalid_files": 0, "duplicate_sop_uid_count": int(duplicate)})


class DatasetCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.write_manifest(make_index())

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, index):
        save_manifest(index, self.manifest)

    def test_help_and_legacy_smoke(self):
        result = run_cli("--help")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("data", result.stdout)
        self.assertIn("synthetic", result.stdout)
        smoke = run_cli("configs/experiments/smoke.toml")
        self.assertEqual(0, smoke.returncode, smoke.stderr)
        self.assertTrue(json.loads(smoke.stdout)["synthetic"])

    def test_synthetic_smoke_help_and_leakage_injection_cli(self):
        help_result = run_cli("synthetic", "smoke", "--help")
        self.assertEqual(0, help_result.returncode, help_result.stderr)
        self.assertIn("--inject", help_result.stdout)
        with tempfile.TemporaryDirectory() as temp:
            result = run_cli("synthetic", "smoke", "--seed", 42,
                             "--inject", "patient-leakage", "--keep", temp)
            self.assertEqual(0, result.returncode, result.stderr)
            output = json.loads(result.stdout)
            self.assertEqual("EXPECTED_FAILURE", output["status"])
            self.assertIn("PATIENT_CROSS_FOLD", output["issue_types"])
            self.assertFalse((Path(temp) / "prepared" / "prepared-dataset.json").exists())

    def test_summary_human_json_and_manifest_inspection(self):
        human = run_cli("data", "summary", "--manifest", self.manifest)
        self.assertEqual(0, human.returncode, human.stderr)
        self.assertIn("Studies: 1", human.stdout)
        output = run_cli("data", "summary", "--manifest", self.manifest, "--json")
        self.assertEqual(0, output.returncode, output.stderr)
        parsed = json.loads(output.stdout)
        self.assertEqual(1, parsed["studies"])
        self.assertEqual(make_index().index_id, parsed["dataset_index_id"])
        self.assertEqual("unavailable", parsed["dataset_version_binding"])
        inspection = run_cli("data", "inspect-manifest", "--manifest", self.manifest, "--json")
        self.assertEqual(0, inspection.returncode, inspection.stderr)
        self.assertEqual(make_index().index_id, json.loads(inspection.stdout)["index_id"])

    def test_entity_inspections_and_missing_entity(self):
        for command, id_flag, uid, key in (
                ("inspect-study", "--study-id", "study-1", "n_series"),
                ("inspect-series", "--series-id", "series-1", "n_slices"),
                ("inspect-slice", "--sop-id", "slice-1-1", "InstanceNumber")):
            result = run_cli("data", command, "--manifest", self.manifest, id_flag, uid, "--json")
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn(key, json.loads(result.stdout))
        missing = run_cli("data", "inspect-study", "--manifest", self.manifest,
                          "--study-id", "absent", "--json")
        self.assertEqual(1, missing.returncode)
        self.assertEqual("INVALID_INPUT", json.loads(missing.stdout)["errors"][0]["code"])

    def test_missing_and_corrupt_manifest(self):
        missing = run_cli("data", "summary", "--manifest", self.root / "missing.json")
        self.assertEqual(3, missing.returncode)
        self.assertIn("ERROR:", missing.stderr)
        corrupt_path = self.root / "broken.json"
        corrupt_path.write_text("{broken", encoding="utf-8")
        corrupt = run_cli("data", "validate", "--manifest", corrupt_path, "--json")
        self.assertEqual(3, corrupt.returncode)
        self.assertFalse(json.loads(corrupt.stdout)["passed"])

    def test_validation_duplicate_uid_and_warning_policy(self):
        self.write_manifest(make_index(duplicate=True))
        failed = run_cli("data", "validate", "--manifest", self.manifest, "--json")
        self.assertEqual(1, failed.returncode)
        self.assertIn("DUPLICATE_SOP_UID", {item["code"] for item in json.loads(failed.stdout)["errors"]})
        self.write_manifest(make_index())
        warning = run_cli("data", "validate", "--manifest", self.manifest, "--json")
        self.assertEqual(0, warning.returncode)
        self.assertTrue(json.loads(warning.stdout)["warnings"])

    def test_stats_sample_filters_and_empty_data(self):
        stats = run_cli("data", "stats", "--manifest", self.manifest, "--json")
        self.assertEqual(0, stats.returncode, stats.stderr)
        self.assertEqual(2, json.loads(stats.stdout)["slices_per_series"]["count"])
        one = run_cli("data", "sample", "--manifest", self.manifest, "--entity", "series", "--count", 1, "--seed", 42, "--json")
        two = run_cli("data", "sample", "--manifest", self.manifest, "--entity", "series", "--count", 1, "--seed", 42, "--json")
        self.assertEqual(json.loads(one.stdout), json.loads(two.stdout))
        filtered = run_cli("data", "summary", "--manifest", self.manifest, "--plane", "sagittal", "--json")
        self.assertEqual(1, json.loads(filtered.stdout)["series"])
        self.write_manifest(make_index(empty=True))
        empty = run_cli("data", "stats", "--manifest", self.manifest, "--json")
        self.assertEqual(0, empty.returncode, empty.stderr)
        self.assertEqual(0, json.loads(empty.stdout)["slices_per_series"]["count"])

    def test_artifact_hash_mismatch_and_json_stdout_purity(self):
        payload_file = self.root / "payload.json"
        payload_file.write_text('{"value":1}', encoding="utf-8")
        artifact = self.root / "artifact.json"
        artifact.write_text(json.dumps({"schema_version": 1,
                                        "files": [{"path": payload_file.name, "sha256": "0" * 64}]}), encoding="utf-8")
        result = run_cli("artifact", "validate", artifact, "--json")
        self.assertEqual(1, result.returncode)
        self.assertIn("HASH_MISMATCH", {item["code"] for item in json.loads(result.stdout)["errors"]})
        self.assertEqual("", result.stderr)

    def test_config_defaults_cli_override_quiet_and_invalid_usage(self):
        config = self.root / "config.toml"
        config.write_text(f'[cli]\nmanifest = "{self.manifest.as_posix()}"\noutput_format = "json"\n', encoding="utf-8")
        configured = run_cli("--config", config, "data", "summary")
        self.assertEqual(0, configured.returncode, configured.stderr)
        self.assertEqual(1, json.loads(configured.stdout)["studies"])
        override = self.root / "other.json"
        save_manifest(make_index(empty=True), override)
        overridden = run_cli("--config", config, "data", "summary", "--manifest", override, "--json")
        self.assertEqual(0, overridden.returncode, overridden.stderr)
        self.assertEqual(0, json.loads(overridden.stdout)["studies"])
        quiet = run_cli("--quiet", "data", "summary", "--manifest", self.manifest, "--json")
        self.assertEqual(0, quiet.returncode, quiet.stderr)
        invalid = run_cli("data", "does-not-exist")
        self.assertEqual(2, invalid.returncode)


if __name__ == "__main__":
    unittest.main()
