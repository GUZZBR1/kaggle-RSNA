import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "rsna", *map(str, args)], cwd=ROOT,
                          text=True, capture_output=True, check=False)


class DatasetCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.payload = {
            "dataset_id": "synthetic",
            "schema_version": 1,
            "studies": [{"study_uid": "study-1", "patient_id": "patient-1"}],
            "series": [
                {"series_uid": "series-1", "study_uid": "study-1", "patient_id": "patient-1",
                 "modality": "MR", "plane": "sagittal", "laterality": "right", "n_slices": 2},
                {"series_uid": "series-2", "study_uid": "study-1", "patient_id": "patient-1",
                 "plane": "axial", "laterality": "left", "n_slices": 1},
            ],
            "slices": [
                {"sop_uid": "slice-1", "study_uid": "study-1", "series_uid": "series-1",
                 "path": "image-1.dcm", "instance_number": 1, "position": [0, 0, 0],
                 "orientation": [1, 0, 0, 0, 1, 0], "spacing": [0.5, 0.5]},
                {"sop_uid": "slice-2", "study_uid": "study-1", "series_uid": "series-1",
                 "path": "image-2.dcm", "instance_number": 2, "position": [0, 0, 1],
                 "orientation": [1, 0, 0, 0, 1, 0], "spacing": [0.5, 0.5]},
                {"sop_uid": "slice-3", "study_uid": "study-1", "series_uid": "series-2",
                 "path": "image-3.dcm", "instance_number": 1, "position": [0, 0, 0],
                 "orientation": [1, 0, 0, 0, 1, 0], "spacing": [0.5, 0.5]},
            ],
        }
        self.write_manifest(self.payload)

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, value):
        self.manifest.write_text(json.dumps(value), encoding="utf-8")

    def test_help_and_legacy_smoke(self):
        result = run_cli("--help")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("data", result.stdout)
        smoke = run_cli("configs/experiments/smoke.toml")
        self.assertEqual(0, smoke.returncode, smoke.stderr)
        self.assertTrue(json.loads(smoke.stdout)["synthetic"])

    def test_summary_human_json_and_manifest_inspection(self):
        human = run_cli("data", "summary", "--manifest", self.manifest)
        self.assertEqual(0, human.returncode, human.stderr)
        self.assertIn("Studies: 1", human.stdout)
        output = run_cli("data", "summary", "--manifest", self.manifest, "--json")
        self.assertEqual(0, output.returncode, output.stderr)
        parsed = json.loads(output.stdout)
        self.assertEqual(1, parsed["studies"])
        inspection = run_cli("data", "inspect-manifest", "--manifest", self.manifest, "--json")
        self.assertEqual(0, inspection.returncode, inspection.stderr)
        self.assertEqual(2, json.loads(inspection.stdout)["entity_tables"]["series"])

    def test_entity_inspections_and_missing_entity(self):
        commands = [
            ("inspect-study", "--study-id", "study-1", "n_series"),
            ("inspect-series", "--series-id", "series-1", "n_slices"),
            ("inspect-slice", "--sop-id", "slice-1", "instance_number"),
        ]
        for command, id_flag, uid, key in commands:
            result = run_cli("data", command, "--manifest", self.manifest,
                             id_flag, uid, "--json")
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
        duplicate = dict(self.payload)
        duplicate["slices"] = self.payload["slices"] + [dict(self.payload["slices"][0])]
        self.write_manifest(duplicate)
        failed = run_cli("data", "validate", "--manifest", self.manifest, "--json")
        self.assertEqual(1, failed.returncode)
        self.assertIn("DUPLICATE_UID", {item["code"] for item in json.loads(failed.stdout)["errors"]})
        self.write_manifest(self.payload)
        warning = run_cli("data", "validate", "--manifest", self.manifest, "--json")
        self.assertEqual(0, warning.returncode)
        self.assertTrue(json.loads(warning.stdout)["warnings"])

    def test_stats_sample_filters_and_empty_data(self):
        stats = run_cli("data", "stats", "--manifest", self.manifest, "--json")
        self.assertEqual(0, stats.returncode, stats.stderr)
        self.assertEqual(2, json.loads(stats.stdout)["slices_per_series"]["count"])
        one = run_cli("data", "sample", "--manifest", self.manifest, "--entity", "series",
                      "--count", 1, "--seed", 42, "--json")
        two = run_cli("data", "sample", "--manifest", self.manifest, "--entity", "series",
                      "--count", 1, "--seed", 42, "--json")
        self.assertEqual(json.loads(one.stdout), json.loads(two.stdout))
        filtered = run_cli("data", "summary", "--manifest", self.manifest,
                           "--plane", "sagittal", "--json")
        self.assertEqual(1, json.loads(filtered.stdout)["series"])
        empty_path = self.root / "empty.json"
        empty_path.write_text(json.dumps({"studies": [], "series": [], "slices": []}), encoding="utf-8")
        empty = run_cli("data", "stats", "--manifest", empty_path, "--json")
        self.assertEqual(0, empty.returncode, empty.stderr)
        empty_stats = json.loads(empty.stdout)
        self.assertEqual(0, empty_stats["series_per_study"]["count"])
        self.assertEqual(0, empty_stats["slices_per_series"]["count"])

    def test_artifact_hash_mismatch_and_json_stdout_purity(self):
        payload_file = self.root / "payload.json"
        payload_file.write_text('{"value":1}', encoding="utf-8")
        artifact = self.root / "artifact.json"
        artifact.write_text(json.dumps({"schema_version": 1,
                                        "files": [{"path": payload_file.name,
                                                   "sha256": "0" * 64}]}), encoding="utf-8")
        result = run_cli("artifact", "validate", artifact, "--json")
        self.assertEqual(1, result.returncode)
        self.assertIn("HASH_MISMATCH", {item["code"] for item in json.loads(result.stdout)["errors"]})
        self.assertEqual("", result.stderr)

    def test_config_defaults_cli_override_quiet_and_invalid_usage(self):
        config = self.root / "config.toml"
        config.write_text(f'[cli]\nmanifest = "{self.manifest.as_posix()}"\noutput_format = "json"\n',
                          encoding="utf-8")
        configured = run_cli("--config", config, "data", "summary")
        self.assertEqual(0, configured.returncode, configured.stderr)
        self.assertEqual(1, json.loads(configured.stdout)["studies"])
        override = self.root / "other.json"
        override.write_text(json.dumps({"studies": [], "series": [], "slices": []}), encoding="utf-8")
        overridden = run_cli("--config", config, "data", "summary", "--manifest", override, "--json")
        self.assertEqual(0, overridden.returncode, overridden.stderr)
        self.assertEqual(0, json.loads(overridden.stdout)["studies"])
        quiet = run_cli("--quiet", "data", "summary", "--manifest", self.manifest, "--json")
        self.assertEqual(0, quiet.returncode, quiet.stderr)
        invalid = run_cli("data", "does-not-exist")
        self.assertEqual(2, invalid.returncode)


if __name__ == "__main__":
    unittest.main()
