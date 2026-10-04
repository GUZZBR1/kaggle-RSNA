"""End-to-end synthetic pipeline checks; these never use Kaggle or GPU data."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rsna.data.smoke import run_data_smoke
from rsna.data.synthetic import SyntheticConfig
from rsna.targets import TARGET_REGISTRY, TARGET_REGISTRY_ID


class SyntheticSmokeIntegrationTests(unittest.TestCase):
    def run_smoke(self, directory: str, *, seed: int = 42, injection: str | None = None) -> dict:
        config = SyntheticConfig(seed=seed, patients=5, studies_per_patient=2,
            series_per_study=1, min_slices=2, max_slices=3,
            missing_label_rate=0.05, injection=injection)
        return run_data_smoke(directory, seed=seed, injection=injection, synthetic_config=config)

    def test_happy_path_and_reproducible_material_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = self.run_smoke(str(Path(temp) / "first"))
            second = self.run_smoke(str(Path(temp) / "second"))
        self.assertEqual(first["status"], "READY")
        self.assertEqual(first["patients"], 5)
        self.assertEqual(first["folds"], 5)
        self.assertEqual(first["target_order"], list(TARGET_REGISTRY.names))
        self.assertEqual(first["target_schema_id"], TARGET_REGISTRY_ID)
        self.assertTrue(all(first["planes"].values()))
        self.assertTrue(all(first["laterality"].values()))
        for key in ("dataset_version_id", "source_identity", "fold_plan_id", "preprocessing_id",
                    "target_schema_id", "selection_ids", "label_identity", "leakage_report_id"):
            self.assertEqual(first[key], second[key])
        self.assertEqual(first["cache"]["cold"], "cold-build")
        self.assertEqual(first["cache"]["warm"], "warm-load")
        self.assertEqual(first["leakage_counts"], {
            "patient_leakage": 0, "study_leakage": 0, "series_leakage": 0,
            "slice_leakage": 0, "status": "PASS",
        })
        self.assertEqual(first["leakage_report_id"], second["leakage_report_id"])
        with tempfile.TemporaryDirectory() as temp:
            artifact = self.run_smoke(temp)
            ready_path = Path(temp) / "prepared" / "prepared-dataset.json"
            persisted = json.loads(ready_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "READY")
            self.assertEqual(persisted["dataset_index_id"], artifact["source_identity"])
            self.assertEqual(persisted["target_registry_id"], artifact["target_schema_id"])
            self.assertEqual(persisted["fold_plan_id"], artifact["fold_plan_id"])
            self.assertEqual(persisted["seed"], 42)
            self.assertFalse(persisted["model_trained"])
            self.assertFalse(persisted["production_ready"])

    def test_different_seed_changes_raw_and_label_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            baseline = self.run_smoke(str(Path(temp) / "seed-42"), seed=42)
            changed = self.run_smoke(str(Path(temp) / "seed-43"), seed=43)
        self.assertNotEqual(baseline["source_identity"], changed["source_identity"])
        self.assertNotEqual(baseline["dataset_version_id"], changed["dataset_version_id"])
        self.assertNotEqual(baseline["label_identity"], changed["label_identity"])
        self.assertNotEqual(baseline["fold_plan_id"], changed["fold_plan_id"])
        self.assertNotEqual(baseline["selection_ids"], changed["selection_ids"])
        self.assertEqual(baseline["target_schema_id"], changed["target_schema_id"])

    def test_failure_injections_are_detected(self) -> None:
        expected_detection = {
            "patient-leakage": "blocked by LeakageGuard",
            "duplicate-sop": "duplicate SOPInstanceUID detected",
            "orientation-conflict": "orientation and laterality conflicts detected",
            "missing-position": "physical-span fallback warning recorded",
            "missing-metadata": "missing PixelSpacing metadata detected",
            "spacing-irregular": "irregular physical slice spacing detected",
            "corrupted-cache": "cache validation rejected corruption",
        }
        for injection in ("patient-leakage", "duplicate-sop", "orientation-conflict",
                          "missing-position", "missing-metadata", "spacing-irregular", "corrupted-cache"):
            with self.subTest(injection=injection), tempfile.TemporaryDirectory() as temp:
                result = self.run_smoke(temp, injection=injection)
                self.assertNotEqual(result["status"], "READY")
                self.assertIn(expected_detection[injection], result["detected"])
                self.assertFalse((Path(temp) / "prepared" / "prepared-dataset.json").exists())
                if injection == "missing-position":
                    self.assertEqual(result["status"], "EXPECTED_FAILURE")
                    self.assertIn("fallback", result["detected"])
                    self.assertGreater(result["fallback_series"], 0)
                else:
                    self.assertEqual(result["status"], "EXPECTED_FAILURE")
                    self.assertEqual(result["injection"], injection)
                    if injection == "patient-leakage":
                        self.assertIn("leakage_report_id", result)
                        self.assertIn("PATIENT_CROSS_FOLD", result["issue_types"])
                    elif injection == "missing-metadata":
                        self.assertGreater(result["missing_fields"], 0)
                    elif injection == "spacing-irregular":
                        self.assertGreater(result["warning_count"], 0)

    def test_patient_leakage_cannot_emit_ready_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_smoke(temp, injection="patient-leakage")
            self.assertEqual(result["status"], "EXPECTED_FAILURE")
            self.assertNotEqual(result["status"], "READY")
            self.assertIn("PATIENT_CROSS_FOLD", result["issue_types"])
            self.assertFalse((Path(temp) / "prepared" / "prepared-dataset.json").exists())


if __name__ == "__main__":
    unittest.main()
