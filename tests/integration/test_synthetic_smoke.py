"""End-to-end synthetic pipeline checks; these never use Kaggle or GPU data."""

from __future__ import annotations

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
                    "target_schema_id", "selection_ids"):
            self.assertEqual(first[key], second[key])
        self.assertEqual(first["cache"]["cold"], "cold-build")
        self.assertEqual(first["cache"]["warm"], "warm-load")
        self.assertEqual(first["leakage_counts"], {
            "patient_leakage": 0, "study_leakage": 0, "series_leakage": 0,
            "slice_leakage": 0, "status": "PASS",
        })
        self.assertEqual(first["leakage_report_id"], second["leakage_report_id"])

    def test_different_seed_changes_raw_and_label_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            baseline = self.run_smoke(str(Path(temp) / "seed-42"), seed=42)
            changed = self.run_smoke(str(Path(temp) / "seed-43"), seed=43)
        self.assertNotEqual(baseline["source_identity"], changed["source_identity"])
        self.assertNotEqual(baseline["dataset_version_id"], changed["dataset_version_id"])
        self.assertNotEqual(baseline["label_identity"], changed["label_identity"])
        self.assertEqual(baseline["target_schema_id"], changed["target_schema_id"])

    def test_failure_injections_are_detected(self) -> None:
        for injection in ("patient-leakage", "duplicate-sop", "orientation-conflict",
                          "missing-position", "missing-metadata", "spacing-irregular", "corrupted-cache"):
            with self.subTest(injection=injection), tempfile.TemporaryDirectory() as temp:
                result = self.run_smoke(temp, injection=injection)
                if injection == "missing-position":
                    self.assertEqual(result["status"], "EXPECTED_FAILURE")
                    self.assertIn("fallback", result["detected"])
                else:
                    self.assertEqual(result["status"], "EXPECTED_FAILURE")
                    self.assertEqual(result["injection"], injection)
                    if injection == "patient-leakage":
                        self.assertIn("leakage_report_id", result)
                        self.assertIn("PATIENT_CROSS_FOLD", result["issue_types"])


if __name__ == "__main__":
    unittest.main()
