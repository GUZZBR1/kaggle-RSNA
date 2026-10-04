import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rsna.cli import main
from rsna.contracts import FoldPlan
from rsna.folds import digest, load_dataset, make_plan, read_plan, validate


class FoldCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset.json"
        studies = []
        for i in range(20):
            studies.append({"study_instance_uid": f"study-{i}", "patient_id": f"patient-{i // 2}",
                            "series": [{"slices": [{}, {}]}], "labels": {"target": i % 2}})
        self.dataset.write_text(json.dumps({"manifest_schema_version": 1, "studies": studies}), encoding="utf-8")
        self.studies = studies
        self.output = self.root / "folds.json"

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, *args):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["folds", *map(str, args)])
        return code, stdout.getvalue()

    def generate(self, *extra):
        return self.invoke("generate", "--dataset-manifest", self.dataset, "--output", self.output,
                           "--n-folds", "5", *extra)

    def test_generate_validate_json_inspect_and_patient_lookup(self):
        code, output = self.generate("--json")
        self.assertEqual(0, code)
        report = json.loads(output)
        self.assertTrue(report["passed"])
        self.assertEqual(1.0, report["coverage"]["coverage_fraction"])
        self.assertEqual(20, report["coverage"]["assigned_studies"])
        plan = read_plan(self.output)
        self.assertEqual(5, plan["n_folds"])
        self.assertEqual([0.5, 0.5], report["balance"]["target_prevalence"]["target"]["prevalence_range"])
        code, output = self.invoke("validate", "--fold-plan", self.output,
                                   "--dataset-manifest", self.dataset, "--json")
        self.assertEqual(0, code)
        self.assertTrue(json.loads(output)["passed"])
        code, output = self.invoke("inspect", "--fold-plan", self.output,
                                   "--study-id", "study-0", "--dataset-manifest", self.dataset, "--json")
        self.assertEqual(plan["assignments"]["study-0"], json.loads(output)["study"]["fold"])
        code, output = self.invoke("inspect", "--fold-plan", self.output,
                                   "--patient-id", "patient-0", "--dataset-manifest", self.dataset, "--json")
        self.assertEqual([plan["assignments"]["study-0"]], json.loads(output)["patient"]["folds"])
        self.assertTrue(json.loads(output)["patient"]["folds"])
        code, output = self.invoke("inspect", "--fold-plan", self.output, "--fold",
                                   plan["assignments"]["study-0"], "--limit", "2", "--json")
        self.assertEqual(0, code)
        self.assertLessEqual(len(json.loads(output)["studies"]), 2)
        self.assertEqual(0, self.invoke("stats", "--fold-plan", self.output,
                                        "--dataset-manifest", self.dataset, "--json")[0])

    def test_dry_run_and_overwrite_protection(self):
        code, output = self.generate("--dry-run", "--json")
        self.assertEqual(0, code)
        self.assertFalse(self.output.exists())
        self.generate()
        code, _ = self.generate("--json")
        self.assertEqual(2, code)
        code, output = self.generate("--force", "--json")
        self.assertEqual(0, code)
        self.assertTrue(json.loads(output)["overwrote"])
        self.invoke("lock", "--fold-plan", self.output)
        code, _ = self.generate("--force")
        self.assertEqual(2, code)

    def test_dataset_mismatch_missing_unknown_and_patient_leakage(self):
        _, _ = self.generate()
        original = read_plan(self.output)
        code, _ = self.invoke("validate", "--fold-plan", self.output,
                              "--dataset-manifest", self._other_dataset(), "--json")
        self.assertEqual(4, code)
        broken = dict(original)
        broken["assignments"] = dict(original["assignments"])
        broken["assignments"].pop("study-19")
        broken = self.reidentify(broken)
        report = validate(broken, studies=self._normalized_studies())
        self.assertFalse(report["passed"])
        self.assertEqual(["study-19"], report["coverage"]["unassigned_studies"])
        broken["assignments"]["not-in-dataset"] = "fold_0"
        broken = self.reidentify(broken)
        self.assertIn("assignments contain unknown studies", validate(broken, studies=self._normalized_studies())["errors"])
        broken["assignments"].pop("not-in-dataset")
        broken["assignments"]["study-1"] = "fold_1"
        broken = self.reidentify(broken)
        report = validate(broken, studies=self._normalized_studies())
        self.assertTrue(report["leakage"]["patient_leakage_groups"])

    def test_corrupt_plan_and_duplicate_import_assignments(self):
        self.generate()
        self.output.write_text(self.output.read_text().replace("study-0", "study-X"), encoding="utf-8")
        with self.assertRaises(ValueError):
            read_plan(self.output)
        assignments = self.root / "external.csv"
        assignments.write_text("study_id,fold\nstudy-0,fold_0\nstudy-0,fold_1\n", encoding="utf-8")
        code, _ = self.invoke("import", "--assignments", assignments, "--dataset-manifest", self.dataset,
                              "--output", self.root / "import.json")
        self.assertEqual(3, code)

    def test_index_manifest_identity_and_nested_record_corruption(self):
        metadata = {"PatientID": "p-1"}
        slice_record = {"relative_path": "p-1/s-1/image.dcm", "file_size": 12,
                        "metadata": metadata, "warnings": []}
        slice_record["slice_id"] = digest({key: slice_record[key] for key in
                                           ("relative_path", "file_size", "metadata", "warnings")})
        series = {"series_instance_uid": "series-1", "study_instance_uid": "study-1",
                  "slices": [slice_record], "warnings": []}
        series["series_id"] = digest({"series_instance_uid": series["series_instance_uid"],
                                      "study_instance_uid": series["study_instance_uid"],
                                      "slice_ids": [slice_record["slice_id"]], "warnings": []})
        study = {"study_instance_uid": "study-1", "patient_id": "p-1",
                 "series": [series], "warnings": []}
        study["study_id"] = digest({"study_instance_uid": "study-1", "patient_id": "p-1",
                                    "series_ids": [series["series_id"]], "warnings": []})
        manifest = {"manifest_schema_version": 1, "root_identity": "a" * 64,
                    "discovery_version": "1", "studies": [study], "warnings": [],
                    "statistics": {"n_studies": 1, "n_series": 1, "n_slices": 1},
                    "metadata_files": []}
        manifest["index_id"] = digest({"root_identity": manifest["root_identity"],
                                       "discovery_version": manifest["discovery_version"],
                                       "study_ids": [study["study_id"]], "warnings": [],
                                       "statistics": manifest["statistics"], "metadata_files": []})
        path = self.root / "index.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        _, studies = load_dataset(path)
        self.assertEqual("study-1", studies[0]["study_id"])
        manifest["studies"][0]["series"][0]["slices"][0]["metadata"]["PatientID"] = "changed"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "slice_id"):
            load_dataset(path)

    def test_module_entrypoint_emits_parseable_json_and_reproduce_passes(self):
        command = [sys.executable, "-m", "rsna", "folds", "generate",
                   "--dataset-manifest", str(self.dataset), "--output", str(self.output),
                   "--n-folds", "5", "--json"]
        generated = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(0, generated.returncode, generated.stderr)
        self.assertTrue(json.loads(generated.stdout)["passed"])
        validated = subprocess.run([sys.executable, "-m", "rsna", "folds", "validate",
                                    "--fold-plan", str(self.output), "--dataset-manifest",
                                    str(self.dataset), "--reproduce", "--json"],
                                   capture_output=True, text=True, check=False)
        self.assertEqual(0, validated.returncode, validated.stderr)
        self.assertTrue(json.loads(validated.stdout)["reproduction"]["passed"])

    def test_diff_exact_moved_and_semantic_permutation(self):
        _, _ = self.generate()
        left = read_plan(self.output)
        right = dict(left)
        right["assignments"] = dict(left["assignments"])
        right["assignments"]["study-0"], right["assignments"]["study-2"] = right["assignments"]["study-2"], right["assignments"]["study-0"]
        right = self.reidentify(right)
        other = self.root / "other.json"
        other.write_text(json.dumps(right), encoding="utf-8")
        code, output = self.invoke("diff", self.output, self.output, "--json")
        self.assertEqual(0, code)
        self.assertEqual(0, json.loads(output)["studies_moved"])
        code, output = self.invoke("diff", self.output, other, "--show-moved", "--json")
        self.assertGreater(json.loads(output)["studies_moved"], 0)
        permuted = dict(left)
        remap = {f"fold_{i}": f"fold_{(i + 1) % 5}" for i in range(5)}
        permuted["assignments"] = {k: remap[v] for k, v in left["assignments"].items()}
        permuted = self.reidentify(permuted)
        other.write_text(json.dumps(permuted), encoding="utf-8")
        _, output = self.invoke("diff", self.output, other, "--json")
        report = json.loads(output)
        self.assertFalse(report["exact_equality"])
        self.assertTrue(report["semantic_partition_equality"])

    def test_exports_and_reproduce_determinism(self):
        self.generate()
        first = read_plan(self.output)
        again = make_plan(first["dataset_version_id"], self._normalized_studies(), 5, "group", 42, "patient_id")
        self.assertEqual(first["fold_plan_id"], again["fold_plan_id"])
        csv_path, json_path = self.root / "assignments.csv", self.root / "assignments.json"
        self.assertEqual(0, self.invoke("export", "--fold-plan", self.output, "--format", "csv", "--output", csv_path, "--dataset-manifest", self.dataset)[0])
        with csv_path.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(20, len(rows)); self.assertIn("patient_id", rows[0])
        self.assertEqual(0, self.invoke("export", "--fold-plan", self.output, "--format", "json", "--output", json_path)[0])
        self.assertEqual(20, len(json.loads(json_path.read_text())))
        imported = self.root / "imported.json"
        self.assertEqual(0, self.invoke("import", "--assignments", csv_path,
                                        "--dataset-manifest", self.dataset,
                                        "--output", imported, "--json")[0])
        self.assertEqual(first["assignments"], read_plan(imported)["assignments"])
        self.assertEqual(0, self.invoke("validate", "--fold-plan", self.output,
                                        "--dataset-manifest", self.dataset, "--reproduce")[0])

    def _normalized_studies(self):
        return [{"study_id": s["study_instance_uid"], "patient_id": s["patient_id"], "n_series": 1,
                 "n_slices": 2, "labels": s["labels"]} for s in self.studies]

    def _other_dataset(self):
        other = self.root / "other-dataset.json"
        other.write_text(json.dumps({"studies": [{"study_instance_uid": "other", "patient_id": "other"}]}), encoding="utf-8")
        return other

    @staticmethod
    def reidentify(plan):
        plan["configuration"]["statistics_sha256"] = digest(plan["statistics"])
        plan["assignment_manifest_sha256"] = digest(plan["assignments"])
        core = FoldPlan(plan["dataset_version_id"], plan["strategy"], tuple(plan["fold_ids"]),
                        plan["random_state"], configuration=plan.get("configuration", {}),
                        assignment_manifest_sha256=digest(plan["assignments"]))
        plan["fold_plan_id"] = core.fold_plan_id
        return plan


if __name__ == "__main__":
    unittest.main()
