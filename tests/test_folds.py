import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

from rsna.cli import main
from rsna.folds import generate_fold_plan, load_fold_plan, save_fold_plan
from rsna.folds.validate import validate_leakage
from rsna.data.models import DatasetIndex, StudyRecord
from rsna.labels import LabelRecord
from rsna.targets import TARGETS, TARGET_REGISTRY_ID


def studies(n=100):
    return [{"study_id": f"study_{i:03d}", "patient_id": f"patient_{i:03d}"}
            for i in range(n)]


class FoldPlanTests(unittest.TestCase):
    def test_hundred_studies_make_five_nonempty_folds(self):
        plan = generate_fold_plan(studies())
        self.assertEqual(5, len(plan.statistics["folds"]))
        self.assertEqual(100, plan.statistics["total"]["n_studies"])
        self.assertIsNone(plan.statistics["total"]["n_series"])
        self.assertTrue(all(stats["n_studies"] == 20 for stats in plan.statistics["folds"].values()))

    def test_input_order_does_not_change_assignments_or_identity(self):
        rows = studies()
        first = generate_fold_plan(rows)
        second = generate_fold_plan(list(reversed(rows)))
        self.assertEqual(first.assignments, second.assignments)
        self.assertEqual(first.fold_plan_id, second.fold_plan_id)

    def test_dataset_index_records_are_accepted_directly(self):
        dataset = DatasetIndex("root", "1", tuple(
            StudyRecord(f"study_{i}", f"patient_{i}", ()) for i in range(5)), (), {"n_studies": 5})
        plan = generate_fold_plan(dataset, n_folds=5)
        self.assertEqual(5, plan.statistics["total"]["n_studies"])
        self.assertEqual(5, len(plan.assignments))

    def test_canonical_study_labels_join_by_uid_not_content_hash(self):
        dataset = DatasetIndex("root", "1", tuple(
            StudyRecord(f"uid-{i}", f"patient-{i}", ()) for i in range(4)), (), {"n_studies": 4})
        labels = [LabelRecord(f"uid-{i}", dict.fromkeys(TARGETS, i % 2), "manual_review")
                  for i in range(4)]
        plan = generate_fold_plan(dataset, n_folds=2, labels=labels)
        self.assertEqual({f"uid-{i}" for i in range(4)}, set(plan.assignments))
        hash_keyed = [LabelRecord(study.study_id, dict.fromkeys(TARGETS, 0), "manual_review")
                      for study in dataset.studies]
        with self.assertRaisesRegex(ValueError, "unknown study"):
            generate_fold_plan(dataset, n_folds=2, labels=hash_keyed)
        with self.assertRaisesRegex(ValueError, "content study hash"):
            generate_fold_plan([{"study_id": "a" * 64}], n_folds=2)
        with self.assertRaisesRegex(ValueError, "duplicate label records"):
            generate_fold_plan(dataset, n_folds=2, labels=[labels[0], labels[0]])

    def test_label_content_is_identity_bearing(self):
        rows = studies(10)
        first_labels = {row["study_id"]: {"ACL": 0} for row in rows}
        second_labels = dict(first_labels)
        second_labels["study_000"] = {"ACL": 1}
        first = generate_fold_plan(rows, n_folds=2, labels=first_labels)
        second = generate_fold_plan(rows, n_folds=2, labels=second_labels)
        self.assertNotEqual(first.fold_plan_id, second.fold_plan_id)

    def test_same_patient_studies_stay_together(self):
        rows = [{"study_id": f"s{p}_{i}", "patient_id": f"p{p}"}
                for p in range(10) for i in range(3)]
        plan = generate_fold_plan(rows)
        for patient in range(10):
            assigned = {plan.fold_for_study(f"s{patient}_{i}") for i in range(3)}
            self.assertEqual(1, len(assigned))
        self.assertEqual(10, sum(v["n_patients"] for v in plan.statistics["folds"].values()))

    def test_missing_patient_id_falls_back_to_study_grouping(self):
        plan = generate_fold_plan([{"study_id": f"s{i}"} for i in range(5)], n_folds=2)
        self.assertEqual("study_id", plan.grouping_key)
        self.assertTrue(any("lack PatientID" in warning for warning in plan.warnings))

    def test_patient_conflict_is_rejected(self):
        row = {"study_instance_uid": "s1", "warnings": ["conflicting PatientID values in study s1"]}
        with self.assertRaisesRegex(ValueError, "inconsistent PatientID"):
            generate_fold_plan([row, {"study_id": "s2"}], n_folds=2)

    def test_multiple_series_in_study_stay_together(self):
        plan = generate_fold_plan([{"study_id": f"s{i}", "series": [f"s{i}_a", f"s{i}_b"]} for i in range(8)], n_folds=4)
        self.assertEqual(16, plan.statistics["total"]["n_series"])

    def test_reused_series_across_studies_links_groups(self):
        rows = [{"study_id": "s1", "patient_id": "p1", "series": ["shared"]},
                {"study_id": "s2", "patient_id": "p2", "series": ["shared"]},
                *[{"study_id": f"s{i}", "patient_id": f"p{i}"} for i in range(3, 7)]]
        plan = generate_fold_plan(rows, n_folds=2)
        self.assertEqual(plan.fold_for_study("s1"), plan.fold_for_study("s2"))

    def test_duplicate_study_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate study"):
            generate_fold_plan([{"study_id": "s1"}, {"study_id": "s1"}], n_folds=2)

    def test_group_count_bounds(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed number of studies"):
            generate_fold_plan(studies(4), n_folds=5)
        with self.assertRaisesRegex(ValueError, "indivisible groups"):
            generate_fold_plan([{"study_id": f"s{i}", "patient_id": "same"} for i in range(5)], n_folds=2)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            generate_fold_plan(studies(5), n_folds=1)

    def test_fold_counts_supported_without_code_changes(self):
        for count in (2, 3, 4, 5, 10):
            plan = generate_fold_plan(studies(20), n_folds=count)
            self.assertEqual(count, len(plan.statistics["folds"]))

    def test_seed_controls_identity_and_assignments(self):
        first = generate_fold_plan(studies(100), random_state=42)
        repeat = generate_fold_plan(studies(100), random_state=42)
        other = generate_fold_plan(studies(100), random_state=7)
        self.assertEqual(first.fold_plan_id, repeat.fold_plan_id)
        self.assertNotEqual(first.fold_plan_id, other.fold_plan_id)
        self.assertNotEqual(first.assignments, other.assignments)

    def test_dataset_version_is_identity_bearing(self):
        first = generate_fold_plan(studies(10), n_folds=2, dataset_version_id="a" * 64)
        other = generate_fold_plan(studies(10), n_folds=2, dataset_version_id="b" * 64)
        self.assertNotEqual(first.fold_plan_id, other.fold_plan_id)

    def test_reload_is_exact_and_locked_plan_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.json"
            plan = generate_fold_plan(studies(), locked=True)
            save_fold_plan(plan, path)
            loaded = load_fold_plan(path)
            self.assertEqual(plan.assignments, loaded.assignments)
            self.assertEqual(plan.fold_plan_id, loaded.fold_plan_id)
            with self.assertRaisesRegex(ValueError, "locked"):
                save_fold_plan(generate_fold_plan(studies(), random_state=13, locked=True), path)

    def test_manifest_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.json"
            save_fold_plan(generate_fold_plan(studies(10), n_folds=2), path)
            data = json.loads(path.read_text())
            data["assignments"]["study_000"] = ("fold_1" if data["assignments"]["study_000"] != "fold_1" else "fold_0")
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "does not match|conflicts with its indivisible group"):
                load_fold_plan(path)

    def test_dataset_mismatch_and_new_study_fail_strictly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.json"
            plan = generate_fold_plan(studies(10), n_folds=2)
            save_fold_plan(plan, path)
            with self.assertRaisesRegex(ValueError, "DatasetVersion mismatch"):
                load_fold_plan(path, dataset_version_id="f" * 64)
            with self.assertRaisesRegex(ValueError, "unassigned studies"):
                load_fold_plan(path, dataset=[*studies(10), {"study_id": "new"}])

    def test_changed_patient_or_series_identity_is_rejected_on_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.json"
            rows = studies(10)
            plan = generate_fold_plan(rows, n_folds=2)
            save_fold_plan(plan, path)
            changed = [dict(row) for row in rows]
            changed[0]["patient_id"] = "different-patient"
            with self.assertRaisesRegex(ValueError, "study identity changed"):
                load_fold_plan(path, dataset=changed)

    def test_missing_expected_studies_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.json"
            plan = generate_fold_plan(studies(10), n_folds=2)
            save_fold_plan(plan, path)
            loaded = load_fold_plan(path, dataset=studies(9))
            self.assertTrue(any("study_009" in warning for warning in loaded.application_warnings))

    def test_complete_and_partial_label_statistics_preserve_unknowns(self):
        rows = studies(10)
        labels = {row["study_id"]: {"ACL": i % 2, "Medial Meniscus": None if i < 2 else int(i % 3 == 0)}
                  for i, row in enumerate(rows)}
        plan = generate_fold_plan(rows, n_folds=2, labels=labels)
        total = plan.statistics["total"]["targets"]
        self.assertEqual(5, total["ACL"]["positive"])
        self.assertEqual(5, total["ACL"]["negative"])
        self.assertEqual(2, total["Medial Meniscus"]["missing"])
        self.assertEqual(8, total["Medial Meniscus"]["positive"] + total["Medial Meniscus"]["negative"])

    def test_multilabel_strategy_is_explicit_and_requires_observed_labels(self):
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            generate_fold_plan(studies(10), n_folds=2, strategy="multilabel_group_stratified")
        labels = {row["study_id"]: {"ACL": i % 2} for i, row in enumerate(studies(10))}
        plan = generate_fold_plan(studies(10), n_folds=2,
                                  strategy="multilabel_group_stratified", labels=labels)
        self.assertEqual("multilabel_group_stratified", plan.strategy)
        self.assertEqual(10, plan.statistics["total"]["targets"]["ACL"]["positive"] +
                         plan.statistics["total"]["targets"]["ACL"]["negative"])

    def test_multilabel_stratification_improves_rare_target_balance_without_leakage(self):
        rows, labels = [], {}
        for patient in range(50):
            for visit in range(2):
                study = f"s{patient:02d}-{visit}"
                rows.append({"study_id": study, "patient_id": f"p{patient:02d}"})
                labels[study] = {"ACL": int(patient < 5)}
        grouped = generate_fold_plan(rows, n_folds=5, labels=labels)
        stratified = generate_fold_plan(rows, n_folds=5, strategy="multilabel_group_stratified",
                                        labels=labels)
        group_range = grouped.statistics["diagnostics"]["target_prevalence_range"]["ACL"]
        stratified_range = stratified.statistics["diagnostics"]["target_prevalence_range"]["ACL"]
        self.assertLess(stratified_range, group_range)
        self.assertEqual([20] * 5, [value["n_studies"] for value in stratified.statistics["folds"].values()])
        validate_leakage(rows, stratified.assignments)

    def test_rare_target_generates_warning(self):
        rows = studies(10)
        labels = {row["study_id"]: {"ACL": int(i == 0)} for i, row in enumerate(rows)}
        plan = generate_fold_plan(rows, n_folds=5, labels=labels)
        self.assertTrue(any("rare target ACL" in warning for warning in plan.warnings))

    def test_label_records_preserve_official_order_hard_soft_and_missing_semantics(self):
        rows = studies(10)
        hard = LabelRecord("study_000", {name: (1 if name == "ACL" else None) for name in TARGETS},
                           "manual_review", allow_partial=True)
        soft = LabelRecord("study_001", {name: (0.25 if name == "ACL" else None) for name in TARGETS},
                           "manual_review", label_type="soft", allow_partial=True, allow_soft=True)
        plan = generate_fold_plan(rows, n_folds=2, labels=[hard, soft])
        targets = plan.statistics["total"]["targets"]
        self.assertEqual(TARGETS, tuple(targets))
        self.assertEqual(TARGETS, tuple(plan.configuration["target_order"]))
        self.assertEqual(TARGET_REGISTRY_ID, plan.configuration["target_registry_id"])
        self.assertEqual(1, targets["ACL"]["positive"])
        self.assertEqual(1, targets["ACL"]["soft_count"])
        self.assertEqual(0.25, targets["ACL"]["soft_probability_sum"])
        self.assertEqual(8, targets["ACL"]["missing"])
        self.assertNotEqual(
            generate_fold_plan(rows, n_folds=2, labels=[hard]).fold_plan_id,
            generate_fold_plan(rows, n_folds=2, labels=[soft]).fold_plan_id,
        )

    def test_unknown_targets_and_unapproved_soft_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown target"):
            generate_fold_plan(studies(5), n_folds=2,
                               labels={"study_000": {"not a target": 1}})
        with self.assertRaisesRegex(ValueError, "must be 0, 1, or null"):
            generate_fold_plan(studies(5), n_folds=2,
                               labels={"study_000": {"ACL": 0.25}})

    def test_serialized_label_record_order_and_identity_are_validated(self):
        record = LabelRecord("study_000", {name: (1 if name == "ACL" else None) for name in TARGETS},
                             "manual_review", allow_partial=True).to_dict()
        plan = generate_fold_plan(studies(5), n_folds=2, labels={"study_000": record})
        self.assertEqual(TARGETS, tuple(plan.configuration["target_order"]))
        record["targets"] = list(reversed(record["targets"]))
        with self.assertRaisesRegex(ValueError, "target order"):
            generate_fold_plan(studies(5), n_folds=2, labels={"study_000": record})

    def test_mixed_patient_and_missing_id_uses_study_fallback_warning(self):
        plan = generate_fold_plan([{"study_id": "s1", "patient_id": "p1"},
                                   {"study_id": "s2"}, {"study_id": "s3"}], n_folds=2)
        self.assertEqual("patient_id_with_study_fallback", plan.grouping_key)
        self.assertEqual("study:s2", plan.study_groups["s2"])
        self.assertTrue(any("lack PatientID" in warning for warning in plan.warnings))

    def test_artificial_patient_and_study_leakage_fail(self):
        rows = [{"study_id": "s1", "patient_id": "p1"},
                {"study_id": "s2", "patient_id": "p1"}]
        with self.assertRaisesRegex(ValueError, "patient leakage"):
            validate_leakage(rows, {"s1": "fold_0", "s2": "fold_1"})
        rows = [{"study_id": "s1"}, {"study_id": "s1"}]
        with self.assertRaisesRegex(ValueError, "duplicate study records"):
            validate_leakage(rows, {"s1": "fold_0"})

    def test_cli_creates_and_reloads_five_fold_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "dataset.json", Path(directory) / "folds.json"
            source.write_text(json.dumps({"index_id": "dataset-v1", "studies": studies(20)}))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["folds", "--dataset-manifest", str(source), "--output", str(output)])
            self.assertEqual(0, code)
            self.assertTrue(output.exists())
            self.assertIn("Leakage: PASS", stdout.getvalue())
            self.assertEqual(20, load_fold_plan(output).statistics["total"]["n_studies"])

    def test_python_module_cli_generates_fold_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dataset.json"
            output = Path(directory) / "folds.json"
            source.write_text(json.dumps({"studies": studies(10)}))
            result = subprocess.run([sys.executable, "-m", "rsna", "folds",
                                     "--dataset-manifest", str(source), "--output", str(output),
                                     "--n-folds", "5"], check=True, capture_output=True, text=True)
            self.assertIn("Leakage: PASS", result.stdout)
            self.assertEqual(10, load_fold_plan(output).statistics["total"]["n_studies"])

    def test_fold_statistics_and_schema(self):
        plan = generate_fold_plan(studies(10), n_folds=2)
        self.assertEqual(10, sum(v["n_studies"] for v in plan.statistics["folds"].values()))
        schema = json.loads(Path("schemas/fold-plan-manifest.schema.json").read_text())
        self.assertEqual("FoldPlan Manifest v1", schema["title"])
        self.assertIn("assignments", schema["required"])
        self.assertEqual(set(schema["required"]), set(plan.to_dict()))


if __name__ == "__main__":
    unittest.main()
