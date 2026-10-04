from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from rsna.artifacts import JsonArtifactStore
from rsna.contracts import (ArtifactReference, DatasetVersion, ExperimentSpec,
                            ModelCandidate, PredictionArtifact)
from rsna.evaluation.oof import binary_roc_auc, evaluate_oof
from rsna.folds.manifest import save_fold_plan
from rsna.folds.models import FoldPlanManifest
from rsna.identity import digest
from rsna.labels import LabelRecord
from rsna.targets import TARGETS


class OOFEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input_store = JsonArtifactStore(self.root / "input")
        self.output_store = JsonArtifactStore(self.root / "output")
        self.dataset = DatasetVersion("synthetic", "v1", "a" * 64, "prep-v1", TARGETS,
                                      synthetic=True)
        self.candidates = [
            ModelCandidate("perfect", {"fixture": "perfect"}),
            ModelCandidate("inverse", {"fixture": "inverse"}),
        ]
        assignments = {f"study-{i:02}": f"fold_{i % 5}" for i in range(10)}
        study_groups = {study: f"group-{study}" for study in assignments}
        self.plan = FoldPlanManifest(
            dataset_version_id=self.dataset.dataset_version_id,
            generator_version="test-v1", input_fingerprint=digest("fold-input"),
            provenance={"source": "unit-test"}, strategy="imported", n_folds=5,
            random_state=7, grouping_key="study_id", assignments=assignments,
            group_assignments={group: assignments[study] for study, group in study_groups.items()},
            study_groups=study_groups,
            study_fingerprints={study: digest(study) for study in assignments},
        )
        self.spec = ExperimentSpec(
            name="oof-test", dataset_version_id=self.dataset.dataset_version_id,
            fold_plan_id=self.plan.fold_plan_id,
            model_candidate_ids=tuple(item.model_candidate_id for item in self.candidates),
            fold_ids=tuple(f"fold_{index}" for index in range(5)),
            class_names=TARGETS, synthetic=True,
        )
        self.labels = []
        for index in range(10):
            values = dict.fromkeys(TARGETS)
            values[TARGETS[0]] = index % 2
            if index < 8:
                values[TARGETS[1]] = index % 2
            values[TARGETS[2]] = 1
            self.labels.append(LabelRecord(
                f"study-{index:02}", values, "official_gold", allow_partial=True,
                patient_id=f"patient-{index:02}",
            ))
        self.predictions = self._predictions()

    def tearDown(self):
        self.temp.cleanup()

    def _predictions(self, *, payload_transform=None, metadata_transform=None):
        result = []
        for candidate_index, candidate in enumerate(self.candidates):
            for fold_index in range(5):
                fold_id = f"fold_{fold_index}"
                studies = [study for study, assigned in self.plan.assignments.items()
                           if assigned == fold_id]
                rows = []
                for study in studies:
                    label_index = int(study.split("-")[1])
                    positive = label_index % 2
                    acl_score = float(positive if candidate_index == 0 else 1 - positive)
                    rows.append({"StudyInstanceUID": study,
                                 "scores": [acl_score, acl_score, 0.5, *([0.5] * 9)]})
                payload = {"schema_version": 1, "class_names": list(TARGETS), "rows": rows}
                if payload_transform:
                    payload = payload_transform(candidate_index, fold_index, payload)
                reference = self.input_store.put_json(payload)
                options = {"kind": "oof", "model_candidate_id": candidate.model_candidate_id,
                           "dataset_version_id": self.dataset.dataset_version_id,
                           "checkpoint_id": digest(f"checkpoint-{candidate_index}-{fold_index}"),
                           "class_names": TARGETS, "row_count": len(payload["rows"]),
                           "fold_plan_id": self.plan.fold_plan_id, "fold_id": fold_id,
                           "synthetic": True, "provenance": "external_fixture"}
                if metadata_transform:
                    options = metadata_transform(candidate_index, fold_index, options)
                result.append(PredictionArtifact(reference, **options))
        return result

    def _evaluate(self, predictions=None, labels=None):
        return evaluate_oof(self.spec, self.plan, labels or self.labels,
                            predictions or self.predictions, self.output_store)

    def test_five_fold_oof_scores_mask_labels_and_write_expected_artifacts(self):
        report = self._evaluate()
        evaluations = {item["model_candidate_id"]: item for item in report["evaluations"]}
        perfect_id, inverse_id = [candidate.model_candidate_id for candidate in self.candidates]
        self.assertEqual(1.0, evaluations[perfect_id]["auc_by_class"][TARGETS[0]])
        self.assertEqual(0.0, evaluations[inverse_id]["auc_by_class"][TARGETS[0]])
        self.assertEqual(1.0, evaluations[perfect_id]["auc_by_class"][TARGETS[1]])
        self.assertIsNone(evaluations[perfect_id]["auc_by_class"][TARGETS[2]])
        self.assertIsNone(evaluations[perfect_id]["auc_by_class"][TARGETS[3]])
        self.assertIsNone(evaluations[perfect_id]["macro_auc"])
        self.assertEqual({"oof_aggregate", "target_auc_table", "comparison_report", "evaluations"},
                         set(report["artifacts"]))
        table = JsonArtifactStore.load_json(ArtifactReference.from_dict(
            report["artifacts"]["target_auc_table"]))
        mcl_row = next(row for row in table["rows"]
                       if row["model_candidate_id"] == perfect_id and row["target"] == TARGETS[1])
        self.assertEqual((8, 4, 4), (mcl_row["observed_count"], mcl_row["positive_count"],
                                    mcl_row["negative_count"]))
        comparison = JsonArtifactStore.load_json(ArtifactReference.from_dict(
            report["artifacts"]["comparison_report"]))
        self.assertEqual(1, len(comparison["pairwise_comparisons"]))
        self.assertIsNone(comparison["pairwise_comparisons"][0]["macro_auc_delta"])
        summary = comparison["candidate_metrics"][perfect_id]
        self.assertEqual("NO_NEGATIVES", summary["auc_by_target"][TARGETS[2]]["unavailable_reason"])
        self.assertEqual("NO_SUPERVISION", summary["auc_by_target"][TARGETS[3]]["unavailable_reason"])
        self.assertEqual(8, summary["auc_by_target"][TARGETS[1]]["supervised_count"])
        self.assertGreater(report["performance"]["rows_per_second"], 0)
        self.assertGreater(report["performance"]["peak_traced_bytes"], 0)

    def test_rank_auc_is_tie_aware_and_returns_none_without_both_classes(self):
        self.assertEqual(0.5, binary_roc_auc([0, 1], [0.2, 0.2]))
        self.assertEqual(1.0, binary_roc_auc([0, 1], [0.1, 0.9]))
        self.assertEqual(0.0, binary_roc_auc([0, 1], [0.9, 0.1]))
        self.assertEqual(0.75, binary_roc_auc([1, 0, 1, 0], [0.9, 0.8, 0.4, 0.1]))
        self.assertIsNone(binary_roc_auc([1, 1], [0.1, 0.2]))
        self.assertIsNone(binary_roc_auc([], []))

    def test_artifacts_are_deterministic_across_reruns(self):
        first = self._evaluate()
        second = self._evaluate()
        self.assertEqual(first["input_identity"], second["input_identity"])
        self.assertEqual(first["evaluations"], second["evaluations"])
        for name in ("oof_aggregate", "target_auc_table", "comparison_report"):
            self.assertEqual(first["artifacts"][name]["artifact_id"],
                             second["artifacts"][name]["artifact_id"])
        shuffled = self._evaluate(list(reversed(self.predictions)))
        self.assertEqual(first["evaluations"], shuffled["evaluations"])
        self.assertEqual(first["input_identity"], shuffled["input_identity"])
        for name in ("oof_aggregate", "target_auc_table", "comparison_report"):
            self.assertEqual(first["artifacts"][name]["artifact_id"],
                             shuffled["artifacts"][name]["artifact_id"])

    def test_prediction_row_order_does_not_change_auc_or_aggregate_order(self):
        shuffled = self._predictions(payload_transform=lambda _c, _f, payload:
                                     {**payload, "rows": list(reversed(payload["rows"]))})
        first = self._evaluate()
        second = self._evaluate(shuffled)
        self.assertEqual(first["evaluations"], second["evaluations"])
        first_rows = JsonArtifactStore.load_json(ArtifactReference.from_dict(
            first["artifacts"]["oof_aggregate"]))["rows"]
        second_rows = JsonArtifactStore.load_json(ArtifactReference.from_dict(
            second["artifacts"]["oof_aggregate"]))["rows"]
        self.assertEqual(first_rows, second_rows)

    def test_missing_coverage_and_duplicate_rows_fail(self):
        missing = self._predictions(payload_transform=lambda _c, f, payload:
                                    {**payload, "rows": payload["rows"][:-1]}
                                    if f == 0 else payload)
        with self.assertRaisesRegex(ValueError, "OOF study coverage mismatch"):
            self._evaluate(missing)

        duplicate = self._predictions(payload_transform=lambda _c, f, payload:
                                      {**payload, "rows": [*payload["rows"], payload["rows"][0]]}
                                      if f == 0 else payload)
        with self.assertRaisesRegex(ValueError, "duplicate study"):
            self._evaluate(duplicate)

    def test_foreign_study_and_cross_fold_study_fail(self):
        foreign = self._predictions(payload_transform=lambda _c, f, payload:
                                    {**payload, "rows": [
                                        {**payload["rows"][0], "StudyInstanceUID": "foreign-study"},
                                        *payload["rows"][1:]
                                    ]} if f == 0 else payload)
        with self.assertRaisesRegex(ValueError, "foreign prediction study"):
            self._evaluate(foreign)

        cross_fold = self._predictions(payload_transform=lambda _c, f, payload:
                                       {**payload, "rows": [
                                           {**payload["rows"][0], "StudyInstanceUID": "study-01"},
                                           *payload["rows"][1:]
                                       ]} if f == 0 else payload)
        with self.assertRaisesRegex(ValueError, "assigned to fold_1, not artifact fold fold_0"):
            self._evaluate(cross_fold)

    def test_duplicate_rows_invalid_scores_and_content_tampering_fail(self):
        duplicate = self._predictions(payload_transform=lambda _c, f, payload:
                                      {**payload, "rows": [payload["rows"][0], payload["rows"][0]]}
                                      if f == 0 else payload)
        invalid_score = self._predictions(payload_transform=lambda _c, f, payload:
                                          {**payload, "rows": [
                                              {**payload["rows"][0], "scores": [2.0, *payload["rows"][0]["scores"][1:]]},
                                              *payload["rows"][1:]
                                          ]} if f == 0 else payload)
        with self.assertRaisesRegex(ValueError, "finite probability"):
            self._evaluate(invalid_score)

        artifact_path = Path(self.predictions[0].artifact.uri)
        artifact_path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            self._evaluate()

    def test_candidate_fold_and_target_order_alignment_are_required(self):
        wrong_fold = self._predictions(metadata_transform=lambda _c, f, options:
                                       {**options, "fold_id": "fold_4"}
                                       if f == 0 else options)
        with self.assertRaisesRegex(ValueError, "multiple prediction artifacts"):
            self._evaluate(wrong_fold)

        reversed_targets = self._predictions(metadata_transform=lambda _c, f, options:
                                              {**options, "class_names": tuple(reversed(TARGETS))}
                                              if f == 0 else options)
        with self.assertRaisesRegex(ValueError, "target order"):
            self._evaluate(reversed_targets)

        inference = self._predictions(metadata_transform=lambda _c, f, options:
                                      {**options, "kind": "inference", "fold_plan_id": None,
                                       "fold_id": None} if f == 0 else options)
        with self.assertRaisesRegex(ValueError, "inference PredictionArtifact"):
            self._evaluate(inference)

    def test_labels_must_cover_plan_and_be_hard_binary(self):
        with self.assertRaisesRegex(ValueError, "label study coverage mismatch"):
            self._evaluate(labels=self.labels[:-1])
        soft_values = dict.fromkeys(TARGETS, None)
        soft_values[TARGETS[0]] = 0.5
        soft = LabelRecord("study-00", soft_values, "manual_review", label_type="soft",
                           allow_soft=True, allow_partial=True)
        with self.assertRaisesRegex(ValueError, "not supported for ROC AUC"):
            self._evaluate(labels=[soft, *self.labels[1:]])
        soft_values[TARGETS[0]] = 1.0
        endpoint_soft = LabelRecord("study-00", soft_values, "manual_review", label_type="soft",
                                    allow_soft=True, allow_partial=True)
        with self.assertRaisesRegex(ValueError, "not supported for ROC AUC"):
            self._evaluate(labels=[endpoint_soft, *self.labels[1:]])

    def test_cli_job_loads_existing_contract_serializations(self):
        from rsna.cli import main

        plan_path = self.root / "fold-plan.json"
        save_fold_plan(self.plan, plan_path)
        labels_path = self.root / "labels.json"
        labels_path.write_text(json.dumps([label.to_dict() for label in self.labels]), encoding="utf-8")
        job_path = self.root / "oof-job.json"
        job_path.write_text(json.dumps({
            "experiment": self.spec.to_dict(),
            "fold_plan": str(plan_path),
            "labels": str(labels_path),
            "prediction_artifacts": [prediction.to_dict() for prediction in self.predictions],
        }), encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["evaluate", "oof", "--job", str(job_path),
                           "--output-dir", str(self.root / "cli-output"), "--json"])
        self.assertEqual(0, result)
        self.assertEqual("ready", json.loads(output.getvalue())["status"])


if __name__ == "__main__":
    unittest.main()
