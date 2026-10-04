import json
from pathlib import Path
import unittest

from rsna import (ArtifactReference, DatasetVersion, Evaluation, LabelRecord,
                  ModelCandidate, PredictionArtifact, SubmissionArtifact, TARGETS,
                  StudyDatasetRecord, StudyMetadata, TARGET_REGISTRY, TargetRegistry, label_records_to_rows,
                  rows_to_label_records, validate_submission_columns, validate_targets)
from rsna.targets import Target


class TargetAndLabelTests(unittest.TestCase):
    def test_official_registry_count_names_order_and_lookups(self):
        self.assertEqual(12, TARGET_REGISTRY.count)
        self.assertEqual(12, len(set(TARGETS)))
        self.assertEqual("ACL", TARGET_REGISTRY.target_name(0))
        self.assertEqual(0, TARGET_REGISTRY.target_index("ACL"))
        self.assertEqual(TARGETS, validate_targets(TARGETS))
        self.assertEqual(TARGETS, tuple(item["name"] for item in TARGET_REGISTRY.to_dict()["targets"]))

    def test_aliases_and_ambiguous_alias_rejection(self):
        self.assertEqual(0, TARGET_REGISTRY.target_index("Anterior Cruciate Ligament Injury"))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            TargetRegistry((
                Target("A", ("same",)),
                Target("B", ("same",)),
            ))
        with self.assertRaisesRegex(ValueError, "unknown target"):
            TARGET_REGISTRY.target_index("ACL/MCL")

    def test_order_missing_extra_and_duplicates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "order mismatch"):
            validate_targets((*TARGETS[1:], TARGETS[0]))
        with self.assertRaisesRegex(ValueError, "order mismatch"):
            validate_targets(TARGETS[:-1])
        with self.assertRaisesRegex(ValueError, "order mismatch"):
            validate_targets((*TARGETS, "Other"))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_targets((*TARGETS[:-1], TARGETS[0]), allow_synthetic=True)

    def test_hard_soft_missing_and_mask(self):
        labels = dict.fromkeys(TARGETS, 0)
        labels["ACL"] = 1
        labels["MCL"] = None
        record = LabelRecord("study-1", labels, "official_gold", allow_partial=True)
        self.assertEqual(1, record.values["ACL"])
        self.assertIsNone(record.values["MCL"])
        self.assertFalse(record.mask[1])
        self.assertEqual(11, sum(record.mask))
        self.assertEqual("official_gold", record.to_dict()["provenance"])
        self.assertEqual(1, record.to_row()["ACL"])
        with self.assertRaisesRegex(ValueError, "hard label"):
            LabelRecord("study-1", {**labels, "ACL": 2}, "official_gold")
        with self.assertRaisesRegex(ValueError, "soft label"):
            LabelRecord("study-1", {**labels, "ACL": 1.2}, "report_llm", "soft", allow_partial=True, allow_soft=True)
        soft = LabelRecord("study-1", {**labels, "ACL": 0.7}, "report_llm", "soft", allow_partial=True, allow_soft=True)
        self.assertEqual(0.7, soft.values["ACL"])
        with self.assertRaisesRegex(ValueError, "soft labels are disabled"):
            LabelRecord("study-1", labels, "report_llm", "soft")
        with self.assertRaisesRegex(ValueError, "required"):
            LabelRecord("study-1", {"ACL": 1}, "official_gold", allow_partial=False)

    def test_label_alias_duplicates_and_versions(self):
        values = {name: 0 for name in TARGETS}
        values["Anterior Cruciate Ligament"] = values.pop("ACL")
        normalized = LabelRecord("s", values, "official_gold", allow_partial=True)
        self.assertEqual(0, normalized.values["ACL"])
        self.assertNotIn("Anterior Cruciate Ligament", normalized.values)
        duplicate = dict(values)
        duplicate["ACL"] = duplicate.pop("Anterior Cruciate Ligament")
        duplicate["Anterior Cruciate Ligament"] = 0
        with self.assertRaisesRegex(ValueError, "duplicate target"):
            LabelRecord("s", duplicate, "official_gold", allow_partial=True)
        with self.assertRaisesRegex(ValueError, "unsupported target schema"):
            LabelRecord("s", dict.fromkeys(TARGETS, 0), "official_gold", target_schema_version=2)

    def test_registry_alias_collision_rejected(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            TargetRegistry((Target("One", ("same",)), Target("Two", ("SAME",))))

    def test_registry_identity_tracks_alias_definitions(self):
        changed = TargetRegistry((Target("ACL", ("different alias",)), *TARGET_REGISTRY.targets[1:]))
        self.assertNotEqual(TARGET_REGISTRY.registry_id, changed.registry_id)

    def test_dataset_synthetic_gate_and_identity(self):
        args = ("dataset", "v1", "a" * 64, "prep", tuple(f"class_{i:02d}" for i in range(1, 13)))
        with self.assertRaisesRegex(ValueError, "target order mismatch"):
            DatasetVersion(*args)
        data = DatasetVersion(*args, synthetic=True)
        self.assertTrue(data.synthetic)
        official = DatasetVersion("official", "v1", "a" * 64, "prep", TARGETS)
        self.assertNotEqual(data.dataset_version_id, official.dataset_version_id)

    def test_prediction_and_evaluation_require_official_order(self):
        digest = "d" * 64
        ref = ArtifactReference(digest, "application/json", "mock://pred", digest)
        candidate = ModelCandidate("m", {})
        dataset = DatasetVersion("d", "1", "a" * 64, "p", TARGETS)
        pred = PredictionArtifact(ref, "inference", candidate.model_candidate_id,
                                  dataset.dataset_version_id, "e" * 64, TARGETS, 1)
        self.assertEqual(TARGETS, pred.class_names)
        with self.assertRaisesRegex(ValueError, "order mismatch"):
            PredictionArtifact(ref, "inference", candidate.model_candidate_id,
                dataset.dataset_version_id, "e" * 64, (*TARGETS[1:], TARGETS[0]), 1)
        auc = dict.fromkeys(TARGETS, 0.5)
        with self.assertRaisesRegex(ValueError, "exactly every"):
            Evaluation("f" * 64, dataset.dataset_version_id, candidate.model_candidate_id,
                       ("b" * 64,), TARGETS, {name: auc[name] for name in TARGETS[:-1]})
        evaluation = Evaluation("f" * 64, dataset.dataset_version_id, candidate.model_candidate_id,
                                (pred.prediction_artifact_id,), TARGETS, auc)
        self.assertEqual(0.5, evaluation.macro_auc)
        submission = SubmissionArtifact(ref, candidate.model_candidate_id,
            dataset.dataset_version_id, evaluation.evaluation_id,
            (pred.prediction_artifact_id,), "csv")
        self.assertEqual(TARGETS, tuple(submission.to_dict()["class_names"]))

    def test_row_interop_preserves_missing_as_none(self):
        import math
        row = {name: 0 for name in TARGETS}
        row.update(StudyInstanceUID="s", label_provenance="report_regex", label_type="hard")
        row["MCL"] = math.nan
        records = rows_to_label_records([row])
        self.assertIsNone(records[0].values["MCL"])
        self.assertFalse(records[0].mask[1])
        self.assertIsNone(label_records_to_rows(records)[0]["MCL"])

    def test_study_dataset_metadata_is_separate_from_labels(self):
        labels = LabelRecord("s", dict.fromkeys(TARGETS, 0), "official_gold")
        record = StudyDatasetRecord(StudyMetadata("s", "patient-1", {"modality": "MR"}),
                                    labels, "a" * 64)
        self.assertEqual("patient-1", record.to_dict()["study"]["patient_id"])
        self.assertIn("values", record.to_dict()["labels"])
        with self.assertRaisesRegex(ValueError, "same study"):
            StudyDatasetRecord(StudyMetadata("different"), labels, "a" * 64)

    def test_submission_order_and_json_schemas(self):
        self.assertEqual(("StudyInstanceUID", *TARGETS),
                         validate_submission_columns(("StudyInstanceUID", *TARGETS)))
        with self.assertRaisesRegex(ValueError, "submission columns mismatch"):
            validate_submission_columns(("StudyInstanceUID", *reversed(TARGETS)))
        schemas = Path("schemas")
        for path in schemas.glob("*.schema.json"):
            parsed = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual("object", parsed["type"], path.name)
            self.assertIn("title", parsed)
        registry = json.loads((schemas / "target-registry.schema.json").read_text())
        self.assertEqual(1, registry["properties"]["schema_version"]["const"])


if __name__ == "__main__":
    unittest.main()
