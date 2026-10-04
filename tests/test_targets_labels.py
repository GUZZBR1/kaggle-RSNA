import json
from pathlib import Path
import unittest
from types import MappingProxyType

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
        self.assertEqual(TARGET_REGISTRY.registry_id,
                         TargetRegistry.from_dict(TARGET_REGISTRY.to_dict()).registry_id)
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
        self.assertEqual(record.record_id, LabelRecord.from_dict(
            json.loads(json.dumps(record.to_dict()))).record_id)
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

    def test_values_are_immutable_and_all_serializations_are_canonical(self):
        values = dict.fromkeys(reversed(TARGETS), 0)
        record = LabelRecord("uid-1", values, "official_gold")
        canonical = LabelRecord("uid-1", dict.fromkeys(TARGETS, 0), "official_gold")
        record_id = record.record_id
        self.assertIsInstance(record.values, MappingProxyType)
        with self.assertRaises(TypeError):
            record.values["ACL"] = 1
        values["ACL"] = 1
        self.assertEqual(record_id, record.record_id)
        self.assertEqual(TARGETS, tuple(record.values))
        self.assertEqual(TARGETS, tuple(record.to_dict()["values"]))
        self.assertEqual(TARGETS, tuple(name for name in record.to_row() if name in TARGETS))
        self.assertEqual(canonical.values, record.values)
        self.assertEqual(canonical.to_dict(), record.to_dict())
        self.assertEqual(canonical.to_row(), record.to_row())
        self.assertEqual(canonical.record_id, record.record_id)

    def test_masks_and_row_identity_are_validated(self):
        values = dict.fromkeys(TARGETS, 0)
        serialized = LabelRecord("uid", values, "manual_review").to_dict()
        serialized["mask"] = [1] * len(TARGETS)
        with self.assertRaisesRegex(ValueError, "mask"):
            LabelRecord.from_dict(serialized)
        row = LabelRecord("uid", values, "manual_review").to_row()
        row["label_mask"][0] = False
        with self.assertRaisesRegex(ValueError, "label_mask"):
            LabelRecord.from_row(row)
        strict = LabelRecord("uid", values, "manual_review")
        self.assertEqual(strict.record_id, LabelRecord.from_row(strict.to_row()).record_id)

    def test_alias_columns_and_study_index_adapter_use_uid(self):
        row = {name: 0 for name in TARGETS}
        row.pop("ACL")
        row["Anterior Cruciate Ligament Injury"] = 1
        row.update(StudyInstanceUID="uid", PatientID="patient", label_provenance="manual_review")
        self.assertEqual(1, LabelRecord.from_row(row).values["ACL"])
        with self.assertRaisesRegex(ValueError, "conflicting study identity"):
            LabelRecord.from_row({**row, "study_id": "content-hash"})
        with self.assertRaisesRegex(ValueError, "conflicting PatientID"):
            StudyDatasetRecord(StudyMetadata("uid", "p1"),
                LabelRecord("uid", dict.fromkeys(TARGETS, 0), "manual_review", patient_id="p2"), "a" * 64)

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

    def test_registry_snapshots_mutable_inputs(self):
        aliases = ["alias"]
        targets = [Target("One", aliases)]
        registry = TargetRegistry(targets)
        identity = registry.registry_id
        aliases.append("late alias")
        targets.append(Target("Two"))
        self.assertEqual(identity, registry.registry_id)
        with self.assertRaises(TypeError):
            registry._lookup["mutated"] = "One"

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
        with self.assertRaisesRegex(ValueError, "order mismatch"):
            PredictionArtifact(ref, "inference", candidate.model_candidate_id,
                dataset.dataset_version_id, "e" * 64, (*TARGETS, "Extra"), 1)
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
        self.assertEqual(("StudyInstanceUID", *TARGETS), submission.column_order)
        with self.assertRaisesRegex(ValueError, "order mismatch"):
            SubmissionArtifact(ref, candidate.model_candidate_id,
                dataset.dataset_version_id, evaluation.evaluation_id,
                (pred.prediction_artifact_id,), "csv", class_names=(*TARGETS[1:], TARGETS[0]))
        serialized_submission = submission.to_dict()
        serialized_submission["column_order"][1], serialized_submission["column_order"][2] = (
            serialized_submission["column_order"][2], serialized_submission["column_order"][1])
        with self.assertRaisesRegex(ValueError, "column order"):
            SubmissionArtifact.from_dict(serialized_submission)
        self.assertEqual(dataset.dataset_version_id,
                         DatasetVersion.from_dict(json.loads(json.dumps(dataset.to_dict()))).dataset_version_id)
        self.assertEqual(pred.prediction_artifact_id,
                         PredictionArtifact.from_dict(json.loads(json.dumps(pred.to_dict()))).prediction_artifact_id)
        self.assertEqual(evaluation.evaluation_id,
                         Evaluation.from_dict(json.loads(json.dumps(evaluation.to_dict()))).evaluation_id)
        self.assertEqual(submission.submission_artifact_id,
                         SubmissionArtifact.from_dict(json.loads(json.dumps(submission.to_dict()))).submission_artifact_id)

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
        self.assertEqual(TARGET_REGISTRY.to_dict()["targets"],
                         registry["properties"]["targets"]["const"])
        self.assertNotEqual(list(reversed(TARGET_REGISTRY.to_dict()["targets"])),
                            registry["properties"]["targets"]["const"])
        unknown_targets = TARGET_REGISTRY.to_dict()["targets"] + [
            {"name": "Extra", "index": 12, "aliases": []}]
        self.assertNotEqual(unknown_targets, registry["properties"]["targets"]["const"])
        label_schema = json.loads((schemas / "label-record.schema.json").read_text())
        self.assertEqual(TARGET_REGISTRY.registry_id,
                         label_schema["properties"]["target_registry_id"]["const"])
        self.assertEqual([0, 1, None], label_schema["allOf"][0]["then"]["properties"]["values"]
                         ["properties"]["ACL"]["enum"])
        self.assertEqual(["official_gold", "report_regex", "report_llm", "pseudo_label", "manual_review"],
                         label_schema["properties"]["provenance"]["enum"])


if __name__ == "__main__":
    unittest.main()
