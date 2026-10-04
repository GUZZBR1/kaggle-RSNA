import importlib.util
import unittest

from rsna.contracts import DatasetVersion
from rsna.folds.models import FoldPlanManifest
from rsna.identity import digest
from rsna.labels import LabelRecord
from rsna.leakage import LeakageValidationError
from rsna.targets import TARGETS
from rsna.training.data import FoldTensorDataset, StudyTensorRecord

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is an optional training dependency")
class FoldTensorDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def setUp(self):
        self.dataset_version = DatasetVersion("training-data-test", "v1", "a" * 64,
                                              "prep-v1", TARGETS)
        assignments = {"study-a": "fold_0", "study-b": "fold_0",
                       "study-c": "fold_1", "study-d": "fold_1"}
        groups = {"study-a": "patient-a", "study-b": "patient-b",
                  "study-c": "patient-c", "study-d": "patient-d"}
        self.fold_plan = FoldPlanManifest(
            dataset_version_id=self.dataset_version.dataset_version_id,
            generator_version="test", input_fingerprint=digest(assignments), provenance={},
            strategy="imported", n_folds=2, random_state=17, grouping_key="patient_id",
            assignments=assignments,
            group_assignments={patient: assignments[study] for study, patient in groups.items()},
            study_groups=groups, study_fingerprints={study: digest(study) for study in assignments})
        self.records = [self.make_record(study, patient, marker=index)
                        for index, (study, patient) in enumerate(groups.items())]

    def make_record(self, study, patient, marker=0, *, sop_ids=(), file_hashes=()):
        labels = LabelRecord(study, {name: (1 if i == 0 else None)
                                     for i, name in enumerate(TARGETS)},
                             "official_gold", allow_partial=True, patient_id=patient)
        return StudyTensorRecord(study, patient,
                                 self.torch.full((1, 2), float(marker)), labels,
                                 sop_ids=sop_ids, file_hashes=file_hashes)

    def test_fold_boundaries_and_missing_label_mask(self):
        dataset = FoldTensorDataset(self.records, self.dataset_version, self.fold_plan)
        validation = list(dataset.iter_batches("fold_0", split="validation", batch_size=8))
        training = list(dataset.iter_batches("fold_0", split="train", batch_size=8))
        self.assertEqual({"study-a", "study-b"}, set(validation[0].study_ids))
        self.assertEqual({"study-c", "study-d"}, set(training[0].study_ids))
        self.assertEqual((2, len(TARGETS)), tuple(validation[0].targets.shape))
        self.assertEqual((2, len(TARGETS)), tuple(validation[0].mask.shape))
        self.assertTrue(bool(validation[0].mask[:, 0].all()))
        self.assertFalse(bool(validation[0].mask[:, 1:].any()))
        self.assertTrue(bool((validation[0].targets[:, 1:] == 0).all()))

    def test_missing_study_and_patient_plan_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            FoldTensorDataset(self.records[:-1], self.dataset_version, self.fold_plan)
        wrong_patient = [self.make_record("study-a", "other-patient"), *self.records[1:]]
        with self.assertRaisesRegex(ValueError, "does not match FoldPlan"):
            FoldTensorDataset(wrong_patient, self.dataset_version, self.fold_plan)

    def test_leakage_sop_reuse_across_folds_rejected(self):
        records = list(self.records)
        records[0] = self.make_record("study-a", "patient-a", sop_ids=("sop-shared",))
        records[2] = self.make_record("study-c", "patient-c", sop_ids=("sop-shared",))
        with self.assertRaises(LeakageValidationError):
            FoldTensorDataset(records, self.dataset_version, self.fold_plan)

    def test_training_shuffle_is_repeatable_and_validation_is_stable(self):
        dataset = FoldTensorDataset(self.records, self.dataset_version, self.fold_plan)
        def ids(split, seed, epoch):
            return [study for batch in dataset.iter_batches("fold_0", split=split,
                                                            batch_size=1, seed=seed, epoch=epoch)
                    for study in batch.study_ids]
        self.assertEqual(ids("train", 23, 4), ids("train", 23, 4))
        self.assertEqual(ids("validation", 23, 4), ids("validation", 999, 8))


if __name__ == "__main__":
    unittest.main()
