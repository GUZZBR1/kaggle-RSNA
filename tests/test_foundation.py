import json
from pathlib import Path
import tempfile
import tomllib
import unittest

from rsna import (ArtifactReference, DatasetVersion, Evaluation, ExperimentSpec,
                  FoldPlan, ModelCandidate, PredictionArtifact, TrainingJob)
from rsna.artifacts import JsonArtifactStore
from rsna.cli import run_config
from rsna.experiments.plan import plan_jobs
from rsna.experiments.runner import run_jobs
from rsna.providers.local import LocalProvider
from rsna.providers.mock import MockProvider


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = DatasetVersion("dataset", "v1", "a" * 64, "prep-v1", ("normal", "abnormal"))
        self.folds = FoldPlan(self.dataset.dataset_version_id, "declared", ("fold_0", "fold_1"), 21)
        self.candidate = ModelCandidate("baseline-placeholder", {"architecture": "unset"})
        self.spec = ExperimentSpec("test", self.dataset.dataset_version_id,
            self.folds.fold_plan_id, (self.candidate.model_candidate_id,), ("fold_0",),
            self.dataset.class_names)

    def test_dataset_and_model_identity_ignore_locations(self):
        elsewhere = DatasetVersion("dataset", "v1", "a" * 64, "prep-v1",
                                   ("normal", "abnormal"), uri="s3://elsewhere")
        self.assertEqual(self.dataset.dataset_version_id, elsewhere.dataset_version_id)
        self.assertEqual(self.candidate.model_candidate_id,
            ModelCandidate("baseline-placeholder", {"architecture": "unset"},
                           implementation_uri="file:///tmp/model").model_candidate_id)
        self.assertNotEqual(self.candidate.model_candidate_id,
            ModelCandidate("baseline-placeholder", {"architecture": "different"}).model_candidate_id)

    def test_fold_plan_and_training_job_identity(self):
        jobs = plan_jobs(self.spec, self.dataset, self.folds,
                         {self.candidate.model_candidate_id: self.candidate})
        self.assertEqual(1, len(jobs))
        self.assertEqual(jobs[0].training_job_id, TrainingJob(**jobs[0].to_dict()).training_job_id)
        self.assertNotEqual(self.folds.fold_plan_id,
                            FoldPlan(self.dataset.dataset_version_id, "declared", ("fold_0", "fold_1"), 22).fold_plan_id)

    def test_provider_results_are_bound_to_job(self):
        job = plan_jobs(self.spec, self.dataset, self.folds,
                        {self.candidate.model_candidate_id: self.candidate})[0]
        mock_result = run_jobs([job], MockProvider())[0]
        self.assertEqual(job.training_job_id, mock_result.training_job_id)
        self.assertEqual("succeeded", mock_result.status)
        self.assertTrue(mock_result.provenance["synthetic"])
        provider = LocalProvider(lambda _: {"checkpoint_uri": "file:///tmp/checkpoint",
                                            "metrics": {"loss": 0.5}})
        result = run_jobs([job], provider)[0]
        self.assertEqual(job.fold_id, result.checkpoint.fold_id)
        failed = LocalProvider(lambda _: (_ for _ in ()).throw(RuntimeError("expected")))
        self.assertEqual("failed", run_jobs([job], failed)[0].status)

    def test_artifact_hash_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonArtifactStore(directory)
            ref = store.put_json({"x": 1}, manifest={"run": "smoke"})
            self.assertEqual({"x": 1}, store.load_json(ref))
            Path(ref.uri).write_text('{"x":2}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                store.load_json(ref)
        with self.assertRaises(ValueError):
            ArtifactReference("b" * 64, "application/json", "file:///x", "a" * 64)

    def test_prediction_and_evaluation_contract(self):
        ref = ArtifactReference("c" * 64, "application/json", "mock://predictions", "c" * 64)
        pred = PredictionArtifact(ref, "oof", self.candidate.model_candidate_id,
            self.dataset.dataset_version_id, "d" * 64, self.dataset.class_names, 10,
            self.folds.fold_plan_id, "fold_0")
        auc = {"normal": 0.8, "abnormal": 0.6}
        evaluation = Evaluation(self.spec.experiment_id, self.dataset.dataset_version_id,
            self.candidate.model_candidate_id, (pred.prediction_artifact_id,),
            self.dataset.class_names, auc, 0.7)
        self.assertEqual(0.7, evaluation.macro_auc)
        with self.assertRaises(ValueError):
            Evaluation(self.spec.experiment_id, self.dataset.dataset_version_id,
                self.candidate.model_candidate_id, (pred.prediction_artifact_id,),
                self.dataset.class_names, auc, 0.8)

    def test_config_cli_and_schemas(self):
        config = tomllib.loads(Path("configs/experiments/smoke.toml").read_text(encoding="utf-8"))
        self.assertEqual(12, len(config["dataset"]["class_names"]))
        output = run_config("configs/experiments/smoke.toml")
        self.assertTrue(output["synthetic"])
        self.assertEqual(1, len(output["results"]))
        for schema in Path("schemas").glob("*.schema.json"):
            parsed = json.loads(schema.read_text(encoding="utf-8"))
            self.assertEqual("object", parsed["type"])
            self.assertIn("title", parsed)


if __name__ == "__main__":
    unittest.main()
