from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from rsna.training.smoke import build_engine, make_smoke_job, run_training_smoke
from rsna.training.engine import FoldData
from rsna.providers.local import LocalProvider


@unittest.skipIf(torch is None, "torch is an optional training dependency")
class TrainingEngineTests(unittest.TestCase):
    def test_interrupted_resume_matches_uninterrupted_training(self):
        dataset, candidate, job, loader = make_smoke_job()
        with tempfile.TemporaryDirectory() as root:
            baseline = build_engine(dataset, candidate, job, loader, Path(root) / "baseline").run(job)
            interrupted = build_engine(dataset, candidate, job, loader, Path(root) / "resume").run(
                job, stop_after_optimizer_steps=1)
            self.assertEqual("failed", interrupted.status)
            self.assertIsNotNone(interrupted.checkpoint)
            resumed = build_engine(dataset, candidate, job, loader, Path(root) / "resume").run(
                job, resume_from=interrupted.checkpoint)
            self.assertEqual("succeeded", baseline.status)
            self.assertEqual("succeeded", resumed.status)
            expected = torch.load(baseline.checkpoint.artifact.uri, map_location="cpu", weights_only=False)
            actual = torch.load(resumed.checkpoint.artifact.uri, map_location="cpu", weights_only=False)
            stopped = torch.load(interrupted.checkpoint.artifact.uri, map_location="cpu", weights_only=False)
            for name, value in expected["model_state"].items():
                self.assertTrue(torch.equal(value, actual["model_state"][name]), name)
            self.assertEqual(baseline.metrics["train_loss"], resumed.metrics["train_loss"])
            self.assertEqual(baseline.metrics["validation_loss"], resumed.metrics["validation_loss"])
            baseline_steps = baseline.provenance["step_timings_seconds"]
            resumed_steps = resumed.provenance["step_timings_seconds"]
            self.assertGreater(len(baseline_steps), 0)
            self.assertEqual(len(baseline_steps), len(resumed_steps))
            self.assertTrue(all(step > 0 for step in resumed_steps))
            self.assertGreater(resumed.metrics["step_time_mean_seconds"], 0)
            self.assertGreaterEqual(resumed.metrics["training_seconds"],
                                    stopped["progress"]["elapsed_seconds"])
            self.assertEqual(job.fold_plan_id, resumed.provenance["fold_plan_id"])
            self.assertEqual(job.dataset_version_id, resumed.provenance["dataset_version_id"])

    def test_local_provider_accepts_canonical_engine_result(self):
        dataset, candidate, job, loader = make_smoke_job()
        with tempfile.TemporaryDirectory() as root:
            provider = LocalProvider(build_engine(dataset, candidate, job, loader, Path(root)))
            execution_id = provider.submit_training(job)
            result = provider.get_training_result(execution_id)
        self.assertEqual("succeeded", result.status)
        self.assertEqual(job.training_job_id, result.training_job_id)
        self.assertEqual("pytorch-state-v1", result.checkpoint.format)

    def test_checkpoint_corruption_is_rejected_by_content_hash(self):
        dataset, candidate, job, loader = make_smoke_job()
        with tempfile.TemporaryDirectory() as root:
            engine = build_engine(dataset, candidate, job, loader, Path(root))
            result = engine.run(job)
            checkpoint_path = Path(result.checkpoint.artifact.uri)
            checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tampered")
            resumed = engine.run(job, resume_from=result.checkpoint)
        self.assertEqual("failed", resumed.status)
        self.assertIn("content hash mismatch", resumed.failure or "")

    def test_smoke_command_checkpoint_can_resume(self):
        with tempfile.TemporaryDirectory() as root:
            interrupted = run_training_smoke(root, stop_after_optimizer_steps=1)
            self.assertEqual("interrupted", interrupted["status"])
            resumed = run_training_smoke(root, resume_from=interrupted["checkpoint_uri"])
        self.assertEqual("succeeded", resumed["status"])
        self.assertIsNone(resumed["failure"])

    def test_training_job_requires_matching_leakage_report(self):
        dataset, candidate, job, loader = make_smoke_job()
        invalid = type(job)(job.experiment_id, job.dataset_version_id, job.fold_plan_id,
            job.model_candidate_id, job.fold_id, job.random_state, job.resources, job.configuration,
            validated_leakage_report_id="0" * 64)
        with tempfile.TemporaryDirectory() as root:
            result = build_engine(dataset, candidate, invalid, loader, Path(root)).run(invalid)
        self.assertEqual("failed", result.status)
        self.assertIn("LeakageGuard", result.failure or "")

    def test_nonfinite_training_state_returns_structured_failure(self):
        dataset, candidate, job, loader = make_smoke_job()

        def nonfinite_loader(training_job, train_ids, validation_ids):
            data = loader(training_job, train_ids, validation_ids)
            inputs, targets, masks = data.train.tensors
            corrupted_inputs = inputs.clone()
            corrupted_inputs[0, 0] = float("nan")
            train = torch.utils.data.TensorDataset(corrupted_inputs, targets, masks)
            return FoldData(train, data.validation, data.train_study_ids,
                            data.validation_study_ids, data.provenance)

        with tempfile.TemporaryDirectory() as root:
            result = build_engine(dataset, candidate, job, nonfinite_loader, Path(root)).run(job)
        self.assertEqual("failed", result.status)
        self.assertIn("non-finite masked BCE loss", result.failure or "")
        self.assertEqual("NONFINITE_TRAINING_STATE", result.provenance["failure_state"]["code"])
        self.assertEqual(result.checkpoint is not None,
                         result.provenance["failure_state"]["recoverable"])


if __name__ == "__main__":
    unittest.main()
