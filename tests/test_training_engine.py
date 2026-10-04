from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from rsna.training.smoke import build_engine, make_smoke_job, run_training_smoke
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
            for name, value in expected["model_state"].items():
                self.assertTrue(torch.equal(value, actual["model_state"][name]), name)
            self.assertEqual(baseline.metrics["train_loss"], resumed.metrics["train_loss"])
            self.assertEqual(baseline.metrics["validation_loss"], resumed.metrics["validation_loss"])

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


if __name__ == "__main__":
    unittest.main()
