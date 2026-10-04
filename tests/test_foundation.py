import hashlib
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
import zipfile

from keigo import Candidate, ExperimentSpec, SimulationJob, SimulationResult
from keigo.artifacts import JsonArtifactStore
from keigo.evaluation import evaluate
from keigo.experiments.plan import plan_jobs
from keigo.experiments.runner import run_experiment
from keigo.experiments.seeds import SeedRegistry
from keigo.providers.local import LocalProvider
from keigo.providers.mock import MockProvider
from keigo.submission import build_bundle


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.candidate = Candidate("test-candidate", "a" * 64)
        self.candidates = {self.candidate.candidate_id: self.candidate}
        self.spec = ExperimentSpec(
            name="contract-test", candidate_ids=(self.candidate.candidate_id,),
            seeds=(3, 7), opponents=("baseline",), simulator_version="test-engine",
        )

    def test_imports_and_candidate_identity(self):
        self.assertEqual(self.candidate.candidate_id,
                         Candidate("test-candidate", "a" * 64).candidate_id)
        self.assertNotEqual(self.candidate.candidate_id,
                            Candidate("test-candidate", "b" * 64).candidate_id)
        configuration = {"weights": [1, 2]}
        candidate = Candidate("immutable", "c" * 64, configuration=configuration)
        original_id = candidate.candidate_id
        configuration["weights"].append(3)
        self.assertEqual(original_id, candidate.candidate_id)
        with self.assertRaises(AttributeError):
            candidate.configuration["weights"].append(4)
        moved = Candidate("test-candidate", "a" * 64,
                          artifact_uri="s3://different-location/candidate")
        self.assertEqual(self.candidate.candidate_id, moved.candidate_id)

    def test_schema_validation_and_experiment_identity(self):
        self.assertEqual(self.spec.experiment_id, ExperimentSpec.from_dict(
            self.spec.to_dict()).experiment_id)
        with self.assertRaises(ValueError):
            ExperimentSpec(name="bad", candidate_ids=(), seeds=(1,))
        changed = ExperimentSpec(name="contract-test",
                                 candidate_ids=self.spec.candidate_ids, seeds=(3, 8),
                                 simulator_version="test-engine")
        self.assertNotEqual(self.spec.experiment_id, changed.experiment_id)

    def test_job_identity_and_two_seat_plan(self):
        jobs = plan_jobs(self.spec, self.candidates)
        self.assertEqual(4, len(jobs))
        self.assertEqual({0, 1}, {job.seat for job in jobs})
        self.assertEqual(len(jobs), len({job.job_id for job in jobs}))
        self.assertEqual(jobs[0].job_id, SimulationJob(**jobs[0].to_dict()).job_id)

    def test_manifest_result_coverage_and_mock_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation, artifact = run_experiment(self.spec, self.candidates, MockProvider(),
                                                  JsonArtifactStore(directory))
            value = JsonArtifactStore.load_json(artifact)
            self.assertEqual(evaluation.to_dict(), value)
            manifest = json.loads(JsonArtifactStore.serialize(artifact))
            self.assertEqual(artifact.sha256, manifest["sha256"])
            self.assertEqual(self.spec.experiment_id,
                             artifact.manifest["experiment_id"])

    def test_local_provider_contract(self):
        provider = LocalProvider(lambda job: {"score": float(job.seed % 2)})
        job = plan_jobs(self.spec, self.candidates)[0]
        execution_id = provider.submit_simulation(job)
        result = provider.get_result(execution_id)
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(job.job_id, result.job_id)
        self.assertEqual("local", result.provider)

    def test_evaluator_rejects_incomplete_and_failed_results(self):
        job = plan_jobs(self.spec, self.candidates)[0]
        success = SimulationResult(job_id=job.job_id, candidate_id=job.candidate_id,
                                   opponent_id=job.opponent_id, seed=job.seed, seat=job.seat,
                                   status="succeeded", metrics={"score": 1.0},
                                   provider="test", execution_id="1")
        with self.assertRaises(ValueError):
            evaluate(self.spec, [success])
        failed = SimulationResult(job_id=job.job_id, candidate_id=job.candidate_id,
                                  opponent_id=job.opponent_id, seed=job.seed, seat=job.seat,
                                  status="failed", metrics={}, provider="test",
                                  execution_id="2", failure="synthetic failure")
        with self.assertRaises(ValueError):
            evaluate(self.spec, [failed])

    def test_smoke_config_and_fresh_seed_registry(self):
        config_path = Path("configs/experiments/smoke.toml")
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        configured_candidate = Candidate(**config["candidate"])
        experiment = dict(config["experiment"])
        experiment.pop("seed_registry")
        experiment["candidate_ids"] = [configured_candidate.candidate_id]
        self.assertEqual("smoke", ExperimentSpec.from_dict(experiment).name)
        registry = json.loads(Path("configs/seeds/registry.json").read_text())
        self.assertEqual([], registry["seed_splits"]["validation"])
        self.assertEqual([], registry["seed_splits"]["holdout"])
        seed_registry = SeedRegistry("configs/seeds/registry.json")
        seed_registry.validate((11, 23), "development")
        with self.assertRaises(ValueError):
            seed_registry.validate((11,), "validation")

    def test_all_json_schemas_parse(self):
        for path in Path("schemas").glob("*.schema.json"):
            with self.subTest(schema=path.name):
                schema = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual("object", schema["type"])

    def test_artifact_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonArtifactStore(directory)
            artifact = store.put_json({"a": 1})
            Path(artifact.uri).write_text('{"a":2}\n')
            with self.assertRaises(ValueError):
                store.load_json(artifact)

    def test_submission_bundle_preserves_candidate_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "candidate.py"
            source.write_text("def agent(observation): return {}\n", encoding="utf-8")
            candidate = Candidate("bundle-candidate", hashlib.sha256(
                source.read_bytes()).hexdigest())
            destination = Path(directory) / "submission.zip"
            manifest = build_bundle(candidate, source, destination)
            self.assertEqual(candidate.candidate_id, manifest["candidate_id"])
            first_hash = manifest["bundle_sha256"]
            self.assertEqual(first_hash, build_bundle(candidate, source, destination)["bundle_sha256"])
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual({"main.py", "manifest.json"}, set(archive.namelist()))
                bundled = json.loads(archive.read("manifest.json"))
                self.assertEqual(candidate.artifact_sha256,
                                 bundled["entrypoint_sha256"])


if __name__ == "__main__":
    unittest.main()
