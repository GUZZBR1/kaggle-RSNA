import json
from pathlib import Path
import tempfile
import unittest

from rsna.contracts import DatasetVersion, ExperimentSpec, FoldPlan, ModelCandidate, TrainingJob
from rsna.identity import jsonable
from rsna.providers.cloud import build_cloud_training_spec
from rsna.training.telemetry import JsonlTelemetrySink, load_events, utc_timestamp
from rsna.telemetry.events import TelemetryEvent


class JsonlTelemetryTests(unittest.TestCase):
    def test_append_order_roundtrip_and_canonical_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "events.jsonl"
            sink = JsonlTelemetrySink(path)
            events = [
                TelemetryEvent("run.started", "2026-01-01T00:00:00Z", {"z": 1, "a": {"b": True}}),
                TelemetryEvent("run.finished", "2026-01-01T00:00:01Z", {"loss": 0.25}),
            ]
            sink(events[0])
            sink.emit(events[1])

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(2, len(lines))
            self.assertEqual('{"attributes":{"a":{"b":true},"z":1},"name":"run.started","timestamp":"2026-01-01T00:00:00Z"}', lines[0])
            self.assertEqual(events, sink.read_events())
            self.assertEqual(events, load_events(path))

    def test_timestamp_factory_is_utc(self):
        stamp = utc_timestamp()
        self.assertTrue(stamp.endswith("Z"))
        self.assertIn("T", stamp)


def make_job() -> TrainingJob:
    dataset = DatasetVersion("d", "v1", "a" * 64, "prep-v1", ("normal",), synthetic=True)
    folds = FoldPlan(dataset.dataset_version_id, "group", ("fold_0",), 42)
    candidate = ModelCandidate("smoke", {"kind": "synthetic"})
    experiment = ExperimentSpec("smoke", dataset.dataset_version_id, folds.fold_plan_id,
        (candidate.model_candidate_id,), ("fold_0",), dataset.class_names, synthetic=True)
    return TrainingJob(experiment.experiment_id, dataset.dataset_version_id,
        folds.fold_plan_id, candidate.model_candidate_id, "fold_0", 42, {"cpus": 1, "gpus": 0},
        {"batch_size": 2}, validated_leakage_report_id="b" * 64)


class CloudTrainingSpecTests(unittest.TestCase):
    def test_cloud_spec_contains_job_lineage_and_execution_inputs(self):
        job = make_job()
        spec = build_cloud_training_spec(job, input_uris={"prepared": "s3://data/prepared.json"},
            output_uri="s3://runs/run-1", environment={"OMP_NUM_THREADS": "1"},
            command=("python", "-m", "trainer", "--fold", "fold_0"),
            resume_checkpoint_uri="s3://runs/previous/checkpoint.pt")

        self.assertEqual(job.training_job_id, spec["training_job_id"])
        self.assertEqual(job.fold_id, spec["fold_id"])
        self.assertEqual(jsonable(job.resources), spec["resources"])
        self.assertEqual({"prepared": "s3://data/prepared.json"}, spec["inputs"])
        self.assertEqual("s3://runs/run-1", spec["output_uri"])
        self.assertEqual(["python", "-m", "trainer", "--fold", "fold_0"], spec["command"])
        self.assertEqual("s3://runs/previous/checkpoint.pt", spec["resume_checkpoint_uri"])
        json.dumps(spec)

    def test_invalid_cloud_spec_values_are_rejected(self):
        job = make_job()
        valid = {"input_uris": {"dataset": "file:///dataset"}, "output_uri": "file:///out",
                 "environment": {}, "command": ("python", "train.py")}
        for updates in (
                {"input_uris": {}}, {"input_uris": {"data": " "}}, {"output_uri": ""},
                {"environment": {"BAD=NAME": "x"}}, {"environment": {"A": None}},
                {"command": ()}, {"command": "python train.py"},
                {"resume_checkpoint_uri": " "}):
            with self.subTest(updates=updates), self.assertRaises((TypeError, ValueError)):
                build_cloud_training_spec(job, **{**valid, **updates})


if __name__ == "__main__":
    unittest.main()
