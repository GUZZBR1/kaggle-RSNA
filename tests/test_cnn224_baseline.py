import contextlib
import importlib.util
import io
from pathlib import Path
import tomllib
import unittest

from rsna.contracts import ArtifactReference
from rsna.artifacts import JsonArtifactStore
from rsna.baseline.model import CNN224Config, build_cnn224_model, build_model

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is an optional training dependency")
class CNN224ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_shared_slice_encoder_accepts_24_grayscale_slices_and_12_targets(self):
        model = build_cnn224_model({"slice_count": 24, "input_size": 32,
                                    "channels": [2, 4], "pooling": "mean"})
        inputs = self.torch.rand((2, 24, 1, 32, 32))
        logits = model(inputs)
        self.assertEqual((2, 12), tuple(logits.shape))

    def test_mean_max_pooling_is_repeatable_in_eval_mode(self):
        self.torch.manual_seed(4)
        model = build_model(CNN224Config.tiny(slice_count=3, input_size=24,
                                               pooling="mean_max"))
        model.eval()
        inputs = self.torch.rand((2, 3, 1, 24, 24))
        first = model(inputs)
        second = model(inputs)
        self.assertTrue(self.torch.equal(first, second))
        self.assertEqual((2, 12), tuple(first.shape))

    def test_invalid_slice_channel_and_spatial_shapes_fail(self):
        model = build_cnn224_model({"slice_count": 2, "input_size": 24,
                                    "channels": [2, 4]})
        with self.assertRaisesRegex(ValueError, "expected 2 slices"):
            model(self.torch.rand((1, 3, 1, 24, 24)))
        with self.assertRaisesRegex(ValueError, "one grayscale channel"):
            model(self.torch.rand((1, 2, 3, 24, 24)))
        with self.assertRaisesRegex(ValueError, "24x24"):
            model(self.torch.rand((1, 2, 1, 20, 20)))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is an optional training dependency")
class CNN224PipelineSmokeTests(unittest.TestCase):
    def test_cpu_tiny_smoke_trains_checkpoints_infers_and_uses_oof_engine(self):
        from rsna.baseline.runner import run_cnn224_smoke

        first = run_cnn224_smoke()
        second = run_cnn224_smoke()
        self.assertEqual("succeeded", first["status"])
        self.assertTrue(first["synthetic"])
        self.assertEqual(2, len(first["folds"]))
        self.assertEqual(["fold_0", "fold_1"], [row["fold_id"] for row in first["folds"]])
        self.assertTrue(all(row["checkpoint_id"] and row["prediction_artifact_id"]
                            for row in first["folds"]))
        self.assertTrue(all(row["validation_studies"] == 4 for row in first["folds"]))
        self.assertEqual("ready", first["oof"]["status"])
        self.assertEqual(12, len(first["oof"]["evaluations"][0]["auc_by_class"]))
        aucs = list(first["oof"]["evaluations"][0]["auc_by_class"].values())
        self.assertAlmostEqual(sum(aucs) / len(aucs),
                               first["oof"]["evaluations"][0]["macro_auc"])
        self.assertEqual(first["model_candidate"]["model_candidate_id"],
                         second["model_candidate"]["model_candidate_id"])
        self.assertEqual([row["training_job_id"] for row in first["folds"]],
                         [row["training_job_id"] for row in second["folds"]])
        self.assertEqual(first["oof"]["evaluations"][0]["auc_by_class"],
                         second["oof"]["evaluations"][0]["auc_by_class"])
        target_table = JsonArtifactStore.load_json(ArtifactReference.from_dict(
            first["oof"]["artifacts"]["target_auc_table"]))
        self.assertEqual({7}, {row["observed_count"] for row in target_table["rows"]})

    def test_real_run_config_keeps_cpu_smoke_independent_of_pretraining(self):
        path = Path(__file__).parents[1] / "configs" / "models" / "cnn224-baseline.toml"
        config = tomllib.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(224, config["model"]["input_size"])
        self.assertEqual(24, config["model"]["slice_count"])
        self.assertFalse(config["model"]["pretrained"])
        self.assertEqual(5, config["training"]["epochs"])
        self.assertEqual("cuda:0", config["training"]["device"])

    def test_cli_help_and_tiny_smoke_entrypoint(self):
        from rsna.cli import main

        help_output = io.StringIO()
        with contextlib.redirect_stdout(help_output):
            status = main(["baseline", "cnn224", "--help"])
        self.assertEqual(0, status)
        self.assertIn("smoke", help_output.getvalue())


if __name__ == "__main__":
    unittest.main()
