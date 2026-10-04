"""Study-level CNN baselines built on the canonical RSNA contracts."""

from .runner import run_cnn224, run_cnn224_smoke

__all__ = ["run_cnn224", "run_cnn224_smoke"]
