"""Reproducible, group-aware cross-validation fold plans."""

from .generate import generate_fold_plan
from .manifest import FoldPlanManifest, load_fold_plan, save_fold_plan

__all__ = ["FoldPlanManifest", "generate_fold_plan", "load_fold_plan", "save_fold_plan"]
