"""Fold-level counts and imbalance diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from ..targets import TARGETS


def summarize(records: list[dict[str, Any]], assignments: Mapping[str, str], n_folds: int,
              group_for_study: Mapping[str, str], labels: Mapping[str, Mapping[str, Any]] | None = None,
              *, size_deviation_threshold: float = 0.20,
              prevalence_range_threshold: float = 0.20) -> tuple[dict[str, Any], list[str]]:
    labels = labels or {}
    target_names = TARGETS if labels else ()
    series_available = any(row.get("series_available", True) for row in records)
    fold_stats = {f"fold_{i}": {"n_groups": 0, "n_patients": 0, "n_studies": 0,
                                  "n_series": 0, "targets": {}}
                  for i in range(n_folds)}
    all_patients: dict[str, set[str]] = defaultdict(set)
    all_groups: dict[str, set[str]] = defaultdict(set)
    counts = {fold: {target: [0, 0, 0, 0.0, 0] for target in target_names} for fold in fold_stats}
    for row in records:
        fold = assignments[row["study_id"]]
        target = fold_stats[fold]
        target["n_studies"] += 1
        if series_available:
            target["n_series"] += len(row.get("series_ids", ()))
        patient = row.get("patient_id")
        if patient:
            all_patients[fold].add(str(patient))
        all_groups[fold].add(group_for_study[row["study_id"]])
        for name in target_names:
            value = labels.get(row["study_id"], {}).get(name)
            label_type = labels.get(row["study_id"], {}).get("__label_type__", "hard")
            if value is None:
                counts[fold][name][2] += 1
            elif label_type == "soft":
                counts[fold][name][3] += float(value)
                counts[fold][name][4] += 1
            elif value in (True, 1):
                counts[fold][name][0] += 1
            elif value in (False, 0):
                counts[fold][name][1] += 1
            else:
                raise ValueError(f"label {name!r} for study {row['study_id']!r} must be 0, 1, or null")
    for fold, stats in fold_stats.items():
        if not series_available:
            stats["n_series"] = None
        stats["n_groups"] = len(all_groups[fold])
        stats["n_patients"] = len(all_patients[fold])
        for name, (positive, negative, missing, soft_sum, soft_count) in counts[fold].items():
            known = positive + negative + soft_count
            stats["targets"][name] = {"positive": positive, "negative": negative,
                                      "missing": missing,
                                      "soft_count": soft_count, "soft_probability_sum": soft_sum,
                                      "prevalence": (positive + soft_sum) / known if known else None}
    totals = {"n_groups": len(set(group_for_study.values())),
              "n_patients": len({str(r['patient_id']) for r in records if r.get('patient_id')}),
              "n_studies": len(records),
              "n_series": sum(len(r.get("series_ids", ())) for r in records) if series_available else None,
              "targets": {}}
    for name in target_names:
        values = [counts[fold][name] for fold in fold_stats]
        positive, negative, missing, soft_sum, soft_count = (
            sum(value[i] for value in values) for i in range(5))
        known = positive + negative + soft_count
        totals["targets"][name] = {"positive": positive, "negative": negative,
                                   "missing": missing,
                                   "soft_count": soft_count, "soft_probability_sum": soft_sum,
                                   "prevalence": (positive + soft_sum) / known if known else None}
    sizes = [stats["n_studies"] for stats in fold_stats.values()]
    mean_size = sum(sizes) / len(sizes)
    max_size_deviation = max(abs(size - mean_size) / mean_size for size in sizes)
    patient_counts = [stats["n_patients"] for stats in fold_stats.values()]
    mean_patients = sum(patient_counts) / len(patient_counts)
    patient_deviation = (max(abs(value - mean_patients) / mean_patients for value in patient_counts)
                        if mean_patients else 0.0)
    prevalence_ranges = {}
    for name in target_names:
        values = [s["targets"][name]["prevalence"] for s in fold_stats.values()
                  if s["targets"][name]["prevalence"] is not None]
        prevalence_ranges[name] = max(values) - min(values) if len(values) > 1 else 0.0
    diagnostics = {"max_fold_size_deviation": max_size_deviation,
                   "max_patient_count_deviation": patient_deviation,
                   "target_prevalence_range": prevalence_ranges}
    warnings = []
    if max_size_deviation > size_deviation_threshold:
        warnings.append(f"fold size deviation {max_size_deviation:.3f} exceeds {size_deviation_threshold:.3f}")
    for name, spread in prevalence_ranges.items():
        if spread > prevalence_range_threshold:
            warnings.append(f"target {name} prevalence range {spread:.3f} exceeds {prevalence_range_threshold:.3f}")
    for name, target in totals["targets"].items():
        if 0 < target["positive"] < n_folds:
            warnings.append(f"rare target {name} has fewer positives than folds ({target['positive']} < {n_folds})")
    return {"folds": fold_stats, "total": totals, "diagnostics": diagnostics}, warnings
