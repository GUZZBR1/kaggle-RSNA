import json
from pathlib import Path
import tempfile
import unittest

from rsna.inspection.query import (DatasetIndex, DatasetReadError, EntityNotFoundError,
                                   inspect_manifest, inspect_series, inspect_slice,
                                   inspect_study, load_dataset_index, sample_entities)
from rsna.inspection.summary import build_dataset_stats, build_dataset_summary


def dataset():
    return {
        "dataset_id": "synthetic", "dataset_version_id": "v1", "schema_version": "1",
        "provenance": {"source": "fixture"},
        "studies": [{"StudyInstanceUID": "study1", "PatientID": "patient1"}],
        "series": [
            {"SeriesInstanceUID": "series1", "StudyInstanceUID": "study1", "plane": "sagittal",
             "Laterality": "right", "SeriesDescription": "T2", "ProtocolName": "knee", "Modality": "MR"},
            {"SeriesInstanceUID": "series2", "StudyInstanceUID": "study1", "plane": "coronal",
             "Laterality": "left", "warnings": [{"code": "MISSING_POSITION"}]},
        ],
        "slices": [
            {"SOPInstanceUID": "slice1", "SeriesInstanceUID": "series1", "StudyInstanceUID": "study1",
             "relative_path": "a.dcm", "InstanceNumber": 1, "Rows": 320, "Columns": 320,
             "ImagePositionPatient": [0, 0, 0], "PixelSpacing": [0.5, 0.5],
             "ImageOrientationPatient": [1, 0, 0, 0, 1, 0], "SpacingBetweenSlices": 2.0},
            {"SOPInstanceUID": "slice2", "SeriesInstanceUID": "series1", "SpacingBetweenSlices": 3.0},
        ],
    }


class InspectionServiceTests(unittest.TestCase):
    def test_summary_reports_only_observed_metrics(self):
        result = build_dataset_summary(dataset())
        self.assertEqual((1, 2, 2, 1), (result["studies"], result["series"], result["slices"], result["patients"]))
        self.assertEqual({"coronal": 1, "sagittal": 1}, result["planes"])
        self.assertEqual({"MISSING_POSITION": 1}, result["warning_frequency"])
        self.assertNotIn("invalid_files", result)
        self.assertNotIn("missing_metadata", result)
        self.assertNotIn("patients", build_dataset_summary({"studies": [{"study_uid": "s"}]}))
        self.assertNotIn("slices", build_dataset_summary({"dataset_id": "metadata-only"}))

    def test_inspection_normalizes_dicom_aliases_and_relationships(self):
        index = DatasetIndex.from_mapping(dataset())
        study = inspect_study(index, "study1")
        self.assertEqual((2, 2), (study["n_series"], study["n_slices"]))
        self.assertEqual({"source": "fixture"}, study["provenance"])
        self.assertEqual([[1, 0, 0, 0, 1, 0]], study["orientations"])
        self.assertEqual(2, inspect_series(index, "series1")["n_slices"])
        self.assertEqual("T2", inspect_series(index, "series1")["description"])
        row = inspect_slice(index, "slice1")
        self.assertEqual("a.dcm", row["path"])
        self.assertEqual([0.5, 0.5], row["spacing"])
        self.assertEqual(320, row["rows"])
        # Repeated inspection must not mutate indexes when inheriting study parents.
        self.assertEqual(study, inspect_study(index, "study1"))
        self.assertEqual(1, len(index._slices_by_study["study1"]))

    def test_lookup_not_found_or_duplicate_is_clear(self):
        with self.assertRaisesRegex(EntityNotFoundError, "was not found"):
            inspect_study(dataset(), "missing")
        value = dataset()
        value["series"].append(dict(value["series"][0]))
        with self.assertRaisesRegex(EntityNotFoundError, "ambiguous"):
            inspect_series(value, "series1")
        self.assertEqual(1, build_dataset_summary(value)["duplicate_uids"]["series"])

    def test_geometry_requires_complete_consistent_metadata(self):
        value = dataset()
        self.assertNotIn("physical_span_mm", inspect_series(value, "series1"))
        value["slices"][1].update({"position": [0, 0, 3], "orientation": [1, 0, 0, 0, 1, 0]})
        series = inspect_series(value, "series1")
        self.assertEqual(3, series["physical_span_mm"])
        self.assertEqual(3, series["spacing_stats_mm"]["mean"])
        value["slices"][1]["orientation"] = [0, 1, 0, 1, 0, 0]
        self.assertNotIn("physical_span_mm", inspect_series(value, "series1"))

    def test_keyed_and_wrapped_indexes(self):
        index = DatasetIndex.from_mapping({"dataset_id": "x", "index": {
            "studies": {"s": {"patient_id": "p"}},
            "series": {"r": {"study_id": "s", "n_slices": 4}},
        }})
        self.assertEqual("s", inspect_study(index, "s")["study_uid"])
        self.assertEqual(4, inspect_study(index, "s")["n_slices"])
        manifest = inspect_manifest(index)
        self.assertNotIn("index", manifest)
        self.assertEqual({"series": 1, "studies": 1}, manifest["entity_tables"])

    def test_loader_reads_json_metadata_reference_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"dataset_id": "wrapped", "index_path": "index.json"}), encoding="utf-8")
            value = dataset()
            # There are no actual DICOM files, so attempts to load pixels would fail.
            (root / "index.json").write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(2, build_dataset_summary(load_dataset_index(root))["slices"])
            (root / "manifest.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(DatasetReadError, "cannot read dataset manifest"):
                load_dataset_index(root)
            with self.assertRaisesRegex(DatasetReadError, "cannot read dataset manifest"):
                load_dataset_index(root / "absent.json")

    def test_invalid_tables_rejected(self):
        for value in ({"studies": "invalid"}, {"series": [False]}, {"slices": {"id": 5}}, []):
            with self.subTest(value=value), self.assertRaises(DatasetReadError):
                DatasetIndex.from_mapping(value)

    def test_deterministic_sample_and_filters(self):
        value = dataset()
        first = sample_entities(value, count=1, seed=42)
        self.assertEqual(first, sample_entities(value, count=1, seed=42))
        value["series"].reverse()
        self.assertEqual(first, sample_entities(value, count=1, seed=42))
        filtered = sample_entities(value, plane="SAGITTAL", laterality="RIGHT", count=20)
        self.assertEqual(1, filtered["count"])
        self.assertEqual("series1", filtered["records"][0]["series_uid"])
        self.assertEqual(1, sample_entities(value, warning_code="MISSING_POSITION")["count"])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            sample_entities(value, count=-1)

    def test_stats_relationship_counts_and_empty_observations(self):
        stats = build_dataset_stats(dataset())
        self.assertEqual(2.0, stats["series_per_study"]["mean"])
        self.assertEqual(1.0, stats["slices_per_series"]["p50"])
        self.assertEqual(2.5, stats["slice_spacing"]["p50"])
        empty = build_dataset_stats({"studies": [], "series": [], "slices": []})
        self.assertEqual({"count": 0}, empty["slices_per_series"])
        self.assertNotIn("slice_spacing", empty)
        self.assertEqual({}, build_dataset_stats({"dataset_id": "metadata-only"}))

    def test_summary_filter_drops_global_counts(self):
        value = dataset()
        value["statistics"] = {"series": 2000, "invalid_files": 15}
        result = build_dataset_summary(value, plane="coronal")
        self.assertEqual((1, 1, 0), (result["studies"], result["series"], result["slices"]))
        self.assertNotIn("invalid_files", result)
        self.assertEqual(0, build_dataset_stats(value, plane="coronal")["slices_per_series"]["max"])


if __name__ == "__main__":
    unittest.main()
