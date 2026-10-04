import tempfile
import unittest
from pathlib import Path

from rsna.data import DatasetIndex, SeriesRecord, SliceRecord, StudyRecord, save_manifest
from rsna.inspection.query import (DatasetReadError, EntityNotFoundError, inspect_manifest,
                                   inspect_series, inspect_slice, inspect_study, load_dataset_index,
                                   sample_entities)
from rsna.inspection.summary import build_dataset_stats, build_dataset_summary


def dataset():
    one = SliceRecord("a.dcm", 10, {"SOPInstanceUID": "slice1", "StudyInstanceUID": "study1",
        "SeriesInstanceUID": "series1", "InstanceNumber": 1, "Rows": 320, "Columns": 320,
        "ImagePositionPatient": [0, 0, 0], "PixelSpacing": [0.5, 0.5],
        "ImageOrientationPatient": [0, 1, 0, 0, 0, 1], "SeriesDescription": "T2",
        "ProtocolName": "knee", "Modality": "MR", "Laterality": "L"})
    two = SliceRecord("b.dcm", 10, {"SOPInstanceUID": "slice2", "StudyInstanceUID": "study1",
        "SeriesInstanceUID": "series1", "InstanceNumber": 2, "Rows": 320, "Columns": 320,
        "ImagePositionPatient": [1, 0, 0], "PixelSpacing": [0.5, 0.5],
        "ImageOrientationPatient": [0, 1, 0, 0, 0, 1], "SeriesDescription": "T2",
        "ProtocolName": "knee", "Modality": "MR", "Laterality": "L"}, ("MISSING_POSITION",))
    series = SeriesRecord("series1", "study1", (one, two), ("WARN_SERIES",))
    study = StudyRecord("study1", "patient1", (series,), ("WARN_STUDY",))
    return DatasetIndex("root", "test", (study,), ("INDEX_WARNING",),
                        {"n_studies": 1, "n_series": 1, "n_slices": 2, "invalid_files": 0,
                         "duplicate_sop_uid_count": 0})


class InspectionServiceTests(unittest.TestCase):
    def test_summary_reports_canonical_counts_and_known_fields(self):
        index = dataset()
        result = build_dataset_summary(index)
        self.assertEqual((1, 1, 2, 1), (result["studies"], result["series"], result["slices"], result["patients"]))
        self.assertEqual(index.index_id, result["dataset_index_id"])
        self.assertEqual("unavailable", result["dataset_version_binding"])
        self.assertEqual(0, result["invalid_files"])
        self.assertIn("warnings", result)

    def test_inspection_uses_canonical_entities_and_domain_helpers(self):
        index = dataset()
        study = inspect_study(index, "study1")
        self.assertEqual((1, 2), (study["n_series"], study["n_slices"]))
        self.assertEqual("patient1", study["patient_id"])
        series = inspect_series(index, "series1")
        self.assertEqual(2, series["n_slices"])
        self.assertEqual("T2", series["SeriesDescription"])
        self.assertEqual("sagittal", series["orientation"]["plane"])
        self.assertIn("provenance", series)
        row = inspect_slice(index, "slice1")
        self.assertEqual("a.dcm", row["relative_path"])
        self.assertEqual([0.5, 0.5], row["PixelSpacing"])
        self.assertEqual(320, row["Rows"])
        self.assertEqual(index.index_id, inspect_manifest(index)["index_id"])
        self.assertEqual(study, inspect_study(index, index.studies[0].study_id))
        self.assertEqual(series, inspect_series(index, index.studies[0].series[0].series_id))
        self.assertEqual(row, inspect_slice(index, index.studies[0].series[0].slices[0].slice_id))
        self.assertNotEqual(study["study_instance_uid"], study["study_id"])
        self.assertNotEqual(series["series_instance_uid"], series["series_id"])

    def test_saved_manifest_round_trip_reuses_canonical_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            save_manifest(dataset(), path)
            self.assertEqual(dataset(), load_dataset_index(path))
            self.assertEqual(build_dataset_summary(dataset()), build_dataset_summary(load_dataset_index(path)))

    def test_filters_keep_content_ids_distinct_when_series_uid_is_reused(self):
        first = dataset().studies[0]
        item = SliceRecord("other.dcm", 10, {"SOPInstanceUID": "other", "StudyInstanceUID": "study2",
            "SeriesInstanceUID": "series1", "ImageOrientationPatient": [1, 0, 0, 0, 1, 0]})
        second = StudyRecord("study2", "patient2", (SeriesRecord("series1", "study2", (item,)),))
        index = DatasetIndex("root", "test", (first, second), (), {"n_studies": 2, "n_series": 2, "n_slices": 3})
        self.assertEqual(1, build_dataset_summary(index, plane="sagittal")["studies"])
        self.assertEqual(1, build_dataset_stats(index, plane="sagittal")["series_per_study"]["max"])
        with self.assertRaisesRegex(EntityNotFoundError, "ambiguous"):
            inspect_series(index, "series1")
        self.assertEqual("study2", inspect_series(index, second.series[0].series_id)["study_instance_uid"])
        empty_study = StudyRecord("empty", "p", ())
        self.assertEqual(1, build_dataset_summary(DatasetIndex("r", "test", (empty_study,), (), {}))["studies"])

    def test_lookup_and_manifest_errors(self):
        with self.assertRaisesRegex(EntityNotFoundError, "was not found"):
            inspect_study(dataset(), "missing")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(DatasetReadError):
                load_dataset_index(path)
            with self.assertRaises(DatasetReadError):
                load_dataset_index(Path(directory) / "absent.json")

    def test_deterministic_sampling_and_domain_filters(self):
        index = dataset()
        first = sample_entities(index, count=1, seed=42)
        self.assertEqual(first, sample_entities(index, count=1, seed=42))
        filtered = sample_entities(index, laterality="LEFT", count=20)
        self.assertEqual(1, filtered["count"])
        self.assertEqual("series1", filtered["records"][0]["series_instance_uid"])
        filtered = sample_entities(index, warning_code="WARN_SERIES")
        self.assertEqual(1, filtered["count"])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            sample_entities(index, count=-1)

    def test_stats_empty_and_observed_relationships(self):
        stats = build_dataset_stats(dataset())
        self.assertEqual(1.0, stats["series_per_study"]["mean"])
        self.assertEqual(2.0, stats["slices_per_series"]["p50"])
        self.assertEqual({"series_per_study": {"count": 0}, "slices_per_series": {"count": 0}}, build_dataset_stats(DatasetIndex("e", "test", (), (), {"n_studies": 0, "n_series": 0})))

    def test_filtered_summary_omits_global_counts(self):
        result = build_dataset_summary(dataset(), plane="sagittal")
        self.assertEqual((1, 1, 2), (result["studies"], result["series"], result["slices"]))
        self.assertNotIn("invalid_files", build_dataset_summary(
            DatasetIndex("r", "test", dataset().studies, (), {"invalid_files": 15}), plane="coronal"))


if __name__ == "__main__":
    unittest.main()
