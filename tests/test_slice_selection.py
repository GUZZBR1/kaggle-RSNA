import unittest
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from unittest.mock import patch

from rsna.contracts import DatasetVersion
from rsna.targets import TARGET_REGISTRY
from rsna.data.selection import SliceSelectionConfig, SliceSelector, select_slices
from rsna.data.geometry import order_series_slices
from rsna.data.models import SeriesRecord, SliceRecord
from rsna.cli import main


def series(count, positions=None):
    return [{"relative_path": f"slice-{i:03}.dcm", "metadata": {
        "SOPInstanceUID": f"sop-{i:03}",
        **({"physical_position_mm": positions[i]} if positions is not None else {})}}
        for i in range(count)]


class SliceSelectionTests(unittest.TestCase):
    def test_uniform_and_center_cover_expected_indices(self):
        uniform = select_slices(series(40), strategy="uniform", count=8)
        self.assertEqual((0, 6, 11, 17, 22, 28, 33, 39), uniform.selected_indices)
        self.assertEqual(1.0, uniform.coverage_fraction)
        center = select_slices(series(40), strategy="center", count=8)
        self.assertEqual(tuple(range(16, 24)), center.selected_indices)
        self.assertEqual((16, 24, 19.5), tuple(center.to_dict()["parameters"][key]
                         for key in ("start_index", "end_index", "center_index")))

    def test_physical_span_uniform_irregular_and_shuffled(self):
        uniform_positions = list(range(40))
        result = select_slices(series(40, uniform_positions), strategy="physical_span", count=8)
        self.assertEqual((0, 6, 11, 17, 22, 28, 33, 39), result.selected_indices)
        irregular = [float(i * i) for i in range(40)]
        by_uid = lambda r: tuple(item["metadata"]["SOPInstanceUID"] for item in r.selected)
        physical = select_slices(series(40, irregular), strategy="physical_span", count=8)
        indexed = select_slices(series(40, irregular), strategy="uniform", count=8)
        self.assertNotEqual(by_uid(physical), by_uid(indexed))
        shuffled = list(reversed(series(40, irregular)))
        stable = select_slices(shuffled, strategy="physical_span", count=8)
        self.assertEqual(by_uid(physical), by_uid(stable))

    def test_short_series_policies_and_empty_singleton(self):
        short = series(10)
        kept = select_slices(short, count=24)
        self.assertEqual(10, kept.actual_count)
        repeated = select_slices(short, count=24, short_series_policy="repeat_nearest")
        self.assertEqual(24, repeated.actual_count)
        self.assertEqual(10, len(set(repeated.selected_indices)))
        padded = select_slices(short, count=24, short_series_policy="pad_reference")
        self.assertEqual(24, padded.actual_count)
        self.assertEqual(14, padded.selected.count(None))
        self.assertEqual(14, padded.selected_indices.count(None))
        self.assertEqual(24, select_slices(series(1), count=24,
                         short_series_policy="repeat_nearest").actual_count)
        one = select_slices(series(1), count=24, short_series_policy="repeat_nearest")
        self.assertEqual((0,) * 24, one.selected_indices)
        self.assertEqual(0, select_slices([], count=24).actual_count)
        with self.assertRaises(ValueError):
            select_slices(short, count=24, short_series_policy="strict")

    def test_physical_fallback_partial_positions_and_disable(self):
        fallback = select_slices(series(10), strategy="physical_span", count=4)
        self.assertEqual("uniform", fallback.fallback)
        self.assertIsNone(fallback.selected_positions_mm[0])
        partial = series(4, [0, 1, None, 3])
        partial[2]["metadata"].pop("physical_position_mm")
        partial_result = select_slices(partial, strategy="physical_span", count=2)
        self.assertEqual("uniform", partial_result.fallback)
        self.assertIsNone(partial_result.physical_span_source)
        self.assertIsNone(partial_result.physical_span_selected)
        self.assertIsNone(partial_result.mean_selected_spacing)
        self.assertEqual(1.0, partial_result.coverage_fraction)
        with self.assertRaises(ValueError):
            select_slices(series(3), strategy="physical_span", count=2, allow_fallback=False)

    def test_duplicates_and_close_floats_are_deterministic(self):
        items = series(8, [0.0, 1.0, 1.0, 1.000000001, 2.0, 3.0, 4.0, 5.0])
        selected = select_slices(items, strategy="physical_span", count=5)
        again = select_slices(list(reversed(items)), strategy="physical_span", count=5)
        self.assertEqual(tuple(i["metadata"]["SOPInstanceUID"] for i in selected.selected),
                         tuple(i["metadata"]["SOPInstanceUID"] for i in again.selected))
        self.assertTrue(any("duplicate physical" in warning for warning in selected.warnings))
        selected_positions = [p for p in selected.selected_positions_mm if p is not None]
        self.assertEqual(1, sum(0.999 <= p <= 1.001 for p in selected_positions))

    def test_physical_sampling_preserves_order_and_endpoints(self):
        result = select_slices(series(4, [0.0, 1.0, 2.0, 100.0]),
                               strategy="physical_span", count=4)
        self.assertEqual((0, 1, 2, 3), result.selected_indices)
        self.assertEqual((0.0, 1.0, 2.0, 100.0), result.selected_positions_mm)
        self.assertEqual(100.0, result.last_selected_position)
        repeated = select_slices(series(3, [0.0, 0.0, 10.0]), strategy="physical_span",
                                 count=4, short_series_policy="repeat_nearest")
        self.assertEqual(0.0, repeated.first_selected_position)
        self.assertEqual(10.0, repeated.last_selected_position)
        self.assertEqual(1.0, repeated.coverage_fraction)
        duplicate_heavy = select_slices(series(7, [0, 1, 2, 90, 90, 90, 100]),
                                        strategy="physical_span", count=6)
        self.assertEqual((0, 1, 2, 90, 90, 100), duplicate_heavy.selected_positions_mm)
        near_endpoint = select_slices(series(5, [0, 1, 2, 3, 3.0001]),
                                      strategy="physical_span", count=3)
        self.assertEqual(3.0001, near_endpoint.last_selected_position)

    def test_image_position_requires_usable_orientation_for_physical_sampling(self):
        missing_orientation = [{"metadata": {"ImagePositionPatient": [0, 0, i]}}
                               for i in range(4)]
        fallback = select_slices(missing_orientation, strategy="physical_span", count=2)
        self.assertEqual("uniform", fallback.fallback)
        sagittal = [{"metadata": {"ImagePositionPatient": [0, -i, 0],
                    "ImageOrientationPatient": [1, 0, 0, 0, 0, 1],
                    "SOPInstanceUID": str(i)}} for i in range(4)]
        physical = select_slices(sagittal, strategy="physical_span", count=2)
        self.assertIsNone(physical.fallback)
        self.assertEqual(3.0, physical.physical_span_source)
        self.assertEqual(3.0, physical.physical_span_selected)

    def test_count_edges_and_preprocessing_identity(self):
        self.assertEqual((19,), select_slices(series(40), count=1).selected_indices)
        self.assertEqual(40, select_slices(series(40), count=40).actual_count)
        self.assertEqual(tuple(range(40)), select_slices(series(40, list(range(40))),
                         strategy="physical_span", count=40).selected_indices)
        center_position = select_slices(series(4, [0, 1, 50, 100]),
                                        strategy="physical_span", count=1)
        self.assertEqual((2,), center_position.selected_indices)
        for count in (16, 24, 32, 64):
            self.assertEqual(count, select_slices(series(80), count=count).actual_count)
        base = dict(name="data", version="v1", source_manifest_sha256="a" * 64,
                    preprocessing_version="p1", class_names=TARGET_REGISTRY.names)
        identity = lambda config: DatasetVersion(
            **base, preprocessing=config.to_preprocessing_spec()).dataset_version_id
        uniform24 = SliceSelectionConfig(strategy="uniform", count=24)
        self.assertNotEqual(identity(uniform24), identity(SliceSelectionConfig(strategy="uniform", count=32)))
        self.assertNotEqual(identity(uniform24), identity(SliceSelectionConfig(strategy="physical_span", count=24)))
        self.assertNotEqual(identity(uniform24), identity(SliceSelectionConfig(
            strategy="uniform", count=24, short_series_policy="repeat_nearest")))
        self.assertNotEqual(identity(SliceSelectionConfig(strategy="physical_span", count=24)),
                            identity(SliceSelectionConfig(strategy="physical_span", count=24,
                                                          allow_fallback=False)))
        self.assertNotEqual(identity(SliceSelectionConfig(strategy="physical_span", count=24)),
                            identity(SliceSelectionConfig(strategy="physical_span", count=24,
                                                          physical_position_tolerance_mm=0.01)))
        modified_spec = uniform24.to_preprocessing_spec()
        modified_spec["slice_selection"]["ordering_assumption"] = "different_geometry_order_contract"
        alternative = DatasetVersion(**base, preprocessing=modified_spec).dataset_version_id
        self.assertNotEqual(identity(uniform24), alternative)

    def test_series_and_slice_warnings_survive_selection_serialization(self):
        source = {"series_instance_uid": "series-1", "warnings": ["series geometry warning"],
                  "slices": [{"metadata": {"SOPInstanceUID": "sop-1",
                              "metadata_warnings": ["slice geometry warning"]},
                              "warnings": ["slice record warning"]}]}
        result = SliceSelector({"strategy": "uniform", "count": 1}).select(source)
        self.assertEqual(("series geometry warning", "slice record warning",
                          "slice geometry warning", "not all slices have valid image positions",
                          "ordering uses SOPInstanceUID or relative path"), result.warnings)
        self.assertEqual(list(result.warnings), result.to_dict()["warnings"])
        self.assertEqual("series-1", result.to_dict()["series_uid"])
        self.assertEqual("sop-1", result.to_dict()["selected_references"][0]["sop_instance_uid"])
        json.dumps(result.to_dict())

    def test_config_identity_covers_every_selection_setting(self):
        base = SliceSelectionConfig(strategy="physical_span", count=24)
        spec = base.to_preprocessing_spec()["slice_selection"]
        self.assertEqual("uniform", spec["fallback_policy"])
        self.assertIn("ordering_assumption", spec)
        equivalent = SliceSelectionConfig(strategy="physical_span", count=24,
            short_series_policy="keep_all", physical_position_tolerance_mm=1e-3,
            allow_fallback=True)
        self.assertEqual(base.to_preprocessing_spec(), equivalent.to_preprocessing_spec())
        self.assertEqual("geometry.order_series_slices; explicit_preprojected_mm_adapter",
                         spec["ordering_assumption"])
        self.assertIn("geometry_requirement", spec)
        self.assertEqual(SliceSelector(base).select([]).configuration_id,
                         SliceSelector(equivalent).select([]).configuration_id)
        operational = SliceSelectionConfig(strategy="physical_span", count=24,
            physical_position_tolerance_mm=1e-3, allow_fallback=True)
        self.assertEqual(SliceSelector(base).select([]).configuration_id,
                         SliceSelector(operational).select([]).configuration_id)
        self.assertEqual(
            SliceSelector(SliceSelectionConfig(strategy="uniform", count=24)).select([]).configuration_id,
            SliceSelector(SliceSelectionConfig(strategy="uniform", count=24,
                physical_position_tolerance_mm=0.5, allow_fallback=False)).select([]).configuration_id)

    def test_experiment_counts_for_each_strategy(self):
        positioned = series(80, [float(i * 2) for i in range(80)])
        for strategy in ("uniform", "center", "physical_span"):
            for count in (16, 24, 32):
                with self.subTest(strategy=strategy, count=count):
                    result = select_slices(positioned, strategy=strategy, count=count)
                    self.assertEqual(count, result.actual_count)
                    self.assertEqual(count, len(result.selected))

    def test_geometry_ordering_runs_before_physical_span_selection(self):
        slices = tuple(SliceRecord(f"slice-{i:02}.dcm", 10, {
            "SOPInstanceUID": f"sop-{i:02}", "ImagePositionPatient": [0, 0, float(i * 3)],
            "ImageOrientationPatient": [1, 0, 0, 0, 1, 0]}) for i in reversed(range(40)))
        series_record = SeriesRecord("series-geometry", "study-geometry", slices)
        expected_order = order_series_slices(series_record)
        result = SliceSelector(SliceSelectionConfig(strategy="physical_span", count=24)).select(series_record)
        self.assertEqual("geometry", result.geometry_ordering_method)
        self.assertEqual("high", result.geometry_confidence)
        self.assertEqual(117.0, result.physical_span_source)
        self.assertEqual(117.0, result.physical_span_selected)
        selected_ids = [item.slice_id for item in result.selected]
        ordered_ids = [item.slice_id for item in expected_order.slices]
        self.assertEqual(sorted(selected_ids, key=ordered_ids.index), selected_ids)
        self.assertEqual(24, len(selected_ids))
        self.assertTrue(all(item.metadata.get("pixel_array") is None for item in result.selected))

    def test_ordering_result_feeds_physical_span_selection(self):
        unordered = [SliceRecord(f"slice-{i:02}.dcm", 10, {
            "SOPInstanceUID": f"sop-{i:02}", "ImagePositionPatient": [0, 0, float(i * 5)],
            "ImageOrientationPatient": [1, 0, 0, 0, 1, 0]}) for i in reversed(range(40))]
        ordering = order_series_slices(unordered)
        result = SliceSelector(SliceSelectionConfig(strategy="physical_span", count=24)).select(ordering)
        self.assertEqual("geometry", result.geometry_ordering_method)
        self.assertEqual(195.0, result.physical_span_source)
        self.assertEqual(195.0, result.physical_span_selected)
        selected_z = [float(item.metadata["ImagePositionPatient"][2]) for item in result.selected]
        self.assertEqual(sorted(selected_z), selected_z)
        self.assertEqual(24, result.actual_count)
        json.dumps(result.to_dict())

    def test_ordering_result_is_reused_without_reprojecting(self):
        source = [SliceRecord(f"{i}.dcm", 10, {
            "SOPInstanceUID": str(i), "ImagePositionPatient": [0, 0, i],
            "ImageOrientationPatient": [1, 0, 0, 0, 1, 0]}) for i in (3, 0, 2, 1)]
        ordering = order_series_slices(SeriesRecord("series-1", "study-1", tuple(source)))
        with patch("rsna.data.selection.order_series_slices", side_effect=AssertionError("reprojection")):
            result = SliceSelector({"strategy": "physical_span", "count": 2}).select(ordering)
        self.assertEqual((0.0, 3.0), result.selected_positions_mm)
        self.assertEqual("series-1", result.series_uid)
        self.assertEqual("geometry", result.geometry_ordering_method)

    def test_serialized_dicom_geometry_overrides_conflicting_scalar_adapter(self):
        slices = [SliceRecord(f"slice-{i}.dcm", 10, {
            "SOPInstanceUID": str(i), "ImagePositionPatient": f"0\\0\\{i * 3}",
            "ImageOrientationPatient": "1\\0\\0\\0\\1\\0",
            "physical_position_mm": -i * 100}) for i in (3, 1, 2, 0)]
        record = SeriesRecord("series-serialized", "study-1", tuple(slices))
        result = SliceSelector({"strategy": "physical_span", "count": 2}).select(record.to_dict())
        self.assertEqual((0.0, 9.0), result.selected_positions_mm)
        self.assertEqual(("0", "3"), tuple(item["metadata"]["SOPInstanceUID"] for item in result.selected))
        self.assertTrue(all(item["slice_id"] for item in result.to_dict()["selected_references"]))
        json.dumps(result.to_dict())

    def test_inconsistent_orientation_and_slice_location_require_explicit_fallback(self):
        source = [{"metadata": {"SOPInstanceUID": str(i), "InstanceNumber": i,
            "ImagePositionPatient": [0, 0, i], "SliceLocation": i,
            "ImageOrientationPatient": [1, 0, 0, 0, 1, 0] if i < 2 else [1, 0, 0, 0, 0, 1]}}
                  for i in range(3)]
        result = select_slices(source, strategy="physical_span", count=2)
        self.assertEqual("uniform", result.fallback)
        self.assertEqual("slice_location", result.geometry_ordering_method)
        self.assertIsNone(result.physical_span_source)
        self.assertEqual((None, None), result.selected_positions_mm)
        self.assertTrue(any("orientations exceed" in warning for warning in result.warnings))
        with self.assertRaisesRegex(ValueError, "valid projected physical coordinates"):
            select_slices(source, strategy="physical_span", count=2, allow_fallback=False)

    def test_padding_physical_positions_and_geometry_rank_coverage(self):
        padded = select_slices(series(3, [0, 5, 10]), strategy="physical_span", count=5,
                               short_series_policy="pad_reference")
        self.assertEqual((0, 5, 10, None, None), padded.selected_positions_mm)
        self.assertEqual(10.0, padded.physical_span_selected)
        self.assertEqual(5.0, padded.mean_selected_spacing)
        self.assertEqual(1.0, padded.coverage_fraction)
        source = [{"metadata": {"SOPInstanceUID": str(i), "InstanceNumber": i}}
                  for i in (1, 0, 4, 3, 2)]
        result = select_slices(source, count=2)
        self.assertEqual((1, 2), result.selected_indices)
        self.assertEqual(1.0, result.coverage_fraction)

    def test_configuration_id_covers_strategy_count_and_fallback(self):
        base = SliceSelectionConfig(strategy="physical_span", count=24)
        identity = lambda config: SliceSelector(config).select([]).configuration_id
        for changed in (SliceSelectionConfig(strategy="uniform", count=24),
                        SliceSelectionConfig(strategy="physical_span", count=32),
                        SliceSelectionConfig(strategy="physical_span", count=24, allow_fallback=False)):
            self.assertNotEqual(identity(base), identity(changed))
        result = SliceSelector(base).select(series(3))
        self.assertEqual(identity(base), result.configuration_id)
        self.assertEqual("physical_span", result.parameters["strategy"])
        self.assertEqual(24, result.parameters["count"])
        self.assertEqual("uniform", result.fallback)

    def test_selector_never_reads_pixel_data_and_handles_multiple_series(self):
        class PixelGuard:
            metadata = {"SOPInstanceUID": "sop-no-pixels"}
            relative_path = "slice.dcm"

            @property
            def pixel_array(self):
                raise AssertionError("pixel data must not be accessed")

        selector = SliceSelector(SliceSelectionConfig(strategy="uniform", count=1))
        results = [selector.select({"series_instance_uid": uid, "slices": [PixelGuard()]})
                   for uid in ("series-a", "series-b")]
        self.assertEqual(["series-a", "series-b"], [result.to_dict()["series_uid"] for result in results])
        for result in results:
            json.dumps(result.to_dict())

    def test_cli_inspects_multiple_series_manifest_without_pixels(self):
        manifest = {"studies": [{"series": [
            {"series_instance_uid": uid, "slices": series(40, list(range(40)))}
            for uid in ("series-a", "series-b")] }]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                code = main(["select-slices", "--manifest", str(path),
                             "--strategy", "physical_span", "--count", "24"])
        result = json.loads(output.getvalue())
        self.assertEqual(0, code)
        self.assertEqual(2, result["series_count"])
        self.assertEqual({"series-a", "series-b"},
                         {entry["series_uid"] for entry in result["selections"]})


if __name__ == "__main__":
    unittest.main()
