import unittest
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile

from rsna.contracts import DatasetVersion
from rsna.data.selection import SliceSelectionConfig, SliceSelector, select_slices
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
                    preprocessing_version="p1", class_names=("normal", "abnormal"))
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

    def test_series_and_slice_warnings_survive_selection_serialization(self):
        source = {"series_instance_uid": "series-1", "warnings": ["series geometry warning"],
                  "slices": [{"metadata": {"SOPInstanceUID": "sop-1",
                              "metadata_warnings": ["slice geometry warning"]},
                              "warnings": ["slice record warning"]}]}
        result = SliceSelector({"strategy": "uniform", "count": 1}).select(source)
        self.assertEqual(("series geometry warning", "slice record warning",
                          "slice geometry warning"), result.warnings)
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
        self.assertEqual("explicit_mm_or_projected_ipp; tie_break_sop_path; else_preserve_input_order",
                         spec["ordering_assumption"])

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
