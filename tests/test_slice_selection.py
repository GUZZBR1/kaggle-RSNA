import unittest

from rsna.contracts import DatasetVersion
from rsna.data.selection import SliceSelector, select_slices


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
        with self.assertRaises(ValueError):
            select_slices(series(3), strategy="physical_span", count=2, allow_fallback=False)

    def test_duplicates_and_close_floats_are_deterministic(self):
        items = series(8, [0.0, 1.0, 1.0, 1.000000001, 2.0, 3.0, 4.0, 5.0])
        selected = select_slices(items, strategy="physical_span", count=5)
        again = select_slices(list(reversed(items)), strategy="physical_span", count=5)
        self.assertEqual(tuple(i["metadata"]["SOPInstanceUID"] for i in selected.selected),
                         tuple(i["metadata"]["SOPInstanceUID"] for i in again.selected))
        self.assertTrue(any("duplicate physical" in warning for warning in selected.warnings))

    def test_physical_sampling_preserves_order_and_endpoints(self):
        result = select_slices(series(4, [0.0, 1.0, 2.0, 100.0]),
                               strategy="physical_span", count=4)
        self.assertEqual((0, 1, 2, 3), result.selected_indices)
        self.assertEqual((0.0, 1.0, 2.0, 100.0), result.selected_positions_mm)
        self.assertEqual(100.0, result.last_selected_position)

    def test_count_edges_and_preprocessing_identity(self):
        self.assertEqual((19,), select_slices(series(40), count=1).selected_indices)
        self.assertEqual(40, select_slices(series(40), count=40).actual_count)
        for count in (16, 24, 32, 64):
            self.assertEqual(count, select_slices(series(80), count=count).actual_count)
        base = dict(name="data", version="v1", source_manifest_sha256="a" * 64,
                    preprocessing_version="p1", class_names=("normal", "abnormal"))
        identity = lambda strategy, count: DatasetVersion(**base, preprocessing={
            "slice_selection": {"strategy": strategy, "count": count,
                "short_series_policy": "keep_all", "allow_fallback": True}}).dataset_version_id
        self.assertNotEqual(identity("uniform", 24), identity("uniform", 32))
        self.assertNotEqual(identity("uniform", 24), identity("physical_span", 24))


if __name__ == "__main__":
    unittest.main()
