import random
import unittest
from dataclasses import dataclass

from rsna.data.geometry import (GeometryConfig, cross_product, order_series_slices,
                               parse_orientation, project_position)


AXIAL = (1, 0, 0, 0, 1, 0)


@dataclass
class TestSlice:
    relative_path: str
    metadata: dict
    slice_id: str


@dataclass
class TestSeries:
    series_instance_uid: str
    slices: tuple


def slice_at(z=None, *, instance=None, location=None, orientation=AXIAL, uid=None):
    metadata = {"SOPInstanceUID": uid or f"uid-{instance}-{z}-{location}",
                "InstanceNumber": instance, "SliceLocation": location}
    if z is not None:
        metadata["ImagePositionPatient"] = (0, 0, z)
    if orientation is not None:
        metadata["ImageOrientationPatient"] = orientation
    path = f"slice-{metadata['SOPInstanceUID']}.dcm"
    return TestSlice(path, metadata, f"id-{path}")


class DicomGeometryTests(unittest.TestCase):
    def order(self, slices, config=None):
        return order_series_slices(TestSeries("series", tuple(slices)), config)

    def test_physical_order_overrides_instance_number_and_file_order(self):
        slices = [slice_at(10, instance=1), slice_at(0, instance=2), slice_at(5, instance=3)]
        result = self.order(slices)
        self.assertEqual([0, 5, 10], [s.metadata["ImagePositionPatient"][2] for s in result.slices])
        self.assertEqual("geometry", result.method)
        self.assertEqual(5, result.diagnostics.median_spacing)
        self.assertEqual(3, len(result.diagnostics.to_dict()["slice_coordinates"]))

    def test_axial_sagittal_coronal_normals(self):
        self.assertEqual((0, 0, 1), cross_product(*parse_orientation(AXIAL)))
        self.assertEqual((1, 0, 0), cross_product(*parse_orientation((0, 1, 0, 0, 0, 1))))
        self.assertEqual((0, -1, 0), cross_product(*parse_orientation((1, 0, 0, 0, 0, 1))))
        self.assertEqual(4, project_position("1\\2\\4", (0, 0, 1)))

    def test_geometry_order_is_independent_of_input_permutation(self):
        base = [slice_at(z, instance=10 - z) for z in (12, -1, 4, 8, 2)]
        expected = [item.slice_id for item in self.order(base).slices]
        for seed in range(10):
            shuffled = list(base)
            random.Random(seed).shuffle(shuffled)
            self.assertEqual(expected, [item.slice_id for item in self.order(shuffled).slices])

    def test_noise_spacing_irregularity_and_duplicates_are_reported(self):
        noisy = (1.000001, 0, 0, 0, 0.999999, 0)
        result = self.order([slice_at(z, orientation=noisy) for z in (0, 1, 2, 10, 10)])
        codes = {warning.code for warning in result.warnings}
        self.assertIn("duplicate_positions", codes)
        self.assertIn("irregular_spacing", codes)
        self.assertEqual(5, len(result.slices))

    def test_inconsistent_orientation_warns_or_fails_strict(self):
        result = self.order([slice_at(0), slice_at(1, orientation=(0, 1, 0, 0, 0, 1))])
        self.assertIn("inconsistent_orientation", {w.code for w in result.warnings})
        with self.assertRaises(ValueError):
            self.order([slice_at(0), slice_at(1, orientation=(0, 1, 0, 0, 0, 1))],
                       GeometryConfig(strict_geometry=True))

    def test_both_direction_cosines_reversed_still_warns(self):
        in_plane_reversed = (-1, 0, 0, 0, -1, 0)
        result = self.order([slice_at(0), slice_at(1, orientation=in_plane_reversed)])
        self.assertEqual((0, 0, 1), result.diagnostics.normal_vector)
        self.assertFalse(result.diagnostics.orientation_consistency["consistent"])
        self.assertIn("inconsistent_orientation", {w.code for w in result.warnings})

    def test_invalid_and_partial_positions_use_documented_fallbacks(self):
        partial = self.order([slice_at(0, instance=2), slice_at(None, instance=1)])
        self.assertEqual("instance_number", partial.method)
        malformed = slice_at(0)
        malformed.metadata["ImagePositionPatient"] = (float("nan"), 0, 0)
        self.assertIn("invalid_position", {w.code for w in self.order([malformed]).warnings})

    def test_position_only_uses_series_orientation_when_available(self):
        values = [slice_at(0), slice_at(2), slice_at(1)]
        values[-1].metadata.pop("ImageOrientationPatient")
        result = self.order(values)
        self.assertEqual("position_inferred_normal", result.method)
        self.assertEqual([0, 1, 2], [x.metadata["ImagePositionPatient"][2] for x in result.slices])

    def test_duplicate_sop_uid_is_warned_and_ties_are_deterministic(self):
        left = TestSlice("a.dcm", {"SOPInstanceUID": "duplicate", "ImagePositionPatient": (0, 0, 1),
                                   "ImageOrientationPatient": AXIAL}, "a.dcm")
        right = TestSlice("b.dcm", {"SOPInstanceUID": "duplicate", "ImagePositionPatient": (0, 0, 1),
                                    "ImageOrientationPatient": AXIAL}, "b.dcm")
        result = self.order([right, left])
        codes = {warning.code for warning in result.warnings}
        self.assertIn("duplicate_sop_instance_uid", codes)
        self.assertIn("duplicate_positions", codes)
        self.assertEqual(["a.dcm", "b.dcm"], [item.slice_id for item in result.slices])
        self.assertEqual([item.slice_id for item in result.slices],
                         [item.slice_id for item in self.order([left, right]).slices])
        with self.assertRaises(ValueError):
            self.order([left, right], GeometryConfig(duplicate_policy="strict"))

    def test_duplicate_policies_and_non_geometry_spacing_metadata(self):
        values = [slice_at(0, instance=1), slice_at(0, instance=2)]
        values[0].metadata.update({"SliceThickness": "2.5", "SpacingBetweenSlices": "3",
                                   "PixelSpacing": "0.4\\0.6"})
        keep = self.order(values, GeometryConfig(duplicate_policy="keep_all"))
        self.assertEqual(2, len(keep.slices))
        self.assertNotIn("duplicate_positions", {warning.code for warning in keep.warnings})
        self.assertNotIn("zero_spacing", {warning.code for warning in keep.warnings})
        self.assertEqual([2.5, None], keep.diagnostics.to_dict()["spacing_metadata"]["slice_thickness_mm"])
        self.assertEqual([0.4, 0.6], keep.diagnostics.to_dict()["spacing_metadata"]["pixel_spacing_mm"][0])
        warn = self.order(values)
        self.assertIn("zero_spacing", {warning.code for warning in warn.warnings})
        values[0].metadata["PixelSpacing"] = "0\\-1"
        invalid = self.order(values)
        self.assertIn("invalid_spacing_metadata", {warning.code for warning in invalid.warnings})

    def test_spacing_metadata_is_reported_without_driving_slice_distance(self):
        item = slice_at(0)
        item.metadata.update({"SliceThickness": "4", "SpacingBetweenSlices": "-3",
                              "PixelSpacing": "0\\0.6", "Rows": 1, "Columns": 256})
        result = self.order([item])
        summary = result.diagnostics.to_dict()["spacing_metadata"]
        self.assertEqual([4.0], summary["slice_thickness_mm"])
        self.assertEqual([-3.0], summary["spacing_between_slices_mm"])
        self.assertEqual([0.0, 0.6], summary["pixel_spacing_mm"][0])
        self.assertIn("signed_spacing_between_slices", {warning.code for warning in result.warnings})
        self.assertEqual(None, result.diagnostics.median_spacing)

    def test_stable_fallback_uses_relative_path_without_uid(self):
        first = TestSlice("z.dcm", {}, "")
        second = TestSlice("a.dcm", {}, "")
        result = self.order([first, second])
        self.assertEqual("stable_fallback", result.method)
        self.assertEqual(["a.dcm", "z.dcm"], [item.relative_path for item in result.slices])

    def test_slice_location_instance_number_and_stable_fallbacks(self):
        self.assertEqual("slice_location", self.order([
            slice_at(None, instance=2, location=2), slice_at(None, instance=1, location=1)]).method)
        instance_values = [slice_at(None, instance=2), slice_at(None, instance=1)]
        instance_result = self.order(instance_values)
        self.assertEqual("instance_number", instance_result.method)
        self.assertEqual([item.slice_id for item in instance_result.slices],
                         [item.slice_id for item in self.order(list(reversed(instance_values))).slices])
        result = self.order([slice_at(None, uid="b"), slice_at(None, uid="a")])
        self.assertEqual("stable_fallback", result.method)
        self.assertEqual(["a", "b"], [s.metadata["SOPInstanceUID"] for s in result.slices])
        self.assertEqual(result.to_dict(), self.order(list(reversed(result.slices))).to_dict())

    def test_orientation_validation_and_diagnostic_json(self):
        with self.assertRaises(ValueError):
            cross_product((0, 0, 0), (0, 0, 0))
        invalid = slice_at(0, orientation=(0, 0, 0, 0, 0, 0))
        result = self.order([invalid])
        self.assertIn("invalid_orientation", {w.code for w in result.warnings})
        self.assertIsInstance(result.to_dict()["diagnostics"]["warnings"], list)

    def test_nonfinite_cross_product_input_is_rejected(self):
        with self.assertRaises(ValueError):
            cross_product((float("inf"), 0, 0), (0, 1, 0))


if __name__ == "__main__":
    unittest.main()
