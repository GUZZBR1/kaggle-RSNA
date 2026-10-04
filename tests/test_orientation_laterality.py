import json
import unittest

from rsna.data import (OrientationConfig, assess_orientation_consistency,
                       build_orientation_provenance,
                       build_normalization_plan, describe_series_orientation,
                       parse_laterality_text, resolve_series_laterality,
                       resolve_study_laterality)


AXIAL = [1, 0, 0, 0, 1, 0]
SAGITTAL = [0, 1, 0, 0, 0, 1]
CORONAL = [1, 0, 0, 0, 0, 1]


class OrientationTests(unittest.TestCase):
    def test_canonical_planes_and_reversed_normal(self):
        self.assertEqual("axial", describe_series_orientation({"ImageOrientationPatient": AXIAL}).plane)
        self.assertEqual("sagittal", describe_series_orientation({"ImageOrientationPatient": SAGITTAL}).plane)
        self.assertEqual("coronal", describe_series_orientation({"ImageOrientationPatient": CORONAL}).plane)
        reversed_normal = [-1, 0, 0, 0, 1, 0]
        self.assertEqual("axial", describe_series_orientation({"ImageOrientationPatient": reversed_normal}).plane)
        self.assertEqual(-1.0, describe_series_orientation({"ImageOrientationPatient": reversed_normal}).normal[2])

    def test_obliquity_tolerance_boundary(self):
        angle = 10
        import math
        oblique = [1, 0, 0, 0, math.cos(math.radians(angle)), math.sin(math.radians(angle))]
        self.assertEqual("axial", describe_series_orientation(
            {"ImageOrientationPatient": oblique}, OrientationConfig(plane_tolerance_deg=15)).plane)
        self.assertEqual("oblique", describe_series_orientation(
            {"ImageOrientationPatient": oblique}, OrientationConfig(plane_tolerance_deg=5)).plane)

    def test_missing_malformed_and_nonorthogonal_orientation(self):
        self.assertEqual("unknown", describe_series_orientation({}).plane)
        for value in ([1, 0], [0, 0, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0], [float("nan"), 0, 0, 0, 1, 0]):
            descriptor = describe_series_orientation({"ImageOrientationPatient": value})
            self.assertEqual("unknown", descriptor.plane, value)
            self.assertEqual("unknown", descriptor.confidence)

    def test_orientation_consistency_and_determinism(self):
        near = [1, 0, 0, 0, 0.99999, 0.004]
        status, deviation, _ = assess_orientation_consistency([AXIAL, near])
        self.assertIn(status, {"consistent", "minor_variation"})
        self.assertIsNotNone(deviation)
        severe = [0, 1, 0, 0, 0, 1]
        descriptor = describe_series_orientation([{"ImageOrientationPatient": AXIAL},
                                                  {"ImageOrientationPatient": severe}])
        self.assertEqual("inconsistent", descriptor.consistency)
        first = describe_series_orientation({"ImageOrientationPatient": AXIAL}).to_dict()
        self.assertEqual(first, describe_series_orientation({"ImageOrientationPatient": AXIAL}).to_dict())
        json.dumps(first, sort_keys=True, allow_nan=False)


class LateralityTests(unittest.TestCase):
    def test_structured_values_and_text_variants(self):
        self.assertEqual("LEFT", resolve_series_laterality({"Laterality": "L"}).resolved)
        self.assertEqual("RIGHT", resolve_series_laterality({"ImageLaterality": "R"}).resolved)
        concordant = resolve_series_laterality({"Laterality": "R", "SeriesDescription": "RIGHT KNEE SAG PD"})
        self.assertEqual("RIGHT", concordant.resolved)
        self.assertEqual("high", concordant.confidence)
        for text in ("LEFT KNEE", "R KNEE", "RT KNEE", "LT KNEE", "RIGHT KNEE"):
            self.assertIn(parse_laterality_text(text), {"LEFT", "RIGHT"})
        self.assertEqual("LEFT", resolve_series_laterality({"SeriesDescription": "LEFT KNEE"}).resolved)
        self.assertEqual("UNKNOWN", resolve_series_laterality({}).resolved)

    def test_text_false_positives_and_conflicts(self):
        for text in ("BRIGHT PD", "RIGHTEOUS", "CORONAL PD", "T1 MAPPING"):
            self.assertIsNone(parse_laterality_text(text), text)
        conflict = resolve_series_laterality({"Laterality": "R", "SeriesDescription": "LEFT KNEE"})
        self.assertEqual("AMBIGUOUS", conflict.resolved)
        self.assertIn("conflicting", " ".join(conflict.warnings))
        malformed = resolve_series_laterality({"Laterality": "RIGHTISH"})
        self.assertEqual("UNKNOWN", malformed.resolved)
        self.assertTrue(malformed.warnings)

    def test_slice_and_study_resolution(self):
        same = resolve_series_laterality([{"ImageLaterality": "L"}, {"Laterality": "L"}])
        self.assertEqual("LEFT", same.resolved)
        self.assertEqual("LEFT", resolve_study_laterality([
            {"Laterality": "L"}, {"ImageLaterality": "LEFT"}]).resolved)
        self.assertEqual("AMBIGUOUS", resolve_study_laterality([
            {"Laterality": "L"}, {"Laterality": "R"}]).resolved)
        partial = resolve_study_laterality([{"Laterality": "L"}, {}])
        self.assertEqual("LEFT", partial.resolved)
        self.assertEqual("low", partial.confidence)


class NormalizationTests(unittest.TestCase):
    def test_default_is_preserve_native_and_transform_is_only_planned(self):
        source = {"Laterality": "R", "ImageOrientationPatient": SAGITTAL}
        plan = build_normalization_plan(source)
        self.assertEqual("preserve_native", plan.mode)
        self.assertFalse(plan.requires_flip)
        planned = build_normalization_plan(source, config=OrientationConfig(normalization_mode="left_canonical"))
        self.assertTrue(planned.requires_flip)
        self.assertEqual("unresolved", planned.flip_axis)
        self.assertIn("planned", " ".join(planned.warnings))
        json.dumps(planned.to_dict(), sort_keys=True, allow_nan=False)
        provenance = build_orientation_provenance(source, OrientationConfig(normalization_mode="left_canonical"))
        self.assertEqual("sagittal", provenance.resolved_plane)
        self.assertEqual("R", provenance.original_laterality[0]["Laterality"])
        json.dumps(provenance.to_dict(), sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
