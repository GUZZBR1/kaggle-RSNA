# DICOM orientation, laterality and normalization provenance

This layer audits metadata only. It does not decode pixels or apply flips,
rotations, resampling, or intensity operations.

```text
DICOM geometry
      ↓
plane resolution

DICOM structured fields
      +
text fallback
      ↓
laterality resolution

orientation + laterality
      ↓
normalization plan
      ↓
provenance
```

## Geometry and plane convention

`ImageOrientationPatient` contains six DICOM direction cosines: the first three
describe increasing image column index (the direction of a row), and the next
three describe increasing image row index (the direction of a column). Both are
expressed in the patient coordinate system. For the usual BIPED convention the
axes are positive toward patient left, posterior, and head. The normalized cross
product is the slice normal. Axial, coronal, and sagittal labels are determined
by the dominant absolute normal component (head/superior, anterior/posterior,
and left/right respectively). This makes plane classification invariant to the
sign of the normal; the descriptor retains the signed normal for physical
direction and ordering work. Normals more than `plane_tolerance_deg` from their
dominant cardinal axis are labeled `oblique`. Missing, malformed, non-unit, or
non-orthogonal vectors are `unknown`.

The default plane tolerance is 15 degrees. Slice consistency checks compare both
in-plane direction cosines and the signed normal. This detects grid rotations
within the slice plane and opposite slice directions.
`orientation_consistency_tolerance_deg` defaults to 1 degree and
`minor_variation_deg` to 0.25 degrees. Status is `consistent`,
`minor_variation`, `inconsistent`, or `unknown`.

`PatientPosition` (for example, HFS) describes patient pose. It is not used to
override orientation geometry or to infer laterality on its own.

## Laterality and conflicts

`Laterality` and `ImageLaterality` are primary structured evidence. The accepted
values are `L`/`LEFT`/`LT`, `R`/`RIGHT`/`RT`, and explicit bilateral values.
Fallback parsing examines token boundaries in `SeriesDescription`,
`ProtocolName`, and `StudyDescription`; it does not match substrings such as
`BRIGHT`. Structured evidence conflicting with text, or conflicting structured
fields, resolves to `AMBIGUOUS` by default. It never picks a winner based on
count. Evidence items retain source, original value, normalized value, and
weight. Study resolution aggregates all series: incompatible sides yield
`AMBIGUOUS`.

Text-only decisions have low confidence, structured decisions have high
confidence, and missing or conflicting decisions have unknown confidence.
Invalid structured values are reported as warnings and do not become evidence.

## Normalization and identity

`OrientationConfig.normalization_mode` defaults to `preserve_native`; this
produces a plan with no requested pixel transformation. `left_canonical` is an
opt-in planning mode: RIGHT becomes a declared reflection request and LEFT
requires none. No operation is executed. This metadata alone does not alter raw
dataset content identity. If a future preprocessing run applies the plan, its
effective mode and transform parameters must be recorded in the
`DatasetVersion.preprocessing` configuration, which participates in the
dataset identity. The pixel-array axis remains unresolved unless the future
pixel geometry proves which array direction corresponds to patient left/right.

All descriptors and plans expose `to_dict()` and contain JSON-safe scalar,
list, and mapping values suitable for a dataset manifest. Main entry points are
`describe_series_orientation(series)`, `resolve_series_laterality(series)`,
`resolve_study_laterality(series_list)`, and `build_normalization_plan(series)`.
