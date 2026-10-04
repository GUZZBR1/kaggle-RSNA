# DICOM geometry and slice ordering

`rsna.data.geometry.order_series_slices(series)` accepts a `SeriesRecord` (or an
iterable of slice records) whose DICOM attributes are in each slice's `metadata`.
It returns ordered slices together with the method, confidence, structured warnings,
and a serializable geometry summary. Metadata is already loaded; the ordering code
does not read DICOM files or pixel data.

## Physical coordinate

`ImageOrientationPatient` stores the row and column direction cosines. The code
validates and normalizes their small numeric deviations, then computes
`normal = row × column`. For each `ImagePositionPatient`, it calculates
`coordinate = dot(position, normal)`. Slices are always sorted by ascending
projected patient-space coordinate, independent of input and filename order. The
normal's direction is preserved, so opposite direction cosines reverse the
coordinate convention; they are reported as inconsistent within a series.

## Ordering tiers

One tier is selected for the whole series; partial coordinate lists are never
combined with instance or filename keys:

1. `geometry`: every slice has valid position and compatible orientation.
2. `position_inferred_normal`: every slice has position and at least one
   compatible orientation is available elsewhere in the series.
3. `slice_location`: every slice has a finite, non-constant `SliceLocation`.
4. `instance_number`: every slice has a finite `InstanceNumber`.
5. `stable_fallback`: `SOPInstanceUID`, then relative path / stable slice ID.

`SliceLocation` is only a fallback. Invalid or inconsistent orientation prevents
geometry ordering; ordinary mode reports a structured warning and tries a lower
tier, while `strict_geometry=True` raises. Duplicate physical positions are
retained and warned by default. `duplicate_policy="deduplicate_exact"` opts in to
collapsing positions within the configured tolerance; `"strict"` raises.

## Diagnostics and tolerances

`GeometryConfig` centralizes angular tolerance, position and duplicate tolerances,
spacing outlier factor, strictness, and duplicate policy. `SeriesGeometry` reports
normal, coordinate range/span, spacing statistics, orientation consistency,
duplicate positions, and warnings. `OrderingResult.to_dict()` emits plain JSON-safe
containers suitable for provenance logs.

`SliceThickness` and `SpacingBetweenSlices` are retained as source metadata, but
are not substituted for measured distances between adjacent projected positions.
Spacing statistics ignore coincident positions. No orientation canonicalization,
laterality flip, pixel decoding, or slice selection is performed here.
