# MRI slice selection

The selector consumes slice records or metadata mappings for one series. It does not read pixel data. Call it after geometry ordering when that stage is available; otherwise input order is treated as the caller's preordered reference order. When complete physical positions are present, the selector sorts by position (DICOM `ImagePositionPatient` projected onto the slice normal when `ImageOrientationPatient` is present) with a stable SOP/path tie-breaker.

```text
ordered series
     ↓
selection strategy
     ↓
N selected references
     ↓
selection provenance
     ↓
preprocessing identity
```

```python
from rsna.data.selection import SliceSelectionConfig, SliceSelector

selector = SliceSelector(SliceSelectionConfig(strategy="physical_span", count=24))
result = selector.select(series_record)
manifest_entry = result.to_dict()
```

`uniform` distributes references over the ordered index range, including both endpoints when selecting multiple slices. `center` takes a central contiguous window; when the remaining margin is odd, it places the extra slice after the window. `physical_span` places targets evenly between the first and last physical positions and assigns the nearest real slices. Positions within the configured tolerance are treated as equivalent when avoidable, but are never removed from the source. Results include chosen indices, positions, SOP UIDs when supplied, physical spans, mean selected spacing, warnings and fallback provenance.

| Strategy | Sampling space | Typical use |
| --- | --- | --- |
| `uniform` | Ordered slice index | Simple coverage baseline |
| `center` | Ordered slice index | Central anatomy baseline |
| `physical_span` | Millimeters | Reduce sensitivity to irregular spacing |

`physical_span` falls back explicitly to `uniform` if any slice lacks usable position metadata. Set `allow_fallback=false` to fail instead. Partial geometry is never described as physical sampling. Input order is retained for index-based strategies; callers should run geometry ordering first if their records are not ordered.

For a short series, `keep_all` returns all available references. `repeat_nearest` returns exactly the requested number by deterministic endpoint-inclusive resampling, reusing references without copying files. `pad_reference` returns the available references followed by explicit `null`/`None` placeholders for downstream padding. `strict` raises an error. Zero slices produce an empty result. One-slice series and count one are supported.

Selection parameters are part of preprocessing identity when stored in `DatasetVersion.preprocessing`; that mapping is included in `dataset_version_id`'s canonical hash. For example:

```python
DatasetVersion(..., preprocessing={"slice_selection": {
    "strategy": "physical_span", "count": 24,
    "short_series_policy": "repeat_nearest",
    "physical_position_tolerance_mm": 0.001, "allow_fallback": True,
}})
```

The `configs/slice-selection/` examples show 16, 24 and 32-slice settings. To inspect a saved dataset index without pixels:

```bash
python -m rsna select-slices --manifest dataset-index.json --strategy physical_span --count 24
```

The CLI prints one selection summary per series. It accepts the indexed manifest shape with `studies[].series[].slices[]`.
