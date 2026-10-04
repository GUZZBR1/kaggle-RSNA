# MRI slice selection

The selector consumes slice records or metadata mappings for one series. It does not read pixel data. Call it after geometry ordering when that stage is available; otherwise input order is used when physical ordering cannot be established. Complete physical positions are sorted by a scalar millimeter position or by projecting DICOM `ImagePositionPatient` onto the slice normal computed from `ImageOrientationPatient`; ties use SOP UID and path. Missing or invalid orientation does not use the z component as a proxy: `physical_span` falls back unless an upstream scalar position is supplied.

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

`physical_span` falls back explicitly to `uniform` if any slice lacks usable position metadata. Set `allow_fallback=false` to fail instead. Partial geometry is never described as physical sampling: physical span and spacing metrics are omitted, while the result retains any known per-slice positions and computes coverage in index space. When complete positions exist, the selector orders slices by physical position before applying every strategy; otherwise index-based strategies preserve caller order, so callers should run geometry ordering first.

For a short series, `keep_all` returns all available references. `repeat_nearest` returns exactly the requested number by deterministic endpoint-inclusive resampling, reusing references without copying files. For `physical_span`, targets remain uniform in millimeters and map to nearest available physical positions; for index-based strategies, targets are uniform across the ordered index range. `pad_reference` returns the available references followed by explicit `null`/`None` placeholders for downstream padding. `strict` raises an error. Zero slices produce an empty result. One-slice series and count one are supported.

Selection parameters, fallback policy, and ordering assumption are part of preprocessing identity through `SliceSelectionConfig.to_preprocessing_spec()`. Pass that JSON-compatible mapping as `DatasetVersion.preprocessing`; the existing canonical hash includes it. The realized fallback and selected references remain in each result manifest. For example:

```python
DatasetVersion(..., preprocessing=SliceSelectionConfig(
    strategy="physical_span", count=24, short_series_policy="repeat_nearest",
).to_preprocessing_spec())
```

The `configs/slice-selection/` examples show 16, 24 and 32-slice settings. To inspect a saved dataset index without pixels:

```bash
python -m rsna select-slices --manifest dataset-index.json --strategy physical_span --count 24
```

The CLI prints one selection summary per series. It accepts the indexed manifest shape with `studies[].series[].slices[]`.
