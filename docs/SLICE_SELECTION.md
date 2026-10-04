# MRI slice selection

The selector consumes canonical `SeriesRecord`/`SliceRecord` values or a geometry `OrderingResult` for one series. It does not read pixel data. It calls `geometry.order_series_slices` for raw inputs and reuses a supplied result directly. Physical coordinates and source ordering come only from the canonical Geometry result; scalar position fields outside Geometry are ignored.

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

`uniform` distributes references over the ordered index range, including both endpoints when selecting multiple slices. `center` takes a central contiguous window; when two windows are equally centered, it chooses the lower-index window (equivalently, an odd leftover margin leaves the extra slice after the window). `physical_span` places targets evenly between the first and last physical positions and assigns the nearest real slices. Geometry owns duplicate-position handling through `GeometryConfig`; Selection does not regroup or discard positions. Results include chosen indices, positions, SOP UIDs when supplied, physical spans, warnings and fallback provenance.

| Strategy | Sampling space | Typical use |
| --- | --- | --- |
| `uniform` | Ordered slice index | Simple coverage baseline |
| `center` | Ordered slice index | Central anatomy baseline |
| `physical_span` | Millimeters | Reduce sensitivity to irregular spacing |

`physical_span` falls back explicitly to `uniform` if any slice lacks usable position metadata. Set `allow_fallback=false` to fail instead. Partial geometry is never described as physical sampling: physical span and spacing metrics are omitted, while the result retains any known per-slice positions and computes coverage in index space. The geometry module orders slices before every strategy, including its `SliceLocation`, `InstanceNumber`, and stable identifier fallbacks. Those fallback keys are not treated as validated physical coordinates. Index coverage is computed in this ordered series; selected indices still refer to the source input. Padding references are excluded from physical span and spacing metrics. Geometry tolerances and duplicate policies are configured through `GeometryConfig` and are included in selection and dataset preprocessing identity.

For a short series, `keep_all` returns all available references. `repeat_nearest` returns exactly the requested number by deterministic endpoint-inclusive resampling, reusing references without copying files; for an empty series it returns no real refs and records a warning. For `physical_span`, targets remain uniform in millimeters and map to nearest available physical positions; for index-based strategies, targets are uniform across the ordered index range. `pad_reference` returns available slices followed by explicitly tagged `padding_reference` entries with no DICOM identity. `strict` raises an error, including for an empty series. One-slice series and count one are supported.

Selection strategy, count, short-series policy, ordering assumption, and the full `GeometryConfig` are part of preprocessing identity through `SliceSelectionConfig.to_preprocessing_spec()`. The physical fallback policy is included only for `physical_span`. Pass the JSON-compatible mapping as `DatasetVersion.preprocessing`; the existing canonical hash includes it. Each result records a canonical `configuration_id`, geometry method and confidence, source and ordered counts, coverage basis, lineage IDs, the realized fallback, and selected references. For example:

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
