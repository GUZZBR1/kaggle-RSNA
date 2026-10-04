# Dataset CLI

The CLI reads metadata manifests and JSON artifacts without decoding DICOM pixel
arrays. The dataset manifest adapter accepts JSON records containing studies,
series, and slices; records may be nested or supplied as top-level collections.
See the manifest's original producer for which metadata fields are available.

```bash
python -m rsna --help
python -m rsna data summary --manifest artifacts/dataset-index/manifest.json
python -m rsna data validate --manifest artifacts/dataset-index/manifest.json
python -m rsna data validate --manifest artifacts/dataset-index/manifest.json --level full --json
python -m rsna data inspect-study --manifest manifest.json --study-id 1.2.840...
python -m rsna data inspect-series --manifest manifest.json --series-id 1.2.840...
python -m rsna data inspect-slice --manifest manifest.json --sop-id 1.2.840...
python -m rsna data inspect-manifest --manifest manifest.json --json
python -m rsna data sample --manifest manifest.json --entity series --count 20 --seed 42
python -m rsna data stats --manifest manifest.json --plane sagittal --laterality right --json
python -m rsna artifact validate artifacts/dataset-index/manifest.json
```

Every inspection command supports `--format human|json` (or `--json`). Validation
reports distinguish errors, warnings, and informational findings. Exit status is
`0` for a pass, `1` for validation failure or a missing entity, `2` for invalid
CLI usage, and `3` for unreadable or corrupted input. JSON results go to stdout;
operational errors go to stderr. `--debug` adds a traceback for operational errors.

An optional TOML config can provide defaults in a `[cli]` section, such as
`output_format = "json"`, `manifest = "manifest.json"`, and `validation_level =
"basic"`. Explicit command line flags take precedence. `--quiet` suppresses
informational logging; result output remains available for automation.

The original smoke invocation remains supported:

```bash
python -m rsna configs/experiments/smoke.toml
```
