# Command line

The legacy smoke runner remains available as `python -m rsna CONFIG.toml`. Fold-plan management is available under `python -m rsna folds`:

```text
generate  create, validate, summarize, and optionally save assignments
validate  validate plan identity, dataset match, coverage, and leakage
inspect   show plan metadata, fold statistics, or study/patient lookup
stats     report per-fold counts and balance diagnostics
diff      compare assignments and detect semantic split equality
export    write assignments as CSV or JSON
import    validate external CSV assignments and bind them to a dataset
lock      mark a plan as frozen for experiment reuse
```

Examples and the recommended freeze-and-reuse workflow are in [FOLDS.md](FOLDS.md). Add `--json` to any folds command for machine-readable output. Commands return `0` for success, `1` for validation failure, `2` for usage/output conflict, `3` for corrupt input, and `4` for dataset mismatch.
