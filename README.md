# Keigo Kaggriculture

Cloud-first research infrastructure for candidate generation, Kaggriculture simulations,
evaluation, tournaments, training and reproducible Kaggle artifacts.

This repository is intentionally a clean foundation. It does not carry forward historical
agent policies or competition tuning. The first executable path uses an injected local
runner or a deterministic mock provider; production simulation requires a configured cloud
or Ray provider and the official environment integration.

## Quick start

```bash
python -m unittest discover -s tests
python -m keigo configs/experiments/smoke.toml
```

Python 3.11 or newer is required. The core has no third-party runtime dependencies. Install
the optional Ray extra only for a Ray cluster:

```bash
pip install -e '.[ray]'
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for boundaries and [docs/MIGRATION.md](docs/MIGRATION.md)
for the source audit and migration decisions.
