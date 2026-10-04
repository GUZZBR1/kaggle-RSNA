# Architecture

## First execution path

`ExperimentSpec -> Candidate -> Provider -> SimulationJob -> SimulationResult -> Evaluation -> Artifact`

The core records are immutable dataclasses with schema versions and SHA-256 identities.
`ExperimentSpec` determines the planned jobs; each job includes candidate, opponent, seed,
seat and simulator version. A provider returns an execution ID and later a result carrying
both deterministic job identity and provider provenance. Evaluation rejects failed,
duplicated, unexpected or missing seed/opponent/seat results before writing its artifact.

## Boundaries

- `keigo/contracts.py` defines versioned data contracts and deterministic IDs.
- `keigo/experiments/` validates, expands and orchestrates experiment jobs.
- `keigo/candidates/` owns candidate identity. Agent policy source remains an artifact,
  outside the orchestration core.
- `keigo/providers/` owns compute submission and result retrieval. `LocalProvider` is for
  smoke tests; `MockProvider` is deterministic test infrastructure; `RayProvider` connects
  to an already available Ray runtime; `CloudProvider` adapts an injected cloud transport.
- `keigo/evaluation/` consumes only declared results. Promotion policy belongs in explicit
  configuration and will evolve into champion/challenger gates.
- `keigo/artifacts/` provides content-addressed JSON artifacts for local development.
  Production deployments should inject an object-store implementation.
- `keigo/training/` defines the future trainer boundary; GPU training is a workload, not a
  provider concern.
- `keigo/tournament/` provides stable ordering for paired seat and opponent workloads.
- `keigo/telemetry/` stores provider-neutral structured events.

```text
Keigo orchestration
  -> Provider interface
       -> Cloud transport -> cloud workers / object storage
       -> Ray provider -> configured Ray cluster
       -> Local provider -> short smoke simulation only
```

No cloud vendor was selected. Credentials, queue names, cluster endpoints and deployment
scripts therefore stay out of the core. `RayProvider` does not provision or assume a
particular cluster. Candidate workers never own authoritative experiment state; that
responsibility belongs to the orchestrator or a future transactional storage adapter.

## Reproducibility and promotion

Candidate, experiment, job, result and artifact IDs are hashes of canonical JSON. A
candidate change therefore creates a new ID. The candidate artifact content hash is
identity-bearing; its URI is only a location and can change when copied to cloud storage.
The experiment declaration pins seed list,
opponents, simulator version, configuration, evaluation policy and resources. Result
manifests retain the exact result IDs consumed by evaluation. The fresh seed registry
starts empty; distributed seed admission must be implemented with transactional shared
storage before validation seeds are consumed in cloud runs.

The first evaluator checks full coverage and aggregates the generic `score` metric. It is
an infrastructure contract, not a competitive promotion gate. Tournament statistics,
paired confidence intervals, champion/challenger state and submission gates will be added
as explicit policies after the official Kaggriculture engine adapter is selected.
