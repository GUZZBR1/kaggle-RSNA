# Migration record

## Sources and destination

- Structural source: `GUZZBR1/kaggle-projects`, subdirectory `kaggriculture`, inspected at
  commit `5125245ab2915146ac2d1b379cbdac28c542a5bd`.
- Destination: the existing `GUZZBR1/kaggle-RSNA` repository, per the user's clarification.
- The source checkout remained read-only. The unrelated `GUZZBR1/IA_Farm` and this
  destination's old README were not used as implementation sources.

## Reused and adapted

- `eval/comparison.py::digest` informed canonical JSON SHA-256 identity in
  `keigo/identity.py`.
- `arena/runspec.py` informed immutable experiment identity, explicit material fields and
  evidence binding in `keigo/contracts.py` and the experiment runner.
- `arena/batch.py`, `arena/jobs.py` and `arena/ray_transport.py` informed provider/job
  identity separation, result completeness checks, coordinator ownership and the optional
  Ray adapter. Their old simulator imports, strategy-bound agent lookup, local paths and
  machine setup were not copied.
- `eval/pairing.py`, `eval/metrics.py`, `arena/seeds.py` and the promotion gate audit informed
  complete two-seat coverage, explicit fresh seeds, and the future tournament/evaluation
  policy boundary. Historical seed consumption and local lock files were not retained.
- The new smoke configuration and tests are synthetic and do not contain old candidate
  parameters or historical outcomes.

## Discarded

- Old strategy modules (`agent/planner.py`, `economy.py`, `market.py`, `params.py`,
  `xliq.py`, `champion.py`, `challenger.py`) and old `arena/agents.py` variant resolution.
- Old candidate versions, frozen agents, opponent ratings/panels, strategy experiments,
  competitive replays, score reports and historical seed registry.
- `scripts/ray_cluster.py` as a default execution path because it assumes two local Linux/
  WSL PCs, Tailscale and a user systemd service; no cloud provisioning was present.
- The old submission bundler, which assembled the strategy package and imported tuned
  defaults.
- Game economics telemetry and strategy-specific tests that do not establish the new
  provider or result contracts.

## Kept under `legacy/`

Nothing. The source repository remains available as the historical reference, so copying
old code into this repository would duplicate strategy and results without helping the
first clean execution flow.

## Decisions and known boundaries

- No cloud vendor or Kaggriculture engine version was assumed. The example marks the
  simulator as `unconfigured`; real cloud execution requires an official environment
  adapter and transport.
- Candidate IDs include the candidate artifact hash and configuration, while the artifact
  URI is a movable location passed through the planned job.
- The source Ray transport connects to a cluster but does not create one. Ray is optional
  and its configuration contains no machine addresses.
- The old job store forced SQLite `journal_mode=MEMORY` to work around a WSL/DrvFs mount.
  This foundation does not copy that durability tradeoff; a cloud coordinator store still
  needs to be chosen.
- Champion/challenger in the old repository were agent implementations that imported the
  old planner. The new design reserves those names for artifact IDs, evaluation evidence
  and explicit promotion decisions.
