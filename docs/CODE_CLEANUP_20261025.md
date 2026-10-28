# Source cleanup — October 25, 2026

Scope: the mutable Cosmos `/workspace/project/roborender` and Leo
`/data1/workspaces/roborender/source` source copies. This is not a replacement
of live Isaac owners, frozen source releases, containers or model assets.
The [production contract](END_TO_END_PIPELINE_CONTRACT.md) remains authoritative.

## Changes

The first test relocation below is historical. The subsequent user-requested
consolidation puts **all first-party tests in repository-root `test/`**, grouped
as `repository/`, `pipeline/`, `deployment/` and `rtx/`. Use
`bash test/run_cpu_checks.sh`; see [current layout](../test/README.md).
Production modules and frozen source-transition allowlists are not renamed.

- Move 85 deployment unit/regression/functional test modules into
  `pipeline/tests/deployment/`, retaining their subgroups and test cases.
  Move the CPU runner to `tests/run_cpu_checks.sh`; update fixture imports and
  source-path resolution in test-only helpers. The relocation preserves all
  722 test functions and 1,664 parameterized cases in that moved suite.
- Discover all three dedicated test trees by default, including root `tests/`.
  Add layout/import regression guards; production must not import tests,
  pytest or unittest. Keep the pinned vendor reference unchanged.
- Remove 407 unreachable body lines from six already-retired paths:
  `beta_site._evaluate`, the legacy `resident_campaign_site` evaluator,
  the returned branch in `deployment/plan.py`, and the old paired, small-batch
  and diagnostic student-success command bodies. Keep their explicit errors,
  shared helpers, public signatures and current paired/scaled dispatch.
- Replace the old `run_raven_bc_collection_8gpu.sh`, `run_raven_bc_rank.sh` and
  `run_raven_raytraced_300_eval_h200.sh` shortcuts with immediate, explanatory
  exit-2 stubs. Sentinel-command tests verify they cannot execute tools, access
  assets or launch work. New BC collection uses the continuous protocol;
  student evaluation uses the admitted generated-input paired path.
- Make source snapshots handle unstaged file relocations: inventory the actual
  working tree rather than attempting to read deleted index paths. Verification
  against an older manifest still rejects the changed/deleted paths.
- Refresh the pipeline map, test instructions and skill pointers. Historical
  release receipts and source-transition allowlists retain their old paths;
  moving tests does not waive a frozen-source identity guard.

These are bounded removals, not wholesale commenting-out of old files. Dead
implementations remain recoverable in Git and the pre-cleanup source snapshot.

## Retained deliberately

| Code family | Why it is not safe to delete wholesale |
|---|---|
| BC quota/coverage helpers | Current continuous launch and ingestion reuse storage/identity validators; old receipts also need readers. |
| DAgger coverage, MSE and validation-named modules | New fixed100/continuous schemas and current service/runtime helpers still import them. Their names do not imply a current held-out stage. |
| v2/v3 transport | v4 and current trainers import lower-level codecs and authenticated transport. |
| Native diagnostics/resource checks | Reset, memory, camera and ownership guards are runtime safety dependencies, not unit-test code. |
| Monolithic DAgger shells | Reference validators inspect their historical mixed/native-RGB recipe. They are not current production defaults; replacing them requires a separate provenance-contract migration. |
| Historical overnight supervisor | Has old defaults and service restart behavior; not a current launcher. Retiring its CLI requires preserving shared helpers and ledger compatibility. |
| Standalone proxy/tunnel utilities | Ignored host SSH/runtime configuration can call them; no Python import is not proof of obsolescence. |

## BC payload retirement

The fresh scoped re-audit found all 27,035 previously retired BC data/checkpoint
paths absent from active locations (26,484 data entries, 551 checkpoint entries;
these are files/entries, not episode counts). No new active BC payload candidate
was found. The recoverable trash remains intact; approximately 1.097 TB has **not**
been purged or reclaimed. All 21 protected model paths retain recorded identities.

The same 1,974 root-owned DAgger service directories remain inaccessible to the
ordinary-SSH rescan; their earlier privileged-helper inventory is retained, not
claimed as a fresh exhaustive scan. The 55 known root-owned retired files were
independently checked. Do not restore or purge payloads automatically.

## Evidence and limits

- `runtime/bc-retirement-20261025/RETIREMENT.json`: original scoped retirement.
- `runtime/bc-retirement-reaudit-20261025T211251Z/SUMMARY.json`: fresh checks.
- `runtime/source-cleanup-20261025-BYIi13/test-move-map.json`: exact old/new paths.
- `runtime/source-cleanup-20261025-BYIi13/`: CPU test results, source manifests,
  cross-host verification and recoverable move receipts.
- `test/README.md`: supported test locations and commands.

CPU tests/source matching are not a new live GPU certification. No collection,
training, inference, native-owner transition or compute release is part of this
cleanup. Current-source production still needs its existing live admissions.
