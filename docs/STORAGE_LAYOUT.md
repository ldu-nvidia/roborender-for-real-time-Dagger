# Storage layout and safe migration

October26 update: the standalone BC paths below supersede the historical
October24 simulation-asset aliases. See the current BC section at the end.

This is a storage/source organization change, not a change to
reset, rendering, policy, sample selection, success metrics or compute ownership.

## Locations

| Host / disk | Location | Responsibility |
|---|---|---|
| Cosmos shared volume | `/workspace/project/roborender` | Authoritative editable Git checkout; CPU orchestration |
| Cosmos shared volume | `/workspace/project/roborender-artifacts` | Model/data catalog, retained checkpoints, new releases/runs |
| Leo `/data1` NVMe | `/data1/workspaces/roborender/source` | Working Git checkout synchronized by an explicit source manifest |
| Leo `/data1` NVMe | `/data1/workspaces/roborender/{code_releases,python_envs,runs,caches}` | Executable releases, environment references, new runs and rebuildable caches |
| Leo `/data1` NVMe | `/data1/workspaces/roborender/simulation_assets` | Canonical BC simulation bundle and anchors |
| Leo `/data2` NVMe | `/data2/library/roborender` | Remaining historical policy/checkpoint/dataset/image-archive/catalog entries; simulation aliases are superseded |

`/workspace` is Cosmos shared storage, not a Leo host directory. Docker's daemon
storage stays on Leo's root disk in this phase. Moving Docker requires separate
daemon maintenance; never move its live directories as ordinary asset files.

The tracked authority for **new location resolution** is
[`configs/storage_paths.json`](../configs/storage_paths.json), read by
`roborender.infrastructure.paths`. Select the host explicitly. Paths are
locators, not proof of artifact identity, existence, readiness or scientific
equivalence. Keep SHA-256/source/renderer/panel identities in their own manifests.
Each new run must freeze its resolved paths and registry SHA before admission;
later edits to the registry cannot redirect a frozen run.

For example, from the current source checkout:

```bash
python3 -m roborender.infrastructure.paths resolve \
  --registry configs/storage_paths.json --host leo --key bc_epoch116
python3 -m roborender.infrastructure.paths snapshot \
  --registry configs/storage_paths.json --host leo \
  --output /absolute/existing/new-run/storage-paths.json
```

`verify --registry ORIGINAL_REGISTRY --snapshot SNAPSHOT` checks against the
preserved original registry bytes, not today's mutable configuration. Existing
consumers are not globally redirected: adopting these paths is explicit in a
fresh run configuration. Static executable-source jails are a separate security
boundary; changing a registry entry cannot widen them.

## Incremental cutover, not a bulk rename

1. Preserve the original roots and existing run/checkpoint/source evidence.
2. Create new named roots and a catalog of exact existing artifacts. Link entries
   are explicitly references, not claims of physically moved data. Copy only
   selected small immutable artifacts with pre/post SHA verification.
3. Establish a fresh Leo Git checkout using a Git bundle plus the explicit
   source-only working-tree snapshot; never synchronize `.git`, overwrite the
   modified imported clone, or discard uncommitted source changes.
4. Teach new source launchers about the exact new release jail, retaining legacy
   admission. Freeze a new source release; old releases are never patched.
5. Integrate newly admitted launchers with frozen path resolutions and qualify
   native execution. Only then consider moving path-bound physical payloads.

Existing source/data validators bind canonical paths and sometimes reject
symlinks. Consequently historical `runtime/source-releases`, BC data, renderer
runtime/environments and frozen run roots remain physically where they were.
A compatibility symlink alone does **not** certify their relocation. Do not
mass-rewrite paths inside immutable evidence or make old results appear newly run.
The new `legacy_path_bound` entries point to that preserved history, not new runs.

Leo's old `/data1/workspaces/roborender/source` alias originally referred to the
September imported Git clone with local modifications, not current deployed
source. Preserve its exact link under `source_legacy_20261024` before replacing
the active `source` directory with a verified new checkout. The imported physical
clone and `/home/ldu/roborender` remain untouched.

Migration plans, before/after receipts and source hashes are retained under
`runtime/storage-reorganization-20261024` on Cosmos and the new `catalog` roots.
The migration tool defaults to preview; explicit application refuses conflicting
destinations. A partially applied plan must be reconciled/idempotently resumed,
never repaired by deleting broad roots. Rollback means restoring the recorded
source alias and returning new consumers to their prior frozen locators; it
does not delete datasets or relinquish compute.

Tools: `pipeline.deployment.prepare_storage_catalog` produces a dated additive
plan; `roborender.infrastructure.storage_layout --plan ABSOLUTE_PLAN` previews
it, and `--execute --receipt ABSOLUTE_RECEIPT` applies it without overwrites.
`roborender.infrastructure.source_checkout` handles the narrowly bound Leo
source-alias cutover, preserving both checkouts with before/after receipts.
The working-tree source manifest includes uncommitted changes; equal Git HEADs
alone are insufficient. No commit or remote push is implied by this migration.

## Logical folders

Leo uses `simulation_assets/{robots,objects,motions,render_anchors}`,
`policy_models/{manipulation_teacher,locomotion_controller,visual_student_bootstrap}`,
`student_checkpoints/{behavior_cloning,dagger}`, `datasets`, `container_images`,
`archived_runs` and `catalog`.

Cosmos uses `render_models/{foundation,offline_bc_50step,online_policy_image3f}`,
`policy_models`, `student_checkpoints/{behavior_cloning,dagger}`,
`datasets/{teacher_demonstrations,rendered_bc,split_manifests}`, `code_releases`,
`python_envs`, `runs`, `caches` and `catalog`. SM90 engines stay on Cosmos; they
are not RTX engines. Student training checkpoints remain authoritative there;
Leo receives hash-verified inference copies.

The current algorithm implementation remains under `pipeline/`. The new
`roborender/infrastructure/` package owns path/layout utilities. Extracting
simulation, rendering, training and evaluation modules is a separate incremental
refactor with compatibility imports and regression/native checks; folder creation
does not mean that refactor or the scaled evaluator is runtime-certified.

No migration command allocates, cancels, releases, scales down or suspends
H200 compute, starts/stops Isaac, or creates artificial GPU activity.
# October26: current standalone BC storage interface

New Leo BC collection uses `configs/storage_paths.json` directly from its pinned
source release. Simulation bundle/anchors resolve physically under
`/data1/workspaces/roborender/simulation_assets`; caches use `caches/isaac/leo-gpu0`
and `leo-gpu1`, and run state uses `runs/bc-continuous/<run-id>`. Each executed run
freezes its path snapshot. The current launcher no longer admits home-alias output
paths. Historical records and other legacy consumers retain their old identities.

The shared adapter is `pipeline/deployment/rtx_actor/bc_runtime.py`; no algorithm
is moved out of `pipeline/bc_collection`. See its [runtime guide](../pipeline/deployment/rtx_actor/BC_RUNTIME.md).
Docker still stores images at `/var/lib/docker`: administrator migration is
pending, not implied by the application-path changes. Other unmigrated artifact
categories retain their registry entries; this update is not a whole-host move.
