# Leo BC runtime ownership and control

## Responsibilities

- `pipeline/bc_collection/collector.py` owns collection ordering and
  `chunk_writer.py` owns chunk assembly. Neither manages Docker.
- `pipeline/bc_collection/run_continuous.py` is the collection entrypoint. It
  owns source/capacity/storage admission, the host lock, subprocess lifecycle,
  exact-container cleanup and terminal receipts. Dry-run is the default.
- `pipeline/deployment/rtx_actor/bc_runtime.py` is the shared interface for
  registry-bound paths and BC Docker command assembly. Historical cohort callers
  import the same assembler; their scheduler is not imported by new collection.
- `qualify_native_isaac6.py` supplies shared container configuration. Explicit
  mount paths are arguments, not mutations of module globals. Legacy callers
  retain their defaults; frozen releases are unchanged.
- `materialize_isaac6.py` creates each run's hash-bound compatibility overlay;
  it does not modify the original Raven dependency or image in place.

## New BC paths on Leo

`configs/storage_paths.json` is the path registry. The source manifest must
attest it and its resolver; execution saves an immutable `storage-paths.json`
snapshot. The registry cannot widen the execution/source admission roots.

| Purpose | Path |
|---|---|
| Editable source | `/data1/workspaces/roborender/source` |
| Pinned executable release | `/data1/workspaces/roborender/code_releases/<release>/code` or `frozen` |
| Raven bundle and teacher dependencies | `/data1/workspaces/roborender/simulation_assets/rtx_BC_transfer` |
| Anchors | `/data1/workspaces/roborender/simulation_assets/anchor_pool_2048` |
| Per-GPU rebuildable caches | `/data1/workspaces/roborender/caches/isaac/leo-gpu0` and `leo-gpu1` |
| Logs, overlay, raw chunks and receipts | `/data1/workspaces/roborender/runs/bc-continuous/<run-id>` |

New BC does not select `/home/ldu` aliases or `runs/legacy_home`. Historical
consumers can retain compatibility symlinks; these are not new source defaults
or permission to replay old commands. Other stages' historical launchers have
not all been migrated to this interface.

## Container and control

The existing image is pinned by immutable ID in `qualify_native_isaac6.py`.
Isaac/dependencies stay in the image; project source/assets are mounted read-only.
Only the exact run and cache directories are writable. Ordinary Python edits
require a new source release, not a rebuild of the roughly52GB image. Internal
paths `/isaac-sim`, `/isaac6-port`, `/dagger-runtime` are not extra host copies.

Use `python -m pipeline.bc_collection.run_continuous --help`. No simulator starts
without `--execute`; production also requires genuine matching capacity evidence.
This refactor does not qualify B32. The output must be a fresh direct child of
the canonical `bc-continuous` directory, not an old home-path alias.

The launcher writes `intent.json`, `launch.json`, `container.cid`,
`memory-samples.jsonl`, `host-result.json` and `continuous-execution.json`.
Missing/uncertain acceptance is UNKNOWN, never restart permission. For a
requested stop, verify the exact launcher process identity and deliver SIGTERM
(or Ctrl-C in its foreground terminal). Its existing cleanup checks the full
Docker ID, image and owner label before stopping that workload. Interrupted
collection is partial, not resumable from a chunk number. No broad Docker stop,
prune, automatic restart or compute-allocation deletion is appropriate.

## Docker storage: separate administrator task

Docker still uses `/var/lib/docker` on the system disk. Leo's account lacks
passwordless sudo; no daemon setting was changed. Migration requires an
administrator and maintenance window:

1. Inventory all containers/images/volumes and verify ownership, inactivity and
   automatic-start settings. Preserve daemon configuration and inventories.
2. Verify capacity/filesystem support at
   `/data1/workspaces/roborender/container_storage/docker`.
3. Stop Docker and its socket; copy the quiescent store preserving hardlinks,
   ownership, permissions and xattrs. Check for a separate containerd store.
4. Merge `data-root` without losing NVIDIA runtime settings. Restart and verify
   identical image/container/volume identities and no unintended workload start.
5. Retain the old store for rollback; retirement requires separate cleanup approval.

Do not use a privileged container to bypass administrator access. H200 compute
is outside this host-local task and must remain allocated.
