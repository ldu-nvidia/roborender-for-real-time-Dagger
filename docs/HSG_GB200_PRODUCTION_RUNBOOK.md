# HSG GB200 256-environment production runbook

This runbook covers the fail-closed 256-environment, 300-step DAgger campaign.
It requests one packed `batch_long` allocation: 15 nodes × 4 GB200s, with 52
Image3F renderer tasks and eight synchronized trainer ranks. The absolute
training target is 500 outer updates and checkpoints are saved every five.

## 1. Build and select the immutable campaign snapshot

Production rejects the mutable compatibility checkout. `ROOT` must be the
canonical, non-symlink path below `src_snapshots`, its basename must end in
`_sealed`, and every inode in both source trees must be read-only. The nested
GR00T path is fixed.

```bash
export STAGE_ROOT=/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ldu/roborender_gb200_sm100
export ROOT="$STAGE_ROOT/src_snapshots/fast_roborender_<fast7>_<gr00t7>_sealed"
export GR00T_ROOT="$ROOT/GR00T-VisualSim2Real"
test "$(realpath -e "$ROOT")" = "$ROOT"
test "$(realpath -e "$GR00T_ROOT")" = "$ROOT/GR00T-VisualSim2Real"
test -z "$(find "$ROOT" -xdev -perm /222 -print -quit)"
```

Build it only from the clean stage-zero Git indexes. The checked-in builder copies ordinary files
from their exact Git blobs and accepts an LFS file only when its hydrated bytes match the committed
pointer's SHA-256 and size. It rejects non-allowlisted untracked files, symlinks, special files,
caches and Python bytecode; rechecks HEAD/index/worktrees after copying; preserves executable bits;
writes both revision markers; removes all write bits; and verifies the inventory, deterministic
archive, and both source-tree hashes locally and again on HSG before atomic publication:

```bash
bash scripts/scaling/stage_hsg_source_snapshot.sh
# Export the REMOTE_ROOT path printed by the builder as ROOT.
export ROOT="$STAGE_ROOT/src_snapshots/fast_roborender_<fast7>_<gr00t7>_sealed"
export GR00T_ROOT="$ROOT/GR00T-VisualSim2Real"
```

When HSG already contains an older sealed snapshot with the same large LFS assets, set
`REMOTE_SEED_ROOT` to that canonical snapshot path. The builder clones it server-side, uses
checksum-mode rsync only for byte differences, and still rejects publication unless the complete
remote tree, inventory, deterministic archive, revisions, and permissions match the new local
attestation exactly.

The snapshot must contain a 40-hex `.source_revision` in each repository, plus
the builder's internal v2 manifest. Beside `ROOT`, the builder publishes
`$ROOT.snapshot_attestation.json` and `$ROOT.inventory.json`. The launcher
validates their exact schemas, revisions, Git index trees, per-file hashes and
modes, then binds both external artifact hashes into the production run
identity, submission manifest, predecessor-smoke evidence, allocation-time
preflight, and final pre-postrun check. It also hashes the complete trees before
submission, after the queue wait, and immediately before postrun. Every
container step overlays both source paths read-only even though the broader
stage root is writable.

## 2. Promote prerequisites in order

Do not submit production until all of these gates have passed. The camera and
scale-smoke evidence is bound to the same source-tree and immutable-asset
hashes; the W&B smoke is a separate authorized-team connectivity check:

1. W&B online validation as `ldu@nvidia.com` under exact team entity `nvidia`, with
   scalar, image, and video remote readback.
2. Three-view camera calibration plus explicit human acceptance. Acceptance is
   bound to both source-tree hashes, the immutable model/engine/anchor/teacher
   asset contract, panel bytes, and ordered camera semantics.
3. `smoke_32x300`, proving one optimizer update with seven renderers.
4. `smoke_256x300`, proving one synchronized eight-rank update, all 52 endpoint
   identities and placements, policy-parameter change, W&B trainer scalar, and
   sidecar scalar/image/video readback.

```bash
export WANDB_ACCEPTANCE="$STAGE_ROOT/runtime/validation/wandb_online_smoke_nvidia_acceptance.json"
sbatch "$ROOT/scripts/scaling/hsg_gb200_wandb_validate.sbatch"
# A PASS atomically creates $WANDB_ACCEPTANCE from the job-specific JSON. A
# different existing acceptance is never overwritten.

RUN_PROFILE=calibration_3view bash "$ROOT/scripts/scaling/submit_hsg_gb200_smoke.sh"
# Review both generated panels, then create the source/asset-bound acceptance:
export CALIBRATION_MANIFEST=/absolute/run/path/validation/camera_calibration_manifest.json
export FAST_TREE_SHA256=$(PYTHONDONTWRITEBYTECODE=1 python \
  "$ROOT/scripts/scaling/hsg_gb200_smoke_gate.py" tree-hash --root "$ROOT")
export GR00T_TREE_SHA256=$(PYTHONDONTWRITEBYTECODE=1 python \
  "$ROOT/scripts/scaling/hsg_gb200_smoke_gate.py" tree-hash --root "$GR00T_ROOT")
export ASSET_CONTRACT_SHA256=$(python - "$CALIBRATION_MANIFEST" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["immutable_asset_contract_sha256"])
PY
)
export SOURCE_SNAPSHOT_ATTESTATION_SHA256=$(python - "$CALIBRATION_MANIFEST" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["source_snapshot_attestation_sha256"])
PY
)
export SOURCE_SNAPSHOT_INVENTORY_SHA256=$(python - "$CALIBRATION_MANIFEST" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["source_snapshot_inventory_sha256"])
PY
)
python "$ROOT/scripts/scaling/camera_calibration_gate.py" accept \
  --manifest "$CALIBRATION_MANIFEST" \
  --output "$STAGE_ROOT/runtime/validation/camera_calibration_acceptance.json" \
  --reviewer ldu@nvidia.com \
  --confirm 'I reviewed both three-view calibration panels and accept the camera mapping.' \
  --expected-fast-roborender-revision "$(cat "$ROOT/.source_revision")" \
  --expected-gr00t-revision "$(cat "$GR00T_ROOT/.source_revision")" \
  --expected-fast-source-tree-sha256 "$FAST_TREE_SHA256" \
  --expected-gr00t-source-tree-sha256 "$GR00T_TREE_SHA256" \
  --expected-source-snapshot-attestation-sha256 "$SOURCE_SNAPSHOT_ATTESTATION_SHA256" \
  --expected-source-snapshot-inventory-sha256 "$SOURCE_SNAPSHOT_INVENTORY_SHA256" \
  --expected-asset-contract-sha256 "$ASSET_CONTRACT_SHA256"

RUN_PROFILE=smoke_32x300 bash "$ROOT/scripts/scaling/submit_hsg_gb200_smoke.sh"
RUN_PROFILE=smoke_256x300 bash "$ROOT/scripts/scaling/submit_hsg_gb200_smoke.sh"
```

The camera acceptance and smoke manifests are immutable evidence. A changed
source file, foundation shard, adapter, TensorRT engine, teacher checkpoint, or
anchor file invalidates them and requires re-promotion.
The W&B validation also retains its job-specific JSON. If a different canonical
acceptance already exists, the validation fails closed instead of replacing it;
select a reviewed job JSON explicitly with `WANDB_ONLINE_VALIDATION=/absolute/path`
for the new campaign.

## 3. Validate submission without consuming the logical run ID

The production dry run performs the full login-node preflight and scheduler
request construction under a clearly disposable run root. It never consumes
the intended logical `RUN_ID`.

```bash
DRY_RUN=1 bash "$ROOT/scripts/scaling/submit_hsg_gb200_production.sh"
```

## 4. Submit a fresh production run

Choose a stable run ID. Fresh production must use `CKPT=none`; the student is
initialized from the base model, not Raven's example `last.pt`.

```bash
export RUN_ID=roborender_dagger_256x300_$(date -u +%Y%m%dT%H%M%SZ)
export RUN_ROOT="$STAGE_ROOT/runtime/$RUN_ID"
bash "$ROOT/scripts/scaling/submit_hsg_gb200_production.sh"
```

Only one production attempt may own a run root. The atomic
`$RUN_ROOT/.active_attempt` lease is bound to the exact run, attempt directory,
and Slurm job ID. A submitted lease is reaped only after Slurm explicitly
reports a terminal state. A cross-host unbound submission lease requires
operator review and is never expired by age alone.

## 5. Resume after an interrupted allocation

The `USR1@300` handler publishes `ATTEMPT_FAILED` and the best recovery record
before bounded TERM/KILL and W&B cleanup. After all writers stop it republishes
the newest periodic checkpoint, so the candidate is always latest. Use the
candidate from the failed attempt verbatim:

```bash
export RUN_ID=<original-run-id>
export RUN_ROOT="$STAGE_ROOT/runtime/$RUN_ID"
export RESUME_TRAINING=1
export CKPT="$RUN_ROOT/train/model_step_NNNNNN.pt"
export EXPECTED_TRAIN_CHECKPOINT_SHA256=$(sha256sum "$CKPT" | awk '{print $1}')
export RESUME_CANDIDATE="$RUN_ROOT/attempts/<failed-attempt>/resume_candidate.txt"
bash "$ROOT/scripts/scaling/submit_hsg_gb200_production.sh"
```

Resume rejects a non-latest checkpoint, a candidate not produced by a failed
attempt, changed run identity, changed W&B run IDs, changed source/assets, or a
checkpoint outside the original training directory. Optimizer, scheduler,
trainer state, per-rank DAgger stream state, and W&B continuity are verified.

## 6. Validation-only recovery at step 500

If training produced the exact step-500 checkpoint but telemetry drain or a
postrun gate failed, do not retrain. Use the dedicated validation entrypoint:

```bash
export RUN_ID=<original-run-id>
export RUN_ROOT="$STAGE_ROOT/runtime/$RUN_ID"
export CKPT="$RUN_ROOT/train/model_step_000500.pt"
export EXPECTED_TRAIN_CHECKPOINT_SHA256=$(sha256sum "$CKPT" | awk '{print $1}')
export RESUME_CANDIDATE="$RUN_ROOT/attempts/<failed-attempt>/resume_candidate.txt"
bash "$ROOT/scripts/scaling/submit_hsg_gb200_production_validate.sh"
```

This action starts no renderer fleet and performs no optimizer update. It only
finishes W&B drain/remote readback, checkpoint/policy validation, continuity
evidence, and postrun publication. `RUN_DONE` is created only after all gates
pass; otherwise the attempt remains failed and auditable.

## 7. Expected durable evidence

Inspect the attempt and training directories for the submission/run identity,
runtime preflight, renderer-placement attestation, endpoint capability report,
finite BC-loss and DDP markers, initial/final policy-state digests, W&B drain
and remote-readback reports, resume snapshots, and terminal attempt marker.
Secrets are never written to these artifacts; W&B authentication remains in
the mode-0600 runtime `.netrc`.
