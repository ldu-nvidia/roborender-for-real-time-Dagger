# CUDA learner update and restart-equivalence gate

This bounded gate exercises the real three-view `VisionActorObjPred` learner on
one GPU without Isaac, PhysX, cameras, RoboRender, or W&B.  It intentionally
uses one immutable two-step bundle:

1. stop after exactly one CUDA optimizer update and require only `restart.pt`;
2. run the same two updates uninterrupted and publish a candidate/result;
3. resume the one-update checkpoint in a new container, perform update two, and
   publish a candidate/result;
4. require bit-exact semantic equality of model, optimizer, RNG, lineage, loss,
   and result frontiers.

Do not reuse a replay snapshot from an eval cohort.  `SNAPSHOT` must contain
exactly four sealed `purpose=train` cohorts for `CAMPAIGN` and macro-round zero,
and its current-round manifests must carry the fresh parent policy and runtime
digests.

```bash
set -Eeuo pipefail

ROOT=/mnt/data1/ldu/research/fast_roborender
GR00T_ROOT=$ROOT/GR00T-VisualSim2Real
HOST_PY=/mnt/data2/ldu/lacwm_runtime/envs/lacwm-b200-py310/bin/python
IMAGE=roborender-round-actor:isaaclab2.3.0-local-v2
CONTAINER_PY=/opt/round-dagger-rtx/bin/python
BOOT=/mnt/data2/ldu/roborender_round_dagger_validation/fresh_bootstrap_v5_20260825T0456Z
PARENT=$BOOT/student-v0.pt
PARENT_SHA=3b3a9ac168eaf163a566772679a5bb4b8ec73c9947c9b51668ac08f9a014abae
RUNTIME_SHA=fd1fd8f290fd603e05251f3798f40fd29b8028041e701e6240ac1ae4541f67b6
POLICY_SPEC=$BOOT/student-v0.pt.policy_spec.json

# Set these from the four completed train-smoke cohorts.  CODE_SHA must be the
# orchestrator-pinned digest for the exact learner source closure, not a Git
# branch name and not a placeholder hash.
CAMPAIGN=rtx-train-smoke-v1
SNAPSHOT=/absolute/path/to/replay_snapshot_round_000000.json
CODE_SHA=replace_with_exact_64_character_learner_source_digest

GATE=/mnt/data2/ldu/roborender_round_dagger_validation/learner_cuda_restart_gate_v1
BUNDLE=$GATE/learner_bundle.json
PARTIAL=$GATE/partial_then_resumed
FULL=$GATE/uninterrupted
mkdir -p "$GATE"

SNAPSHOT_FILE_SHA=$(sha256sum "$SNAPSHOT" | awk '{print $1}')
test "$(sha256sum "$PARENT" | awk '{print $1}')" = "$PARENT_SHA"

cd "$ROOT"
"$HOST_PY" - "$SNAPSHOT" "$SNAPSHOT_FILE_SHA" "$CAMPAIGN" \
  "$PARENT_SHA" "$RUNTIME_SHA" <<'PY'
import sys
from round_dagger.artifacts import load_strict_json
from round_dagger.replay import load_snapshot

snapshot_path, file_sha, campaign, parent_sha, runtime_sha = sys.argv[1:]
snapshot = load_snapshot(snapshot_path, expected_file_sha256=file_sha)
expected_view = {
    "dtype": "uint8", "layout": "vertical", "shape": [324, 192, 3],
    "views": ["head_color", "hand_left", "hand_right"],
}
if snapshot["campaign_id"] != campaign or snapshot["through_macro_round"] != 0:
    raise SystemExit("gate replay campaign/frontier mismatch")
if snapshot["expected_train_cohorts"] != 4 or snapshot["cohort_count"] != 4:
    raise SystemExit("gate replay must contain exactly four train cohorts")
if snapshot["transition_count"] != 8:
    raise SystemExit("gate replay must contain exactly eight transitions")
if snapshot["view_contract"] != expected_view:
    raise SystemExit("gate replay is not the production three-view contract")
if snapshot["action_contract"] != {"dtype": "float32", "shape": [26]}:
    raise SystemExit("gate replay action contract mismatch")
if len(snapshot["resolved_shards"]) < 1:
    raise SystemExit("gate replay has no learner shard")
global_env_ids = set()
for expected_index, cohort in enumerate(snapshot["cohorts"]):
    if (
        cohort["macro_round"] != 0
        or cohort["cohort_index"] != expected_index
        or cohort["transition_count"] != 2
    ):
        raise SystemExit("gate cohort identity/count mismatch")
    manifest = load_strict_json(cohort["cohort_manifest_path"])
    expected = {
        "campaign_id": campaign,
        "macro_round": 0,
        "cohort_index": expected_index,
        "purpose": "train",
        "policy_sha256": parent_sha,
        "runtime_policy_sha256": runtime_sha,
        "sealed": True,
        "transition_count": 2,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise SystemExit("gate cohort provenance mismatch")
    ids = manifest.get("global_env_ids")
    if not isinstance(ids, list) or len(ids) != 1 or ids[0] in global_env_ids:
        raise SystemExit("gate cohorts must cover four distinct environments")
    global_env_ids.add(ids[0])
print("LEARNER_GATE_REPLAY_PREFLIGHT_OK")
PY

"$HOST_PY" scripts/round_dagger/prepare_gr00t_learner_bundle.py \
  --base-policy-spec "$POLICY_SPEC" \
  --parent-checkpoint "$PARENT" \
  --expected-parent-checkpoint-sha256 "$PARENT_SHA" \
  --runtime-policy-sha256 "$RUNTIME_SHA" \
  --campaign-id "$CAMPAIGN" \
  --macro-round 0 \
  --code-sha256 "$CODE_SHA" \
  --batch-size 8 \
  --max-steps 2 \
  --save-every 1 \
  --learning-rate 2.5e-5 \
  --weight-decay 0.01 \
  --seed 17 \
  --partition-seed 23 \
  --output "$BUNDLE"

learner_container() {
  local container_name=$1
  shift
  docker run --rm \
    --name "$container_name" \
    --gpus '"device=0"' \
    --ipc=private \
    --shm-size=16g \
    --network=none \
    --workdir "$ROOT" \
    --mount type=bind,src=/mnt/data1,dst=/mnt/data1,readonly \
    --mount type=bind,src=/mnt/data2,dst=/mnt/data2 \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --env PYTHONPATH="$ROOT:$GR00T_ROOT" \
    "$IMAGE" \
    "$CONTAINER_PY" -m torch.distributed.run \
      --nnodes 1 --nproc-per-node 1 \
      "$ROOT/scripts/round_dagger/train_learner.py" \
      --bundle "$BUNDLE" \
      --snapshot "$SNAPSHOT" \
      --model-factory round_dagger.gr00t_policy:create_vision_actor_obj_pred \
      --adapter-factory round_dagger.gr00t_policy:vision_actor_obj_pred_adapter_factory \
      --initial-checkpoint "$PARENT" \
      --expected-snapshot-file-sha256 "$SNAPSHOT_FILE_SHA" \
      --max-steps 2 \
      --save-every 1 \
      --device cuda \
      "$@"
}

# Rung 1: exactly one real CUDA optimizer update, intentionally unpublished.
learner_container rd-learner-partial-v1 \
  --output-dir "$PARTIAL" \
  --telemetry-jsonl "$PARTIAL/telemetry.jsonl" \
  --validation-stop-at-step 1 \
  --no-publish-result
test -f "$PARTIAL/restart.pt"
test ! -e "$PARTIAL/candidate_round_000000.pt"
test ! -e "$PARTIAL/learner_result_round_000000.json"

# Preserve the intentional interruption frontier before resume overwrites the
# canonical restart path.
cp --reflink=auto "$PARTIAL/restart.pt" "$GATE/restart_step_000001.pt"

# Rung 2a: uninterrupted two-update reference, including immutable publication.
learner_container rd-learner-full-v1 \
  --output-dir "$FULL" \
  --telemetry-jsonl "$FULL/telemetry.jsonl"

# Rung 2b: new process/container resumes update one and reaches update two.
learner_container rd-learner-resume-v1 \
  --output-dir "$PARTIAL" \
  --telemetry-jsonl "$PARTIAL/telemetry.jsonl" \
  --resume-from "$PARTIAL/restart.pt"

"$HOST_PY" scripts/round_dagger/verify_learner_restart_equivalence.py \
  --uninterrupted-checkpoint "$FULL/restart.pt" \
  --resumed-checkpoint "$PARTIAL/restart.pt" \
  --expected-completed-steps 2 \
  --uninterrupted-result "$FULL/learner_result_round_000000.json" \
  --resumed-result "$PARTIAL/learner_result_round_000000.json" \
  --output "$GATE/restart_equivalence_report.json"
```

The only accepted terminal marker is
`LEARNER_RESTART_EQUIVALENT ...`.  A mismatch leaves both evidence directories
intact and must block candidate promotion.  The first partial invocation must
never create a learner result or candidate.

The learner additionally enables strict PyTorch deterministic algorithms,
disables cuDNN benchmarking and TF32, pins cuDNN deterministic mode, and uses
the non-foreach/non-fused AdamW path.  A CUDA invocation without the exact
pre-start `CUBLAS_WORKSPACE_CONFIG=:4096:8` contract fails before loading replay
or executing an optimizer update.  Every rerun after a source change must use a
fresh gate root and a freshly computed learner source-closure digest; v1
failure evidence must remain immutable.

## Promotable policy bundle

A completed learner publication is not a checkpoint-only handoff.  It commits
four immutable files in one output directory:

- `candidate_round_NNNNNN.pt`
- `candidate_round_NNNNNN.pt.policy_spec.json`
- `candidate_round_NNNNNN.pt.manifest.json`
- `learner_result_round_NNNNNN.json`

The v2 learner result binds every file digest and the serialization-independent
runtime policy digest.  Candidate staging validates the source bundle, copies
the checkpoint and spec to the exact co-located `checkpoint.pt*` names, rewrites
the basename-bound actor manifest, revalidates the staged bundle, and publishes
the v2 candidate manifest last.  The promoted runtime/spec/manifest digests are
then frozen in the v2 next-round contract; an operator-supplied runtime digest
that differs from that contract fails before collection.
