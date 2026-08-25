#!/usr/bin/env bash
# Launch the camera-free GR00T learner for one immutable DAgger macro-round.
set -Eeuo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
H100_PYTHON=${ROUND_DAGGER_H100_PYTHON:-${H100_PYTHON:-}}
DRY_RUN=${DRY_RUN:-0}

required=(
    ROUND_DAGGER_CAMPAIGN_ROOT ROUND_DAGGER_CAMPAIGN_ID
    ROUND_DAGGER_MACRO_ROUND ROUND_DAGGER_POLICY_CHECKPOINT
    ROUND_DAGGER_POLICY_SHA256 ROUND_DAGGER_RUNTIME_POLICY_SHA256
    ROUND_DAGGER_REPLAY_SNAPSHOT ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256
    ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256 ROUND_DAGGER_LEARNER_OUTPUT_DIR
    ROUND_DAGGER_LEARNER_RESULT ROUND_DAGGER_LEARNER_CODE_SHA256
    ROUND_DAGGER_LEARNER_CONFIG_SHA256
    ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256
    ROUND_DAGGER_BASE_POLICY_SPEC ROUND_DAGGER_LEARNER_BATCH_SIZE
    ROUND_DAGGER_LEARNER_MAX_STEPS ROUND_DAGGER_LEARNER_SAVE_EVERY
    ROUND_DAGGER_LEARNER_LEARNING_RATE ROUND_DAGGER_LEARNER_WEIGHT_DECAY
    ROUND_DAGGER_LEARNER_SEED ROUND_DAGGER_LEARNER_PARTITION_SEED
)
for name in "${required[@]}"; do
    [[ -n ${!name:-} ]] || fatal "missing required environment: $name"
done
[[ -n $H100_PYTHON ]] || fatal "set ROUND_DAGGER_H100_PYTHON"
[[ $H100_PYTHON == /* && -x $H100_PYTHON ]] || \
    fatal "H100 Python must be an absolute executable path"
# Virtual-environment Python executables are normally symlinks.  Permit that
# expected form while requiring the final target to be a regular executable.
H100_PYTHON_TARGET=$(readlink -f -- "$H100_PYTHON") || \
    fatal "cannot resolve H100 Python"
[[ -f $H100_PYTHON_TARGET && -x $H100_PYTHON_TARGET ]] || \
    fatal "H100 Python target is missing or unsafe"
[[ -d $GR00T_ROOT && ! -L $GR00T_ROOT ]] || fatal "GR00T root is missing or unsafe"
[[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || fatal "DRY_RUN must be exactly 0 or 1"

for name in \
    ROUND_DAGGER_CAMPAIGN_ROOT ROUND_DAGGER_POLICY_CHECKPOINT \
    ROUND_DAGGER_REPLAY_SNAPSHOT ROUND_DAGGER_LEARNER_OUTPUT_DIR \
    ROUND_DAGGER_LEARNER_RESULT ROUND_DAGGER_BASE_POLICY_SPEC; do
    [[ ${!name} == /* ]] || fatal "$name must be absolute"
done
for path in \
    "$ROUND_DAGGER_POLICY_CHECKPOINT" "$ROUND_DAGGER_REPLAY_SNAPSHOT" \
    "$ROUND_DAGGER_BASE_POLICY_SPEC"; do
    [[ -f $path && ! -L $path ]] || fatal "unsafe or missing input file: $path"
done
[[ -d $ROUND_DAGGER_CAMPAIGN_ROOT && ! -L $ROUND_DAGGER_CAMPAIGN_ROOT ]] || \
    fatal "campaign root is missing or unsafe"
[[ ! -L $ROUND_DAGGER_LEARNER_OUTPUT_DIR ]] || fatal "learner output is a symlink"

[[ $(sha256sum "$ROUND_DAGGER_POLICY_CHECKPOINT" | awk '{print $1}') == \
    "$ROUND_DAGGER_POLICY_SHA256" ]] || fatal "parent checkpoint SHA-256 mismatch"
[[ $ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256 == \
    "$ROUND_DAGGER_POLICY_SHA256" ]] || fatal "learner lineage is not the active parent"

# Bit-exact CUDA restart requires deterministic cuBLAS reductions.  Pin this
# before any Python/CUDA process starts and reject a conflicting inherited
# value rather than silently changing the numerical contract mid-campaign.
if [[ -n ${CUBLAS_WORKSPACE_CONFIG:-} && \
      $CUBLAS_WORKSPACE_CONFIG != :4096:8 ]]; then
    fatal "CUBLAS_WORKSPACE_CONFIG must be exactly :4096:8"
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8

NNODES=${ROUND_DAGGER_LEARNER_NNODES:-1}
NPROC_PER_NODE=${ROUND_DAGGER_LEARNER_NPROC_PER_NODE:-8}
NODE_RANK=${ROUND_DAGGER_LEARNER_NODE_RANK:-${SLURM_NODEID:-0}}
MASTER_ADDR=${ROUND_DAGGER_LEARNER_MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${ROUND_DAGGER_LEARNER_MASTER_PORT:-29631}
for pair in \
    "$NNODES:nnodes" "$NPROC_PER_NODE:nproc_per_node" \
    "$ROUND_DAGGER_LEARNER_BATCH_SIZE:batch_size" \
    "$ROUND_DAGGER_LEARNER_MAX_STEPS:max_steps" \
    "$ROUND_DAGGER_LEARNER_SAVE_EVERY:save_every"; do
    value=${pair%%:*}; label=${pair#*:}
    [[ $value =~ ^[1-9][0-9]*$ ]] || fatal "$label must be a positive integer"
done
[[ $NODE_RANK =~ ^[0-9]+$ && $NODE_RANK -lt $NNODES ]] || fatal "invalid node rank"
[[ $MASTER_PORT =~ ^[1-9][0-9]*$ && $MASTER_PORT -le 65535 ]] || \
    fatal "invalid master port"
[[ $ROUND_DAGGER_MACRO_ROUND =~ ^[0-9]+$ ]] || fatal "invalid macro round"
[[ $ROUND_DAGGER_LEARNER_SEED =~ ^[0-9]+$ ]] || fatal "invalid learner seed"
[[ $ROUND_DAGGER_LEARNER_PARTITION_SEED =~ ^[0-9]+$ ]] || \
    fatal "invalid partition seed"

# These are redundant with the coordinator topology gate, but retaining them
# here prevents a manually invoked learner hook from training on a different
# logical production contract.
[[ ${ROUND_DAGGER_TARGET_ENVIRONMENTS:-} == 256 ]] || fatal "target environments must be 256"
[[ ${ROUND_DAGGER_COHORT_COUNT:-} == 4 ]] || fatal "cohort count must be 4"
[[ ${ROUND_DAGGER_ENVIRONMENTS_PER_COHORT:-} == 64 ]] || \
    fatal "environments per cohort must be 64"
[[ ${ROUND_DAGGER_STEPS_PER_ENVIRONMENT:-} == 300 ]] || \
    fatal "steps per environment must be 300"
[[ ${ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND:-} == 76800 ]] || \
    fatal "transitions per macro-round must be 76800"

printf -v ROUND_PADDED '%06d' "$((10#$ROUND_DAGGER_MACRO_ROUND))"
BUNDLE="$ROUND_DAGGER_LEARNER_OUTPUT_DIR/learner_bundle_round_${ROUND_PADDED}.json"
EXPECTED_RESULT="$ROUND_DAGGER_LEARNER_OUTPUT_DIR/learner_result_round_${ROUND_PADDED}.json"
[[ $ROUND_DAGGER_LEARNER_RESULT == "$EXPECTED_RESULT" ]] || \
    fatal "orchestrator learner result path is non-canonical"
[[ ! -e $ROUND_DAGGER_LEARNER_RESULT && ! -L $ROUND_DAGGER_LEARNER_RESULT ]] || \
    fatal "learner result already exists; coordinator should verify it instead"

PREPARE=(
    "$H100_PYTHON" "$ROOT/scripts/round_dagger/prepare_gr00t_learner_bundle.py"
    --base-policy-spec "$ROUND_DAGGER_BASE_POLICY_SPEC"
    --parent-checkpoint "$ROUND_DAGGER_POLICY_CHECKPOINT"
    --expected-parent-checkpoint-sha256 "$ROUND_DAGGER_POLICY_SHA256"
    --runtime-policy-sha256 "$ROUND_DAGGER_RUNTIME_POLICY_SHA256"
    --campaign-id "$ROUND_DAGGER_CAMPAIGN_ID"
    --macro-round "$ROUND_DAGGER_MACRO_ROUND"
    --code-sha256 "$ROUND_DAGGER_LEARNER_CODE_SHA256"
    --batch-size "$ROUND_DAGGER_LEARNER_BATCH_SIZE"
    --max-steps "$ROUND_DAGGER_LEARNER_MAX_STEPS"
    --save-every "$ROUND_DAGGER_LEARNER_SAVE_EVERY"
    --learning-rate "$ROUND_DAGGER_LEARNER_LEARNING_RATE"
    --weight-decay "$ROUND_DAGGER_LEARNER_WEIGHT_DECAY"
    --seed "$ROUND_DAGGER_LEARNER_SEED"
    --partition-seed "$ROUND_DAGGER_LEARNER_PARTITION_SEED"
    --output "$BUNDLE"
    --expected-bundle-sha256 "$ROUND_DAGGER_LEARNER_CONFIG_SHA256"
)
if [[ $DRY_RUN == 1 ]]; then
    "${PREPARE[@]}" --dry-run >/dev/null
else
    mkdir -p "$ROUND_DAGGER_LEARNER_OUTPUT_DIR"
    "${PREPARE[@]}" >/dev/null
fi

WORLD_SIZE=$((10#$NNODES * 10#$NPROC_PER_NODE))
RESTART="$ROUND_DAGGER_LEARNER_OUTPUT_DIR/restart.pt"
export PYTHONPATH="$ROOT:$GR00T_ROOT${PYTHONPATH:+:$PYTHONPATH}"
RESUME_PATH=$("$H100_PYTHON" - \
    "$BUNDLE" "$DRY_RUN" "$ROUND_DAGGER_LEARNER_CONFIG_SHA256" \
    "$ROUND_DAGGER_REPLAY_SNAPSHOT" \
    "$ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256" \
    "$ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256" "$ROUND_DAGGER_CAMPAIGN_ID" \
    "$ROUND_DAGGER_MACRO_ROUND" "$WORLD_SIZE" \
    "$ROUND_DAGGER_LEARNER_MAX_STEPS" "$RESTART" \
    "$ROUND_DAGGER_POLICY_SHA256" "$ROUND_DAGGER_RUNTIME_POLICY_SHA256" \
    "$ROUND_DAGGER_LEARNER_PARTITION_SEED" \
    "$ROUND_DAGGER_BASE_POLICY_SPEC" <<'PY'
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from round_dagger.artifacts import load_strict_json
from round_dagger.learner import (
    CHECKPOINT_SCHEMA,
    LearnerBundleConfig,
    partition_shards,
)
from round_dagger.gr00t_policy import load_policy_spec
from round_dagger.replay import load_snapshot

(
    bundle_path, dry_run, expected_bundle_sha, snapshot_path,
    snapshot_file_sha, snapshot_semantic_sha, campaign_id, macro_round,
    world_size, max_steps, restart_path, parent_policy_sha,
    runtime_policy_sha, partition_seed, base_policy_spec_path,
) = sys.argv[1:]
dry_run = dry_run == "1"
macro_round = int(macro_round)
world_size = int(world_size)
max_steps = int(max_steps)
partition_seed = int(partition_seed)
bundle = None
path = Path(bundle_path)
if path.exists() or path.is_symlink():
    if path.is_symlink() or not path.is_file():
        raise SystemExit("learner bundle is unsafe")
    bundle = LearnerBundleConfig.load(path)
    if bundle.sha256 != expected_bundle_sha:
        raise SystemExit("learner bundle digest mismatch")
    if bundle.campaign_id != campaign_id or bundle.macro_round != macro_round:
        raise SystemExit("learner bundle round lineage mismatch")
    if bundle.max_steps != max_steps:
        raise SystemExit("learner bundle max_steps mismatch")
elif not dry_run:
    raise SystemExit("learner bundle was not published")
policy_spec = (
    bundle.to_mapping()["model_config"]
    if bundle is not None
    else load_policy_spec(base_policy_spec_path)
)
replay_action_dim = policy_spec["output_contract"]["replay_action_dim"]

snapshot = load_snapshot(snapshot_path, expected_file_sha256=snapshot_file_sha)
if snapshot["snapshot_sha256"] != snapshot_semantic_sha:
    raise SystemExit("replay snapshot semantic digest mismatch")
if snapshot["campaign_id"] != campaign_id:
    raise SystemExit("replay snapshot campaign mismatch")
if snapshot["through_macro_round"] != macro_round:
    raise SystemExit("replay snapshot macro-round mismatch")
if snapshot["expected_train_cohorts"] != 4:
    raise SystemExit("replay snapshot must contain four cohorts per round")
if snapshot["view_contract"] != {
    "dtype": "uint8", "layout": "vertical", "shape": [324, 192, 3],
    "views": ["head_color", "hand_left", "hand_right"],
}:
    raise SystemExit("replay snapshot three-view contract mismatch")
if snapshot["action_contract"] != {
    "dtype": "float32", "shape": [replay_action_dim]
}:
    raise SystemExit("replay snapshot action contract mismatch")
if len(snapshot["resolved_shards"]) < world_size:
    raise SystemExit("fewer replay shards than distributed learner ranks")
empty_ranks = [
    rank
    for rank in range(world_size)
    if not partition_shards(
        snapshot["resolved_shards"],
        rank=rank,
        world_size=world_size,
        seed=partition_seed,
    )
]
if empty_ranks:
    raise SystemExit(f"replay partition leaves ranks without data: {empty_ranks}")

# Cumulative replay intentionally contains older policy versions.  The four
# newly collected cohorts, however, must be demonstrations for this exact
# active parent/runtime pair before this learner may consume the snapshot.
current_cohorts = [
    cohort for cohort in snapshot["cohorts"]
    if cohort["macro_round"] == macro_round
]
if len(current_cohorts) != 4:
    raise SystemExit("replay snapshot lacks exactly four current-round cohorts")
for cohort in current_cohorts:
    manifest = load_strict_json(cohort["cohort_manifest_path"])
    expected_current = {
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "purpose": "train",
        "policy_sha256": parent_policy_sha,
        "runtime_policy_sha256": runtime_policy_sha,
        "sealed": True,
    }
    if any(manifest.get(key) != value for key, value in expected_current.items()):
        raise SystemExit("current-round replay provenance does not match active policy")

restart = Path(restart_path)
if not restart.exists() and not restart.is_symlink():
    print("")
    raise SystemExit(0)
if restart.is_symlink() or not restart.is_file():
    raise SystemExit("restart checkpoint is unsafe")
if bundle is None:
    raise SystemExit("cannot validate restart without a published learner bundle")
import torch

document = torch.load(restart, map_location="cpu", weights_only=False)
if not isinstance(document, Mapping) or document.get("schema") != CHECKPOINT_SCHEMA:
    raise SystemExit("restart checkpoint schema mismatch")
expected = {
    "campaign_id": campaign_id,
    "macro_round": macro_round,
    "model_lineage_sha256": bundle.model_lineage_sha256,
    "code_sha256": bundle.code_sha256,
    "learner_config_sha256": bundle.sha256,
    "replay_snapshot_sha256": snapshot_semantic_sha,
    "replay_snapshot_file_sha256": snapshot_file_sha,
    "world_size": world_size,
}
if any(document.get(key) != value for key, value in expected.items()):
    raise SystemExit("restart checkpoint lineage mismatch")
completed = document.get("completed_steps")
if type(completed) is not int or not 0 <= completed <= max_steps:
    raise SystemExit("restart completed_steps is invalid")
initial_loss = document.get("initial_loss")
if (
    isinstance(initial_loss, bool)
    or not isinstance(initial_loss, (int, float))
    or not math.isfinite(float(initial_loss))
    or float(initial_loss) < 0.0
):
    raise SystemExit("restart initial_loss is invalid")
states = document.get("rng_states")
if not isinstance(states, Sequence) or len(states) != world_size:
    raise SystemExit("restart rank RNG frontier mismatch")
if not isinstance(document.get("model_state_dict"), Mapping) or not isinstance(
    document.get("optimizer_state_dict"), Mapping
):
    raise SystemExit("restart model/optimizer state is malformed")
print(str(restart))
PY
)

COMMAND=(
    "$H100_PYTHON" -m torch.distributed.run
    --nnodes "$NNODES" --nproc-per-node "$NPROC_PER_NODE"
    --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR"
    --master-port "$MASTER_PORT"
    "$ROOT/scripts/round_dagger/train_learner.py"
    --bundle "$BUNDLE"
    --snapshot "$ROUND_DAGGER_REPLAY_SNAPSHOT"
    --output-dir "$ROUND_DAGGER_LEARNER_OUTPUT_DIR"
    --model-factory round_dagger.gr00t_policy:create_vision_actor_obj_pred
    --adapter-factory round_dagger.gr00t_policy:vision_actor_obj_pred_adapter_factory
    --initial-checkpoint "$ROUND_DAGGER_POLICY_CHECKPOINT"
    --expected-snapshot-file-sha256 "$ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256"
    --max-steps "$ROUND_DAGGER_LEARNER_MAX_STEPS"
    --save-every "$ROUND_DAGGER_LEARNER_SAVE_EVERY"
    # This is the production GPU learner.  Never silently fall back to a CPU
    # execution that could publish a lineage-valid result without exercising
    # the intended CUDA/NCCL runtime.
    --device cuda
    --telemetry-jsonl "$ROUND_DAGGER_LEARNER_OUTPUT_DIR/telemetry.jsonl"
)
[[ -z $RESUME_PATH ]] || COMMAND+=(--resume-from "$RESUME_PATH")

WANDB_PROJECT=${ROUND_DAGGER_WANDB_PROJECT:-}
WANDB_ENTITY=${ROUND_DAGGER_WANDB_ENTITY:-}
WANDB_RUN_ID=${ROUND_DAGGER_WANDB_RUN_ID:-}
if [[ -n $WANDB_PROJECT ]]; then
    COMMAND+=(--wandb-project "$WANDB_PROJECT")
    [[ -z $WANDB_ENTITY ]] || COMMAND+=(--wandb-entity "$WANDB_ENTITY")
    if [[ -z $WANDB_RUN_ID ]]; then
        WANDB_RUN_ID="${ROUND_DAGGER_CAMPAIGN_ID}-learner-m${ROUND_PADDED}"
    fi
    COMMAND+=(--wandb-run-id "$WANDB_RUN_ID")
elif [[ -n $WANDB_ENTITY || -n $WANDB_RUN_ID ]]; then
    fatal "W&B entity/run ID requires ROUND_DAGGER_WANDB_PROJECT"
fi

if [[ $DRY_RUN == 1 ]]; then
    printf 'DRY_RUN'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    echo "H100_LEARNER_DRY_RUN_OK bundle_sha256=$ROUND_DAGGER_LEARNER_CONFIG_SHA256 world_size=$WORLD_SIZE resume=${RESUME_PATH:-none} cuda_determinism=$CUBLAS_WORKSPACE_CONFIG"
    exit 0
fi

"${COMMAND[@]}"

"$H100_PYTHON" - \
    "$ROUND_DAGGER_LEARNER_RESULT" "$ROUND_DAGGER_CAMPAIGN_ID" \
    "$ROUND_DAGGER_MACRO_ROUND" "$ROUND_DAGGER_POLICY_SHA256" \
    "$ROUND_DAGGER_LEARNER_CODE_SHA256" \
    "$ROUND_DAGGER_LEARNER_CONFIG_SHA256" \
    "$ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256" \
    "$ROUND_DAGGER_LEARNER_MAX_STEPS" <<'PY'
import math
import sys
from pathlib import Path

from round_dagger.artifacts import load_strict_json, sha256_file
from round_dagger.learner import RESULT_SCHEMA
from round_dagger.policy_bundle import validate_policy_bundle_identity

(
    result_path, campaign_id, macro_round, parent_sha, code_sha, config_sha,
    snapshot_sha, max_steps,
) = sys.argv[1:]
path = Path(result_path)
if path.is_symlink() or not path.is_file():
    raise SystemExit("learner result is missing or unsafe")
result = load_strict_json(path)
expected = {
    "schema": RESULT_SCHEMA,
    "campaign_id": campaign_id,
    "macro_round": int(macro_round),
    "model_lineage_sha256": parent_sha,
    "code_sha256": code_sha,
    "learner_config_sha256": config_sha,
    "replay_snapshot_sha256": snapshot_sha,
    "optimizer_steps": int(max_steps),
}
if any(result.get(key) != value for key, value in expected.items()):
    raise SystemExit("learner result lineage mismatch")
if set(result) != {
    "schema", "campaign_id", "macro_round", "checkpoint_path",
    "checkpoint_sha256", "runtime_policy_sha256", "policy_spec_path",
    "policy_spec_sha256", "policy_bundle_manifest_path",
    "policy_bundle_manifest_sha256", "model_lineage_sha256", "replay_snapshot_sha256",
    "code_sha256", "learner_config_sha256", "optimizer_steps",
    "initial_loss", "final_loss",
}:
    raise SystemExit("learner result fields mismatch")
for key in ("initial_loss", "final_loss"):
    if isinstance(result[key], bool) or not math.isfinite(float(result[key])):
        raise SystemExit(f"learner {key} is non-finite")
checkpoint = path.parent / result["checkpoint_path"]
if checkpoint.is_symlink() or not checkpoint.is_file():
    raise SystemExit("learner candidate checkpoint is missing or unsafe")
if checkpoint.parent != path.parent or checkpoint.name != result["checkpoint_path"]:
    raise SystemExit("learner checkpoint path is non-canonical")
if sha256_file(checkpoint) != result["checkpoint_sha256"]:
    raise SystemExit("learner checkpoint SHA-256 mismatch")
spec = path.parent / result["policy_spec_path"]
bundle_manifest = path.parent / result["policy_bundle_manifest_path"]
if spec != Path(f"{checkpoint}.policy_spec.json") or bundle_manifest != Path(
    f"{checkpoint}.manifest.json"
):
    raise SystemExit("learner policy sidecar path is non-canonical")
for sidecar, digest_key in (
    (spec, "policy_spec_sha256"),
    (bundle_manifest, "policy_bundle_manifest_sha256"),
):
    if sidecar.is_symlink() or not sidecar.is_file():
        raise SystemExit("learner policy sidecar is missing or unsafe")
    if sha256_file(sidecar) != result[digest_key]:
        raise SystemExit("learner policy sidecar SHA-256 mismatch")
validate_policy_bundle_identity(
    checkpoint=checkpoint,
    policy_spec=spec,
    manifest=bundle_manifest,
    expected_checkpoint_sha256=result["checkpoint_sha256"],
    expected_runtime_policy_sha256=result["runtime_policy_sha256"],
)
PY

COMPLETION_ATTESTATION="$ROUND_DAGGER_LEARNER_OUTPUT_DIR/learner_completion_round_${ROUND_PADDED}.json"
WANDB_READBACK="$ROUND_DAGGER_LEARNER_OUTPUT_DIR/wandb_learner_readback_round_${ROUND_PADDED}.json"
[[ ! -e $COMPLETION_ATTESTATION && ! -L $COMPLETION_ATTESTATION ]] || \
    fatal "learner completion attestation already exists"
[[ ! -e $WANDB_READBACK && ! -L $WANDB_READBACK ]] || \
    fatal "learner W&B readback already exists"
"$H100_PYTHON" "$ROOT/scripts/round_dagger/learner_completion_attestation.py" publish \
    --attestation "$COMPLETION_ATTESTATION" \
    --result "$ROUND_DAGGER_LEARNER_RESULT" \
    --output-dir "$ROUND_DAGGER_LEARNER_OUTPUT_DIR" \
    --bundle "$(basename "$BUNDLE")" --restart restart.pt \
    --telemetry telemetry.jsonl \
    --parent-checkpoint "$ROUND_DAGGER_POLICY_CHECKPOINT" \
    --parent-sha256 "$ROUND_DAGGER_POLICY_SHA256" \
    --snapshot "$ROUND_DAGGER_REPLAY_SNAPSHOT" \
    --snapshot-file-sha256 "$ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256" \
    --snapshot-sha256 "$ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256" \
    --campaign-id "$ROUND_DAGGER_CAMPAIGN_ID" \
    --macro-round "$ROUND_DAGGER_MACRO_ROUND" \
    --code-sha256 "$ROUND_DAGGER_LEARNER_CODE_SHA256" \
    --config-sha256 "$ROUND_DAGGER_LEARNER_CONFIG_SHA256" \
    --wandb-readback "$WANDB_READBACK" \
    --wandb-entity "$ROUND_DAGGER_WANDB_ENTITY" \
    --wandb-project "$ROUND_DAGGER_WANDB_PROJECT" \
    --wandb-run-id "$ROUND_DAGGER_WANDB_RUN_ID" \
    --wandb-expected-email "$ROUND_DAGGER_WANDB_EXPECTED_EMAIL" \
    --wandb-timeout-seconds "${ROUND_DAGGER_WANDB_READBACK_TIMEOUT_SECONDS:-300}" \
    >/dev/null

echo "H100_LEARNER_COMPLETE result=$ROUND_DAGGER_LEARNER_RESULT"
