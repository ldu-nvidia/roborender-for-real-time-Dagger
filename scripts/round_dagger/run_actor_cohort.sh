#!/usr/bin/env bash
# Collect one isolated 32-environment rank of a frozen 64 x 300 RTX cohort.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
GR00T_PYTHON=${GR00T_PYTHON:?set GR00T_PYTHON to the validated Isaac/GR00T Python}
DRY_RUN=${DRY_RUN:-0}
ACTOR_RANK=${ROUND_DAGGER_ACTOR_RANK:?set the isolated actor rank (0 or 1)}
ACTOR_WORLD_SIZE=${ROUND_DAGGER_ACTOR_WORLD_SIZE:?set actor world size to 2}
[[ $ACTOR_RANK == 0 || $ACTOR_RANK == 1 ]] || {
    echo "production actor rank must be exactly 0 or 1" >&2; exit 2;
}
[[ $ACTOR_WORLD_SIZE == 2 ]] || {
    echo "production actor world size must be exactly 2" >&2; exit 2;
}

required=(
    ROUND_DAGGER_COHORT_DIR ROUND_DAGGER_CAMPAIGN_ID
    ROUND_DAGGER_MACRO_ROUND ROUND_DAGGER_COHORT_INDEX
    ROUND_DAGGER_POLICY_CHECKPOINT ROUND_DAGGER_POLICY_SHA256
    ROUND_DAGGER_POLICY_SPEC_SHA256 ROUND_DAGGER_POLICY_MANIFEST_SHA256
    ROUND_DAGGER_RUNTIME_POLICY_SHA256 ROUND_DAGGER_TEACHER_CHECKPOINT
    ROUND_DAGGER_TEACHER_SHA256 ROUND_DAGGER_RENDERER_SHA256
    ROUND_DAGGER_BETA ROUND_DAGGER_BETA_SHA256 ROUND_DAGGER_CONFIG_SHA256
    ROUND_DAGGER_IMAGE3F_ATTEMPT_NAMESPACE ROUND_DAGGER_IMAGE3F_STREAM_PLAN
    IMAGE3F_ENDPOINT_MANIFEST ANCHORS TRAIN_RUN_DIR
)
for name in "${required[@]}"; do
    [ -n "${!name:-}" ] || { echo "missing required environment: $name" >&2; exit 2; }
done

case "$ROUND_DAGGER_COHORT_DIR:$ROUND_DAGGER_POLICY_CHECKPOINT:$ROUND_DAGGER_TEACHER_CHECKPOINT:$IMAGE3F_ENDPOINT_MANIFEST:$ROUND_DAGGER_IMAGE3F_STREAM_PLAN:$ANCHORS:$TRAIN_RUN_DIR" in
    /*:/*:/*:/*:/*:/*:/*) ;;
    *) echo "all actor artifact paths must be absolute" >&2; exit 2 ;;
esac
for file in "$ROUND_DAGGER_POLICY_CHECKPOINT" "$ROUND_DAGGER_TEACHER_CHECKPOINT" \
    "$IMAGE3F_ENDPOINT_MANIFEST" "$ROUND_DAGGER_IMAGE3F_STREAM_PLAN"; do
    [ -f "$file" ] && [ ! -L "$file" ] || { echo "unsafe or missing file: $file" >&2; exit 2; }
done
[ -d "$ANCHORS" ] && [ ! -L "$ANCHORS" ] || { echo "unsafe anchor directory" >&2; exit 2; }
[ -x "$GR00T_PYTHON" ] || { echo "GR00T_PYTHON is not executable" >&2; exit 2; }

[ "$(sha256sum "$ROUND_DAGGER_POLICY_CHECKPOINT" | awk '{print $1}')" = "$ROUND_DAGGER_POLICY_SHA256" ] || {
    echo "actor checkpoint SHA-256 mismatch" >&2; exit 2;
}
[ "$(sha256sum "$ROUND_DAGGER_TEACHER_CHECKPOINT" | awk '{print $1}')" = "$ROUND_DAGGER_TEACHER_SHA256" ] || {
    echo "teacher checkpoint SHA-256 mismatch" >&2; exit 2;
}

# Bind the byte checkpoint to the exact portable architecture and runtime
# identity before inspecting the cohort or constructing an Isaac command.  The
# validator is deliberately control-plane-only: it imports neither PyTorch nor
# GR00T/Isaac runtime modules.
POLICY_SPEC_PATH=${ROUND_DAGGER_POLICY_SPEC:-${ROUND_DAGGER_POLICY_CHECKPOINT}.policy_spec.json}
POLICY_MANIFEST_PATH=${ROUND_DAGGER_POLICY_MANIFEST:-${ROUND_DAGGER_POLICY_CHECKPOINT}.manifest.json}
[[ $POLICY_SPEC_PATH == "${ROUND_DAGGER_POLICY_CHECKPOINT}.policy_spec.json" ]] || {
    echo "policy spec must be exactly co-located with checkpoint" >&2; exit 2;
}
[[ $POLICY_MANIFEST_PATH == "${ROUND_DAGGER_POLICY_CHECKPOINT}.manifest.json" ]] || {
    echo "policy manifest must be exactly co-located with checkpoint" >&2; exit 2;
}
case "$POLICY_SPEC_PATH:$POLICY_MANIFEST_PATH" in
    /*:/*) ;;
    *) echo "policy spec and manifest paths must be absolute" >&2; exit 2 ;;
esac
for file in "$POLICY_SPEC_PATH" "$POLICY_MANIFEST_PATH"; do
    [ -f "$file" ] && [ ! -L "$file" ] || {
        echo "unsafe or missing policy bundle file: $file" >&2; exit 2;
    }
done
[[ $(sha256sum "$POLICY_SPEC_PATH" | awk '{print $1}') == "$ROUND_DAGGER_POLICY_SPEC_SHA256" ]] || {
    echo "policy spec SHA-256 differs from frozen round contract" >&2; exit 2;
}
[[ $(sha256sum "$POLICY_MANIFEST_PATH" | awk '{print $1}') == "$ROUND_DAGGER_POLICY_MANIFEST_SHA256" ]] || {
    echo "policy manifest SHA-256 differs from frozen round contract" >&2; exit 2;
}
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m round_dagger.policy_bundle \
    --checkpoint "$ROUND_DAGGER_POLICY_CHECKPOINT" \
    --policy-spec "$POLICY_SPEC_PATH" \
    --manifest "$POLICY_MANIFEST_PATH" \
    --expected-checkpoint-sha256 "$ROUND_DAGGER_POLICY_SHA256" \
    --expected-runtime-policy-sha256 "$ROUND_DAGGER_RUNTIME_POLICY_SHA256" \
    >/dev/null

export ROUND_DAGGER_GLOBAL_ENV_OFFSET=$((10#$ROUND_DAGGER_COHORT_INDEX * 64))
export ROUND_DAGGER_ENVS_PER_SHARD=${ROUND_DAGGER_ENVS_PER_SHARD:-4}
export ROUND_DAGGER_ROLLOUT_ID=${ROUND_DAGGER_ROLLOUT_ID:-0}
export ROUND_DAGGER_PURPOSE=train

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import os
from pathlib import Path
from round_dagger.artifacts import load_strict_json
from round_dagger.gr00t_policy import REPLAY_ACTION_DIM

root = Path(os.environ["ROUND_DAGGER_COHORT_DIR"])
contract = load_strict_json(root / "contract.json")
offset = int(os.environ["ROUND_DAGGER_GLOBAL_ENV_OFFSET"])
expected = {
    "campaign_id": os.environ["ROUND_DAGGER_CAMPAIGN_ID"],
    "macro_round": int(os.environ["ROUND_DAGGER_MACRO_ROUND"]),
    "cohort_index": int(os.environ["ROUND_DAGGER_COHORT_INDEX"]),
    "purpose": os.environ["ROUND_DAGGER_PURPOSE"],
    "expected_ranks": [0, 1],
    "expected_global_env_ids": list(range(offset, offset + 64)),
    "expected_transitions": 64 * 300,
    "policy_sha256": os.environ["ROUND_DAGGER_POLICY_SHA256"],
    "runtime_policy_sha256": os.environ["ROUND_DAGGER_RUNTIME_POLICY_SHA256"],
    "teacher_sha256": os.environ["ROUND_DAGGER_TEACHER_SHA256"],
    "renderer_sha256": os.environ["ROUND_DAGGER_RENDERER_SHA256"],
    "beta_sha256": os.environ["ROUND_DAGGER_BETA_SHA256"],
    "config_sha256": os.environ["ROUND_DAGGER_CONFIG_SHA256"],
}
for key, value in expected.items():
    if contract.get(key) != value:
        raise SystemExit(f"cohort contract mismatch for {key}")
if contract.get("view_contract") != {
    "dtype": "uint8", "layout": "vertical", "shape": [324, 192, 3],
    "views": ["head_color", "hand_left", "hand_right"],
}:
    raise SystemExit("cohort three-view contract mismatch")
if contract.get("action_contract") != {
    "dtype": "float32", "shape": [REPLAY_ACTION_DIM]
}:
    raise SystemExit("cohort action contract mismatch")
PY

EXPECTED_RENDERERS=${ROUND_DAGGER_EXPECTED_RENDERERS:-2}
EP_LIST=$(python3 - "$IMAGE3F_ENDPOINT_MANIFEST" "$EXPECTED_RENDERERS" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
expected = int(sys.argv[2])
lines = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(lines) != expected:
    raise SystemExit(f"endpoint manifest has {len(lines)} entries, expected {expected}")
for value in lines:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise SystemExit(f"unsafe endpoint descriptor: {value}")
print(json.dumps(lines, separators=(",", ":")))
PY
)

ANCHOR_COUNT=$(find "$ANCHORS" -maxdepth 1 -type f -name '*.npz' -printf . | wc -c)
[ "$ANCHOR_COUNT" -ge "${ROUND_DAGGER_MIN_ANCHORS:-2048}" ] || {
    echo "anchor pool is incomplete: $ANCHOR_COUNT" >&2; exit 2;
}
BASE_TRAIN_RUN_DIR=$TRAIN_RUN_DIR
printf -v ACTOR_RANK_PADDED '%04d' "$ACTOR_RANK"
RANK_RUN_DIR="$BASE_TRAIN_RUN_DIR/ranks/rank_${ACTOR_RANK_PADDED}"
mkdir -p "$RANK_RUN_DIR" "$RANK_RUN_DIR/hydra" "$RANK_RUN_DIR/output"

export PYTHONPATH="$ROOT:$GR00T_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export IMAGE_DAGGER_EXPECT_PRIMARY_VIEW=head_color
export IMAGE_DAGGER_EXPECT_POLICY_VIEWS=head_color,hand_left,hand_right
export IMAGE_DAGGER_EXPECT_POLICY_VIEW_COUNT=3
export IMAGE_DAGGER_EXPECT_POLICY_VIEW_LAYOUT=vertical
export IMAGE_DAGGER_EXPECT_POLICY_IMAGE_SHAPE=324,192,3
export IMAGE_DAGGER_EXPECT_ROLLOUT_ACTION_SOURCE=beta
export IMAGE_DAGGER_EXPECT_BETA_SCHEDULE=constant
export IMAGE_DAGGER_EXPECT_BETA_START="$ROUND_DAGGER_BETA"
export IMAGE_DAGGER_EXPECT_BETA_END="$ROUND_DAGGER_BETA"
export IMAGE_DAGGER_EXPECT_TEACHER_WARMUP_STEPS=0
export IMAGE_DAGGER_EXPECT_ENFORCE_TEACHER=false
export IMAGE_DAGGER_EXPECT_TEACHER_RATIO=0.0
export IMAGE_DAGGER_EXPECT_LEARNING_EPOCHS=0
export IMAGE_DAGGER_EXPECT_VALUE_LOSS_COEF=0.0
export IMAGE_DAGGER_EXPECT_ENTROPY_COEF=0.0
export IMAGE_DAGGER_VERIFY_DDP_SYNC=0
export IMAGE_DAGGER_TELEMETRY_REQUIRED=1
export IMAGE_DAGGER_SIMULATION_APP_SKIP_CLEANUP=1
export IMAGE_DAGGER_TELEMETRY_REQUIRE_VIDEO=1
export IMAGE_DAGGER_TELEMETRY_SCALAR_EVERY=${IMAGE_DAGGER_TELEMETRY_SCALAR_EVERY:-10}
export IMAGE_DAGGER_TELEMETRY_IMAGE_EVERY=${IMAGE_DAGGER_TELEMETRY_IMAGE_EVERY:-300}
export IMAGE_DAGGER_TELEMETRY_CLIP_EVERY=${IMAGE_DAGGER_TELEMETRY_CLIP_EVERY:-300}
export IMAGE_DAGGER_TELEMETRY_CLIP_FRAMES=${IMAGE_DAGGER_TELEMETRY_CLIP_FRAMES:-32}
export IMAGE_DAGGER_TELEMETRY_CLIP_FPS=${IMAGE_DAGGER_TELEMETRY_CLIP_FPS:-10}
PLAN_IDENTITY=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
    -m round_dagger.image3f_lifecycle verify-rank \
    --plan "$ROUND_DAGGER_IMAGE3F_STREAM_PLAN" --rank "$ACTOR_RANK" \
    --expected-namespace "$ROUND_DAGGER_IMAGE3F_ATTEMPT_NAMESPACE" \
    --endpoint-manifest "$IMAGE3F_ENDPOINT_MANIFEST" \
    --renderer-sha256 "$ROUND_DAGGER_RENDERER_SHA256" \
    --local-environments 32 --actor-world-size "$ACTOR_WORLD_SIZE")
PLAN_NAMESPACE=$(python3 -c \
    'import json,sys; print(json.load(sys.stdin)["attempt_namespace"])' \
    <<<"$PLAN_IDENTITY")
[[ "$PLAN_NAMESPACE" == "$ROUND_DAGGER_IMAGE3F_ATTEMPT_NAMESPACE" ]] || {
    echo "actor attempt namespace differs from immutable stream plan" >&2; exit 2;
}
export IMAGE3F_RUN_ID="$ROUND_DAGGER_IMAGE3F_ATTEMPT_NAMESPACE"
export IMAGE3F_REFERENCE_LAG=256
export IMAGE3F_LOCAL_ENVS=32
export ROBORENDER_TELEMETRY_DIR="$BASE_TRAIN_RUN_DIR/telemetry"
export ROUND_DAGGER_ACTOR_RANK=$ACTOR_RANK
export ROUND_DAGGER_ACTOR_WORLD_SIZE=$ACTOR_WORLD_SIZE
# Frozen actor ranks are independent Isaac processes.  They must not inherit a
# launcher environment that makes Accelerate create a CUDA/NCCL process group.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
unset MASTER_ADDR MASTER_PORT

COMMAND=(
    "$GR00T_PYTHON" "$GR00T_ROOT/gr00t/rl/train_agent_trl.py"
    +exp=loco_manip/wsdpt_round_actor_3view
    num_envs=32 algo.config.num_steps_per_env=300
    algo.config.num_learning_epochs=0
    algo.config.rollout_with_teacher_num_steps=0
    algo.config.enforce_teacher_rollout=false
    algo.config.ratio_teacher_rollout=0.0
    algo.config.videogen.rollout_action_source=beta
    ++algo.config.videogen.beta_schedule=constant
    ++algo.config.videogen.beta_start="$ROUND_DAGGER_BETA"
    ++algo.config.videogen.beta_end="$ROUND_DAGGER_BETA"
    ++algo.config.videogen.beta_decay_iters=1
    ++algo.config.videogen.reference_lag=256
    algo.trl.num_total_batches=1 algo.trl.total_episodes=null
    checkpoint="$ROUND_DAGGER_POLICY_CHECKPOINT" resume=false
    teacher_actor_path="$ROUND_DAGGER_TEACHER_CHECKPOINT"
    algo.config.videogen.raytraced_mix_prob=0.0
    algo.config.videogen.image3f_endpoint_files="$EP_LIST"
    algo.config.videogen.anchor_pool_dir="$ANCHORS"
    algo.config.videogen.anchor_index=0
    simulator.config.cameras.policy_views="[head_color,hand_left,hand_right]"
    policy_view_count=3 ++env.config.image_dagger_policy_view_count=3
    simulator.config.render_results=false headless=true use_wandb=false
    experiment_name="actor-m${ROUND_DAGGER_MACRO_ROUND}-c${ROUND_DAGGER_COHORT_INDEX}-r${ACTOR_RANK}"
    experiment_dir="$RANK_RUN_DIR"
    hydra.run.dir="$RANK_RUN_DIR/hydra"
    output_dir="$RANK_RUN_DIR/output"
)

if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY_RUN actor_rank=%q actor_world_size=%q image3f_run_id=%q' \
        "$ACTOR_RANK" "$ACTOR_WORLD_SIZE" "$IMAGE3F_RUN_ID"
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
else
    cd "$GR00T_ROOT"
    "${COMMAND[@]}"
    # Hydra/Kit can log a fatal traceback yet allow the top-level interpreter
    # to return zero.  Process status alone is therefore not a collection
    # success signal.  Require this logical rank's exact immutable shard bytes
    # and coverage before returning success to the outer pair launcher.
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
        -m round_dagger.actor_pair rank \
        --cohort-dir "$ROUND_DAGGER_COHORT_DIR" \
        --rank "$ACTOR_RANK" >/dev/null
    ATTEMPT_DIR=$(dirname "$ROUND_DAGGER_IMAGE3F_STREAM_PLAN")
    printf -v COMPLETION_PATH '%s/rank_%04d_stream_completion.json' \
        "$ATTEMPT_DIR" "$ACTOR_RANK"
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
        -m round_dagger.image3f_lifecycle complete-rank \
        --output "$COMPLETION_PATH" \
        --plan "$ROUND_DAGGER_IMAGE3F_STREAM_PLAN" \
        --cohort-dir "$ROUND_DAGGER_COHORT_DIR" \
        --rank "$ACTOR_RANK" >/dev/null
    echo "ROUND_ACTOR_RANK_OK rank=$ACTOR_RANK purpose=train"
fi
