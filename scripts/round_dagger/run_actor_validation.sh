#!/usr/bin/env bash
# Bounded real-runtime rung for one frozen round actor process.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
GR00T_PYTHON=${GR00T_PYTHON:?set GR00T_PYTHON to the validated Isaac/GR00T Python}
ACCELERATE=${ACCELERATE:-$(dirname "$GR00T_PYTHON")/accelerate}
RANKS=${ROUND_DAGGER_VALIDATION_RANKS:-1}
ENVS_PER_RANK=${ROUND_DAGGER_VALIDATION_ENVS_PER_RANK:-1}
STEPS=${ROUND_DAGGER_VALIDATION_STEPS:-2}
DRY_RUN=${DRY_RUN:-0}
PURPOSE=${ROUND_DAGGER_VALIDATION_PURPOSE:-eval}
TRAIN_ACK=${ROUND_DAGGER_VALIDATION_TRAIN_ACK:-}
ACTOR_RANK=${ROUND_DAGGER_VALIDATION_ACTOR_RANK:-}

case "$PURPOSE" in
    eval) ;;
    train)
        [[ $TRAIN_ACK == bounded-train-replay ]] || {
            echo "bounded train validation requires ROUND_DAGGER_VALIDATION_TRAIN_ACK=bounded-train-replay" >&2
            exit 2
        }
        ;;
    *) echo "validation purpose must be exactly eval or train" >&2; exit 2 ;;
esac

case "$RANKS:$ENVS_PER_RANK:$STEPS" in
    1:*:*|2:*:*) ;;
    *) echo "validation ranks must be exactly 1 or 2" >&2; exit 2 ;;
esac
for value in "$ENVS_PER_RANK" "$STEPS"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || {
        echo "validation environment and step counts must be positive integers" >&2
        exit 2
    }
done
((ENVS_PER_RANK <= 32)) || { echo "validation envs/rank must be <= 32" >&2; exit 2; }
((STEPS <= 300)) || { echo "validation steps must be <= 300" >&2; exit 2; }
if [[ -n $ACTOR_RANK ]]; then
    [[ $ACTOR_RANK =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo "validation actor rank must be a canonical non-negative integer" >&2
        exit 2
    }
    ((ACTOR_RANK < RANKS)) || {
        echo "validation actor rank must be smaller than validation ranks" >&2
        exit 2
    }
elif ((RANKS > 1)); then
    echo "multi-rank validation must launch isolated actor ranks independently" >&2
    exit 2
fi

required=(
    ROUND_DAGGER_COHORT_DIR ROUND_DAGGER_CAMPAIGN_ID
    ROUND_DAGGER_MACRO_ROUND ROUND_DAGGER_COHORT_INDEX
    ROUND_DAGGER_POLICY_CHECKPOINT ROUND_DAGGER_POLICY_SHA256
    ROUND_DAGGER_RUNTIME_POLICY_SHA256 ROUND_DAGGER_TEACHER_CHECKPOINT
    ROUND_DAGGER_TEACHER_SHA256 ROUND_DAGGER_RENDERER_SHA256
    ROUND_DAGGER_BETA ROUND_DAGGER_BETA_SHA256 ROUND_DAGGER_CONFIG_SHA256
    IMAGE3F_ENDPOINT_MANIFEST ANCHORS TRAIN_RUN_DIR
)
for name in "${required[@]}"; do
    [[ -n ${!name:-} ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
case "$ROUND_DAGGER_COHORT_DIR:$ROUND_DAGGER_POLICY_CHECKPOINT:$ROUND_DAGGER_TEACHER_CHECKPOINT:$IMAGE3F_ENDPOINT_MANIFEST:$ANCHORS:$TRAIN_RUN_DIR" in
    /*:/*:/*:/*:/*:/*) ;;
    *) echo "all validation artifact paths must be absolute" >&2; exit 2 ;;
esac
for file in "$ROUND_DAGGER_POLICY_CHECKPOINT" "$ROUND_DAGGER_TEACHER_CHECKPOINT" "$IMAGE3F_ENDPOINT_MANIFEST"; do
    [[ -f $file && ! -L $file ]] || { echo "unsafe or missing file: $file" >&2; exit 2; }
done
[[ -d $ANCHORS && ! -L $ANCHORS ]] || { echo "unsafe anchor directory" >&2; exit 2; }
[[ -x $GR00T_PYTHON ]] || { echo "GR00T_PYTHON is not executable" >&2; exit 2; }
[[ -x $ACCELERATE ]] || { echo "accelerate executable is missing" >&2; exit 2; }
[[ $(sha256sum "$ROUND_DAGGER_POLICY_CHECKPOINT" | awk '{print $1}') == "$ROUND_DAGGER_POLICY_SHA256" ]] || {
    echo "actor checkpoint SHA-256 mismatch" >&2; exit 2;
}
[[ $(sha256sum "$ROUND_DAGGER_TEACHER_CHECKPOINT" | awk '{print $1}') == "$ROUND_DAGGER_TEACHER_SHA256" ]] || {
    echo "teacher checkpoint SHA-256 mismatch" >&2; exit 2;
}

# The observer remains inaccessible to ordinary production and train-smoke
# invocations.  A shadow selection is accepted only when every process value
# is re-derived from one immutable, validation-only A/B plan.  Partial A/B
# metadata and standalone ambient shadow flags fail before Isaac is imported.
RUNTIME_AB_HINT=${ROUND_DAGGER_RUNTIME_AB_PLAN:-}${ROUND_DAGGER_RUNTIME_AB_PLAN_SHA256:-}${ROUND_DAGGER_RUNTIME_AB_ARM:-}${ROUND_DAGGER_RUNTIME_AB_ARM_CONTRACT:-}${ROUND_DAGGER_RUNTIME_AB_ARM_CONTRACT_SHA256:-}${ROUND_DAGGER_RUNTIME_AB_EVIDENCE:-}
RUNTIME_AB_ACTIVE=0
if [[ -n $RUNTIME_AB_HINT ]]; then
    for name in ROUND_DAGGER_RUNTIME_AB_PLAN ROUND_DAGGER_RUNTIME_AB_PLAN_SHA256 \
        ROUND_DAGGER_RUNTIME_AB_ARM ROUND_DAGGER_RUNTIME_AB_ARM_CONTRACT \
        ROUND_DAGGER_RUNTIME_AB_ARM_CONTRACT_SHA256 \
        ROUND_DAGGER_RUNTIME_ADAPTER_MODE \
        ROUND_DAGGER_RUNTIME_ADAPTER_VALIDATION_ACK \
        ROUND_DAGGER_RUNTIME_ADAPTER_EVIDENCE; do
        [[ -n ${!name:-} ]] || {
            echo "runtime adapter A/B launch is missing $name" >&2
            exit 2
        }
    done
    RUNTIME_AB_ACTIVE=1
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
        -m round_dagger.runtime_ab authorize \
        --plan "$ROUND_DAGGER_RUNTIME_AB_PLAN" \
        --arm "$ROUND_DAGGER_RUNTIME_AB_ARM" \
        --expected-plan-sha256 "$ROUND_DAGGER_RUNTIME_AB_PLAN_SHA256" \
        >/dev/null
else
    [[ ${ROUND_DAGGER_RUNTIME_ADAPTER_MODE:-legacy} == legacy ]] || {
        echo "shadow runtime requires an immutable runtime A/B plan" >&2
        exit 2
    }
    [[ -z ${ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK:-} && \
       -z ${ROUND_DAGGER_RUNTIME_ADAPTER_VALIDATION_ACK:-} && \
       -z ${ROUND_DAGGER_RUNTIME_ADAPTER_EVIDENCE:-} ]] || {
        echo "runtime adapter validation metadata is forbidden without an A/B plan" >&2
        exit 2
    }
    export ROUND_DAGGER_RUNTIME_ADAPTER_MODE=legacy
    unset ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK
    unset ROUND_DAGGER_RUNTIME_ADAPTER_VALIDATION_ACK
    unset ROUND_DAGGER_RUNTIME_ADAPTER_EVIDENCE
fi

# A checkpoint byte digest alone cannot prove that the actor, replay and
# simulator agree on their different action widths.  Bind this validation rung
# to the immutable bootstrap bundle and validate the portable policy spec before
# constructing (let alone launching) an Isaac command.
POLICY_SPEC_PATH=${ROUND_DAGGER_POLICY_SPEC:-${ROUND_DAGGER_POLICY_CHECKPOINT}.policy_spec.json}
POLICY_MANIFEST_PATH=${ROUND_DAGGER_POLICY_MANIFEST:-${ROUND_DAGGER_POLICY_CHECKPOINT}.manifest.json}
case "$POLICY_SPEC_PATH:$POLICY_MANIFEST_PATH" in
    /*:/*) ;;
    *) echo "policy spec and manifest paths must be absolute" >&2; exit 2 ;;
esac
for file in "$POLICY_SPEC_PATH" "$POLICY_MANIFEST_PATH"; do
    [[ -f $file && ! -L $file ]] || { echo "unsafe or missing policy bundle file: $file" >&2; exit 2; }
done
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$ROUND_DAGGER_POLICY_CHECKPOINT" "$POLICY_SPEC_PATH" "$POLICY_MANIFEST_PATH" <<'PY'
import os
import sys
from pathlib import Path

from round_dagger.artifacts import load_strict_json, sha256_file
from round_dagger.gr00t_policy import (
    ENV_COMMAND_DIM,
    HOMIE_ACTION_DIM,
    POLICY_ACTION_DIM,
    REPLAY_ACTION_DIM,
    TRAIN_ACTION_SLICE,
    load_policy_spec,
)

checkpoint, spec_path, manifest_path = map(Path, sys.argv[1:])
spec = load_policy_spec(spec_path)
manifest = load_strict_json(manifest_path)
expected_manifest_keys = {
    "schema", "checkpoint_path", "checkpoint_sha256", "runtime_policy_sha256",
    "policy_spec_path", "policy_spec_sha256",
}
if set(manifest) != expected_manifest_keys:
    raise SystemExit("bootstrap policy manifest key-set mismatch")
if manifest.get("schema") != "fast_roborender.round_dagger.bootstrap_policy.v1":
    raise SystemExit("bootstrap policy manifest schema mismatch")
if manifest.get("checkpoint_path") != checkpoint.name:
    raise SystemExit("bootstrap policy manifest checkpoint path mismatch")
if manifest.get("policy_spec_path") != spec_path.name:
    raise SystemExit("bootstrap policy manifest spec path mismatch")
if manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
    raise SystemExit("bootstrap policy manifest/checkpoint mismatch")
if manifest.get("policy_spec_sha256") != sha256_file(spec_path):
    raise SystemExit("bootstrap policy manifest/spec mismatch")
if manifest.get("checkpoint_sha256") != os.environ["ROUND_DAGGER_POLICY_SHA256"]:
    raise SystemExit("bootstrap policy identity differs from validation cohort")
if manifest.get("runtime_policy_sha256") != os.environ["ROUND_DAGGER_RUNTIME_POLICY_SHA256"]:
    raise SystemExit("bootstrap runtime policy identity differs from validation cohort")
if spec.get("checkpoint_sha256") != manifest["checkpoint_sha256"]:
    raise SystemExit("portable policy spec/checkpoint mismatch")
if spec.get("runtime_policy_sha256") != manifest["runtime_policy_sha256"]:
    raise SystemExit("portable policy spec/runtime identity mismatch")
expected_outputs = {
    "policy_action_dim": POLICY_ACTION_DIM,
    "homie_action_dim": HOMIE_ACTION_DIM,
    "replay_action_dim": REPLAY_ACTION_DIM,
    "env_command_dim": ENV_COMMAND_DIM,
    "train_action_slice": list(TRAIN_ACTION_SLICE),
    "object_prediction_dim": 3,
}
if spec.get("output_contract") != expected_outputs:
    raise SystemExit("portable policy 11/15/26/31 action contract mismatch")
PY

# Ambient ROUND_DAGGER_PURPOSE is never authoritative.  The bounded validation
# selector above is explicit and train additionally requires an acknowledgement.
export ROUND_DAGGER_PURPOSE="$PURPOSE"
export ROUND_DAGGER_GLOBAL_ENV_OFFSET=${ROUND_DAGGER_GLOBAL_ENV_OFFSET:-0}
export ROUND_DAGGER_ENVS_PER_SHARD=${ROUND_DAGGER_ENVS_PER_SHARD:-4}
export ROUND_DAGGER_ROLLOUT_ID=${ROUND_DAGGER_ROLLOUT_ID:-0}
EXPECTED_ENVS=$((RANKS * ENVS_PER_RANK))
EXPECTED_TRANSITIONS=$((EXPECTED_ENVS * STEPS))

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$RANKS" "$ENVS_PER_RANK" "$STEPS" "$PURPOSE" <<'PY'
import os
import sys
from pathlib import Path
from round_dagger.artifacts import load_strict_json
from round_dagger.gr00t_policy import REPLAY_ACTION_DIM

ranks, envs_per_rank, steps = map(int, sys.argv[1:4])
purpose = sys.argv[4]
root = Path(os.environ["ROUND_DAGGER_COHORT_DIR"])
contract = load_strict_json(root / "contract.json")
offset = int(os.environ["ROUND_DAGGER_GLOBAL_ENV_OFFSET"])
expected_envs = ranks * envs_per_rank
expected = {
    "campaign_id": os.environ["ROUND_DAGGER_CAMPAIGN_ID"],
    "macro_round": int(os.environ["ROUND_DAGGER_MACRO_ROUND"]),
    "cohort_index": int(os.environ["ROUND_DAGGER_COHORT_INDEX"]),
    "purpose": purpose,
    "expected_ranks": list(range(ranks)),
    "expected_global_env_ids": list(range(offset, offset + expected_envs)),
    "expected_transitions": expected_envs * steps,
    "expected_steps_per_env": steps,
    "policy_sha256": os.environ["ROUND_DAGGER_POLICY_SHA256"],
    "runtime_policy_sha256": os.environ["ROUND_DAGGER_RUNTIME_POLICY_SHA256"],
    "teacher_sha256": os.environ["ROUND_DAGGER_TEACHER_SHA256"],
    "renderer_sha256": os.environ["ROUND_DAGGER_RENDERER_SHA256"],
    "beta_sha256": os.environ["ROUND_DAGGER_BETA_SHA256"],
    "config_sha256": os.environ["ROUND_DAGGER_CONFIG_SHA256"],
}
for key, value in expected.items():
    if contract.get(key) != value:
        raise SystemExit(f"validation cohort contract mismatch for {key}")
if contract.get("view_contract") != {
    "dtype": "uint8", "layout": "vertical", "shape": [324, 192, 3],
    "views": ["head_color", "hand_left", "hand_right"],
}:
    raise SystemExit("validation three-view contract mismatch")
if contract.get("action_contract") != {
    "dtype": "float32", "shape": [REPLAY_ACTION_DIM]
}:
    raise SystemExit("validation replay-action contract mismatch")
PY

EXPECTED_RENDERERS=${ROUND_DAGGER_EXPECTED_RENDERERS:-$RANKS}
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
[[ $ANCHOR_COUNT -ge ${ROUND_DAGGER_MIN_ANCHORS:-2048} ]] || {
    echo "anchor pool is incomplete: $ANCHOR_COUNT" >&2; exit 2;
}
BASE_TRAIN_RUN_DIR=$TRAIN_RUN_DIR
if [[ -n $ACTOR_RANK ]]; then
    printf -v ACTOR_RANK_PADDED '%04d' "$ACTOR_RANK"
    RANK_RUN_DIR="$BASE_TRAIN_RUN_DIR/ranks/rank_${ACTOR_RANK_PADDED}"
else
    RANK_RUN_DIR=$BASE_TRAIN_RUN_DIR
fi
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
export IMAGE_DAGGER_TELEMETRY_REQUIRE_VIDEO=${IMAGE_DAGGER_TELEMETRY_REQUIRE_VIDEO:-0}
export IMAGE_DAGGER_TELEMETRY_SCALAR_EVERY=${IMAGE_DAGGER_TELEMETRY_SCALAR_EVERY:-1}
export IMAGE_DAGGER_TELEMETRY_IMAGE_EVERY=${IMAGE_DAGGER_TELEMETRY_IMAGE_EVERY:-1}
export IMAGE_DAGGER_TELEMETRY_CLIP_EVERY=${IMAGE_DAGGER_TELEMETRY_CLIP_EVERY:-0}
export IMAGE_DAGGER_TELEMETRY_CLIP_FRAMES=${IMAGE_DAGGER_TELEMETRY_CLIP_FRAMES:-32}
export IMAGE_DAGGER_TELEMETRY_CLIP_FPS=${IMAGE_DAGGER_TELEMETRY_CLIP_FPS:-10}
if ((RUNTIME_AB_ACTIVE)); then
    # authorize_runtime_ab already proved that this namespace is arm-specific.
    export IMAGE3F_RUN_ID
else
    export IMAGE3F_RUN_ID="validation-${ROUND_DAGGER_CAMPAIGN_ID}-${ROUND_DAGGER_COHORT_INDEX}"
fi
export IMAGE3F_REFERENCE_LAG=256
export IMAGE3F_LOCAL_ENVS="$ENVS_PER_RANK"
export ROBORENDER_TELEMETRY_DIR="$BASE_TRAIN_RUN_DIR/telemetry"
export ROLLOUT_STEPS=$STEPS
if [[ -n $ACTOR_RANK ]]; then
    export ROUND_DAGGER_ACTOR_RANK=$ACTOR_RANK
    export ROUND_DAGGER_ACTOR_WORLD_SIZE=$RANKS
    # The logical actor rank belongs to the immutable cohort, not to Torch
    # distributed.  Each process gets one isolated GPU from its outer runtime.
    unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
    unset MASTER_ADDR MASTER_PORT
else
    unset ROUND_DAGGER_ACTOR_RANK ROUND_DAGGER_ACTOR_WORLD_SIZE
fi

TRAINER=(
    "$GR00T_ROOT/gr00t/rl/train_agent_trl.py"
    +exp=loco_manip/wsdpt_round_actor_3view
    seed="${ROUND_DAGGER_SIMULATOR_SEED:-0}"
    ++algo.config.videogen.seed="${ROUND_DAGGER_IMAGE3F_SEED:-42}"
    num_envs="$ENVS_PER_RANK" algo.config.num_steps_per_env="$STEPS"
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
    experiment_name="round-actor-validation-r${RANKS}-e${ENVS_PER_RANK}-s${STEPS}"
    experiment_dir="$RANK_RUN_DIR"
    hydra.run.dir="$RANK_RUN_DIR/hydra"
    output_dir="$RANK_RUN_DIR/output"
)
# Image3F's lag history is correctly scoped to an Isaac episode. A validation
# rollout crossing the lag boundary therefore needs one deliberately
# uninterrupted environment; ordinary actor/train rollouts retain every normal
# task and safety termination. At 200 Hz with control decimation 4, the base
# 8-second horizon is 400 control steps, beyond this launcher's <=300 bound.
if ((STEPS > 256)); then
    [[ $PURPOSE == eval ]] || {
        echo "validation rollouts crossing lag 256 must use purpose=eval" >&2
        exit 2
    }
    TRAINER+=(
        env.config.max_episode_length_s=8
        ++env.config.reset_on_overtime=false
        env.config.reset_on_complete=false
        env.config.termination.terminate_object_by_low_height=false
        env.config.termination.terminate_by_face_mul=false
        env.config.termination.terminate_by_throw_object=false
        env.config.termination.terminate_by_fast_right_arm_qvel=false
        env.config.termination.terminate_by_hold_object_tray_distance=false
        env.config.termination.terminate_after_finish_turning_for_student=false
    )
fi
COMMAND=("$GR00T_PYTHON" "${TRAINER[@]}")
if ((DRY_RUN)); then
    printf 'ROUND_ACTOR_VALIDATION_DRY_RUN ranks=%q actor_rank=%q purpose=%q envs_per_rank=%q steps=%q transitions=%q' \
        "$RANKS" "${ACTOR_RANK:-none}" "$PURPOSE" "$ENVS_PER_RANK" "$STEPS" "$EXPECTED_TRANSITIONS"
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
else
    cd "$GR00T_ROOT"
    "${COMMAND[@]}"
    CHECK_RANK=${ACTOR_RANK:-0}
    # A zero trainer exit is insufficient: Hydra/Kit has paths that print a
    # fatal traceback but still unwind with status zero.  Prove that this exact
    # logical rank committed all expected, provenance-bound shard bytes before
    # a success marker, lag audit, or cohort seal is allowed.
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
        -m round_dagger.actor_pair rank \
        --cohort-dir "$ROUND_DAGGER_COHORT_DIR" \
        --rank "$CHECK_RANK" >/dev/null
    # A validation rollout that crosses lag 256 must leave an immutable,
    # exact proof that rank zero used the anchor through step 255 and generated
    # t-256 beginning at step 256. Sparse or malformed telemetry fails the rung.
    if ((STEPS > 256)) && [[ -z $ACTOR_RANK || $ACTOR_RANK == 0 ]]; then
        mapfile -t REFERENCE_SHARDS < <(
            find "$ROUND_DAGGER_COHORT_DIR/shards" -maxdepth 1 -type f \
                -name 'rank-0000_rollout-*_part-0000.npz' -print | LC_ALL=C sort
        )
        [[ ${#REFERENCE_SHARDS[@]} == 1 ]] || {
            echo "lag validation requires exactly one rank-0 part-0 rollout shard" >&2
            exit 2
        }
        PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
            "$ROOT/scripts/round_dagger/audit_reference_lag.py" \
            --scalar-dir "$BASE_TRAIN_RUN_DIR/telemetry/scalars" \
            --lag 256 \
            --expected-steps "$STEPS" \
            --rollout-shard "${REFERENCE_SHARDS[0]}" \
            --output "$BASE_TRAIN_RUN_DIR/reference_lag_validation.json"
    fi
    if [[ -n $ACTOR_RANK ]]; then
        echo "ROUND_ACTOR_VALIDATION_RANK_OK rank=$ACTOR_RANK purpose=$PURPOSE"
        exit 0
    fi
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
        "$ROUND_DAGGER_COHORT_DIR" "$EXPECTED_TRANSITIONS" "$PURPOSE" <<'PY'
import sys
from round_dagger.artifacts import seal_cohort

sealed = seal_cohort(sys.argv[1])
if sealed["purpose"] != sys.argv[3] or sealed["transition_count"] != int(sys.argv[2]):
    raise SystemExit("sealed validation cohort coverage mismatch")
print(
    "ROUND_ACTOR_VALIDATION_OK "
    f"transitions={sealed['transition_count']} shards={len(sealed['shards'])}"
)
PY
fi
