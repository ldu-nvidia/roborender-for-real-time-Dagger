#!/bin/bash
# Distributed production DAgger: 256 global Isaac environments, 300 rollout steps.
# This remains the validated pure-BC DAgger objective (teacher action supervision),
# not PPO reward optimization. Submit via scripts/scaling/submit_production.sh.
#SBATCH --partition=grizzly
#SBATCH --account=general_cs_infra
#SBATCH --job-name=rr_dagger_256x300
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
set -Eeuo pipefail

# This launcher may be submitted with a restricted sbatch --export contract.
# Child srun steps still need variables derived after allocation (endpoint
# manifests, per-rank environment counts, telemetry controls, and seeds).
export SLURM_EXPORT_ENV=ALL

STAGE_ROOT=${STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
ANCHORS=${ANCHORS:-$STAGE_ROOT/assets/anchor_pool_2048}
TEACHER_CKPT=${TEACHER_CKPT:-$STAGE_ROOT/assets/policies/expert_labeler/model_step_044500.pt}
EXPECTED_TEACHER_CKPT_SHA256=${EXPECTED_TEACHER_CKPT_SHA256:-943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38}
FLEET_DIR=${FLEET_DIR:?FLEET_DIR is required; use scripts/scaling/submit_production.sh}
RUN_ID=${RUN_ID:?RUN_ID is required; use a sanitized launcher}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]{1,128}$ ]] || {
    echo "FATAL: RUN_ID must match [A-Za-z0-9_-]{1,128}" >&2
    exit 2
}
GLOBAL_NUM_ENVS=${GLOBAL_NUM_ENVS:-256}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-300}
MIX=${MIX:-0.0}
REFERENCE_LAG=${REFERENCE_LAG:-0}
ENVS_PER_RENDERER=${ENVS_PER_RENDERER:-5}
NUM_MINI_BATCHES=${NUM_MINI_BATCHES:-1}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
if [ -n "${NUM_TOTAL_BATCHES:-}" ] && [ -n "${NUM_ROLLOUT_ITERATIONS:-}" ] && \
   [ "$NUM_TOTAL_BATCHES" != "$NUM_ROLLOUT_ITERATIONS" ]; then
    echo "FATAL: NUM_TOTAL_BATCHES and NUM_ROLLOUT_ITERATIONS disagree"
    exit 2
fi
NUM_ROLLOUT_ITERATIONS=${NUM_ROLLOUT_ITERATIONS:-${NUM_TOTAL_BATCHES:-1}}
NUM_LEARNING_EPOCHS=${NUM_LEARNING_EPOCHS:-1}
BEHAVIOR_SOURCE=${BEHAVIOR_SOURCE:-annealed}
TEACHER_ROLLOUT_RATIO=${TEACHER_ROLLOUT_RATIO:-0.0}
BETA_SCHEDULE=${BETA_SCHEDULE:-linear}
BETA_START=${BETA_START:-1.0}
BETA_END=${BETA_END:-0.0}
BETA_DECAY_ITERS=${BETA_DECAY_ITERS:-500}
VG_PRIMARY_VIEW=${VG_PRIMARY_VIEW:-head_color}
VG_POLICY_VIEWS=${VG_POLICY_VIEWS:-head_color:hand_left:hand_right}
VG_POLICY_VIEW_COUNT=${VG_POLICY_VIEW_COUNT:-3}
VG_POLICY_VIEW_LAYOUT=${VG_POLICY_VIEW_LAYOUT:-vertical}
VG_POLICY_IMAGE_SHAPE=${VG_POLICY_IMAGE_SHAPE:-324x192x3}
SAVE_FREQUENCY=${SAVE_FREQUENCY:-5}
TEST_STOP_AFTER_CHECKPOINT_STEP=${TEST_STOP_AFTER_CHECKPOINT_STEP:-0}
TRAINER_CPUS_PER_TASK=${TRAINER_CPUS_PER_TASK:-16}
TRAINER_MEM_PER_NODE=${TRAINER_MEM_PER_NODE:-128G}
TRAINER_CONTAINER_IMAGE=${TRAINER_CONTAINER_IMAGE:-nvcr.io#nvidia/isaac-sim:5.1.0}
TRAINER_CONTAINER_MOUNTS=${TRAINER_CONTAINER_MOUNTS:-$STAGE_ROOT:$STAGE_ROOT}
WANDB_EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_ENTITY=${WANDB_ENTITY:-nvidia}
WANDB_AUTHORIZATION_SCOPE=${WANDB_AUTHORIZATION_SCOPE:-team}
USE_WANDB=${USE_WANDB:-1}
VIRAL_ENV=${VIRAL_ENV:-$STAGE_ROOT/runtime/envs/viral}
VIRAL_PYTHON=${VIRAL_PYTHON:-$VIRAL_ENV/bin/python}
VIRAL_CONDA_SH=${VIRAL_CONDA_SH:-}
VIRAL_ENV_NAME=${VIRAL_ENV_NAME:-viral}
VIRAL_VENV_ACTIVATE=${VIRAL_VENV_ACTIVATE:-$STAGE_ROOT/runtime/envs/activate_viral_isaac.sh}
WANDB_ENV=${WANDB_ENV:-$STAGE_ROOT/runtime/envs/wandb-host}
WANDB_PYTHON=${WANDB_PYTHON:-$WANDB_ENV/bin/python}
WANDB_WATCHER_POLL_S=${WANDB_WATCHER_POLL_S:-15}
WANDB_WATCHER_READY_TIMEOUT_S=${WANDB_WATCHER_READY_TIMEOUT_S:-120}
WANDB_WATCHER_DRAIN_TIMEOUT_S=${WANDB_WATCHER_DRAIN_TIMEOUT_S:-240}
TRAINER_NODES=${SLURM_NNODES:?must run under Slurm}
RUN_PROFILE=${RUN_PROFILE:-production_256x300}
RENDER_READY_TIMEOUT_S=${RENDER_READY_TIMEOUT_S:-3600}
RENDER_JOB_ID=${RENDER_JOB_ID:-}
[[ -z "$RENDER_JOB_ID" || "$RENDER_JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "FATAL: RENDER_JOB_ID must be the numeric array parent job id"
    exit 2
}
case "$USE_WANDB" in
    1) ;;
    0)
        [ "${CHECKPOINT_MECHANICS_SMOKE:-0}" = 1 ] || {
            echo "FATAL: USE_WANDB=0 is restricted to CHECKPOINT_MECHANICS_SMOKE=1"
            exit 2
        }
        [ "$RUN_PROFILE" != production_256x300 ] || {
            echo "FATAL: production_256x300 requires online W&B telemetry"
            exit 2
        }
        ;;
    *) echo "FATAL: USE_WANDB must be 0 or 1"; exit 2 ;;
esac
if [ "$USE_WANDB" = 1 ]; then
    [[ "$WANDB_WATCHER_DRAIN_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] && \
        [ "$WANDB_WATCHER_DRAIN_TIMEOUT_S" -le 270 ] || {
        echo "FATAL: WANDB_WATCHER_DRAIN_TIMEOUT_S must be an integer in [1,270]"
        exit 2
    }
fi

SRUN_PID=""
WANDB_WATCHER_PID=""
PRE_TIMEOUT_SIGNALLED=0

_terminate_srun() {
    if [ -z "$SRUN_PID" ] || ! kill -0 "$SRUN_PID" 2>/dev/null; then
        return 0
    fi
    kill -TERM "$SRUN_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
        kill -0 "$SRUN_PID" 2>/dev/null || return 0
        sleep 1
    done
    echo "WARNING: srun did not exit after SIGTERM; sending SIGKILL"
    kill -KILL "$SRUN_PID" 2>/dev/null || true
}

_cancel_owned_renderer_fleet() {
    if [ -z "$RENDER_JOB_ID" ]; then
        return 0
    fi
    echo "Cancelling owned renderer array $RENDER_JOB_ID"
    if ! scancel "$RENDER_JOB_ID"; then
        echo "WARNING: scancel could not cancel renderer array $RENDER_JOB_ID (it may already be terminal)"
        return 1
    fi
}

_handle_pre_timeout() {
    PRE_TIMEOUT_SIGNALLED=1
    local marker_dir=${TRAIN_RUN_DIR:-$FLEET_DIR}
    printf 'received USR1 at %s; stopping after latest periodic checkpoint\n' \
        "$(date --iso-8601=seconds)" > "$marker_dir/PRE_TIMEOUT_WARNING"
    echo "PRE-TIMEOUT: stopping the trainer step with five minutes left for cleanup and W&B drain"
    if [ -n "$SRUN_PID" ] && kill -0 "$SRUN_PID" 2>/dev/null; then
        kill -TERM "$SRUN_PID" 2>/dev/null || true
    else
        exit 124
    fi
}

_production_cleanup() {
    local rc=$?
    local watcher_rc=0
    trap - EXIT USR1

    _terminate_srun
    if [ -n "$SRUN_PID" ]; then
        wait "$SRUN_PID" 2>/dev/null || true
        SRUN_PID=""
    fi

    if [ -n "${TRAIN_RUN_DIR:-}" ] && [ -n "$WANDB_WATCHER_PID" ]; then
        : > "$TRAIN_RUN_DIR/WANDB_UPLOAD_STOP"
        local watcher_deadline=$((SECONDS + WANDB_WATCHER_DRAIN_TIMEOUT_S))
        while kill -0 "$WANDB_WATCHER_PID" 2>/dev/null && \
              [ "$SECONDS" -lt "$watcher_deadline" ]; do
            sleep 1
        done
        if kill -0 "$WANDB_WATCHER_PID" 2>/dev/null; then
            echo "ERROR: W&B media watcher did not finish its final drain within ${WANDB_WATCHER_DRAIN_TIMEOUT_S}s"
            kill -TERM "$WANDB_WATCHER_PID" 2>/dev/null || true
            watcher_rc=70
        fi
        local watcher_wait_rc=0
        if wait "$WANDB_WATCHER_PID" 2>/dev/null; then
            watcher_wait_rc=0
        else
            watcher_wait_rc=$?
        fi
        if [ "$watcher_rc" -eq 0 ] && [ "$watcher_wait_rc" -ne 0 ]; then
            watcher_rc=$watcher_wait_rc
        fi
        WANDB_WATCHER_PID=""
    fi
    if [ "$rc" -eq 0 ] && [ "$watcher_rc" -ne 0 ]; then
        rc=70
    fi

    _cancel_owned_renderer_fleet || true
    if [ -n "${TRAIN_RUN_DIR:-}" ]; then
        if [ "$rc" -eq 0 ]; then
            : > "$TRAIN_RUN_DIR/RUN_DONE"
        else
            printf '%s\n' "$rc" > "$TRAIN_RUN_DIR/RUN_FAILED"
        fi
    fi
    exit "$rc"
}
trap _production_cleanup EXIT
trap _handle_pre_timeout USR1

FAST_REVISION=$(git -C "$ROOT" rev-parse HEAD)
FAST_DIRTY=$(git -C "$ROOT" status --porcelain)
if [ -n "$FAST_DIRTY" ] && [ "${ALLOW_DIRTY_FAST_ROBORENDER:-0}" != "1" ]; then
    echo "FATAL: fast_roborender worktree is dirty; commit this scaling branch first."
    exit 2
fi
if [ -n "${EXPECTED_FAST_ROBORENDER_REVISION:-}" ] && \
   [ "$FAST_REVISION" != "$EXPECTED_FAST_ROBORENDER_REVISION" ]; then
    echo "FATAL: fast_roborender revision $FAST_REVISION != $EXPECTED_FAST_ROBORENDER_REVISION"
    exit 2
fi
: "${EXPECTED_GR00T_REVISION:?EXPECTED_GR00T_REVISION must pin the integrated trainer}"

PLAN=(
    --global-envs "$GLOBAL_NUM_ENVS"
    --rollout-steps "$ROLLOUT_STEPS"
    --num-rollout-iterations "$NUM_ROLLOUT_ITERATIONS"
    --trainer-nodes "$TRAINER_NODES"
    --raytraced-mix "$MIX"
    --envs-per-renderer "$ENVS_PER_RENDERER"
    --reference-lag "$REFERENCE_LAG"
    --num-mini-batches "$NUM_MINI_BATCHES"
    --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --num-learning-epochs "$NUM_LEARNING_EPOCHS"
    --behavior-source "$BEHAVIOR_SOURCE"
    --teacher-rollout-ratio "$TEACHER_ROLLOUT_RATIO"
    --beta-schedule "$BETA_SCHEDULE"
    --beta-start "$BETA_START"
    --beta-end "$BETA_END"
    --beta-decay-iters "$BETA_DECAY_ITERS"
    --primary-view "$VG_PRIMARY_VIEW"
)
LOCAL_NUM_ENVS=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field local_envs)
REQUIRED_RENDERERS=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field required_renderers)
GRADIENT_ACCUMULATION_STEPS=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field gradient_accumulation_steps)
TRANSITIONS_PER_ROLLOUT_ITERATION=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field transitions_per_rollout_iteration)
TRANSITIONS_PER_OPTIMIZER_STEP=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field transitions_per_optimizer_step)
OPTIMIZER_STEPS_PER_ROLLOUT_ITERATION=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field optimizer_steps_per_rollout_iteration)
ENFORCE_TEACHER_ROLLOUT=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field enforce_teacher_rollout)
VG_ROLLOUT_ACTION_SOURCE=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field rollout_action_source)
VG_BETA_SCHEDULE=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field beta_schedule)
VG_BETA_START=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field beta_start)
VG_BETA_END=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field beta_end)
VG_BETA_DECAY_ITERS=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field beta_decay_iters)
[[ "$NUM_ROLLOUT_ITERATIONS" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: NUM_ROLLOUT_ITERATIONS must be a positive integer, got $NUM_ROLLOUT_ITERATIONS"
    exit 2
}
[[ "$SAVE_FREQUENCY" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: SAVE_FREQUENCY must be a positive integer, got $SAVE_FREQUENCY"
    exit 2
}
[ $((NUM_ROLLOUT_ITERATIONS % SAVE_FREQUENCY)) -eq 0 ] || {
    echo "FATAL: NUM_ROLLOUT_ITERATIONS must be divisible by SAVE_FREQUENCY so the exact final checkpoint is persisted"
    exit 2
}
[[ "$TEST_STOP_AFTER_CHECKPOINT_STEP" =~ ^[0-9]+$ ]] || {
    echo "FATAL: TEST_STOP_AFTER_CHECKPOINT_STEP must be a non-negative integer"
    exit 2
}
if [ "$TEST_STOP_AFTER_CHECKPOINT_STEP" -gt 0 ]; then
    [ "${CHECKPOINT_MECHANICS_SMOKE:-0}" = 1 ] || {
        echo "FATAL: test checkpoint stop is restricted to CHECKPOINT_MECHANICS_SMOKE=1"
        exit 2
    }
    [ "$USE_WANDB" = 0 ] || {
        echo "FATAL: test checkpoint stop requires USE_WANDB=0"
        exit 2
    }
    [ "$RUN_PROFILE" != production_256x300 ] || {
        echo "FATAL: production may not use the test checkpoint stop"
        exit 2
    }
    [ "$TEST_STOP_AFTER_CHECKPOINT_STEP" -lt "$NUM_ROLLOUT_ITERATIONS" ] || {
        echo "FATAL: test checkpoint stop must be earlier than the absolute target"
        exit 2
    }
    [ $((TEST_STOP_AFTER_CHECKPOINT_STEP % SAVE_FREQUENCY)) -eq 0 ] || {
        echo "FATAL: test checkpoint stop must be divisible by SAVE_FREQUENCY"
        exit 2
    }
fi
[ "$VG_PRIMARY_VIEW" = head_color ] || {
    echo "FATAL: three-view production requires primary_view=head_color"
    exit 2
}
[ "$VG_POLICY_VIEWS" = head_color:hand_left:hand_right ] || {
    echo "FATAL: three-view production requires ordered policy views head_color:hand_left:hand_right"
    exit 2
}
[ "$VG_POLICY_VIEW_COUNT" = 3 ] || {
    echo "FATAL: three-view production requires policy_view_count=3"
    exit 2
}
[ "$VG_POLICY_VIEW_LAYOUT" = vertical ] || {
    echo "FATAL: three-view production requires a vertical tile layout"
    exit 2
}
[ "$VG_POLICY_IMAGE_SHAPE" = 324x192x3 ] || {
    echo "FATAL: three-view production requires policy image shape 324x192x3"
    exit 2
}

# Until Raven's private trainer proves an episode-sticky mixed cohort, lagged
# streams are production-safe only when every env renders every tick (MIX=0)
# or no env renders (MIX=1). Per-tick Bernoulli mixing creates step gaps.
if [ "$MIX" != "0" ] && [ "$MIX" != "0.0" ] && \
   [ "$MIX" != "1" ] && [ "$MIX" != "1.0" ] && \
   [ "${ALLOW_EXPERIMENTAL_MIXED_LAG:-0}" != "1" ]; then
    echo "FATAL: MIX=$MIX is not validated with contiguous lag-$REFERENCE_LAG streams."
    echo "       Use MIX=0.0 (production default), MIX=1.0, or explicitly set"
    echo "       ALLOW_EXPERIMENTAL_MIXED_LAG=1 after verifying an episode-sticky mask."
    exit 2
fi

ALLOW_DIRTY_ARGS=()
if [ "${ALLOW_DIRTY_GR00T:-0}" = "1" ]; then
    ALLOW_DIRTY_ARGS+=(--allow-dirty)
fi
EXPECTED_REV_ARGS=()
if [ -n "${EXPECTED_GR00T_REVISION:-}" ]; then
    EXPECTED_REV_ARGS+=(--expected-revision "$EXPECTED_GR00T_REVISION")
fi
python3 "$ROOT/scripts/scaling/install_gr00t_integration.py" \
    --gr00t-root "$GR00T_ROOT" "${ALLOW_DIRTY_ARGS[@]}" "${EXPECTED_REV_ARGS[@]}"

[ -d "$GR00T_ROOT" ] || { echo "FATAL: GR00T tree not found: $GR00T_ROOT"; exit 2; }
CKPT=${CKPT:-none}
RESUME_TRAINING=${RESUME_TRAINING:-0}
: "${EXPECTED_TRAIN_CHECKPOINT_SHA256:?launcher must pin EXPECTED_TRAIN_CHECKPOINT_SHA256}"
if [ "$CKPT" = "none" ] || [ "$CKPT" = "null" ]; then
    [ "$RESUME_TRAINING" = 0 ] || {
        echo "FATAL: RESUME_TRAINING=1 requires CKPT"
        exit 2
    }
    INITIALIZATION_MODE=base
    ACTUAL_TRAIN_CHECKPOINT_SHA256=NONE
    CKPT_ARG=checkpoint=null
    RESUME_ARG=resume=false
else
    case "$RESUME_TRAINING" in
        0) INITIALIZATION_MODE=weight_warm_start; RESUME_ARG=resume=false ;;
        1) INITIALIZATION_MODE=resume_checkpoint; RESUME_ARG=resume=true ;;
        *) echo "FATAL: RESUME_TRAINING must be 0 or 1"; exit 2 ;;
    esac
    [ -f "$CKPT" ] || { echo "FATAL: checkpoint not found: $CKPT"; exit 2; }
    ACTUAL_TRAIN_CHECKPOINT_SHA256=$(sha256sum "$CKPT" | awk '{print $1}')
    CKPT_ARG="checkpoint=$CKPT"
fi
[ "$ACTUAL_TRAIN_CHECKPOINT_SHA256" = "$EXPECTED_TRAIN_CHECKPOINT_SHA256" ] || {
    echo "FATAL: training initialization changed after production submission"
    exit 2
}
[ -s "$TEACHER_CKPT" ] || {
    echo "FATAL: expert labeler checkpoint not found: $TEACHER_CKPT"
    exit 2
}
ACTUAL_TEACHER_CKPT_SHA256=$(sha256sum "$TEACHER_CKPT" | awk '{print $1}')
[ "$ACTUAL_TEACHER_CKPT_SHA256" = "$EXPECTED_TEACHER_CKPT_SHA256" ] || {
    echo "FATAL: expert labeler checkpoint hash mismatch"
    exit 2
}

SELECTED_ENDPOINT_MANIFEST="$FLEET_DIR/selected_endpoints.txt"
if [ "$REQUIRED_RENDERERS" -gt 0 ]; then
    : "${EXPECTED_IMAGE3F_RELEASE_ID:?EXPECTED_IMAGE3F_RELEASE_ID is required}"
    : "${EXPECTED_IMAGE3F_ADAPTER_SHA256:?EXPECTED_IMAGE3F_ADAPTER_SHA256 is required}"
    : "${EXPECTED_IMAGE3F_INFERENCE_STEPS:?EXPECTED_IMAGE3F_INFERENCE_STEPS is required}"
    : "${EXPECTED_IMAGE3F_TEACACHE_THRESHOLD:?EXPECTED_IMAGE3F_TEACACHE_THRESHOLD is required}"
    : "${EXPECTED_IMAGE3F_CFG_SCALE:?EXPECTED_IMAGE3F_CFG_SCALE is required}"
    : "${EXPECTED_IMAGE3F_CUDA_GRAPH:?EXPECTED_IMAGE3F_CUDA_GRAPH is required}"
    REFERENCE_PROBE_ARGS=()
    if [ "$REFERENCE_LAG" -gt 0 ]; then
        REFERENCE_PROBE_ARGS+=(--require-reference-history --required-reference-lag "$REFERENCE_LAG")
    fi
    EP_LIST=$(python3 "$ROOT/scripts/scaling/image3f_probe.py" \
        --endpoint-dir "$FLEET_DIR/endpoints" \
        --expected "$REQUIRED_RENDERERS" \
        --wait-timeout-s "$RENDER_READY_TIMEOUT_S" \
        "${REFERENCE_PROBE_ARGS[@]}" \
        --expected-release-id "$EXPECTED_IMAGE3F_RELEASE_ID" \
        --expected-adapter-sha256 "$EXPECTED_IMAGE3F_ADAPTER_SHA256" \
        --expected-inference-steps "$EXPECTED_IMAGE3F_INFERENCE_STEPS" \
        --expected-teacache-threshold "$EXPECTED_IMAGE3F_TEACACHE_THRESHOLD" \
        --expected-cfg-scale "$EXPECTED_IMAGE3F_CFG_SCALE" \
        --expected-cuda-graph "$EXPECTED_IMAGE3F_CUDA_GRAPH" \
        --expected-output-views "${VG_POLICY_VIEWS//:/,}" \
        --expected-output-layout "$VG_POLICY_VIEW_LAYOUT" \
        --expected-output-tile-shape 240,416,3 \
        --expected-output-stacked-shape 720,416,3 \
        --require-tensorrt-engine \
        --write-manifest "$SELECTED_ENDPOINT_MANIFEST" \
        --write-report "$FLEET_DIR/renderer_capabilities.json" --hydra-list)
else
    EP_LIST='[]'
    : > "$SELECTED_ENDPOINT_MANIFEST"
fi

PROJ=${PROJ:-roborender_dagger_scale}
EXP=${EXP:-dagger_${GLOBAL_NUM_ENVS}env_${ROLLOUT_STEPS}step_mix${MIX}}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:-$GR00T_ROOT/logs_rl/$PROJ/${EXP}-${SLURM_JOB_ID}}
mkdir -p "$TRAIN_RUN_DIR/slurm" "$TRAIN_RUN_DIR/output"
if [ "$RESUME_TRAINING" = 1 ]; then
    CHECKPOINT_DIR=$(dirname "$(realpath "$CKPT")")
    [ "$(realpath "$TRAIN_RUN_DIR")" = "$CHECKPOINT_DIR" ] || {
        echo "FATAL: strict resume must reuse the checkpoint experiment directory"
        exit 2
    }
    if [ "$USE_WANDB" = 1 ]; then
        [ -s "$TRAIN_RUN_DIR/meta.yaml" ] || {
            echo "FATAL: strict resume is missing the original trainer W&B metadata"
            exit 2
        }
        [ -s "$TRAIN_RUN_DIR/telemetry/WANDB_SIDECAR_RUN_ID" ] || {
            echo "FATAL: strict resume is missing the original media W&B run ID"
            exit 2
        }
    fi
fi
printf -v EXPECTED_FINAL_CHECKPOINT '%s/model_step_%06d.pt' \
    "$TRAIN_RUN_DIR" "$NUM_ROLLOUT_ITERATIONS"
printf -v EXPECTED_TEST_STOP_CHECKPOINT '%s/model_step_%06d.pt' \
    "$TRAIN_RUN_DIR" "$TEST_STOP_AFTER_CHECKPOINT_STEP"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((15000 + SLURM_JOB_ID % 20000))

export HOME=${HOME_OVERRIDE:-$STAGE_ROOT/runtime/home}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$STAGE_ROOT/runtime/cache/xdg}
export XDG_DATA_HOME=${XDG_DATA_HOME:-$STAGE_ROOT/runtime/cache/share}
export TMPDIR=${TMPDIR:-$STAGE_ROOT/runtime/tmp}
export OMNI_KIT_ACCEPT_EULA=YES
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export IMAGE3F_ENDPOINT_DIR="$FLEET_DIR/endpoints"
export IMAGE3F_ENDPOINT_MANIFEST="$SELECTED_ENDPOINT_MANIFEST"
export IMAGE3F_LOCAL_ENVS="$LOCAL_NUM_ENVS"
export IMAGE3F_REFERENCE_LAG="$REFERENCE_LAG"
export IMAGE3F_EXPECT_OUTPUT_VIEWS="${VG_POLICY_VIEWS//:/,}"
export IMAGE3F_EXPECT_OUTPUT_SHAPE=240,416,3
export IMAGE3F_RUN_ID=${IMAGE3F_RUN_ID:-$SLURM_JOB_ID}
export IMAGE3F_REQUEST_TIMEOUT_S=${IMAGE3F_REQUEST_TIMEOUT_S:-120}
export BASE_RENDER_SEED=${BASE_RENDER_SEED:-42}
export IMAGE_DAGGER_EXPECT_PRIMARY_VIEW="$VG_PRIMARY_VIEW"
export IMAGE_DAGGER_EXPECT_POLICY_VIEWS="${VG_POLICY_VIEWS//:/,}"
export IMAGE_DAGGER_EXPECT_POLICY_VIEW_COUNT="$VG_POLICY_VIEW_COUNT"
export IMAGE_DAGGER_EXPECT_POLICY_VIEW_LAYOUT="$VG_POLICY_VIEW_LAYOUT"
export IMAGE_DAGGER_EXPECT_POLICY_IMAGE_SHAPE="${VG_POLICY_IMAGE_SHAPE//x/,}"
export IMAGE_DAGGER_DUMP_SIM_RGB=${IMAGE_DAGGER_DUMP_SIM_RGB:-0}
case "$IMAGE_DAGGER_DUMP_SIM_RGB" in 0|1) ;; *)
    echo "FATAL: IMAGE_DAGGER_DUMP_SIM_RGB must be 0 or 1"
    exit 2
esac
if [ -n "${IMAGE_DAGGER_DUMP_MAX_STEPS:-}" ]; then
    [[ "$IMAGE_DAGGER_DUMP_MAX_STEPS" =~ ^[0-9]+$ ]] || {
        echo "FATAL: IMAGE_DAGGER_DUMP_MAX_STEPS must be a non-negative integer"
        exit 2
    }
    export IMAGE_DAGGER_DUMP_MAX_STEPS
fi
if [ -n "${IMAGE_DAGGER_DUMP_MAX_ENVS:-}" ]; then
    [[ "$IMAGE_DAGGER_DUMP_MAX_ENVS" =~ ^[0-9]+$ ]] || {
        echo "FATAL: IMAGE_DAGGER_DUMP_MAX_ENVS must be a non-negative integer"
        exit 2
    }
    export IMAGE_DAGGER_DUMP_MAX_ENVS
fi
case "${IMAGE_DAGGER_VERIFY_DDP_SYNC:-0}" in
    0) unset IMAGE_DAGGER_VERIFY_DDP_SYNC ;;
    1)
        [ "$TRAINER_NODES" -gt 1 ] || {
            echo "FATAL: DDP synchronization verification requires multiple trainer ranks"
            exit 2
        }
        export IMAGE_DAGGER_VERIFY_DDP_SYNC=1
        ;;
    *) echo "FATAL: IMAGE_DAGGER_VERIFY_DDP_SYNC must be 0 or 1"; exit 2 ;;
esac
case "${IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS:-}" in
    1) export IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS=1 ;;
    *)
        echo "FATAL: packed production profiles require IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS=1"
        exit 2
        ;;
esac
export IMAGE_DAGGER_EXPECT_ROLLOUT_ACTION_SOURCE="$VG_ROLLOUT_ACTION_SOURCE"
export IMAGE_DAGGER_EXPECT_BETA_SCHEDULE="$VG_BETA_SCHEDULE"
export IMAGE_DAGGER_EXPECT_BETA_START="$VG_BETA_START"
export IMAGE_DAGGER_EXPECT_BETA_END="$VG_BETA_END"
export IMAGE_DAGGER_EXPECT_BETA_DECAY_ITERS="$VG_BETA_DECAY_ITERS"
export IMAGE_DAGGER_EXPECT_TEACHER_WARMUP_STEPS=0
export IMAGE_DAGGER_EXPECT_ENFORCE_TEACHER="$ENFORCE_TEACHER_ROLLOUT"
export IMAGE_DAGGER_EXPECT_TEACHER_RATIO="$TEACHER_ROLLOUT_RATIO"
export IMAGE_DAGGER_EXPECT_LEARNING_EPOCHS="$NUM_LEARNING_EPOCHS"
export IMAGE_DAGGER_EXPECT_VALUE_LOSS_COEF=0.0
export IMAGE_DAGGER_EXPECT_ENTROPY_COEF=0.0
export WANDB_EXPECTED_EMAIL WANDB_MODE WANDB_ENTITY WANDB_AUTHORIZATION_SCOPE WANDB_PYTHON USE_WANDB
export ROOT GR00T_ROOT FLEET_DIR GLOBAL_NUM_ENVS ROLLOUT_STEPS MIX REFERENCE_LAG
export RUN_PROFILE
export STAGE_ROOT ANCHORS TEACHER_CKPT EXPECTED_TEACHER_CKPT_SHA256
export LOCAL_NUM_ENVS REQUIRED_RENDERERS GRADIENT_ACCUMULATION_STEPS EP_LIST CKPT
export PROJ EXP TRAIN_RUN_DIR MASTER_ADDR MASTER_PORT TRAINER_NODES
export VIRAL_PYTHON VIRAL_CONDA_SH VIRAL_ENV_NAME VIRAL_VENV_ACTIVATE
export NUM_MINI_BATCHES PER_DEVICE_TRAIN_BATCH_SIZE
export NUM_ROLLOUT_ITERATIONS NUM_LEARNING_EPOCHS
export BEHAVIOR_SOURCE VG_ROLLOUT_ACTION_SOURCE VG_BETA_SCHEDULE VG_BETA_START VG_BETA_END VG_BETA_DECAY_ITERS
export TEACHER_ROLLOUT_RATIO ENFORCE_TEACHER_ROLLOUT VG_PRIMARY_VIEW
export VG_POLICY_VIEWS VG_POLICY_VIEW_COUNT VG_POLICY_VIEW_LAYOUT VG_POLICY_IMAGE_SHAPE
export INITIALIZATION_MODE CKPT_ARG RESUME_ARG RESUME_TRAINING
export SAVE_FREQUENCY TEST_STOP_AFTER_CHECKPOINT_STEP
export TRAINER_CPUS_PER_TASK TRAINER_MEM_PER_NODE

export ROBORENDER_TELEMETRY_DIR="$TRAIN_RUN_DIR/telemetry"
export IMAGE_DAGGER_TELEMETRY_SCALAR_EVERY=${IMAGE_DAGGER_TELEMETRY_SCALAR_EVERY:-10}
export IMAGE_DAGGER_TELEMETRY_IMAGE_EVERY=${IMAGE_DAGGER_TELEMETRY_IMAGE_EVERY:-300}
export IMAGE_DAGGER_TELEMETRY_CLIP_EVERY=${IMAGE_DAGGER_TELEMETRY_CLIP_EVERY:-300}
export IMAGE_DAGGER_TELEMETRY_CLIP_FRAMES=${IMAGE_DAGGER_TELEMETRY_CLIP_FRAMES:-32}
export IMAGE_DAGGER_TELEMETRY_CLIP_FPS=${IMAGE_DAGGER_TELEMETRY_CLIP_FPS:-10}
export IMAGE_DAGGER_TELEMETRY_REQUIRED=1
export IMAGE_DAGGER_TELEMETRY_REQUIRE_VIDEO=1
mkdir -p "$ROBORENDER_TELEMETRY_DIR/scalars" \
    "$ROBORENDER_TELEMETRY_DIR/images" \
    "$ROBORENDER_TELEMETRY_DIR/videos"
WANDB_WATCHER_LOG="$TRAIN_RUN_DIR/wandb_media_watcher.log"
WANDB_WATCHER_READY="$TRAIN_RUN_DIR/telemetry/WANDB_WATCHER_READY.json"
# These markers belong to the previous watcher process.  The stable run ID
# and upload-state files remain; removing only lifecycle markers lets the new
# process resume that same media run instead of immediately final-draining.
rm -f "$WANDB_WATCHER_READY" "$TRAIN_RUN_DIR/WANDB_UPLOAD_STOP"
if [ "$USE_WANDB" = 1 ]; then
    [ "$WANDB_EXPECTED_EMAIL" = ldu@nvidia.com ] || {
        echo "FATAL: W&B account binding changed"; exit 2;
    }
    [ "$WANDB_ENTITY" = nvidia ] || {
        echo "FATAL: W&B team entity binding changed"; exit 2;
    }
    [ "$WANDB_AUTHORIZATION_SCOPE" = team ] || {
        echo "FATAL: W&B authorization scope changed"; exit 2;
    }
    [ -x "$WANDB_PYTHON" ] || {
        echo "FATAL: trainer W&B Python is not executable: $WANDB_PYTHON"
        exit 2
    }
    WANDB_PREFLIGHT=("$WANDB_PYTHON" "$ROOT/scripts/scaling/wandb_preflight.py"
        --expected-email "$WANDB_EXPECTED_EMAIL" --entity "$WANDB_ENTITY"
        --authorization-scope "$WANDB_AUTHORIZATION_SCOPE" --format shell)
    WANDB_IDENTITY=$(WANDB_MODE="$WANDB_MODE" "${WANDB_PREFLIGHT[@]}")
    case "$WANDB_IDENTITY" in
        WANDB_ENTITY=?*) WANDB_ENTITY=${WANDB_IDENTITY#WANDB_ENTITY=} ;;
        *) echo "FATAL: rank-zero W&B identity preflight failed"; exit 2 ;;
    esac
    export WANDB_ENTITY
    TRAINER_USE_WANDB=true
    "$WANDB_PYTHON" "$ROOT/scripts/scaling/wandb_run_watcher.py" \
        --run-dir "$TRAIN_RUN_DIR" \
        --project "$PROJ" \
        --name "${EXP}-media-${SLURM_JOB_ID}" \
        --expected-email "$WANDB_EXPECTED_EMAIL" \
        --entity "$WANDB_ENTITY" \
        --authorization-scope "$WANDB_AUTHORIZATION_SCOPE" \
        --poll-interval-s "$WANDB_WATCHER_POLL_S" \
        --ready-marker "$WANDB_WATCHER_READY" \
        --max-consecutive-error-scans 4 \
        --finish-marker WANDB_UPLOAD_STOP \
        >>"$WANDB_WATCHER_LOG" 2>&1 &
    WANDB_WATCHER_PID=$!

    WATCHER_READY_DEADLINE=$((SECONDS + WANDB_WATCHER_READY_TIMEOUT_S))
    while [ ! -s "$WANDB_WATCHER_READY" ] && \
          kill -0 "$WANDB_WATCHER_PID" 2>/dev/null && \
          [ "$SECONDS" -lt "$WATCHER_READY_DEADLINE" ]; do
        sleep 1
    done
    if [ ! -s "$WANDB_WATCHER_READY" ]; then
        if kill -0 "$WANDB_WATCHER_PID" 2>/dev/null; then
            kill -TERM "$WANDB_WATCHER_PID" 2>/dev/null || true
        fi
        if wait "$WANDB_WATCHER_PID"; then
            WATCHER_START_RC=0
        else
            WATCHER_START_RC=$?
        fi
        WANDB_WATCHER_PID=""
        if [ "$WATCHER_START_RC" -eq 0 ]; then
            WATCHER_START_RC=70
        fi
        echo "FATAL: W&B media watcher did not become ready; see $WANDB_WATCHER_LOG"
        exit "$WATCHER_START_RC"
    fi
else
    WANDB_ENTITY=DISABLED
    TRAINER_USE_WANDB=false
fi
export WANDB_ENTITY TRAINER_USE_WANDB

python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --format json \
    | tee "$TRAIN_RUN_DIR/scale_plan.json"
cat > "$TRAIN_RUN_DIR/launch_manifest.txt" <<EOF
fast_roborender_revision=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
fast_roborender_dirty_status_sha256=$(git -C "$ROOT" status --porcelain | sha256sum | awk '{print $1}')
gr00t_revision=$(git -C "$GR00T_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
gr00t_dirty_status_sha256=$(git -C "$GR00T_ROOT" status --porcelain | sha256sum | awk '{print $1}')
initialization_mode=$INITIALIZATION_MODE
strict_resume=$RESUME_TRAINING
checkpoint=$CKPT
checkpoint_sha256=$ACTUAL_TRAIN_CHECKPOINT_SHA256
expert_labeler_checkpoint=$TEACHER_CKPT
expert_labeler_checkpoint_sha256=$ACTUAL_TEACHER_CKPT_SHA256
fleet_dir=$FLEET_DIR
trainer_nodes=$TRAINER_NODES
local_envs=$LOCAL_NUM_ENVS
global_envs=$GLOBAL_NUM_ENVS
rollout_steps=$ROLLOUT_STEPS
transitions_per_rollout_iteration=$TRANSITIONS_PER_ROLLOUT_ITERATION
transitions_per_optimizer_step=$TRANSITIONS_PER_OPTIMIZER_STEP
optimizer_steps_per_rollout_iteration=$OPTIMIZER_STEPS_PER_ROLLOUT_ITERATION
renderers=$REQUIRED_RENDERERS
reference_lag=$REFERENCE_LAG
trainer_num_total_batches=$NUM_ROLLOUT_ITERATIONS
num_rollout_iterations=$NUM_ROLLOUT_ITERATIONS
num_learning_epochs=$NUM_LEARNING_EPOCHS
behavior_source=$BEHAVIOR_SOURCE
videogen_rollout_action_source=$VG_ROLLOUT_ACTION_SOURCE
videogen_beta_schedule=$VG_BETA_SCHEDULE
videogen_beta_start=$VG_BETA_START
videogen_beta_end=$VG_BETA_END
videogen_beta_decay_iters=$VG_BETA_DECAY_ITERS
rollout_with_teacher_num_steps=0
enforce_teacher_rollout=$ENFORCE_TEACHER_ROLLOUT
teacher_rollout_ratio=$TEACHER_ROLLOUT_RATIO
primary_view=$VG_PRIMARY_VIEW
policy_views=${VG_POLICY_VIEWS//:/,}
policy_view_count=$VG_POLICY_VIEW_COUNT
policy_view_layout=$VG_POLICY_VIEW_LAYOUT
policy_image_shape=$VG_POLICY_IMAGE_SHAPE
initial_policy_checkpoint=$TRAIN_RUN_DIR/initial_policy.pt
expected_final_checkpoint=$EXPECTED_FINAL_CHECKPOINT
camera_calibration_acceptance=${CAMERA_CALIBRATION_ACCEPTANCE:-NONE}
camera_calibration_acceptance_sha256=${EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256:-NONE}
camera_calibration_run_id=${CAMERA_CALIBRATION_RUN_ID:-NONE}
prerequisite_profile=${PREREQUISITE_PROFILE_EXPECTED:-NONE}
prerequisite_profile_validation=${PREREQUISITE_PROFILE_VALIDATION:-NONE}
prerequisite_profile_validation_sha256=${EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256:-NONE}
finite_bc_loss_gate=$IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS
checkpoint_frequency_rollout_iterations=$SAVE_FREQUENCY
test_stop_after_checkpoint_step=$TEST_STOP_AFTER_CHECKPOINT_STEP
value_loss_coef=0.0
entropy_coef=0.0
wandb_expected_email=$WANDB_EXPECTED_EMAIL
wandb_enabled=$USE_WANDB
wandb_entity=$WANDB_ENTITY
wandb_project=$PROJ
wandb_media_run=${EXP}-media-${SLURM_JOB_ID}
wandb_watcher_drain_timeout_s=$WANDB_WATCHER_DRAIN_TIMEOUT_S
EOF

echo "=================================================================="
echo " distributed DAgger: ${TRAINER_NODES} ranks x ${LOCAL_NUM_ENVS} envs = ${GLOBAL_NUM_ENVS}"
echo " rollout: ${ROLLOUT_STEPS} steps, ${TRANSITIONS_PER_ROLLOUT_ITERATION} transitions/rollout iteration"
echo " learning: ${NUM_LEARNING_EPOCHS} epoch(s), ${OPTIMIZER_STEPS_PER_ROLLOUT_ITERATION} optimizer steps/rollout iteration"
echo " optimizer step: ${TRANSITIONS_PER_OPTIMIZER_STEP} global transitions"
echo " rollout iterations: ${NUM_ROLLOUT_ITERATIONS} (trainer num_total_batches)"
echo " behavior: ${BEHAVIOR_SOURCE}, videogen rollout source=${VG_ROLLOUT_ACTION_SOURCE}, beta=${VG_BETA_SCHEDULE}:${VG_BETA_START}->${VG_BETA_END}/${VG_BETA_DECAY_ITERS}, legacy teacher mixer=false, warmup teacher steps=0"
echo " objective: pure DAgger BC (value_loss_coef=0.0, entropy_coef=0.0)"
echo " actor RGB input: generated policy views=${VG_POLICY_VIEWS} (${VG_POLICY_VIEW_COUNT} views, ${VG_POLICY_VIEW_LAYOUT}, ${VG_POLICY_IMAGE_SHAPE})"
echo " renderers: ${REQUIRED_RENDERERS}, ray-traced mix=${MIX}, reference policy=$([ "$REFERENCE_LAG" -eq 0 ] && echo fixed-anchor || echo generated-lag), RGB lag=${REFERENCE_LAG}"
echo " initialization: $INITIALIZATION_MODE ($CKPT)"
echo " output: $TRAIN_RUN_DIR"
echo " W&B enabled: $USE_WANDB ($WANDB_ENTITY/$PROJ)"
echo "=================================================================="

TRAINER_CONTAINER_ARGS=()
if [ -n "$TRAINER_CONTAINER_IMAGE" ]; then
    TRAINER_CONTAINER_ARGS+=(--container-image="$TRAINER_CONTAINER_IMAGE")
fi
if [ -n "$TRAINER_CONTAINER_MOUNTS" ]; then
    TRAINER_CONTAINER_ARGS+=(--container-mounts="$TRAINER_CONTAINER_MOUNTS")
fi
srun --nodes="$TRAINER_NODES" --ntasks="$TRAINER_NODES" --ntasks-per-node=1 \
    --gpus-per-task=1 --cpus-per-task="$TRAINER_CPUS_PER_TASK" \
    --mem="$TRAINER_MEM_PER_NODE" --exact --exclusive \
    --gpu-bind=closest --cpu-bind=cores --kill-on-bad-exit=1 \
    "${TRAINER_CONTAINER_ARGS[@]}" \
    --output="$TRAIN_RUN_DIR/slurm/rank_%t.out" \
    --error="$TRAIN_RUN_DIR/slurm/rank_%t.err" \
    bash -lc '
        set -Eeuo pipefail
        if [ -n "$VIRAL_VENV_ACTIVATE" ]; then
            [ -f "$VIRAL_VENV_ACTIVATE" ] || {
                echo "FATAL: VIRAL_VENV_ACTIVATE not found: $VIRAL_VENV_ACTIVATE" >&2
                exit 2
            }
            source "$VIRAL_VENV_ACTIVATE"
        else
            [ -f "$VIRAL_CONDA_SH" ] || {
                echo "FATAL: VIRAL_CONDA_SH not found: $VIRAL_CONDA_SH" >&2
                exit 2
            }
            source "$VIRAL_CONDA_SH"
            conda activate "$VIRAL_ENV_NAME"
        fi
        cd "$GR00T_ROOT"
        RANK_RENDER_SEED=$((BASE_RENDER_SEED + SLURM_PROCID))
        if [ "$TRAINER_NODES" -eq 1 ]; then
            ACCELERATE_ARGS=(
                --num_processes 1
                --num_cpu_threads_per_process "$TRAINER_CPUS_PER_TASK"
            )
        else
            ACCELERATE_ARGS=(
                --multi_gpu
                --num_machines "$TRAINER_NODES"
                --num_processes "$TRAINER_NODES"
                --machine_rank "$SLURM_PROCID"
                --main_process_ip "$MASTER_ADDR"
                --main_process_port "$MASTER_PORT"
                --same_network --rdzv_backend static --max_restarts 0
                --num_cpu_threads_per_process "$TRAINER_CPUS_PER_TASK"
            )
        fi
        exec accelerate launch "${ACCELERATE_ARGS[@]}" \
            gr00t/rl/train_agent_trl.py \
                +exp=loco_manip/wsdpt_image_dagger_3view \
                num_envs="$LOCAL_NUM_ENVS" \
                algo.config.num_steps_per_env="$ROLLOUT_STEPS" \
                algo.config.num_mini_batches="$NUM_MINI_BATCHES" \
                algo.config.num_learning_epochs="$NUM_LEARNING_EPOCHS" \
                algo.config.rollout_with_teacher_num_steps=0 \
                algo.config.enforce_teacher_rollout="$ENFORCE_TEACHER_ROLLOUT" \
                algo.config.ratio_teacher_rollout="$TEACHER_ROLLOUT_RATIO" \
                algo.config.value_loss_coef=0.0 \
                algo.config.entropy_coef=0.0 \
                algo.config.videogen.rollout_action_source="$VG_ROLLOUT_ACTION_SOURCE" \
                ++algo.config.videogen.beta_schedule="$VG_BETA_SCHEDULE" \
                ++algo.config.videogen.beta_start="$VG_BETA_START" \
                ++algo.config.videogen.beta_end="$VG_BETA_END" \
                ++algo.config.videogen.beta_decay_iters="$VG_BETA_DECAY_ITERS" \
                algo.trl.per_device_train_batch_size="$PER_DEVICE_TRAIN_BATCH_SIZE" \
                algo.trl.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
                algo.trl.num_total_batches="$NUM_ROLLOUT_ITERATIONS" \
                algo.trl.total_episodes=null \
                callbacks.model_save.save_frequency="$SAVE_FREQUENCY" \
                callbacks.model_save.save_initial_checkpoint=true \
                callbacks.model_save.test_stop_after_checkpoint_step="$TEST_STOP_AFTER_CHECKPOINT_STEP" \
                headless=true "$CKPT_ARG" "$RESUME_ARG" \
                teacher_actor_path="$TEACHER_CKPT" \
                algo.config.videogen.raytraced_mix_prob="$MIX" \
                ++algo.config.videogen.seed="$RANK_RENDER_SEED" \
                ++algo.config.videogen.reference_lag="$REFERENCE_LAG" \
                ++algo.config.videogen.primary_view="$VG_PRIMARY_VIEW" \
                simulator.config.cameras.policy_views="[head_color,hand_left,hand_right]" \
                policy_view_count="$VG_POLICY_VIEW_COUNT" \
                ++env.config.image_dagger_policy_view_count="$VG_POLICY_VIEW_COUNT" \
                algo.config.videogen.image3f_endpoint_files="$EP_LIST" \
                algo.config.videogen.anchor_pool_dir="$ANCHORS" \
                algo.config.videogen.anchor_index="${ANCHOR_IDX:-0}" \
                simulator.config.render_results=false \
                use_wandb="$TRAINER_USE_WANDB" \
                wandb.wandb_entity="$WANDB_ENTITY" \
                wandb.wandb_group=null \
                project_name="$PROJ" experiment_name="$EXP" \
                experiment_dir="$TRAIN_RUN_DIR" \
                hydra.run.dir="$TRAIN_RUN_DIR/.hydra/rank_${SLURM_PROCID}" \
                output_dir="$TRAIN_RUN_DIR/output/rank_${SLURM_PROCID}"
    ' &
SRUN_PID=$!

# W&B media is a production gate, not a best-effort sidecar.  Check it while
# the trainer step is live and stop the distributed step if observability dies.
while kill -0 "$SRUN_PID" 2>/dev/null; do
    if [ "$USE_WANDB" = 1 ] && ! kill -0 "$WANDB_WATCHER_PID" 2>/dev/null; then
        if wait "$WANDB_WATCHER_PID"; then
            WATCHER_RUN_RC=0
        else
            WATCHER_RUN_RC=$?
        fi
        WANDB_WATCHER_PID=""
        echo "FATAL: W&B media watcher exited during training (rc=$WATCHER_RUN_RC); see $WANDB_WATCHER_LOG"
        _terminate_srun
        if wait "$SRUN_PID"; then
            TRAIN_RC=0
        else
            TRAIN_RC=$?
        fi
        SRUN_PID=""
        exit 70
    fi
    sleep 5
done

if wait "$SRUN_PID"; then
    TRAIN_RC=0
else
    TRAIN_RC=$?
fi
SRUN_PID=""
if [ "$PRE_TIMEOUT_SIGNALLED" -eq 1 ]; then
    # Normalize the expected SIGTERM status (normally 143) so manifests and
    # monitors distinguish a bounded wall-time stop from an arbitrary failure.
    TRAIN_RC=124
fi

# Close the race where the watcher and trainer exit between the final poll.
if [ "$USE_WANDB" = 1 ] && ! kill -0 "$WANDB_WATCHER_PID" 2>/dev/null; then
    if wait "$WANDB_WATCHER_PID"; then
        WATCHER_RUN_RC=0
    else
        WATCHER_RUN_RC=$?
    fi
    WANDB_WATCHER_PID=""
    echo "FATAL: W&B media watcher exited before final drain (rc=$WATCHER_RUN_RC)"
    if [ "$TRAIN_RC" -eq 0 ]; then
        exit 70
    fi
fi

if [ "$TRAIN_RC" -eq 0 ]; then
    if [ ! -s "$TRAIN_RUN_DIR/initial_policy.pt" ]; then
        echo "FATAL: successful trainer did not create initial_policy.pt" >&2
        TRAIN_RC=71
    fi
    if [ "$TEST_STOP_AFTER_CHECKPOINT_STEP" -gt 0 ]; then
        if [ ! -s "$EXPECTED_TEST_STOP_CHECKPOINT" ]; then
            echo "FATAL: checkpoint mechanics phase A did not create $EXPECTED_TEST_STOP_CHECKPOINT" >&2
            TRAIN_RC=71
        fi
        if [ -e "$EXPECTED_FINAL_CHECKPOINT" ]; then
            echo "FATAL: checkpoint mechanics phase A unexpectedly reached the absolute target" >&2
            TRAIN_RC=71
        fi
    elif [ ! -s "$EXPECTED_FINAL_CHECKPOINT" ]; then
        echo "FATAL: successful trainer did not create exact final checkpoint $EXPECTED_FINAL_CHECKPOINT" >&2
        TRAIN_RC=71
    fi
fi

# If a pre-timeout run produced any trained checkpoint, training necessarily
# began and the iteration-zero baseline must also exist. Do not misclassify a
# job killed before trainer initialization as an artifact-integrity failure.
if [ "$TRAIN_RC" -eq 124 ] && compgen -G "$TRAIN_RUN_DIR/model_step_*.pt" >/dev/null; then
    if [ ! -s "$TRAIN_RUN_DIR/initial_policy.pt" ]; then
        echo "FATAL: checkpointed pre-timeout run is missing initial_policy.pt" >&2
        TRAIN_RC=71
    fi
fi

if [ "$TRAIN_RC" -eq 0 ]; then
    case "$RUN_PROFILE" in
        calibration_3view|smoke_32x300|smoke_256x300)
            FINAL_CHECKPOINT="$EXPECTED_FINAL_CHECKPOINT"
            if [ ! -s "$FINAL_CHECKPOINT" ]; then
                echo "FATAL: validation profile did not create model_step_000001.pt" >&2
                TRAIN_RC=71
            else
                INITIAL_SHA256=$(sha256sum "$TRAIN_RUN_DIR/initial_policy.pt" | awk '{print $1}')
                FINAL_SHA256=$(sha256sum "$FINAL_CHECKPOINT" | awk '{print $1}')
                BC_LOSS_MARKER="$TRAIN_RUN_DIR/validation/dagger_bc_loss_step_000000001.json"
                if [ ! -s "$BC_LOSS_MARKER" ]; then
                    echo "FATAL: validation profile is missing its finite BC-loss marker" >&2
                    exit 71
                fi
                if ! python3 - "$BC_LOSS_MARKER" "$TRAINER_NODES" <<'PY'
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "schema": "fast_roborender.dagger_bc_loss.v1",
    "update": 1,
    "world_size": int(sys.argv[2]),
    "finite_verified": True,
    "nonnegative_verified": True,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"FATAL: invalid BC-loss marker {key}={payload.get(key)!r}")
loss = payload.get("loss")
if isinstance(loss, bool) or not isinstance(loss, (int, float)):
    raise SystemExit("FATAL: BC-loss marker has no numeric loss")
if not math.isfinite(float(loss)) or float(loss) < 0.0:
    raise SystemExit(f"FATAL: BC-loss marker has invalid loss {loss!r}")
PY
                then
                    echo "FATAL: finite BC-loss marker validation failed" >&2
                    exit 71
                fi
                BC_LOSS_MARKER_SHA256=$(sha256sum "$BC_LOSS_MARKER" | awk '{print $1}')
                DDP_MARKER=""
                DDP_MARKER_SHA256=""
                if [ "$RUN_PROFILE" = smoke_256x300 ]; then
                    DDP_MARKER="$TRAIN_RUN_DIR/validation/ddp_policy_sync_step_000000001.json"
                    if [ ! -s "$DDP_MARKER" ]; then
                        echo "FATAL: distributed smoke is missing its DDP synchronization marker" >&2
                        exit 71
                    fi
                    if ! python3 - "$DDP_MARKER" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "schema": "fast_roborender.ddp_policy_sync.v1",
    "verified": True,
    "optimizer_update": 1,
    "world_size": 8,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(
            f"FATAL: invalid DDP synchronization marker {key}={payload.get(key)!r}"
        )
PY
                    then
                        echo "FATAL: distributed smoke DDP marker validation failed" >&2
                        exit 71
                    fi
                    DDP_MARKER_SHA256=$(sha256sum "$DDP_MARKER" | awk '{print $1}')
                fi
                CAMERA_CALIBRATION_MANIFEST=""
                CAMERA_CALIBRATION_MANIFEST_SHA256=""
                VALIDATION_STATUS=validated
                if [ "$RUN_PROFILE" = calibration_3view ]; then
                    CAMERA_PANEL_0="$TRAIN_RUN_DIR/render_dumps/step000_env0.png"
                    CAMERA_PANEL_1="$TRAIN_RUN_DIR/render_dumps/step001_env0.png"
                    for dump in "$CAMERA_PANEL_0" "$CAMERA_PANEL_1"; do
                        if [ ! -s "$dump" ]; then
                            echo "FATAL: calibration profile is missing camera panel $dump" >&2
                            exit 71
                        fi
                        if ! python3 - "$dump" <<'PY'
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("rb") as handle:
    signature = handle.read(8)
    length = struct.unpack(">I", handle.read(4))[0]
    chunk_type = handle.read(4)
    ihdr = handle.read(length)
if signature != b"\x89PNG\r\n\x1a\n" or chunk_type != b"IHDR" or length != 13:
    raise SystemExit(f"FATAL: malformed calibration PNG {path}")
width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
    ">IIBBBBB", ihdr
)
expected = (1248, 720, 8, 2, 0, 0, 0)
actual = (width, height, bit_depth, color_type, compression, filtering, interlace)
if actual != expected:
    raise SystemExit(
        f"FATAL: calibration panel contract mismatch for {path}: "
        f"expected {expected}, got {actual}"
    )
PY
                        then
                            echo "FATAL: calibration panel validation failed" >&2
                            exit 71
                        fi
                    done
                    CAMERA_PANEL_0_SHA256=$(sha256sum "$CAMERA_PANEL_0" | awk '{print $1}')
                    CAMERA_PANEL_1_SHA256=$(sha256sum "$CAMERA_PANEL_1" | awk '{print $1}')
                    CAMERA_CALIBRATION_MANIFEST="$TRAIN_RUN_DIR/validation/camera_calibration_manifest.json"
                    CAMERA_TMP="$CAMERA_CALIBRATION_MANIFEST.tmp.$$"
                    cat > "$CAMERA_TMP" <<EOF
{
  "schema": "fast_roborender.camera_calibration.v2",
  "status": "awaiting_visual_acceptance",
  "run_id": "$RUN_ID",
  "fast_roborender_revision": "$FAST_REVISION",
  "gr00t_revision": "$EXPECTED_GR00T_REVISION",
  "policy_views": ["head_color", "hand_left", "hand_right"],
  "columns": ["head_color", "hand_left", "hand_right"],
  "rows": ["depth_control", "simulator_rgb", "generated_rgb"],
  "tile_shape_hwc": [240, 416, 3],
  "panel_shape_hwc": [720, 1248, 3],
  "panels": [
    {"path": "$CAMERA_PANEL_0", "sha256": "$CAMERA_PANEL_0_SHA256"},
    {"path": "$CAMERA_PANEL_1", "sha256": "$CAMERA_PANEL_1_SHA256"}
  ]
}
EOF
                    mv "$CAMERA_TMP" "$CAMERA_CALIBRATION_MANIFEST"
                    CAMERA_CALIBRATION_MANIFEST_SHA256=$(sha256sum "$CAMERA_CALIBRATION_MANIFEST" | awk '{print $1}')
                    VALIDATION_STATUS=awaiting_camera_visual_acceptance
                fi
                VALIDATION_TMP="$TRAIN_RUN_DIR/profile_validation_manifest.txt.tmp.$$"
                cat > "$VALIDATION_TMP" <<EOF
run_profile=$RUN_PROFILE
fast_roborender_revision=$FAST_REVISION
gr00t_revision=$EXPECTED_GR00T_REVISION
global_envs=$GLOBAL_NUM_ENVS
rollout_steps=$ROLLOUT_STEPS
trainer_nodes=$TRAINER_NODES
policy_views=${VG_POLICY_VIEWS//:/,}
policy_view_count=$VG_POLICY_VIEW_COUNT
policy_view_layout=$VG_POLICY_VIEW_LAYOUT
policy_image_shape=$VG_POLICY_IMAGE_SHAPE
initial_policy_checkpoint=$TRAIN_RUN_DIR/initial_policy.pt
initial_policy_sha256=$INITIAL_SHA256
final_checkpoint=$FINAL_CHECKPOINT
final_checkpoint_sha256=$FINAL_SHA256
finite_bc_loss_marker=$BC_LOSS_MARKER
finite_bc_loss_marker_sha256=$BC_LOSS_MARKER_SHA256
ddp_sync_marker=$DDP_MARKER
ddp_sync_marker_sha256=$DDP_MARKER_SHA256
camera_calibration_manifest=$CAMERA_CALIBRATION_MANIFEST
camera_calibration_manifest_sha256=$CAMERA_CALIBRATION_MANIFEST_SHA256
rollout_iterations=1
status=$VALIDATION_STATUS
EOF
                mv "$VALIDATION_TMP" "$TRAIN_RUN_DIR/profile_validation_manifest.txt"
                if [ "$RUN_PROFILE" = calibration_3view ]; then
                    : > "$TRAIN_RUN_DIR/PROFILE_TECHNICALLY_VALIDATED"
                else
                    : > "$TRAIN_RUN_DIR/PROFILE_VALIDATED"
                fi
            fi
            ;;
    esac
fi

if [ "$TRAIN_RC" -eq 0 ]; then
    echo RUN_DONE
fi
exit "$TRAIN_RC"
