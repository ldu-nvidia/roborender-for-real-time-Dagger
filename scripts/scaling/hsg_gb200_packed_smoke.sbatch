#!/bin/bash
# HSG packed smoke: four GB200s/node, distinct renderer and Isaac/GR00T images.
# Submit only through submit_hsg_gb200_smoke.sh.
#SBATCH --account=llmservice_nemotron_ultra
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=800G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=rr_gb200_smoke
set -Eeuo pipefail

required=(STAGE_ROOT ROOT GR00T_ROOT RUN_PROFILE RUN_ID RUN_ROOT FLEET_DIR TRAIN_RUN_DIR
          SLURM_BIN_DIR
          PACKED_NODES PACKED_GPUS_PER_NODE GLOBAL_NUM_ENVS ROLLOUT_STEPS TRAINER_RANKS
          EXPECTED_RENDERERS RENDERER_CONTAINER TRAINER_CONTAINER IMAGE3F_ROOT
          ROBORENDER_MODEL_BASE IMAGE3F_GENERATED_ROOT ANCHORS TEACHER_CKPT
          TORCH_HOME RESNET18_WEIGHTS EXPECTED_RESNET18_SHA256
          IMAGE_DAGGER_FFMPEG
          PACKED_QOS EXPECTED_GR00T_SOURCE_TREE_SHA256
          WANDB_ONLINE_VALIDATION EXPECTED_WANDB_ONLINE_VALIDATION_SHA256
          WANDB_WATCHER_POLL_S WANDB_WATCHER_READY_TIMEOUT_S
          WANDB_WATCHER_DRAIN_TIMEOUT_S
          EXPECTED_TEACHER_CKPT_SHA256 EXPECTED_FAST_ROBORENDER_REVISION
          EXPECTED_GR00T_REVISION EXPECTED_IMAGE3F_RELEASE_ID
          EXPECTED_IMAGE3F_ADAPTER_SHA256 IMAGE_DAGGER_DUMP_SIM_RGB USE_WANDB)
for name in "${required[@]}"; do
    [ -n "${!name:-}" ] || { echo "FATAL: missing sanitized launch variable $name" >&2; exit 2; }
done
for slurm_command in scontrol srun; do
    [ -x "$SLURM_BIN_DIR/$slurm_command" ] || {
        echo "FATAL: missing HSG Slurm command: $SLURM_BIN_DIR/$slurm_command" >&2
        exit 2
    }
done

# The submit wrapper deliberately gives sbatch a sanitized --export list.
# Without overriding Slurm's inherited export policy, child srun steps only
# receive that original list and miss variables computed inside this script
# (for example EP_LIST, LOCAL_NUM_ENVS, and BASE_RENDER_SEED).  The batch
# environment is already sanitized above, so propagate its complete current
# state to every containerized step.
export SLURM_EXPORT_ENV=ALL

fatal() { echo "FATAL: $*" >&2; exit 2; }
[ "$PACKED_GPUS_PER_NODE" -eq 4 ] || fatal "HSG profile requires four GB200s per node"
[ "${SLURM_NNODES:-0}" -eq "$PACKED_NODES" ] || fatal "allocated node count changed"
[ "${SLURM_JOB_PARTITION:-}" = batch ] || fatal "HSG profile requires partition=batch"
ACTUAL_QOS=${SLURM_JOB_QOS:-}
if [ -z "$ACTUAL_QOS" ]; then
    ACTUAL_QOS=$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* QOS=\([^ ]*\).*/\1/p')
fi
[ "$ACTUAL_QOS" = "$PACKED_QOS" ] || fatal "allocated QOS changed: $ACTUAL_QOS != $PACKED_QOS"
ACTUAL_ACCOUNT=${SLURM_JOB_ACCOUNT:-}
if [ -z "$ACTUAL_ACCOUNT" ]; then
    ACTUAL_ACCOUNT=$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* Account=\([^ ]*\).*/\1/p')
fi
[ "$ACTUAL_ACCOUNT" = llmservice_nemotron_ultra ] || \
    fatal "HSG account changed: $ACTUAL_ACCOUNT"
[ "$RENDERER_CONTAINER" != "$TRAINER_CONTAINER" ] || fatal "renderer and trainer images must be distinct"
for timing_name in WANDB_WATCHER_POLL_S WANDB_WATCHER_READY_TIMEOUT_S \
                   WANDB_WATCHER_DRAIN_TIMEOUT_S; do
    timing_value=${!timing_name}
    [[ "$timing_value" =~ ^[1-9][0-9]*$ ]] || \
        fatal "$timing_name must be a positive integer"
done
[ "$WANDB_WATCHER_DRAIN_TIMEOUT_S" -le 270 ] || \
    fatal "WANDB_WATCHER_DRAIN_TIMEOUT_S must be at most 270 seconds"

case "$RUN_PROFILE" in
    minimal_e2e)
        EXPECTED_CONTRACT="1:1:2:1:1:0:0:interactive"
        ;;
    resume_minimal)
        EXPECTED_CONTRACT="1:1:2:1:1:0:0:interactive"
        ;;
    calibration_3view)
        EXPECTED_CONTRACT="1:1:2:1:1:1:0:interactive"
        ;;
    smoke_32x300)
        EXPECTED_CONTRACT="2:32:300:1:7:0:1:interactive"
        ;;
    smoke_256x300)
        EXPECTED_CONTRACT="15:256:300:8:52:0:1:normal"
        ;;
    *) fatal "unsupported HSG profile $RUN_PROFILE" ;;
esac
ACTUAL_CONTRACT="$PACKED_NODES:$GLOBAL_NUM_ENVS:$ROLLOUT_STEPS:$TRAINER_RANKS:$EXPECTED_RENDERERS:$IMAGE_DAGGER_DUMP_SIM_RGB:$USE_WANDB:$PACKED_QOS"
[ "$ACTUAL_CONTRACT" = "$EXPECTED_CONTRACT" ] || fatal "$RUN_PROFILE contract changed: $ACTUAL_CONTRACT"

for file in "$RENDERER_CONTAINER" "$TRAINER_CONTAINER" "$TEACHER_CKPT" \
            "$ROOT/.source_revision" "$GR00T_ROOT/.source_revision"; do
    [ -s "$file" ] || fatal "required file missing: $file"
done
[ "$(tr -d '\r\n' < "$ROOT/.source_revision")" = "$EXPECTED_FAST_ROBORENDER_REVISION" ] || \
    fatal "fast_roborender source revision changed after submission"
[ "$(tr -d '\r\n' < "$GR00T_ROOT/.source_revision")" = "$EXPECTED_GR00T_REVISION" ] || \
    fatal "GR00T source revision changed after submission"
ACTUAL_GR00T_SOURCE_TREE_SHA256=$(python3 "$ROOT/scripts/scaling/hsg_gb200_smoke_gate.py" \
    tree-hash --root "$GR00T_ROOT")
[ "$ACTUAL_GR00T_SOURCE_TREE_SHA256" = "$EXPECTED_GR00T_SOURCE_TREE_SHA256" ] || \
    fatal "GR00T source tree changed after submission"
[ "$(sha256sum "$TEACHER_CKPT" | awk '{print $1}')" = "$EXPECTED_TEACHER_CKPT_SHA256" ] || \
    fatal "expert labeler changed after submission"
[ -s "$RESNET18_WEIGHTS" ] || fatal "pinned ResNet18 weights disappeared after submission"
[ "$(sha256sum "$RESNET18_WEIGHTS" | awk '{print $1}')" = "$EXPECTED_RESNET18_SHA256" ] || \
    fatal "pinned ResNet18 weights changed after submission"
export TORCH_HOME

mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[ "${#ALLOCATED_NODES[@]}" -eq "$PACKED_NODES" ] || fatal "cannot resolve allocated nodes"
TRAINER_START=$((PACKED_NODES - TRAINER_RANKS))
[ "$TRAINER_START" -ge 0 ] || fatal "trainer ranks exceed allocated nodes"
TRAINER_NODES=("${ALLOCATED_NODES[@]:TRAINER_START:TRAINER_RANKS}")
TRAINER_NODELIST=$(IFS=,; echo "${TRAINER_NODES[*]}")
MASTER_ADDR=${TRAINER_NODES[0]}
MASTER_PORT=$((15000 + SLURM_JOB_ID % 20000))
LOCAL_NUM_ENVS=$((GLOBAL_NUM_ENVS / TRAINER_RANKS))

# Validate the exact encoder in the sealed trainer before starting renderer
# servers or importing Isaac/GR00T.  IMAGE_DAGGER_FFMPEG is carried through the
# sanitized sbatch environment and is also passed explicitly at this boundary,
# so a missing Slurm export cannot silently fall back to PATH.
FFMPEG_PREFLIGHT="$RUN_ROOT/trainer_ffmpeg_preflight.txt"
srun --nodes=1 --ntasks=1 --nodelist="$MASTER_ADDR" \
    --cpus-per-task=2 --mem=4G --exact --exclusive \
    --container-image="$TRAINER_CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT" \
    env IMAGE_DAGGER_FFMPEG="$IMAGE_DAGGER_FFMPEG" \
    /usr/local/bin/roborender-trainer-sm100 bash \
    "$ROOT/scripts/scaling/check_image_dagger_ffmpeg.sh" \
    | tee "$FFMPEG_PREFLIGHT"
grep -Fx "IMAGE_DAGGER_FFMPEG=$IMAGE_DAGGER_FFMPEG" "$FFMPEG_PREFLIGHT" >/dev/null || \
    fatal "trainer FFmpeg preflight did not attest the selected encoder"

mkdir -p "$FLEET_DIR/endpoints" "$FLEET_DIR/queues" "$FLEET_DIR/logs" \
         "$TRAIN_RUN_DIR/slurm" "$TRAIN_RUN_DIR/output" "$TRAIN_RUN_DIR/telemetry" \
         "$STAGE_ROOT/runtime/home" "$STAGE_ROOT/runtime/cache" "$STAGE_ROOT/runtime/tmp"

if [ "$USE_WANDB" -eq 1 ]; then
    [ -s "$WANDB_ONLINE_VALIDATION" ] || fatal "W&B online validation disappeared after submission"
    [ "$(sha256sum "$WANDB_ONLINE_VALIDATION" | awk '{print $1}')" = \
      "$EXPECTED_WANDB_ONLINE_VALIDATION_SHA256" ] || \
        fatal "W&B online validation changed after submission"
    ACCEPTED_WANDB_ENTITY=$(python3 - "$WANDB_ONLINE_VALIDATION" "$WANDB_EXPECTED_EMAIL" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("schema") != "fast_roborender.wandb_online_smoke.v1":
    raise SystemExit("invalid W&B validation schema")
if payload.get("status") != "PASS" or payload.get("personal") is not True:
    raise SystemExit("W&B validation is not a successful personal-workspace write")
if payload.get("account_email") != sys.argv[2]:
    raise SystemExit("W&B validation account changed")
required = {"validation/scalar", "validation/image", "validation/video"}
if not required.issubset(payload.get("observed_keys", [])):
    raise SystemExit("W&B validation media readback is incomplete")
print(payload.get("entity", ""))
PY
    ) || fatal "W&B online validation is invalid"
    [ -n "$WANDB_ENTITY" ] && [ "$WANDB_ENTITY" = "$ACCEPTED_WANDB_ENTITY" ] || \
        fatal "W&B entity is not bound to the accepted personal-workspace write"
    WANDB_PREFLIGHT=(python "$ROOT/scripts/scaling/wandb_preflight.py"
        --expected-email "$WANDB_EXPECTED_EMAIL" --entity "$WANDB_ENTITY" --format shell)
    WANDB_PREFLIGHT_OUTPUT=$(srun --nodes=1 --ntasks=1 --nodelist="$MASTER_ADDR" \
        --cpus-per-task=2 --mem=4G --exact --exclusive \
        --container-image="$TRAINER_CONTAINER" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
        --container-workdir=/opt/GR00T-VisualSim2Real \
        env HOME="$STAGE_ROOT/runtime/home" WANDB_MODE=online \
        /usr/local/bin/roborender-trainer-sm100 "${WANDB_PREFLIGHT[@]}")
    WANDB_IDENTITY=$(printf '%s\n' "$WANDB_PREFLIGHT_OUTPUT" | grep -E '^WANDB_ENTITY=[^[:space:]]+$' || true)
    [ "$(printf '%s\n' "$WANDB_IDENTITY" | grep -c .)" -eq 1 ] || \
        fatal "containerized W&B preflight did not return exactly one identity"
    case "$WANDB_IDENTITY" in
        WANDB_ENTITY=?*) WANDB_ENTITY=${WANDB_IDENTITY#WANDB_ENTITY=} ;;
        *) fatal "containerized W&B identity preflight failed: $WANDB_IDENTITY" ;;
    esac
    WANDB_MODE=online
    USE_WANDB_HYDRA=true
else
    WANDB_ENTITY=null
    WANDB_MODE=disabled
    USE_WANDB_HYDRA=false
fi
export WANDB_ENTITY WANDB_MODE

RENDER_STEP_PID=""
TRAIN_STEP_PID=""
WANDB_WATCHER_PID=""

drain_wandb_watcher() {
    [ "$USE_WANDB" -eq 1 ] || return 0
    [ -n "$WANDB_WATCHER_PID" ] || return 0

    : > "$TRAIN_RUN_DIR/WANDB_UPLOAD_STOP"
    local deadline=$((SECONDS + WANDB_WATCHER_DRAIN_TIMEOUT_S))
    while kill -0 "$WANDB_WATCHER_PID" 2>/dev/null && \
          [ "$SECONDS" -lt "$deadline" ]; do
        sleep 1
    done
    if kill -0 "$WANDB_WATCHER_PID" 2>/dev/null; then
        echo "ERROR: W&B media watcher did not finish its final drain within ${WANDB_WATCHER_DRAIN_TIMEOUT_S}s" >&2
        kill -TERM "$WANDB_WATCHER_PID" 2>/dev/null || true
        wait "$WANDB_WATCHER_PID" 2>/dev/null || true
        WANDB_WATCHER_PID=""
        return 70
    fi

    local watcher_rc=0
    if wait "$WANDB_WATCHER_PID" 2>/dev/null; then
        watcher_rc=0
    else
        watcher_rc=$?
    fi
    WANDB_WATCHER_PID=""
    if [ "$watcher_rc" -ne 0 ]; then
        echo "ERROR: W&B media watcher final drain failed (rc=$watcher_rc)" >&2
        return 70
    fi
}

cleanup() {
    local rc=$?
    local watcher_rc=0
    trap - EXIT TERM INT USR1
    for pid in "$TRAIN_STEP_PID" "$RENDER_STEP_PID"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    if drain_wandb_watcher; then
        watcher_rc=0
    else
        watcher_rc=$?
    fi
    if [ "$rc" -eq 0 ] && [ "$watcher_rc" -ne 0 ]; then
        rc=70
    fi
    if [ "$rc" -eq 0 ]; then : > "$RUN_ROOT/RUN_DONE"; else printf '%s\n' "$rc" > "$RUN_ROOT/RUN_FAILED"; fi
    exit "$rc"
}
pre_timeout() {
    : > "$RUN_ROOT/PRE_TIMEOUT_WARNING"
    [ -z "$TRAIN_STEP_PID" ] || kill -TERM "$TRAIN_STEP_PID" 2>/dev/null || true
    exit 124
}
terminate_with_status() {
    local status=$1
    trap - TERM INT
    exit "$status"
}
trap cleanup EXIT
trap 'terminate_with_status 143' TERM
trap 'terminate_with_status 130' INT
trap pre_timeout USR1

export HOME="$STAGE_ROOT/runtime/home"
export IMAGE3F_RUN_ID="$RUN_ID"
export IMAGE3F_MAX_STREAMS=16
export STARTUP_STAGGER_SCOPE=local
export STARTUP_STAGGER_S=2
export PYTHONDONTWRITEBYTECODE=1

srun --nodes="$PACKED_NODES" --ntasks="$EXPECTED_RENDERERS" --ntasks-per-node=4 \
    --distribution=cyclic --gpus="$EXPECTED_RENDERERS" --gpus-per-task=1 \
    --cpus-per-task=8 --mem=512G \
    --exact --exclusive --gpu-bind=closest --cpu-bind=cores --kill-on-bad-exit=1 \
    --container-image="$RENDERER_CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$ROBORENDER_MODEL_BASE:/models:ro,$IMAGE3F_GENERATED_ROOT:/opt/roborender/assets/generated:ro" \
    --container-workdir=/opt/roborender \
    --output="$FLEET_DIR/logs/renderer_%t.out" --error="$FLEET_DIR/logs/renderer_%t.err" \
    bash -lc '
        set -Eeuo pipefail
        export ROBORENDER_CONTAINERIZED=1
        export ROBORENDER_STAGE_ROOT="$STAGE_ROOT"
        export SLURM_ARRAY_TASK_ID=$SLURM_PROCID
        export SLURM_ARRAY_JOB_ID=$SLURM_JOB_ID
        export IMAGE3F_ADVERTISE_HOST=${SLURMD_NODENAME:-$(hostname -f)}
        export IMAGE3F_ROOT=/opt/roborender
        export IMAGE3F_ENV=/opt/roborender-venv
        export IMAGE3F_PYTHON=/opt/roborender-venv/bin/python
        export ROBORENDER_MODEL_BASE=/models
        export DIFFSYNTH_MODEL_BASE_PATH=/models
        export DIFFSYNTH_SKIP_DOWNLOAD=true
        exec bash "$ROOT/scripts/server/image3f_server_array.sbatch"
    ' &
RENDER_STEP_PID=$!

READY_DEADLINE=$((SECONDS + 1800))
while :; do
    READY_COUNT=$(find "$FLEET_DIR/endpoints" -maxdepth 1 -type f -name 'renderer_*.txt' -printf '.' | wc -c)
    [ "$READY_COUNT" -eq "$EXPECTED_RENDERERS" ] && break
    kill -0 "$RENDER_STEP_PID" 2>/dev/null || fatal "renderer step exited before endpoint readiness"
    [ "$SECONDS" -lt "$READY_DEADLINE" ] || fatal "timed out waiting for $EXPECTED_RENDERERS endpoints"
    sleep 5
done

EP_LIST=$(python3 "$ROOT/scripts/scaling/image3f_probe.py" \
    --endpoint-dir "$FLEET_DIR/endpoints" --expected "$EXPECTED_RENDERERS" \
    --wait-timeout-s 60 --expected-release-id "$EXPECTED_IMAGE3F_RELEASE_ID" \
    --expected-adapter-sha256 "$EXPECTED_IMAGE3F_ADAPTER_SHA256" \
    --expected-inference-steps 5 --expected-teacache-threshold 0.3 \
    --expected-cfg-scale 1.0 --expected-cuda-graph disabled \
    --expected-output-views head_color,hand_left,hand_right \
    --expected-output-layout vertical --expected-output-tile-shape 240,416,3 \
    --expected-output-stacked-shape 720,416,3 --require-tensorrt-engine \
    --write-manifest "$FLEET_DIR/selected_endpoints.txt" \
    --write-report "$FLEET_DIR/renderer_capabilities.json" --hydra-list)

TRAIN_ABSOLUTE_TARGET=1
if [ "$RUN_PROFILE" = resume_minimal ]; then
    TRAIN_ABSOLUTE_TARGET=2
fi
PLAN=(--global-envs "$GLOBAL_NUM_ENVS" --rollout-steps "$ROLLOUT_STEPS"
      --num-rollout-iterations "$TRAIN_ABSOLUTE_TARGET" --trainer-nodes "$TRAINER_RANKS"
      --raytraced-mix 0.0 --envs-per-renderer 5 --reference-lag 0
      --num-mini-batches 1 --per-device-train-batch-size 1
      --num-learning-epochs 1 --behavior-source annealed --teacher-rollout-ratio 0.0
      --beta-schedule linear --beta-start 1.0 --beta-end 0.0 --beta-decay-iters 500
      --primary-view head_color)
GRADIENT_ACCUMULATION_STEPS=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field gradient_accumulation_steps)
REQUIRED_RENDERERS=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field required_renderers)
[ "$REQUIRED_RENDERERS" -eq "$EXPECTED_RENDERERS" ] || fatal "renderer capacity plan changed"
python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --format json > "$TRAIN_RUN_DIR/scale_plan.json"

if [ "$ROLLOUT_STEPS" -eq 2 ]; then
    TELEMETRY_SCALAR_EVERY=1 TELEMETRY_IMAGE_EVERY=1 TELEMETRY_CLIP_EVERY=2 TELEMETRY_CLIP_FRAMES=2
else
    TELEMETRY_SCALAR_EVERY=10 TELEMETRY_IMAGE_EVERY=300 TELEMETRY_CLIP_EVERY=300 TELEMETRY_CLIP_FRAMES=32
fi
DDP_GATE=0
[ "$TRAINER_RANKS" -eq 1 ] || DDP_GATE=1

export EP_LIST LOCAL_NUM_ENVS GRADIENT_ACCUMULATION_STEPS MASTER_ADDR MASTER_PORT
export TRAINER_RANKS GLOBAL_NUM_ENVS ROLLOUT_STEPS TRAIN_RUN_DIR FLEET_DIR ANCHORS TEACHER_CKPT
export EXPECTED_GR00T_REVISION USE_WANDB_HYDRA WANDB_ENTITY WANDB_MODE
export ROBORENDER_TELEMETRY_DIR="$TRAIN_RUN_DIR/telemetry"
export IMAGE3F_ENDPOINT_DIR="$FLEET_DIR/endpoints"
export IMAGE3F_ENDPOINT_MANIFEST="$FLEET_DIR/selected_endpoints.txt"
export IMAGE3F_LOCAL_ENVS="$LOCAL_NUM_ENVS"
export IMAGE3F_REFERENCE_LAG=0
export IMAGE3F_EXPECT_OUTPUT_VIEWS=head_color,hand_left,hand_right
export IMAGE3F_EXPECT_OUTPUT_SHAPE=240,416,3
export IMAGE3F_REQUEST_TIMEOUT_S=120
export BASE_RENDER_SEED=42
# A continuity/calibration smoke must exercise the real reset-data path, but
# loading and resampling all 295 production demonstrations obscures failures
# behind minutes of unrelated setup.  Pin one committed, representative motion
# for the one-environment profiles; the 32/256 production profiles continue to
# consume the complete configured motion directory.
export SMOKE_MOTION_FILE=gr00t/rl/data/motions/g1_wsdpt/33demos_675_775/record_2025-10-14_21-28-47_0.675m.npz
[ -s "$GR00T_ROOT/$SMOKE_MOTION_FILE" ] || \
    fatal "pinned smoke motion is missing: $GR00T_ROOT/$SMOKE_MOTION_FILE"
export IMAGE_DAGGER_EXPECT_PRIMARY_VIEW=head_color
export IMAGE_DAGGER_EXPECT_POLICY_VIEWS=head_color,hand_left,hand_right
export IMAGE_DAGGER_EXPECT_POLICY_VIEW_COUNT=3
export IMAGE_DAGGER_EXPECT_POLICY_VIEW_LAYOUT=vertical
export IMAGE_DAGGER_EXPECT_POLICY_IMAGE_SHAPE=324,192,3
export IMAGE_DAGGER_DUMP_SIM_RGB
export IMAGE_DAGGER_DUMP_MAX_STEPS=$([ "$IMAGE_DAGGER_DUMP_SIM_RGB" -eq 1 ] && echo 2 || echo 0)
export IMAGE_DAGGER_DUMP_MAX_ENVS=$([ "$IMAGE_DAGGER_DUMP_SIM_RGB" -eq 1 ] && echo 1 || echo 0)
export IMAGE_DAGGER_DUMP_DIR="$TRAIN_RUN_DIR/render_dumps"
export IMAGE_DAGGER_VERIFY_DDP_SYNC="$DDP_GATE"
export IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS=1
export IMAGE_DAGGER_EXPECT_ROLLOUT_ACTION_SOURCE=beta
export IMAGE_DAGGER_EXPECT_BETA_SCHEDULE=linear
export IMAGE_DAGGER_EXPECT_BETA_START=1.0
export IMAGE_DAGGER_EXPECT_BETA_END=0.0
export IMAGE_DAGGER_EXPECT_BETA_DECAY_ITERS=500
export IMAGE_DAGGER_EXPECT_TEACHER_WARMUP_STEPS=0
export IMAGE_DAGGER_EXPECT_ENFORCE_TEACHER=true
export IMAGE_DAGGER_EXPECT_TEACHER_RATIO=0.0
export IMAGE_DAGGER_EXPECT_LEARNING_EPOCHS=1
export IMAGE_DAGGER_EXPECT_VALUE_LOSS_COEF=0.0
export IMAGE_DAGGER_EXPECT_ENTROPY_COEF=0.0
export IMAGE_DAGGER_TELEMETRY_SCALAR_EVERY="$TELEMETRY_SCALAR_EVERY"
export IMAGE_DAGGER_TELEMETRY_IMAGE_EVERY="$TELEMETRY_IMAGE_EVERY"
export IMAGE_DAGGER_TELEMETRY_CLIP_EVERY="$TELEMETRY_CLIP_EVERY"
export IMAGE_DAGGER_TELEMETRY_CLIP_FRAMES="$TELEMETRY_CLIP_FRAMES"
export IMAGE_DAGGER_TELEMETRY_CLIP_FPS=10
export IMAGE_DAGGER_TELEMETRY_REQUIRED=1
export IMAGE_DAGGER_TELEMETRY_REQUIRE_VIDEO=1
export IMAGE_DAGGER_FFMPEG
# These finite batch-smoke phases exit only after their checkpoint is durable.
# Isaac Sim 5.1's graceful close can then block indefinitely in headless HSG
# jobs, so opt this launcher into its documented immediate-close path.  Other
# entrypoints retain SimulationAppLifecycle's graceful default.
export IMAGE_DAGGER_SIMULATION_APP_SKIP_CLEANUP=1
export PROJ=roborender_dagger_hsg_gb200_smoke
export EXP="${RUN_PROFILE}_${RUN_ID}"
export TRAIN_ABSOLUTE_TARGET

WANDB_WATCHER_LOG="$TRAIN_RUN_DIR/wandb_media_watcher.log"
WANDB_WATCHER_READY="$TRAIN_RUN_DIR/telemetry/WANDB_WATCHER_READY.json"
if [ "$USE_WANDB" -eq 1 ]; then
    # Preserve the durable sidecar run ID and upload-state files across a
    # restarted process.  Only per-process lifecycle markers are disposable.
    rm -f "$WANDB_WATCHER_READY" "$TRAIN_RUN_DIR/WANDB_UPLOAD_STOP"
    srun --nodes=1 --ntasks=1 --nodelist="$MASTER_ADDR" \
        --cpus-per-task=2 --mem=4G --exact --exclusive \
        --container-image="$TRAINER_CONTAINER" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
        --container-workdir=/opt/GR00T-VisualSim2Real \
        env HOME="$STAGE_ROOT/runtime/home" WANDB_MODE="$WANDB_MODE" \
        /usr/local/bin/roborender-trainer-sm100 python \
        "$ROOT/scripts/scaling/wandb_run_watcher.py" \
        --run-dir "$TRAIN_RUN_DIR" --project "$PROJ" --name "${EXP}-media" \
        --expected-email "$WANDB_EXPECTED_EMAIL" --entity "$WANDB_ENTITY" \
        --poll-interval-s "$WANDB_WATCHER_POLL_S" \
        --ready-marker "$WANDB_WATCHER_READY" \
        --max-consecutive-error-scans 4 --finish-marker WANDB_UPLOAD_STOP \
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
        [ "$WATCHER_START_RC" -ne 0 ] || WATCHER_START_RC=70
        echo "FATAL: W&B media watcher did not become ready (rc=$WATCHER_START_RC); see $WANDB_WATCHER_LOG" >&2
        exit "$WATCHER_START_RC"
    fi
fi

run_train_phase() {
    local phase=$1 checkpoint=$2 resume=$3 stop_after_step=$4
    case "$resume" in 0|1) ;; *) fatal "invalid resume flag for $phase" ;; esac
    [[ "$stop_after_step" =~ ^[0-9]+$ ]] || fatal "invalid stop boundary for $phase"
    if [ "$resume" -eq 1 ]; then
        [ -s "$checkpoint" ] || fatal "resume checkpoint is missing: $checkpoint"
        [ "$(realpath -e "$(dirname "$checkpoint")")" = "$(realpath -e "$TRAIN_RUN_DIR")" ] || \
            fatal "resume checkpoint must be inside the reused training directory"
        TRAIN_CHECKPOINT_ARG="checkpoint=$checkpoint"
        TRAIN_RESUME_ARG=resume=true
    else
        [ "$checkpoint" = null ] || fatal "fresh phase must use checkpoint=null"
        TRAIN_CHECKPOINT_ARG=checkpoint=null
        TRAIN_RESUME_ARG=resume=false
    fi
    export TRAIN_PHASE="$phase" TRAIN_CHECKPOINT_ARG TRAIN_RESUME_ARG STOP_AFTER_STEP="$stop_after_step"

    srun --nodes="$TRAINER_RANKS" --ntasks="$TRAINER_RANKS" --ntasks-per-node=1 \
        --nodelist="$TRAINER_NODELIST" --gpus="$TRAINER_RANKS" --gpus-per-task=1 \
        --cpus-per-task=16 --mem=128G \
        --exact --exclusive --gpu-bind=closest --cpu-bind=cores --kill-on-bad-exit=1 \
        --container-image="$TRAINER_CONTAINER" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
        --container-workdir=/opt/GR00T-VisualSim2Real \
        --output="$TRAIN_RUN_DIR/slurm/${phase}_rank_%t.out" \
        --error="$TRAIN_RUN_DIR/slurm/${phase}_rank_%t.err" \
        /usr/local/bin/roborender-trainer-sm100 bash -lc '
            set -Eeuo pipefail
            bash "$ROOT/scripts/scaling/check_image_dagger_ffmpeg.sh" >/dev/null
            [ "$(tr -d "\r\n" < /opt/GR00T-VisualSim2Real/.source_revision)" = "$EXPECTED_GR00T_REVISION" ] || {
                echo "FATAL: sealed trainer GR00T revision mismatch" >&2; exit 2;
            }
            cd /opt/GR00T-VisualSim2Real
            RANK_RENDER_SEED=$((BASE_RENDER_SEED + SLURM_PROCID))
            if [ "$TRAINER_RANKS" -eq 1 ]; then
                ACCELERATE_ARGS=(--num_processes 1 --num_cpu_threads_per_process 16)
            else
                ACCELERATE_ARGS=(--multi_gpu --num_machines "$TRAINER_RANKS" --num_processes "$TRAINER_RANKS"
                    --machine_rank "$SLURM_PROCID" --main_process_ip "$MASTER_ADDR"
                    --main_process_port "$MASTER_PORT" --same_network --rdzv_backend static
                    --max_restarts 0 --num_cpu_threads_per_process 16)
            fi
            SMOKE_DATASET_ARGS=()
            case "$RUN_PROFILE" in
                minimal_e2e|resume_minimal|calibration_3view)
                    SMOKE_DATASET_ARGS=(
                        env.config.reset_from_dataset.use_motion_file_dir=false
                        "env.config.reset_from_dataset.motion_files=[$SMOKE_MOTION_FILE]"
                    )
                    ;;
            esac
            exec accelerate launch "${ACCELERATE_ARGS[@]}" gr00t/rl/train_agent_trl.py \
                +exp=loco_manip/wsdpt_image_dagger_3view \
                num_envs="$LOCAL_NUM_ENVS" algo.config.num_steps_per_env="$ROLLOUT_STEPS" \
                algo.config.num_mini_batches=1 algo.config.num_learning_epochs=1 \
                algo.config.rollout_with_teacher_num_steps=0 \
                algo.config.enforce_teacher_rollout=true algo.config.ratio_teacher_rollout=0.0 \
                algo.config.value_loss_coef=0.0 algo.config.entropy_coef=0.0 \
                algo.config.videogen.rollout_action_source=beta \
                ++algo.config.videogen.beta_schedule=linear ++algo.config.videogen.beta_start=1.0 \
                ++algo.config.videogen.beta_end=0.0 ++algo.config.videogen.beta_decay_iters=500 \
                algo.trl.per_device_train_batch_size=1 \
                algo.trl.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
                algo.trl.num_total_batches="$TRAIN_ABSOLUTE_TARGET" algo.trl.total_episodes=null \
                callbacks.model_save.save_frequency=1 callbacks.model_save.save_initial_checkpoint=true \
                callbacks.model_save.test_stop_after_checkpoint_step="$STOP_AFTER_STEP" \
                headless=true "$TRAIN_CHECKPOINT_ARG" "$TRAIN_RESUME_ARG" \
                teacher_actor_path="$TEACHER_CKPT" \
                "${SMOKE_DATASET_ARGS[@]}" \
                algo.config.videogen.raytraced_mix_prob=0.0 ++algo.config.videogen.seed="$RANK_RENDER_SEED" \
                ++algo.config.videogen.reference_lag=0 ++algo.config.videogen.primary_view=head_color \
                simulator.config.cameras.policy_views="[head_color,hand_left,hand_right]" \
                policy_view_count=3 ++env.config.image_dagger_policy_view_count=3 \
                algo.config.videogen.image3f_endpoint_files="$EP_LIST" \
                algo.config.videogen.anchor_pool_dir="$ANCHORS" algo.config.videogen.anchor_index=0 \
                simulator.config.render_results=false use_wandb="$USE_WANDB_HYDRA" \
                wandb.wandb_entity="$WANDB_ENTITY" wandb.wandb_group=null \
                project_name="$PROJ" experiment_name="$EXP" experiment_dir="$TRAIN_RUN_DIR" \
                hydra.run.dir="$TRAIN_RUN_DIR/.hydra/${TRAIN_PHASE}_rank_${SLURM_PROCID}" \
                output_dir="$TRAIN_RUN_DIR/output/rank_${SLURM_PROCID}"
        ' &
    TRAIN_STEP_PID=$!

    # Media observability is a production gate.  Do not let a trainer phase
    # continue after the uploader has failed, and close the exit race after the
    # trainer step leaves the polling loop.
    while kill -0 "$TRAIN_STEP_PID" 2>/dev/null; do
        if [ "$USE_WANDB" -eq 1 ] && ! kill -0 "$WANDB_WATCHER_PID" 2>/dev/null; then
            local watcher_run_rc=0
            if wait "$WANDB_WATCHER_PID"; then
                watcher_run_rc=0
            else
                watcher_run_rc=$?
            fi
            WANDB_WATCHER_PID=""
            kill -TERM "$TRAIN_STEP_PID" 2>/dev/null || true
            wait "$TRAIN_STEP_PID" 2>/dev/null || true
            TRAIN_STEP_PID=""
            echo "FATAL: W&B media watcher exited during $phase (rc=$watcher_run_rc); see $WANDB_WATCHER_LOG" >&2
            return 70
        fi
        sleep 5
    done
    if wait "$TRAIN_STEP_PID"; then TRAIN_RC=0; else TRAIN_RC=$?; fi
    TRAIN_STEP_PID=""
    [ "$TRAIN_RC" -eq 0 ] || return "$TRAIN_RC"
    if [ "$USE_WANDB" -eq 1 ] && ! kill -0 "$WANDB_WATCHER_PID" 2>/dev/null; then
        local watcher_run_rc=0
        if wait "$WANDB_WATCHER_PID"; then
            watcher_run_rc=0
        else
            watcher_run_rc=$?
        fi
        WANDB_WATCHER_PID=""
        echo "FATAL: W&B media watcher exited before final drain after $phase (rc=$watcher_run_rc)" >&2
        return 70
    fi
}

resume_snapshot() {
    local checkpoint=$1 output=$2
    srun --nodes=1 --ntasks=1 --nodelist="$MASTER_ADDR" \
        --cpus-per-task=2 --mem=4G --exact --exclusive \
        --container-image="$TRAINER_CONTAINER" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
        --container-workdir=/opt/GR00T-VisualSim2Real \
        /usr/local/bin/roborender-trainer-sm100 python \
        "$ROOT/scripts/scaling/resume_continuity_gate.py" snapshot \
        --checkpoint "$checkpoint" --run-dir "$TRAIN_RUN_DIR" \
        --expected-world-size 1 --wandb-mode disabled --output "$output"
}

if [ "$RUN_PROFILE" = resume_minimal ]; then
    run_train_phase phase_a null 0 1
    resume_snapshot "$TRAIN_RUN_DIR/model_step_000001.pt" \
        "$TRAIN_RUN_DIR/resume_before.json"
    run_train_phase phase_b "$TRAIN_RUN_DIR/model_step_000001.pt" 1 0
    resume_snapshot "$TRAIN_RUN_DIR/model_step_000002.pt" \
        "$TRAIN_RUN_DIR/resume_after.json"
    python3 "$ROOT/scripts/scaling/resume_continuity_gate.py" verify \
        --before "$TRAIN_RUN_DIR/resume_before.json" \
        --after "$TRAIN_RUN_DIR/resume_after.json" \
        --expected-before-step 1 --expected-after-step 2 \
        --rollout-steps "$ROLLOUT_STEPS" --expected-world-size 1 \
        --wandb-mode disabled --output "$TRAIN_RUN_DIR/resume_continuity_report.json"
else
    run_train_phase single null 0 0
fi

kill -TERM "$RENDER_STEP_PID" 2>/dev/null || true
wait "$RENDER_STEP_PID" 2>/dev/null || true
RENDER_STEP_PID=""

if [ "$USE_WANDB" -eq 1 ]; then
    if ! kill -0 "$WANDB_WATCHER_PID" 2>/dev/null; then
        if wait "$WANDB_WATCHER_PID"; then
            WATCHER_RUN_RC=0
        else
            WATCHER_RUN_RC=$?
        fi
        WANDB_WATCHER_PID=""
        echo "FATAL: W&B media watcher exited before final drain (rc=$WATCHER_RUN_RC)" >&2
        exit 70
    fi
    if ! drain_wandb_watcher; then
        exit 70
    fi
    python3 - "$TRAIN_RUN_DIR" <<'PY'
import json
import os
import re
import sys
import tempfile
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
telemetry = run_dir / "telemetry"
run_id_path = telemetry / "WANDB_SIDECAR_RUN_ID"
state_path = telemetry / ".wandb_upload_state.json"
run_id = run_id_path.read_text(encoding="utf-8").strip()
if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id) is None:
    raise SystemExit("FATAL: W&B media sidecar run ID is missing or malformed")
state = json.loads(state_path.read_text(encoding="utf-8"))
if state.get("schema_version") != 2 or state.get("wandb_run_id") != run_id:
    raise SystemExit("FATAL: W&B media upload state is not bound to its durable run ID")
artifacts = state.get("artifacts")
if not isinstance(artifacts, dict):
    raise SystemExit("FATAL: W&B media upload state has no artifact inventory")

expected = {}
for kind, suffixes in {
    "scalar": {".json"},
    "image": {".png", ".jpg", ".jpeg", ".webp"},
    "video": {".mp4", ".webm", ".mov"},
}.items():
    directory = telemetry / f"{kind}s"
    files = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in suffixes
    )
    if not files:
        raise SystemExit(f"FATAL: W&B final drain found no {kind} telemetry")
    for path in files:
        expected[str(path.relative_to(run_dir))] = kind
missing = sorted(path for path in expected if path not in artifacts)
if missing:
    raise SystemExit(
        "FATAL: W&B final drain did not attest every local telemetry artifact: "
        + ", ".join(missing[:5])
    )

report = {
    "schema": "fast_roborender.wandb_media_drain.v1",
    "status": "PASS",
    "wandb_run_id": run_id,
    "artifact_count": len(expected),
    "artifact_kinds": {
        kind: sum(value == kind for value in expected.values())
        for kind in ("scalar", "image", "video")
    },
}
destination = run_dir / "validation" / "wandb_media_drain.json"
destination.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
fi

python3 "$ROOT/scripts/scaling/hsg_gb200_smoke_gate.py" postrun \
    --profile "$RUN_PROFILE" --run-dir "$TRAIN_RUN_DIR" \
    --global-envs "$GLOBAL_NUM_ENVS" --rollout-steps "$ROLLOUT_STEPS" \
    --world-size "$TRAINER_RANKS" --fast-revision "$EXPECTED_FAST_ROBORENDER_REVISION" \
    --gr00t-revision "$EXPECTED_GR00T_REVISION" > "$TRAIN_RUN_DIR/postrun_gate.json"

publish_once() {
    local source=$1 destination=$2
    mkdir -p "$(dirname "$destination")"
    if [ -e "$destination" ]; then
        [ "$(sha256sum "$source" | awk '{print $1}')" = "$(sha256sum "$destination" | awk '{print $1}')" ] || \
            fatal "refusing to replace a different validation artifact: $destination"
    else
        cp "$source" "$destination"
    fi
}
publish_symlink_once() {
    local source destination
    source=$(realpath -e "$1")
    destination=$2
    mkdir -p "$(dirname "$destination")"
    if [ -e "$destination" ]; then
        [ "$(realpath -e "$destination")" = "$source" ] || \
            fatal "refusing to replace a different validation link: $destination"
    else
        ln -s "$source" "$destination"
    fi
}

case "$RUN_PROFILE" in
    calibration_3view)
        TECHNICAL="$TRAIN_RUN_DIR/validation/camera_calibration_manifest.json"
        PYTHONPATH="$ROOT/scripts/scaling" python3 - "$TECHNICAL" \
            "$EXPECTED_FAST_ROBORENDER_REVISION" "$EXPECTED_GR00T_REVISION" <<'PY'
import json
import sys
from pathlib import Path
from camera_calibration_gate import validate_technical_manifest
print(json.dumps(validate_technical_manifest(Path(sys.argv[1]), sys.argv[2], sys.argv[3]), indent=2))
PY
        publish_once "$TECHNICAL" "$STAGE_ROOT/runtime/validation/camera_calibration_manifest.json"
        echo "Calibration is technically valid; explicit visual acceptance is still required."
        ;;
    smoke_32x300|smoke_256x300)
        PROFILE_MANIFEST="$TRAIN_RUN_DIR/profile_validation_manifest.txt"
        python3 "$ROOT/scripts/scaling/profile_validation_gate.py" \
            --manifest "$PROFILE_MANIFEST" --expected-profile "$RUN_PROFILE" \
            --expected-fast-revision "$EXPECTED_FAST_ROBORENDER_REVISION" \
            --expected-gr00t-revision "$EXPECTED_GR00T_REVISION" >/dev/null
        publish_symlink_once "$PROFILE_MANIFEST" \
            "$STAGE_ROOT/runtime/validation/${RUN_PROFILE}_profile_validation_manifest.txt"
        ;;
esac
echo "HSG_GB200_DAGGER_SMOKE_PASS profile=$RUN_PROFILE"
