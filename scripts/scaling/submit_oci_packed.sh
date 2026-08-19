#!/bin/bash
# Submit the base-policy 256-environment production run as one packed OCI job.
set -Eeuo pipefail

STAGE_ROOT=${STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
IMAGE3F_ROOT=${IMAGE3F_ROOT:-$STAGE_ROOT/assets/roborender_image_model_accelerated}
ROBORENDER_MODEL_BASE=${ROBORENDER_MODEL_BASE:-$STAGE_ROOT/assets/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control}
ANCHORS=${ANCHORS:-$STAGE_ROOT/assets/anchor_pool_2048}
VIRAL_ENV=${VIRAL_ENV:-$STAGE_ROOT/runtime/envs/viral}
VIRAL_PYTHON=${VIRAL_PYTHON:-$VIRAL_ENV/bin/python}
RUN_ID=${RUN_ID:-oci_base_256x300_$(date -u +%Y%m%dT%H%M%SZ)}
FLEET_DIR=${FLEET_DIR:-$STAGE_ROOT/runtime/$RUN_ID/fleet}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:-$STAGE_ROOT/runtime/$RUN_ID/train}
RUN_WALLTIME=${RUN_WALLTIME:-08:15:00}
WANDB_EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_ENTITY=${WANDB_ENTITY:-}
DRY_RUN=${DRY_RUN:-0}

fatal() {
    echo "FATAL: $*" >&2
    exit 2
}

for path in "$ROOT" "$GR00T_ROOT" "$IMAGE3F_ROOT" "$ROBORENDER_MODEL_BASE" "$ANCHORS"; do
    [ -d "$path" ] || fatal "required directory missing: $path"
done
[ -x "$VIRAL_PYTHON" ] || fatal "trainer environment missing: $VIRAL_PYTHON"
[ -x "$IMAGE3F_ROOT/.venv/bin/python" ] || fatal "Image3F H100 environment is not ready"
[ -s "$IMAGE3F_ROOT/assets/generated/engines/current/engine_build_manifest.json" ] || \
    fatal "Image3F H100 TensorRT engine is not ready"
[ -z "$(git -C "$ROOT" status --porcelain)" ] || fatal "fast_roborender source is dirty"
[ -z "$(git -C "$GR00T_ROOT" status --porcelain)" ] || fatal "GR00T source is dirty"

FAST_REVISION=$(git -C "$ROOT" rev-parse HEAD)
GR00T_REVISION=$(git -C "$GR00T_ROOT" rev-parse HEAD)
if [ "$DRY_RUN" = 1 ]; then
    WANDB_ENTITY=${WANDB_ENTITY:-DRY_RUN_UNVERIFIED}
else
    [ "$WANDB_MODE" = online ] || fatal "production telemetry requires WANDB_MODE=online"
    if [ -z "${WANDB_API_KEY:-}" ] && [ ! -s "${HOME:?}/.netrc" ]; then
        fatal "no W&B credential is available for secure propagation to the job"
    fi
    # The authenticated email and personal entity are verified again inside
    # the Isaac container before distributed training starts.
    WANDB_ENTITY=${WANDB_ENTITY:-}
fi

mkdir -p "$FLEET_DIR/logs" "$TRAIN_RUN_DIR"
EXPORTS="ALL,STAGE_ROOT=$STAGE_ROOT,ROOT=$ROOT,GR00T_ROOT=$GR00T_ROOT,IMAGE3F_ROOT=$IMAGE3F_ROOT,ROBORENDER_MODEL_BASE=$ROBORENDER_MODEL_BASE,ANCHORS=$ANCHORS,VIRAL_ENV=$VIRAL_ENV,VIRAL_PYTHON=$VIRAL_PYTHON,RUN_ID=$RUN_ID,FLEET_DIR=$FLEET_DIR,TRAIN_RUN_DIR=$TRAIN_RUN_DIR,CKPT=none,EXPECTED_TRAIN_CHECKPOINT_SHA256=NONE,GLOBAL_NUM_ENVS=256,ROLLOUT_STEPS=300,TRAINER_NODES=8,MIX=0.0,REFERENCE_LAG=0,NUM_MINI_BATCHES=1,BEHAVIOR_SOURCE=annealed,BETA_SCHEDULE=linear,BETA_START=1.0,BETA_END=0.0,BETA_DECAY_ITERS=500,WANDB_EXPECTED_EMAIL=$WANDB_EXPECTED_EMAIL,WANDB_MODE=$WANDB_MODE,WANDB_ENTITY=$WANDB_ENTITY,EXPECTED_FAST_ROBORENDER_REVISION=$FAST_REVISION,EXPECTED_GR00T_REVISION=$GR00T_REVISION"
CMD=(sbatch --parsable --time="$RUN_WALLTIME" --signal=B:USR1@300
    --output="$FLEET_DIR/logs/packed_%j.out"
    --error="$FLEET_DIR/logs/packed_%j.err"
    --export="$EXPORTS"
    "$ROOT/scripts/scaling/oci_packed_256x300.sbatch")

cat > "$FLEET_DIR/submission_manifest.txt" <<EOF
status=planned
run_id=$RUN_ID
initialization=fresh_student
checkpoint=none
global_envs=256
rollout_steps=300
nodes=8
gpus_per_node=8
renderer_tasks=52
trainer_tasks=8
reference_policy=FT-FIXED
reference_lag=0
fast_roborender_revision=$FAST_REVISION
gr00t_revision=$GR00T_REVISION
wandb_expected_email=$WANDB_EXPECTED_EMAIL
wandb_entity=$WANDB_ENTITY
EOF

if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY RUN:'; printf ' %q' "${CMD[@]}"; printf '\n'
    exit 0
fi

JOB_ID=$("${CMD[@]}")
JOB_ID=${JOB_ID%%;*}
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || fatal "sbatch returned malformed job id: $JOB_ID"
cat >> "$FLEET_DIR/submission_manifest.txt" <<EOF
status=submitted
job_id=$JOB_ID
EOF
echo "Packed 256-environment base-policy job: $JOB_ID"
echo "Monitor: squeue -j $JOB_ID"
echo "Logs: $FLEET_DIR/logs"
