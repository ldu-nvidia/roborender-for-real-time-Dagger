#!/bin/bash
# Submit the base-policy 256-environment production run as one packed OCI job.
set -Eeuo pipefail

STAGE_ROOT=${STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
IMAGE3F_ROOT=${IMAGE3F_ROOT:-$STAGE_ROOT/assets/roborender_image_model_accelerated}
IMAGE3F_ENV=${IMAGE3F_ENV:-$STAGE_ROOT/runtime/envs/image3f-cu128}
IMAGE3F_PYTHON=${IMAGE3F_PYTHON:-$IMAGE3F_ENV/bin/python}
ROBORENDER_MODEL_BASE=${ROBORENDER_MODEL_BASE:-$STAGE_ROOT/assets/models}
ANCHORS=${ANCHORS:-$STAGE_ROOT/assets/anchor_pool_2048}
TEACHER_CKPT=${TEACHER_CKPT:-$STAGE_ROOT/assets/policies/expert_labeler/model_step_044500.pt}
EXPECTED_TEACHER_CKPT_SHA256=${EXPECTED_TEACHER_CKPT_SHA256:-943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38}
VIRAL_ENV=${VIRAL_ENV:-$STAGE_ROOT/runtime/envs/viral}
VIRAL_PYTHON=${VIRAL_PYTHON:-$VIRAL_ENV/bin/python}
WANDB_ENV=${WANDB_ENV:-$STAGE_ROOT/runtime/envs/wandb-host}
WANDB_PYTHON=${WANDB_PYTHON:-$WANDB_ENV/bin/python}
RUN_ID=${RUN_ID:-oci_base_256x300_$(date -u +%Y%m%dT%H%M%SZ)}
FLEET_DIR=${FLEET_DIR:-$STAGE_ROOT/runtime/$RUN_ID/fleet}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:-$STAGE_ROOT/runtime/$RUN_ID/train}
RUN_WALLTIME=${RUN_WALLTIME:-08:15:00}
WANDB_EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_ENTITY=${WANDB_ENTITY:-}
OCI_CONTAINER_IMAGE=${OCI_CONTAINER_IMAGE:-nvcr.io#nvidia/isaac-sim:5.1.0}
PROJ=${PROJ:-roborender_dagger_base_scale_8h}
EXP=${EXP:-dagger_base_256env_300step_${RUN_ID}}
NUM_ROLLOUT_ITERATIONS=${NUM_ROLLOUT_ITERATIONS:-500}
SAVE_FREQUENCY=${SAVE_FREQUENCY:-5}
EXPECTED_IMAGE3F_RELEASE_ID=${EXPECTED_IMAGE3F_RELEASE_ID:-roborender-image3f-rgb030-depth030-step5000-4090-source-1.0.0}
EXPECTED_IMAGE3F_ADAPTER_SHA256=${EXPECTED_IMAGE3F_ADAPTER_SHA256:-c68de259ce5ee8e4df5a45296be8328242b3ebba351990df1d86f4e3a7a2d588}
EXPECTED_IMAGE3F_INFERENCE_STEPS=${EXPECTED_IMAGE3F_INFERENCE_STEPS:-5}
EXPECTED_IMAGE3F_TEACACHE_THRESHOLD=${EXPECTED_IMAGE3F_TEACACHE_THRESHOLD:-0.3}
EXPECTED_IMAGE3F_CFG_SCALE=${EXPECTED_IMAGE3F_CFG_SCALE:-1.0}
EXPECTED_IMAGE3F_CUDA_GRAPH=${EXPECTED_IMAGE3F_CUDA_GRAPH:-disabled}
DRY_RUN=${DRY_RUN:-0}

fatal() {
    echo "FATAL: $*" >&2
    exit 2
}

for path in "$ROOT" "$GR00T_ROOT" "$IMAGE3F_ROOT" "$ROBORENDER_MODEL_BASE" "$ANCHORS"; do
    [ -d "$path" ] || fatal "required directory missing: $path"
done
[ -x "$VIRAL_PYTHON" ] || fatal "trainer environment missing: $VIRAL_PYTHON"
[ -x "$WANDB_PYTHON" ] || fatal "host W&B environment missing: $WANDB_PYTHON"
[ -x "$IMAGE3F_PYTHON" ] || fatal "Image3F H100 environment is not ready: $IMAGE3F_PYTHON"
[ -s "$IMAGE3F_ROOT/assets/generated/engines/current/engine_build_manifest.json" ] || \
    fatal "Image3F H100 TensorRT engine is not ready"
[ -z "$(git -C "$ROOT" status --porcelain)" ] || fatal "fast_roborender source is dirty"
[ -z "$(git -C "$GR00T_ROOT" status --porcelain)" ] || fatal "GR00T source is dirty"
[ -s "$TEACHER_CKPT" ] || fatal "expert labeler checkpoint missing: $TEACHER_CKPT"
ACTUAL_TEACHER_CKPT_SHA256=$(sha256sum "$TEACHER_CKPT" | awk '{print $1}')
[ "$ACTUAL_TEACHER_CKPT_SHA256" = "$EXPECTED_TEACHER_CKPT_SHA256" ] || \
    fatal "expert labeler checkpoint hash mismatch"

FAST_REVISION=$(git -C "$ROOT" rev-parse HEAD)
GR00T_REVISION=$(git -C "$GR00T_ROOT" rev-parse HEAD)
if [ "$DRY_RUN" = 1 ]; then
    WANDB_ENTITY=${WANDB_ENTITY:-DRY_RUN_UNVERIFIED}
else
    [ "$WANDB_MODE" = online ] || fatal "production telemetry requires WANDB_MODE=online"
    NETRC="$STAGE_ROOT/runtime/home/.netrc"
    if [ -z "${WANDB_API_KEY:-}" ]; then
        [ -s "$NETRC" ] || fatal "no W&B credential is available for secure propagation to the job"
        [ "$(stat -c %u "$NETRC")" -eq "$(id -u)" ] || fatal "W&B netrc is not owned by the submitting user"
        case "$(stat -c %a "$NETRC")" in
            400|600) ;;
            *) fatal "W&B netrc must have owner-only permissions (0400 or 0600)" ;;
        esac
    fi
    # The authenticated email and personal entity are verified again inside
    # the Isaac container before distributed training starts.
    WANDB_ENTITY=${WANDB_ENTITY:-}
fi

mkdir -p "$FLEET_DIR/logs" "$TRAIN_RUN_DIR"
EXPORTS="STAGE_ROOT=$STAGE_ROOT,ROOT=$ROOT,GR00T_ROOT=$GR00T_ROOT,IMAGE3F_ROOT=$IMAGE3F_ROOT,IMAGE3F_ENV=$IMAGE3F_ENV,IMAGE3F_PYTHON=$IMAGE3F_PYTHON,ROBORENDER_MODEL_BASE=$ROBORENDER_MODEL_BASE,ANCHORS=$ANCHORS,ANCHOR_IDX=0,ANCHOR_MIN_FILES=2048,TEACHER_CKPT=$TEACHER_CKPT,EXPECTED_TEACHER_CKPT_SHA256=$EXPECTED_TEACHER_CKPT_SHA256,VIRAL_ENV=$VIRAL_ENV,VIRAL_PYTHON=$VIRAL_PYTHON,WANDB_ENV=$WANDB_ENV,WANDB_PYTHON=$WANDB_PYTHON,OCI_CONTAINER_IMAGE=$OCI_CONTAINER_IMAGE,RUN_ID=$RUN_ID,FLEET_DIR=$FLEET_DIR,TRAIN_RUN_DIR=$TRAIN_RUN_DIR,CKPT=none,EXPECTED_TRAIN_CHECKPOINT_SHA256=NONE,GLOBAL_NUM_ENVS=256,ROLLOUT_STEPS=300,TRAINER_NODES=8,ENVS_PER_RENDERER=5,MIX=0.0,REFERENCE_LAG=0,NUM_MINI_BATCHES=1,NUM_ROLLOUT_ITERATIONS=$NUM_ROLLOUT_ITERATIONS,NUM_LEARNING_EPOCHS=1,BEHAVIOR_SOURCE=annealed,TEACHER_ROLLOUT_RATIO=0.0,BETA_SCHEDULE=linear,BETA_START=1.0,BETA_END=0.0,BETA_DECAY_ITERS=500,VG_PRIMARY_VIEW=head_color,SAVE_FREQUENCY=$SAVE_FREQUENCY,PROJ=$PROJ,EXP=$EXP,WANDB_EXPECTED_EMAIL=$WANDB_EXPECTED_EMAIL,WANDB_MODE=$WANDB_MODE,WANDB_ENTITY=$WANDB_ENTITY,EXPECTED_FAST_ROBORENDER_REVISION=$FAST_REVISION,EXPECTED_GR00T_REVISION=$GR00T_REVISION,EXPECTED_IMAGE3F_RELEASE_ID=$EXPECTED_IMAGE3F_RELEASE_ID,EXPECTED_IMAGE3F_ADAPTER_SHA256=$EXPECTED_IMAGE3F_ADAPTER_SHA256,EXPECTED_IMAGE3F_INFERENCE_STEPS=$EXPECTED_IMAGE3F_INFERENCE_STEPS,EXPECTED_IMAGE3F_TEACACHE_THRESHOLD=$EXPECTED_IMAGE3F_TEACACHE_THRESHOLD,EXPECTED_IMAGE3F_CFG_SCALE=$EXPECTED_IMAGE3F_CFG_SCALE,EXPECTED_IMAGE3F_CUDA_GRAPH=$EXPECTED_IMAGE3F_CUDA_GRAPH"
if [ -n "${WANDB_API_KEY:-}" ]; then
    # Export the existing variable by name so the secret is never embedded in
    # the printed dry-run command or the submission manifest.
    export WANDB_API_KEY
    EXPORTS+=",WANDB_API_KEY"
fi
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
expert_labeler_checkpoint=$TEACHER_CKPT
expert_labeler_checkpoint_sha256=$EXPECTED_TEACHER_CKPT_SHA256
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
