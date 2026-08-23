#!/bin/bash
# OCI production topology: 52 H100 Image3F servers and eight DAgger ranks
# share one 8-node/64-GPU allocation. Four GPUs remain as placement slack.
# Submit only through submit_oci_packed.sh so paths and provenance are pinned.
#SBATCH --account=general_cs_infra
#SBATCH --partition=grizzly
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=768G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=rr_base_256x300
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
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3}
WANDB_ENV=${WANDB_ENV:-$STAGE_ROOT/runtime/envs/wandb-host}
VIRAL_VENV_ACTIVATE=${VIRAL_VENV_ACTIVATE:-$STAGE_ROOT/runtime/envs/activate_viral_isaac.sh}
OCI_CONTAINER_IMAGE=${OCI_CONTAINER_IMAGE:-nvcr.io#nvidia/isaac-sim:5.1.0}
RUN_ID=${RUN_ID:-oci_base_256x300_${SLURM_JOB_ID}}
RUN_PROFILE=${RUN_PROFILE:-production_256x300}
FLEET_DIR=${FLEET_DIR:-$STAGE_ROOT/runtime/$RUN_ID/fleet}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:-$STAGE_ROOT/runtime/$RUN_ID/train}
GLOBAL_NUM_ENVS=${GLOBAL_NUM_ENVS:-256}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-300}
TRAINER_NODES=${TRAINER_NODES:-8}
ENVS_PER_RENDERER=${ENVS_PER_RENDERER:-5}
REFERENCE_LAG=${REFERENCE_LAG:-0}
MIX=${MIX:-0.0}
CKPT=${CKPT:-none}
EXPECTED_TRAIN_CHECKPOINT_SHA256=${EXPECTED_TRAIN_CHECKPOINT_SHA256:-NONE}
ANCHOR_MIN_FILES=${ANCHOR_MIN_FILES:-2048}
RENDERER_CPUS_PER_TASK=${RENDERER_CPUS_PER_TASK:-8}
RENDERER_MEM_PER_NODE=${RENDERER_MEM_PER_NODE:-512G}
TRAINER_CPUS_PER_TASK=${TRAINER_CPUS_PER_TASK:-16}
TRAINER_MEM_PER_NODE=${TRAINER_MEM_PER_NODE:-128G}
RENDER_READY_TIMEOUT_S=${RENDER_READY_TIMEOUT_S:-1800}
WANDB_EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PYTHON=${WANDB_PYTHON:-$WANDB_ENV/bin/python}
VG_PRIMARY_VIEW=${VG_PRIMARY_VIEW:-head_color}
VG_POLICY_VIEWS=${VG_POLICY_VIEWS:-head_color:hand_left:hand_right}
VG_POLICY_VIEW_COUNT=${VG_POLICY_VIEW_COUNT:-3}
VG_POLICY_VIEW_LAYOUT=${VG_POLICY_VIEW_LAYOUT:-vertical}
VG_POLICY_IMAGE_SHAPE=${VG_POLICY_IMAGE_SHAPE:-324x192x3}
PACKED_NODES=${PACKED_NODES:-8}
PACKED_GPUS_PER_NODE=${PACKED_GPUS_PER_NODE:-8}
PACKED_PARTITION=${PACKED_PARTITION:-grizzly}
PACKED_CPUS_PER_GPU=${PACKED_CPUS_PER_GPU:-10}
PACKED_MEM=${PACKED_MEM:-768G}
EXPECTED_RENDERERS=${EXPECTED_RENDERERS:-52}
CAMERA_CALIBRATION_ACCEPTANCE=${CAMERA_CALIBRATION_ACCEPTANCE:-NONE}
CAMERA_CALIBRATION_RUN_ID=${CAMERA_CALIBRATION_RUN_ID:-NONE}
EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256=${EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256:-NONE}
PREREQUISITE_PROFILE_VALIDATION=${PREREQUISITE_PROFILE_VALIDATION:-NONE}
EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256=${EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256:-NONE}
PREREQUISITE_PROFILE_EXPECTED=${PREREQUISITE_PROFILE_EXPECTED:-NONE}
NUM_ROLLOUT_ITERATIONS=${NUM_ROLLOUT_ITERATIONS:-}
SAVE_FREQUENCY=${SAVE_FREQUENCY:-}

fatal() {
    echo "FATAL: $*" >&2
    exit 2
}

[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]{1,128}$ ]] || \
    fatal "RUN_ID must match [A-Za-z0-9_-]{1,128}"

case "$RUN_PROFILE" in
    production_256x300)
        NUM_ROLLOUT_ITERATIONS=${NUM_ROLLOUT_ITERATIONS:-500}
        SAVE_FREQUENCY=${SAVE_FREQUENCY:-5}
        [ "$PACKED_NODES:$PACKED_GPUS_PER_NODE:$GLOBAL_NUM_ENVS:$ROLLOUT_STEPS:$TRAINER_NODES:$ENVS_PER_RENDERER:$EXPECTED_RENDERERS" = \
          "8:8:256:300:8:5:52" ] || fatal "production_256x300 topology is immutable"
        [ "$PACKED_PARTITION:$PACKED_CPUS_PER_GPU:$PACKED_MEM:$RENDERER_CPUS_PER_TASK:$RENDERER_MEM_PER_NODE:$TRAINER_CPUS_PER_TASK:$TRAINER_MEM_PER_NODE" = \
          "grizzly:10:768G:8:512G:16:128G" ] || fatal "production_256x300 resources are immutable"
        [ "${IMAGE_DAGGER_DUMP_SIM_RGB:-0}:${IMAGE_DAGGER_DUMP_MAX_STEPS:-0}:${IMAGE_DAGGER_DUMP_MAX_ENVS:-0}:${IMAGE_DAGGER_VERIFY_DDP_SYNC:-}:${IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS:-}" = "0:0:0:1:1" ] || \
            fatal "production profile requires one-time DDP sync/loss gates and forbids camera dumps"
        ;;
    smoke_32x300)
        NUM_ROLLOUT_ITERATIONS=${NUM_ROLLOUT_ITERATIONS:-1}
        SAVE_FREQUENCY=${SAVE_FREQUENCY:-1}
        [ "$PACKED_NODES:$PACKED_GPUS_PER_NODE:$GLOBAL_NUM_ENVS:$ROLLOUT_STEPS:$TRAINER_NODES:$ENVS_PER_RENDERER:$EXPECTED_RENDERERS:${NUM_ROLLOUT_ITERATIONS:-}:${SAVE_FREQUENCY:-}" = \
          "1:8:32:300:1:5:7:1:1" ] || fatal "smoke_32x300 topology is immutable"
        [ "$PACKED_PARTITION:$PACKED_CPUS_PER_GPU:$PACKED_MEM:$RENDERER_CPUS_PER_TASK:$RENDERER_MEM_PER_NODE:$TRAINER_CPUS_PER_TASK:$TRAINER_MEM_PER_NODE" = \
          "backfill_block1:10:768G:8:512G:16:128G" ] || \
            fatal "smoke_32x300 scheduler and step resources are immutable"
        [ "${IMAGE_DAGGER_DUMP_SIM_RGB:-0}:${IMAGE_DAGGER_DUMP_MAX_STEPS:-0}:${IMAGE_DAGGER_DUMP_MAX_ENVS:-0}:${IMAGE_DAGGER_VERIFY_DDP_SYNC:-0}:${IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS:-}" = "0:0:0:0:1" ] || \
            fatal "smoke_32x300 does not enable calibration dumps or DDP collectives"
        ;;
    calibration_3view)
        NUM_ROLLOUT_ITERATIONS=${NUM_ROLLOUT_ITERATIONS:-1}
        SAVE_FREQUENCY=${SAVE_FREQUENCY:-1}
        [ "$PACKED_NODES:$PACKED_GPUS_PER_NODE:$GLOBAL_NUM_ENVS:$ROLLOUT_STEPS:$TRAINER_NODES:$EXPECTED_RENDERERS:${NUM_ROLLOUT_ITERATIONS:-}:${SAVE_FREQUENCY:-}" = \
          "1:2:1:2:1:1:1:1" ] || fatal "calibration_3view topology is immutable"
        [ "$PACKED_PARTITION:$PACKED_CPUS_PER_GPU:$PACKED_MEM:$RENDERER_MEM_PER_NODE:$TRAINER_MEM_PER_NODE" = \
          "backfill_block1:16:256G:64G:128G" ] || fatal "calibration_3view resources are immutable"
        [ "${IMAGE_DAGGER_DUMP_SIM_RGB:-}:${IMAGE_DAGGER_DUMP_MAX_STEPS:-}:${IMAGE_DAGGER_DUMP_MAX_ENVS:-}:${IMAGE_DAGGER_VERIFY_DDP_SYNC:-0}:${IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS:-}" = \
          "1:2:1:0:1" ] || fatal "calibration_3view diagnostic and loss gates are immutable"
        ;;
    smoke_256x300)
        NUM_ROLLOUT_ITERATIONS=${NUM_ROLLOUT_ITERATIONS:-1}
        SAVE_FREQUENCY=${SAVE_FREQUENCY:-1}
        [ "$PACKED_NODES:$PACKED_GPUS_PER_NODE:$GLOBAL_NUM_ENVS:$ROLLOUT_STEPS:$TRAINER_NODES:$EXPECTED_RENDERERS:${NUM_ROLLOUT_ITERATIONS:-}:${SAVE_FREQUENCY:-}" = \
          "8:8:256:300:8:52:1:1" ] || fatal "smoke_256x300 topology is immutable"
        [ "$PACKED_PARTITION:$PACKED_CPUS_PER_GPU:$PACKED_MEM:$RENDERER_CPUS_PER_TASK:$RENDERER_MEM_PER_NODE:$TRAINER_CPUS_PER_TASK:$TRAINER_MEM_PER_NODE" = \
          "grizzly:10:768G:8:512G:16:128G" ] || \
            fatal "smoke_256x300 scheduler and step resources are immutable"
        [ "${IMAGE_DAGGER_DUMP_SIM_RGB:-0}:${IMAGE_DAGGER_DUMP_MAX_STEPS:-0}:${IMAGE_DAGGER_DUMP_MAX_ENVS:-0}:${IMAGE_DAGGER_VERIFY_DDP_SYNC:-}:${IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS:-}" = \
          "0:0:0:1:1" ] || fatal "smoke_256x300 diagnostic, DDP, and loss gates are immutable"
        ;;
    *) fatal "unknown RUN_PROFILE=$RUN_PROFILE" ;;
esac
if [ "$RUN_PROFILE" = calibration_3view ]; then
    OCI_RUN_ROOT=$STAGE_ROOT/runtime/$RUN_ID
    [ "$FLEET_DIR" = "$OCI_RUN_ROOT/fleet" ] || \
        fatal "calibration FLEET_DIR must be bound to RUN_ID"
    [ "$TRAIN_RUN_DIR" = "$OCI_RUN_ROOT/train" ] || \
        fatal "calibration TRAIN_RUN_DIR must be bound to RUN_ID"
    [ -d "$OCI_RUN_ROOT" ] && [ ! -L "$OCI_RUN_ROOT" ] || \
        fatal "calibration RUN_ROOT is missing or unsafe"
    [ "$(realpath -e "$OCI_RUN_ROOT")" = "$OCI_RUN_ROOT" ] || \
        fatal "calibration RUN_ROOT must be canonical"
    for run_child in "$FLEET_DIR" "$TRAIN_RUN_DIR"; do
        [ -d "$run_child" ] && [ ! -L "$run_child" ] || \
            fatal "calibration run directory is missing or unsafe: $run_child"
        [ "$(realpath -e "$run_child")" = "$run_child" ] || \
            fatal "calibration run directory must be canonical: $run_child"
    done
fi
if [ "$RUN_PROFILE" = calibration_3view ]; then
    [ "$CAMERA_CALIBRATION_ACCEPTANCE:$CAMERA_CALIBRATION_RUN_ID:$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256" = \
      NONE:NONE:NONE ] || fatal "calibration_3view cannot consume a prior camera acceptance"
else
    [[ "$CAMERA_CALIBRATION_RUN_ID" =~ ^[A-Za-z0-9_-]{1,128}$ ]] || \
        fatal "CAMERA_CALIBRATION_RUN_ID is malformed"
    [ "$CAMERA_CALIBRATION_ACCEPTANCE" = \
      "$STAGE_ROOT/runtime/validation/camera_calibration_acceptance_${CAMERA_CALIBRATION_RUN_ID}.json" ] || \
        fatal "camera calibration acceptance escaped its exact versioned path"
fi
[ "${SLURM_NNODES:-0}" -eq "$PACKED_NODES" ] || \
    fatal "$RUN_PROFILE requires exactly $PACKED_NODES allocated node(s)"
[ "${SLURM_JOB_PARTITION:-}" = "$PACKED_PARTITION" ] || \
    fatal "$RUN_PROFILE requires partition $PACKED_PARTITION"
[ "$REFERENCE_LAG" -eq 0 ] || fatal "Raven FT-FIXED production uses reference lag 0"
case "$MIX" in 0|0.0) ;; *) fatal "packed production requires generated RGB for all environments (MIX=0.0)" ;; esac
case "$CKPT" in none|null) ;; *) fatal "Round-4 and all other student checkpoints are excluded; set CKPT=none" ;; esac
[ "$EXPECTED_TRAIN_CHECKPOINT_SHA256" = NONE ] || fatal "fresh policy must pin checkpoint hash as NONE"
[ "$VG_PRIMARY_VIEW" = head_color ] || fatal "three-view production requires primary_view=head_color"
[ "$VG_POLICY_VIEWS" = head_color:hand_left:hand_right ] || \
    fatal "three-view production requires ordered views head_color:hand_left:hand_right"
[ "$VG_POLICY_VIEW_COUNT" = 3 ] || fatal "three-view production requires policy_view_count=3"
[ "$VG_POLICY_VIEW_LAYOUT" = vertical ] || fatal "three-view production requires vertical layout"
[ "$VG_POLICY_IMAGE_SHAPE" = 324x192x3 ] || \
    fatal "three-view production requires image shape 324x192x3"
[[ "$NUM_ROLLOUT_ITERATIONS" =~ ^[1-9][0-9]*$ ]] || \
    fatal "NUM_ROLLOUT_ITERATIONS must be a positive integer"
[[ "$SAVE_FREQUENCY" =~ ^[1-9][0-9]*$ ]] || fatal "SAVE_FREQUENCY must be a positive integer"
[ $((NUM_ROLLOUT_ITERATIONS % SAVE_FREQUENCY)) -eq 0 ] || \
    fatal "NUM_ROLLOUT_ITERATIONS must be divisible by SAVE_FREQUENCY"
printf -v EXPECTED_FINAL_CHECKPOINT '%s/model_step_%06d.pt' \
    "$TRAIN_RUN_DIR" "$NUM_ROLLOUT_ITERATIONS"

for path in "$ROOT" "$GR00T_ROOT" "$IMAGE3F_ROOT" "$ROBORENDER_MODEL_BASE" "$ANCHORS"; do
    [ -d "$path" ] || fatal "required directory missing: $path"
done
[ -x "$IMAGE3F_PYTHON" ] || fatal "H100 Image3F environment has not been built: $IMAGE3F_PYTHON"
[ -s "$IMAGE3F_ROOT/assets/generated/engines/current/engine_build_manifest.json" ] || \
    fatal "target-local H100 TensorRT engine has not passed its build gate"
[ -x "$VIRAL_PYTHON" ] || fatal "trainer environment has not been built: $VIRAL_PYTHON"
[ -x "$HOST_PYTHON" ] || fatal "OCI host Python is missing: $HOST_PYTHON"
[ -x "$WANDB_PYTHON" ] || fatal "OCI host W&B environment has not been built: $WANDB_PYTHON"
[ -f "$VIRAL_VENV_ACTIVATE" ] || fatal "trainer activation script missing: $VIRAL_VENV_ACTIVATE"

FAST_REVISION=$(git -C "$ROOT" rev-parse HEAD)
GR00T_REVISION=$(git -C "$GR00T_ROOT" rev-parse HEAD)
[ -z "$(git -C "$ROOT" status --porcelain)" ] || fatal "fast_roborender source is dirty"
[ -z "$(git -C "$GR00T_ROOT" status --porcelain)" ] || fatal "GR00T source is dirty"
[ -z "${EXPECTED_FAST_ROBORENDER_REVISION:-}" ] || \
    [ "$FAST_REVISION" = "$EXPECTED_FAST_ROBORENDER_REVISION" ] || fatal "fast_roborender revision changed"
[ -z "${EXPECTED_GR00T_REVISION:-}" ] || \
    [ "$GR00T_REVISION" = "$EXPECTED_GR00T_REVISION" ] || fatal "GR00T revision changed"
if [ "$RUN_PROFILE" = calibration_3view ]; then
    [ "$CAMERA_CALIBRATION_ACCEPTANCE:$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256" = "NONE:NONE" ] || \
        fatal "calibration_3view cannot consume a prior camera acceptance"
else
    [ -s "$CAMERA_CALIBRATION_ACCEPTANCE" ] || fatal "accepted camera calibration is missing"
    [ "$(sha256sum "$CAMERA_CALIBRATION_ACCEPTANCE" | awk '{print $1}')" = \
      "$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256" ] || \
        fatal "camera calibration acceptance hash mismatch"
    "$HOST_PYTHON" "$ROOT/scripts/scaling/camera_calibration_gate.py" verify \
        --acceptance "$CAMERA_CALIBRATION_ACCEPTANCE" \
        --expected-run-id "$CAMERA_CALIBRATION_RUN_ID" \
        --expected-fast-roborender-revision "$FAST_REVISION" \
        --expected-gr00t-revision "$GR00T_REVISION" \
        --expected-acceptance-sha256 "$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256" \
        >/dev/null || fatal "camera calibration acceptance validation failed"
fi
if [ "$PREREQUISITE_PROFILE_EXPECTED" = NONE ]; then
    [ "$PREREQUISITE_PROFILE_VALIDATION:$EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256" = "NONE:NONE" ] || \
        fatal "$RUN_PROFILE cannot consume an unexpected predecessor profile"
else
    case "$RUN_PROFILE:$PREREQUISITE_PROFILE_EXPECTED" in
        smoke_256x300:smoke_32x300|production_256x300:smoke_256x300) ;;
        *) fatal "invalid predecessor profile transition $PREREQUISITE_PROFILE_EXPECTED -> $RUN_PROFILE" ;;
    esac
    [ -s "$PREREQUISITE_PROFILE_VALIDATION" ] || fatal "predecessor profile validation is missing"
    [ "$(sha256sum "$PREREQUISITE_PROFILE_VALIDATION" | awk '{print $1}')" = \
      "$EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256" ] || \
        fatal "predecessor profile validation hash mismatch"
    "$HOST_PYTHON" "$ROOT/scripts/scaling/profile_validation_gate.py" \
        --manifest "$PREREQUISITE_PROFILE_VALIDATION" \
        --expected-profile "$PREREQUISITE_PROFILE_EXPECTED" \
        --expected-fast-revision "$FAST_REVISION" \
        --expected-gr00t-revision "$GR00T_REVISION" \
        --expected-manifest-sha256 "$EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256" \
        >/dev/null || fatal "predecessor profile validation failed"
fi

[ -s "$TEACHER_CKPT" ] || fatal "expert labeler checkpoint missing"
[ "$(sha256sum "$TEACHER_CKPT" | awk '{print $1}')" = "$EXPECTED_TEACHER_CKPT_SHA256" ] || \
    fatal "expert labeler checkpoint hash mismatch"
[ -s "$GR00T_ROOT/models/model_walk.pt" ] || fatal "Homie walk model missing"
[ -s "$GR00T_ROOT/models/model_stand.pt" ] || fatal "Homie stand model missing"

ANCHOR_COUNT=$(find "$ANCHORS" -maxdepth 1 -type f -name '*.npz' -printf '.' | wc -c)
[ "$ANCHOR_COUNT" -ge "$ANCHOR_MIN_FILES" ] || \
    fatal "FT-FIXED anchor pool has $ANCHOR_COUNT files; expected at least $ANCHOR_MIN_FILES"
"$HOST_PYTHON" - "$ANCHORS" <<'PY'
import sys
import zipfile
from pathlib import Path

path = sorted(Path(sys.argv[1]).glob("*.npz"))[0]
with zipfile.ZipFile(path) as archive:
    keys = {Path(name).stem for name in archive.namelist() if name.endswith(".npy")}
    required = {"gen__head_color", "gen__hand_left", "gen__hand_right"}
    missing = sorted(required.difference(keys))
    if missing:
        raise SystemExit(f"FATAL: anchor schema missing {missing} in {path}")
PY

REQUIRED_RENDERERS=$("$HOST_PYTHON" "$ROOT/scripts/scaling/scale_plan.py" \
    --global-envs "$GLOBAL_NUM_ENVS" --rollout-steps "$ROLLOUT_STEPS" \
    --trainer-nodes "$TRAINER_NODES" --raytraced-mix "$MIX" \
    --envs-per-renderer "$ENVS_PER_RENDERER" --reference-lag "$REFERENCE_LAG" \
    --field required_renderers)
[ "$REQUIRED_RENDERERS" -eq "$EXPECTED_RENDERERS" ] || \
    fatal "capacity plan expected $EXPECTED_RENDERERS renderers, got $REQUIRED_RENDERERS"
ALLOCATED_GPUS=$((PACKED_NODES * PACKED_GPUS_PER_NODE))
[ $((REQUIRED_RENDERERS + TRAINER_NODES)) -le "$ALLOCATED_GPUS" ] || \
    fatal "packed GPU plan exceeds $ALLOCATED_GPUS GPUs"
SPARE_GPUS=$((ALLOCATED_GPUS - REQUIRED_RENDERERS - TRAINER_NODES))

mkdir -p "$FLEET_DIR/endpoints" "$FLEET_DIR/queues" "$FLEET_DIR/logs" \
    "$TRAIN_RUN_DIR" "$STAGE_ROOT/runtime/home" "$STAGE_ROOT/runtime/cache" \
    "$STAGE_ROOT/runtime/tmp"
cat > "$FLEET_DIR/packed_manifest.txt" <<EOF
job_id=$SLURM_JOB_ID
run_profile=$RUN_PROFILE
nodes=$SLURM_NNODES
allocated_gpus=$ALLOCATED_GPUS
partition=$PACKED_PARTITION
cpus_per_gpu=$PACKED_CPUS_PER_GPU
memory=$PACKED_MEM
renderer_tasks=$REQUIRED_RENDERERS
trainer_tasks=$TRAINER_NODES
spare_gpus=$SPARE_GPUS
global_envs=$GLOBAL_NUM_ENVS
rollout_steps=$ROLLOUT_STEPS
initialization=fresh_student
checkpoint=none
expert_labeler_checkpoint=$TEACHER_CKPT
expert_labeler_checkpoint_sha256=$EXPECTED_TEACHER_CKPT_SHA256
reference_policy=FT-FIXED
reference_lag=0
policy_views=${VG_POLICY_VIEWS//:/,}
policy_view_count=$VG_POLICY_VIEW_COUNT
policy_view_layout=$VG_POLICY_VIEW_LAYOUT
policy_image_shape=$VG_POLICY_IMAGE_SHAPE
initial_policy_checkpoint=$TRAIN_RUN_DIR/initial_policy.pt
expected_final_checkpoint=$EXPECTED_FINAL_CHECKPOINT
camera_calibration_acceptance=$CAMERA_CALIBRATION_ACCEPTANCE
camera_calibration_acceptance_sha256=$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256
camera_calibration_run_id=$CAMERA_CALIBRATION_RUN_ID
prerequisite_profile=$PREREQUISITE_PROFILE_EXPECTED
prerequisite_profile_validation=$PREREQUISITE_PROFILE_VALIDATION
prerequisite_profile_validation_sha256=$EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256
anchor_files=$ANCHOR_COUNT
fast_roborender_revision=$FAST_REVISION
gr00t_revision=$GR00T_REVISION
container=$OCI_CONTAINER_IMAGE
EOF

export STAGE_ROOT ROOT GR00T_ROOT IMAGE3F_ROOT IMAGE3F_ENV IMAGE3F_PYTHON
export ROBORENDER_MODEL_BASE ANCHORS
export TEACHER_CKPT EXPECTED_TEACHER_CKPT_SHA256
export FLEET_DIR TRAIN_RUN_DIR GLOBAL_NUM_ENVS ROLLOUT_STEPS TRAINER_NODES
export RUN_PROFILE PACKED_NODES PACKED_GPUS_PER_NODE EXPECTED_RENDERERS
export PACKED_PARTITION PACKED_CPUS_PER_GPU PACKED_MEM
export ENVS_PER_RENDERER REFERENCE_LAG MIX CKPT EXPECTED_TRAIN_CHECKPOINT_SHA256
export VG_PRIMARY_VIEW VG_POLICY_VIEWS VG_POLICY_VIEW_COUNT VG_POLICY_VIEW_LAYOUT VG_POLICY_IMAGE_SHAPE
export IMAGE_DAGGER_DUMP_SIM_RGB IMAGE_DAGGER_DUMP_MAX_STEPS IMAGE_DAGGER_DUMP_MAX_ENVS
export IMAGE_DAGGER_VERIFY_DDP_SYNC
export IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS
export CAMERA_CALIBRATION_ACCEPTANCE EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256
export CAMERA_CALIBRATION_RUN_ID
export PREREQUISITE_PROFILE_VALIDATION EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256
export PREREQUISITE_PROFILE_EXPECTED
export VIRAL_PYTHON VIRAL_VENV_ACTIVATE WANDB_ENV WANDB_PYTHON WANDB_EXPECTED_EMAIL WANDB_MODE
export TRAINER_CPUS_PER_TASK TRAINER_MEM_PER_NODE RENDER_READY_TIMEOUT_S
export TRAINER_CONTAINER_IMAGE="$OCI_CONTAINER_IMAGE"
export TRAINER_CONTAINER_MOUNTS="$STAGE_ROOT:$STAGE_ROOT"
export HOME_OVERRIDE="$STAGE_ROOT/runtime/home"
export XDG_CACHE_HOME="$STAGE_ROOT/runtime/cache/xdg"
export XDG_DATA_HOME="$STAGE_ROOT/runtime/cache/share"
export TORCH_HOME="$STAGE_ROOT/runtime/cache/torch"
export TMPDIR="$STAGE_ROOT/runtime/tmp"
export IMAGE_DAGGER_DUMP_DIR="$TRAIN_RUN_DIR/render_dumps"
# The batch job receives an explicit allowlist from sbatch. Export that already
# sanitized environment, plus the values established above, to nested srun steps.
export SLURM_EXPORT_ENV=ALL
export IMAGE3F_RUN_ID="$RUN_ID"
export IMAGE3F_MAX_STREAMS=16
export STARTUP_STAGGER_SCOPE=local
export STARTUP_STAGGER_S=${STARTUP_STAGGER_S:-2}
export PYTHONDONTWRITEBYTECODE=1
export PROJ=${PROJ:-roborender_dagger_base_scale_4h}
export EXP=${EXP:-dagger_base_256env_300step_${SLURM_JOB_ID}}
export NUM_MINI_BATCHES=${NUM_MINI_BATCHES:-1}
export NUM_ROLLOUT_ITERATIONS
export NUM_LEARNING_EPOCHS=${NUM_LEARNING_EPOCHS:-1}
export BEHAVIOR_SOURCE=${BEHAVIOR_SOURCE:-annealed}
export BETA_SCHEDULE=${BETA_SCHEDULE:-linear}
export BETA_START=${BETA_START:-1.0}
export BETA_END=${BETA_END:-0.0}
export BETA_DECAY_ITERS=${BETA_DECAY_ITERS:-500}
export TEACHER_ROLLOUT_RATIO=${TEACHER_ROLLOUT_RATIO:-0.0}
export SAVE_FREQUENCY
export EXPECTED_FAST_ROBORENDER_REVISION=${EXPECTED_FAST_ROBORENDER_REVISION:-$FAST_REVISION}
export EXPECTED_GR00T_REVISION=${EXPECTED_GR00T_REVISION:-$GR00T_REVISION}
export EXPECTED_IMAGE3F_RELEASE_ID=${EXPECTED_IMAGE3F_RELEASE_ID:-roborender-image3f-rgb030-depth030-step5000-4090-source-1.0.0}
export EXPECTED_IMAGE3F_ADAPTER_SHA256=${EXPECTED_IMAGE3F_ADAPTER_SHA256:-c68de259ce5ee8e4df5a45296be8328242b3ebba351990df1d86f4e3a7a2d588}
export EXPECTED_IMAGE3F_INFERENCE_STEPS=${EXPECTED_IMAGE3F_INFERENCE_STEPS:-5}
export EXPECTED_IMAGE3F_TEACACHE_THRESHOLD=${EXPECTED_IMAGE3F_TEACACHE_THRESHOLD:-0.3}
export EXPECTED_IMAGE3F_CFG_SCALE=${EXPECTED_IMAGE3F_CFG_SCALE:-1.0}
export EXPECTED_IMAGE3F_CUDA_GRAPH=${EXPECTED_IMAGE3F_CUDA_GRAPH:-disabled}

RENDER_STEP_PID=""
TRAIN_PID=""
PRE_TIMEOUT=0

cleanup() {
    local rc=$?
    trap - EXIT USR1 TERM INT
    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
    if [ -n "$RENDER_STEP_PID" ] && kill -0 "$RENDER_STEP_PID" 2>/dev/null; then
        kill -TERM "$RENDER_STEP_PID" 2>/dev/null || true
        wait "$RENDER_STEP_PID" 2>/dev/null || true
    fi
    exit "$rc"
}

forward_pre_timeout() {
    PRE_TIMEOUT=1
    echo "PRE-TIMEOUT: forwarding USR1 to packed trainer"
    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -USR1 "$TRAIN_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT TERM INT
trap forward_pre_timeout USR1

srun --nodes="$PACKED_NODES" --ntasks="$REQUIRED_RENDERERS" --ntasks-per-node=7 \
    --distribution=cyclic --gpus-per-task=1 --cpus-per-task="$RENDERER_CPUS_PER_TASK" \
    --mem="$RENDERER_MEM_PER_NODE" --exact --exclusive --gpu-bind=closest \
    --cpu-bind=cores --kill-on-bad-exit=1 \
    --container-image="$OCI_CONTAINER_IMAGE" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT" \
    --output="$FLEET_DIR/logs/renderer_%t.out" \
    --error="$FLEET_DIR/logs/renderer_%t.err" \
    bash -lc '
        set -Eeuo pipefail
        export ROBORENDER_CONTAINERIZED=1
        export SLURM_ARRAY_TASK_ID=$SLURM_PROCID
        export SLURM_ARRAY_JOB_ID=$SLURM_JOB_ID
        export IMAGE3F_ADVERTISE_HOST=${SLURMD_NODENAME:-$(hostname -f)}
        exec bash "$ROOT/scripts/server/image3f_server_array.sbatch"
    ' &
RENDER_STEP_PID=$!

RENDER_JOB_ID="" bash "$ROOT/scripts/train/dagger_scale_256x300.sbatch" &
TRAIN_PID=$!

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    if ! kill -0 "$RENDER_STEP_PID" 2>/dev/null; then
        if wait "$RENDER_STEP_PID"; then RENDER_RC=0; else RENDER_RC=$?; fi
        RENDER_STEP_PID=""
        echo "FATAL: packed renderer step exited while trainer was live (rc=$RENDER_RC)" >&2
        kill -USR1 "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" 2>/dev/null || true
        TRAIN_PID=""
        exit 70
    fi
    sleep 5
done

if wait "$TRAIN_PID"; then TRAIN_RC=0; else TRAIN_RC=$?; fi
TRAIN_PID=""
if [ "$PRE_TIMEOUT" -eq 1 ] && [ "$TRAIN_RC" -eq 0 ]; then
    TRAIN_RC=124
fi
exit "$TRAIN_RC"
