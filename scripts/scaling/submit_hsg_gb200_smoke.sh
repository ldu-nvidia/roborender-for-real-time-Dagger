#!/bin/bash
# Submit one fail-closed HSG GB200 dual-container DAgger validation profile.
set -Eeuo pipefail

STAGE_ROOT=${STAGE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ldu/roborender_gb200_sm100}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
RUN_PROFILE=${RUN_PROFILE:-minimal}
DRY_RUN=${DRY_RUN:-0}
SLURM_BIN_DIR=${SLURM_BIN_DIR:-/cm/local/apps/slurm/current/bin}

RENDERER_CONTAINER=${RENDERER_CONTAINER:-$STAGE_ROOT/containers/roborender-image3f-arm64-sm100-cu130-trt10.13.3.9.sqsh}
TRAINER_CONTAINER=${TRAINER_CONTAINER:-$STAGE_ROOT/containers/roborender-isaac-gr00t-arm64-sm100-isaac5.1-r5.sqsh}
EXPECTED_RENDERER_CONTAINER_SHA256=${EXPECTED_RENDERER_CONTAINER_SHA256:-4c8261f246a650399c2cdf3d3567e1e14a643d3d65e239ff5a7f9f52442da963}
EXPECTED_TRAINER_CONTAINER_SHA256=${EXPECTED_TRAINER_CONTAINER_SHA256:-65ae3fec98e1bb9097bb852a28604ac43ba0bfeb1b169708e16ce1b28343b852}
IMAGE3F_ROOT=${IMAGE3F_ROOT:-$STAGE_ROOT/assets/roborender_image_model_accelerated}
ROBORENDER_MODEL_BASE=${ROBORENDER_MODEL_BASE:-$STAGE_ROOT/assets/models}
IMAGE3F_GENERATED_ROOT=${IMAGE3F_GENERATED_ROOT:-$STAGE_ROOT/runtime/image3f-generated}
ANCHORS=${ANCHORS:-$STAGE_ROOT/assets/anchor_pool_2048}
TEACHER_CKPT=${TEACHER_CKPT:-$STAGE_ROOT/assets/policies/expert_labeler/model_step_044500.pt}
EXPECTED_TEACHER_CKPT_SHA256=${EXPECTED_TEACHER_CKPT_SHA256:-943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38}
TORCH_HOME=${TORCH_HOME:-$STAGE_ROOT/assets/torch}
RESNET18_WEIGHTS=${RESNET18_WEIGHTS:-$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth}
EXPECTED_RESNET18_SHA256=${EXPECTED_RESNET18_SHA256:-f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec}
HSG_TRAINER_FFMPEG=/opt/viral-sm100/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v7.0.2
IMAGE_DAGGER_FFMPEG=${IMAGE_DAGGER_FFMPEG:-$HSG_TRAINER_FFMPEG}
EXPECTED_IMAGE3F_RELEASE_ID=${EXPECTED_IMAGE3F_RELEASE_ID:-roborender-image3f-rgb030-depth030-step5000-4090-source-1.0.0}
EXPECTED_IMAGE3F_ADAPTER_SHA256=${EXPECTED_IMAGE3F_ADAPTER_SHA256:-c68de259ce5ee8e4df5a45296be8328242b3ebba351990df1d86f4e3a7a2d588}
EXPECTED_FAST_ROBORENDER_REVISION=${EXPECTED_FAST_ROBORENDER_REVISION:-}
EXPECTED_GR00T_REVISION=${EXPECTED_GR00T_REVISION:-}
WANDB_EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_ONLINE_VALIDATION=${WANDB_ONLINE_VALIDATION:-$STAGE_ROOT/runtime/validation/wandb_online_smoke_acceptance.json}
WANDB_WATCHER_POLL_S=${WANDB_WATCHER_POLL_S:-15}
WANDB_WATCHER_READY_TIMEOUT_S=${WANDB_WATCHER_READY_TIMEOUT_S:-120}
WANDB_WATCHER_DRAIN_TIMEOUT_S=${WANDB_WATCHER_DRAIN_TIMEOUT_S:-240}

fatal() { echo "FATAL: $*" >&2; exit 2; }
case "$DRY_RUN" in 0|1) ;; *) fatal "DRY_RUN must be 0 or 1" ;; esac
for timing_name in WANDB_WATCHER_POLL_S WANDB_WATCHER_READY_TIMEOUT_S \
                   WANDB_WATCHER_DRAIN_TIMEOUT_S; do
    timing_value=${!timing_name}
    [[ "$timing_value" =~ ^[1-9][0-9]*$ ]] || \
        fatal "$timing_name must be a positive integer"
done
[ "$WANDB_WATCHER_DRAIN_TIMEOUT_S" -le 270 ] || \
    fatal "WANDB_WATCHER_DRAIN_TIMEOUT_S must be at most 270 seconds"
for slurm_command in scontrol srun; do
    [ -x "$SLURM_BIN_DIR/$slurm_command" ] || \
        fatal "missing HSG Slurm command: $SLURM_BIN_DIR/$slurm_command"
done
[ -s "$RESNET18_WEIGHTS" ] || fatal "pinned ResNet18 weights are missing: $RESNET18_WEIGHTS"
[ "$(sha256sum "$RESNET18_WEIGHTS" | awk '{print $1}')" = "$EXPECTED_RESNET18_SHA256" ] || \
    fatal "pinned ResNet18 weights hash mismatch"
[ "$IMAGE_DAGGER_FFMPEG" = "$HSG_TRAINER_FFMPEG" ] || \
    fatal "HSG trainer FFmpeg path must remain pinned to $HSG_TRAINER_FFMPEG"

case "$RUN_PROFILE" in
    minimal|minimal_e2e)
        RUN_PROFILE=minimal_e2e
        PACKED_NODES=1 GLOBAL_NUM_ENVS=1 ROLLOUT_STEPS=2
        TRAINER_RANKS=1 EXPECTED_RENDERERS=1 RUN_WALLTIME=01:00:00
        PACKED_QOS=interactive IMAGE_DAGGER_DUMP_SIM_RGB=0 USE_WANDB=0
        ;;
    resume|resume_minimal)
        RUN_PROFILE=resume_minimal
        PACKED_NODES=1 GLOBAL_NUM_ENVS=1 ROLLOUT_STEPS=2
        TRAINER_RANKS=1 EXPECTED_RENDERERS=1 RUN_WALLTIME=01:30:00
        PACKED_QOS=interactive IMAGE_DAGGER_DUMP_SIM_RGB=0 USE_WANDB=0
        ;;
    calibration|calibration_3view)
        RUN_PROFILE=calibration_3view
        PACKED_NODES=1 GLOBAL_NUM_ENVS=1 ROLLOUT_STEPS=2
        TRAINER_RANKS=1 EXPECTED_RENDERERS=1 RUN_WALLTIME=01:30:00
        PACKED_QOS=interactive IMAGE_DAGGER_DUMP_SIM_RGB=1 USE_WANDB=0
        ;;
    32|smoke32|smoke_32x300)
        RUN_PROFILE=smoke_32x300
        PACKED_NODES=2 GLOBAL_NUM_ENVS=32 ROLLOUT_STEPS=300
        TRAINER_RANKS=1 EXPECTED_RENDERERS=7 RUN_WALLTIME=02:30:00
        PACKED_QOS=interactive IMAGE_DAGGER_DUMP_SIM_RGB=0 USE_WANDB=1
        ;;
    256|smoke256|smoke_256x300)
        RUN_PROFILE=smoke_256x300
        PACKED_NODES=15 GLOBAL_NUM_ENVS=256 ROLLOUT_STEPS=300
        TRAINER_RANKS=8 EXPECTED_RENDERERS=52 RUN_WALLTIME=04:00:00
        PACKED_QOS=normal IMAGE_DAGGER_DUMP_SIM_RGB=0 USE_WANDB=1
        ;;
    *) fatal "RUN_PROFILE must be minimal, resume, calibration, 32, or 256" ;;
esac

[ $((PACKED_NODES * 4)) -ge $((EXPECTED_RENDERERS + TRAINER_RANKS)) ] || \
    fatal "profile exceeds four GB200s per HSG node"
[ $((GLOBAL_NUM_ENVS % TRAINER_RANKS)) -eq 0 ] || fatal "environments do not divide trainer ranks"

read_revision() {
    local marker=$1
    [ -s "$marker" ] || fatal "immutable source revision marker missing: $marker"
    local value
    value=$(tr -d '\r\n' < "$marker")
    [[ "$value" =~ ^[0-9a-f]{40}$ ]] || fatal "malformed source revision marker: $marker"
    printf '%s' "$value"
}
FAST_REVISION=$(read_revision "$ROOT/.source_revision")
GR00T_REVISION=$(read_revision "$GR00T_ROOT/.source_revision")
if [ -n "$EXPECTED_FAST_ROBORENDER_REVISION" ]; then
    [ "$FAST_REVISION" = "$EXPECTED_FAST_ROBORENDER_REVISION" ] || fatal "fast_roborender revision changed"
fi
if [ -n "$EXPECTED_GR00T_REVISION" ]; then
    [ "$GR00T_REVISION" = "$EXPECTED_GR00T_REVISION" ] || fatal "GR00T revision changed"
fi
EXPECTED_FAST_ROBORENDER_REVISION=$FAST_REVISION
EXPECTED_GR00T_REVISION=$GR00T_REVISION

CAMERA_CALIBRATION_ACCEPTANCE=${CAMERA_CALIBRATION_ACCEPTANCE:-$STAGE_ROOT/runtime/validation/camera_calibration_acceptance.json}
REQUESTED_PREREQUISITE_PROFILE_VALIDATION=${PREREQUISITE_PROFILE_VALIDATION:-}
EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256=NONE
PREREQUISITE_PROFILE_VALIDATION=NONE
EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256=NONE
if [ "$RUN_PROFILE" = smoke_32x300 ] || [ "$RUN_PROFILE" = smoke_256x300 ]; then
    [ -s "$CAMERA_CALIBRATION_ACCEPTANCE" ] || fatal "accepted HSG camera calibration is missing"
    EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256=$(sha256sum "$CAMERA_CALIBRATION_ACCEPTANCE" | awk '{print $1}')
    python3 "$ROOT/scripts/scaling/camera_calibration_gate.py" verify \
        --acceptance "$CAMERA_CALIBRATION_ACCEPTANCE" \
        --expected-fast-roborender-revision "$FAST_REVISION" \
        --expected-gr00t-revision "$GR00T_REVISION" \
        --expected-acceptance-sha256 "$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256" >/dev/null
fi
if [ "$RUN_PROFILE" = smoke_256x300 ]; then
    PREREQUISITE_PROFILE_VALIDATION=${REQUESTED_PREREQUISITE_PROFILE_VALIDATION:-$STAGE_ROOT/runtime/validation/smoke_32x300_profile_validation_manifest.txt}
    [ -s "$PREREQUISITE_PROFILE_VALIDATION" ] || fatal "validated HSG 32-env predecessor is missing"
    EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256=$(sha256sum "$PREREQUISITE_PROFILE_VALIDATION" | awk '{print $1}')
    python3 "$ROOT/scripts/scaling/profile_validation_gate.py" \
        --manifest "$PREREQUISITE_PROFILE_VALIDATION" \
        --expected-profile smoke_32x300 \
        --expected-fast-revision "$FAST_REVISION" \
        --expected-gr00t-revision "$GR00T_REVISION" \
        --expected-manifest-sha256 "$EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256" >/dev/null
fi

if [ "$USE_WANDB" -eq 1 ]; then
    [ "$WANDB_EXPECTED_EMAIL" = ldu@nvidia.com ] || fatal "HSG smoke telemetry is pinned to ldu@nvidia.com"
    NETRC=$STAGE_ROOT/runtime/home/.netrc
    [ -s "$NETRC" ] || fatal "W&B credential missing: $NETRC"
    [ "$(stat -c %u "$NETRC")" -eq "$(id -u)" ] || fatal "W&B netrc owner mismatch"
    case "$(stat -c %a "$NETRC")" in 400|600) ;; *) fatal "W&B netrc must be 0400 or 0600" ;; esac
    [ -s "$WANDB_ONLINE_VALIDATION" ] || \
        fatal "successful personal-workspace W&B scalar/image/video validation is missing: $WANDB_ONLINE_VALIDATION"
    WANDB_ENTITY=$(python3 - "$WANDB_ONLINE_VALIDATION" "$WANDB_EXPECTED_EMAIL" <<'PY'
import json
import re
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "schema": "fast_roborender.wandb_online_smoke.v1",
    "status": "PASS",
    "account_email": sys.argv[2],
    "personal": True,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"W&B acceptance {key} mismatch")
entity = payload.get("entity")
if not isinstance(entity, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", entity) is None:
    raise SystemExit("W&B acceptance entity is malformed")
required = {"validation/scalar", "validation/image", "validation/video"}
if not required.issubset(payload.get("observed_keys", [])):
    raise SystemExit("W&B acceptance did not read back scalar/image/video")
print(entity)
PY
    ) || fatal "W&B online validation manifest is invalid"
    EXPECTED_WANDB_ONLINE_VALIDATION_SHA256=$(sha256sum "$WANDB_ONLINE_VALIDATION" | awk '{print $1}')
else
    WANDB_ONLINE_VALIDATION=NONE
    EXPECTED_WANDB_ONLINE_VALIDATION_SHA256=NONE
fi

RUN_ID=${RUN_ID:-hsg_gb200_${RUN_PROFILE}_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-$STAGE_ROOT/runtime/$RUN_ID}
FLEET_DIR=$RUN_ROOT/fleet
TRAIN_RUN_DIR=$RUN_ROOT/train
mkdir -p "$FLEET_DIR/logs" "$TRAIN_RUN_DIR" "$STAGE_ROOT/runtime/validation"

python3 "$ROOT/scripts/scaling/hsg_gb200_smoke_gate.py" preflight \
    --profile "$RUN_PROFILE" --root "$ROOT" --gr00t-root "$GR00T_ROOT" \
    --fast-revision "$FAST_REVISION" --gr00t-revision "$GR00T_REVISION" \
    --renderer-container "$RENDERER_CONTAINER" --renderer-sha256 "$EXPECTED_RENDERER_CONTAINER_SHA256" \
    --trainer-container "$TRAINER_CONTAINER" --trainer-sha256 "$EXPECTED_TRAINER_CONTAINER_SHA256" \
    --image-root "$IMAGE3F_ROOT" --model-root "$ROBORENDER_MODEL_BASE" \
    --generated-root "$IMAGE3F_GENERATED_ROOT" --anchors "$ANCHORS" --anchor-min-files 2048 \
    --teacher "$TEACHER_CKPT" --teacher-sha256 "$EXPECTED_TEACHER_CKPT_SHA256" \
    --release-id "$EXPECTED_IMAGE3F_RELEASE_ID" --adapter-sha256 "$EXPECTED_IMAGE3F_ADAPTER_SHA256" \
    > "$RUN_ROOT/preflight.json"
EXPECTED_GR00T_SOURCE_TREE_SHA256=$(python3 - "$RUN_ROOT/preflight.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["gr00t_source_tree_sha256"])
PY
)
[[ "$EXPECTED_GR00T_SOURCE_TREE_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
    fatal "preflight returned a malformed GR00T source-tree hash"

EXPORTS="PATH=$SLURM_BIN_DIR:/usr/local/bin:/usr/bin:/bin,SLURM_BIN_DIR=$SLURM_BIN_DIR,HOME=$STAGE_ROOT/runtime/home,TORCH_HOME=$TORCH_HOME,RESNET18_WEIGHTS=$RESNET18_WEIGHTS,EXPECTED_RESNET18_SHA256=$EXPECTED_RESNET18_SHA256,IMAGE_DAGGER_FFMPEG=$IMAGE_DAGGER_FFMPEG,STAGE_ROOT=$STAGE_ROOT,ROOT=$ROOT,GR00T_ROOT=$GR00T_ROOT,RUN_PROFILE=$RUN_PROFILE,RUN_ID=$RUN_ID,RUN_ROOT=$RUN_ROOT,FLEET_DIR=$FLEET_DIR,TRAIN_RUN_DIR=$TRAIN_RUN_DIR,PACKED_NODES=$PACKED_NODES,PACKED_GPUS_PER_NODE=4,PACKED_QOS=$PACKED_QOS,GLOBAL_NUM_ENVS=$GLOBAL_NUM_ENVS,ROLLOUT_STEPS=$ROLLOUT_STEPS,TRAINER_RANKS=$TRAINER_RANKS,EXPECTED_RENDERERS=$EXPECTED_RENDERERS,RENDERER_CONTAINER=$RENDERER_CONTAINER,TRAINER_CONTAINER=$TRAINER_CONTAINER,IMAGE3F_ROOT=$IMAGE3F_ROOT,ROBORENDER_MODEL_BASE=$ROBORENDER_MODEL_BASE,IMAGE3F_GENERATED_ROOT=$IMAGE3F_GENERATED_ROOT,ANCHORS=$ANCHORS,TEACHER_CKPT=$TEACHER_CKPT,EXPECTED_TEACHER_CKPT_SHA256=$EXPECTED_TEACHER_CKPT_SHA256,EXPECTED_FAST_ROBORENDER_REVISION=$FAST_REVISION,EXPECTED_GR00T_REVISION=$GR00T_REVISION,EXPECTED_GR00T_SOURCE_TREE_SHA256=$EXPECTED_GR00T_SOURCE_TREE_SHA256,EXPECTED_IMAGE3F_RELEASE_ID=$EXPECTED_IMAGE3F_RELEASE_ID,EXPECTED_IMAGE3F_ADAPTER_SHA256=$EXPECTED_IMAGE3F_ADAPTER_SHA256,EXPECTED_IMAGE3F_INFERENCE_STEPS=5,EXPECTED_IMAGE3F_TEACACHE_THRESHOLD=0.3,EXPECTED_IMAGE3F_CFG_SCALE=1.0,EXPECTED_IMAGE3F_CUDA_GRAPH=disabled,IMAGE_DAGGER_DUMP_SIM_RGB=$IMAGE_DAGGER_DUMP_SIM_RGB,USE_WANDB=$USE_WANDB,WANDB_EXPECTED_EMAIL=$WANDB_EXPECTED_EMAIL,WANDB_ENTITY=$WANDB_ENTITY,WANDB_ONLINE_VALIDATION=$WANDB_ONLINE_VALIDATION,EXPECTED_WANDB_ONLINE_VALIDATION_SHA256=$EXPECTED_WANDB_ONLINE_VALIDATION_SHA256,WANDB_WATCHER_POLL_S=$WANDB_WATCHER_POLL_S,WANDB_WATCHER_READY_TIMEOUT_S=$WANDB_WATCHER_READY_TIMEOUT_S,WANDB_WATCHER_DRAIN_TIMEOUT_S=$WANDB_WATCHER_DRAIN_TIMEOUT_S,CAMERA_CALIBRATION_ACCEPTANCE=$CAMERA_CALIBRATION_ACCEPTANCE,EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256=$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256,PREREQUISITE_PROFILE_VALIDATION=$PREREQUISITE_PROFILE_VALIDATION,EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256=$EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256"
CMD=(sbatch --parsable --account=llmservice_nemotron_ultra --partition=batch --qos="$PACKED_QOS"
    --nodes="$PACKED_NODES" --gpus-per-node=4 --cpus-per-gpu=32 --mem=800G
    --time="$RUN_WALLTIME" --signal=B:USR1@300
    --job-name="rr_${RUN_PROFILE}_gb200"
    --output="$FLEET_DIR/logs/packed_%j.out" --error="$FLEET_DIR/logs/packed_%j.err"
    --export="$EXPORTS" "$ROOT/scripts/scaling/hsg_gb200_packed_smoke.sbatch")

cat > "$RUN_ROOT/submission_manifest.txt" <<EOF
status=planned
run_profile=$RUN_PROFILE
nodes=$PACKED_NODES
gpus_per_node=4
qos=$PACKED_QOS
renderer_tasks=$EXPECTED_RENDERERS
trainer_tasks=$TRAINER_RANKS
global_envs=$GLOBAL_NUM_ENVS
rollout_steps=$ROLLOUT_STEPS
renderer_container=$RENDERER_CONTAINER
renderer_container_sha256=$EXPECTED_RENDERER_CONTAINER_SHA256
trainer_container=$TRAINER_CONTAINER
trainer_container_sha256=$EXPECTED_TRAINER_CONTAINER_SHA256
resnet18_weights=$RESNET18_WEIGHTS
resnet18_weights_sha256=$EXPECTED_RESNET18_SHA256
image_dagger_ffmpeg=$IMAGE_DAGGER_FFMPEG
fast_roborender_revision=$FAST_REVISION
gr00t_revision=$GR00T_REVISION
gr00t_source_tree_sha256=$EXPECTED_GR00T_SOURCE_TREE_SHA256
wandb_online_validation=$WANDB_ONLINE_VALIDATION
wandb_online_validation_sha256=$EXPECTED_WANDB_ONLINE_VALIDATION_SHA256
wandb_watcher_poll_s=$WANDB_WATCHER_POLL_S
wandb_watcher_ready_timeout_s=$WANDB_WATCHER_READY_TIMEOUT_S
wandb_watcher_drain_timeout_s=$WANDB_WATCHER_DRAIN_TIMEOUT_S
slurm_bin_dir=$SLURM_BIN_DIR
camera_calibration_acceptance_sha256=$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256
prerequisite_profile_validation_sha256=$EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256
EOF

if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:'; printf ' %q' "${CMD[@]}"; printf '\n'
    exit 0
fi
JOB_ID=$("${CMD[@]}")
JOB_ID=${JOB_ID%%;*}
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || fatal "sbatch returned malformed job id: $JOB_ID"
printf 'status=submitted\njob_id=%s\n' "$JOB_ID" >> "$RUN_ROOT/submission_manifest.txt"
echo "HSG GB200 $RUN_PROFILE smoke job: $JOB_ID"
