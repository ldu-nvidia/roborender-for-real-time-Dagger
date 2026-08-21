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
REQUESTED_RUN_ID=${RUN_ID:-}
REQUESTED_RUN_ROOT=${RUN_ROOT:-}
RESUME_TRAINING=${RESUME_TRAINING:-0}
CKPT=${CKPT:-none}
EXPECTED_TRAIN_CHECKPOINT_SHA256=${EXPECTED_TRAIN_CHECKPOINT_SHA256:-NONE}
RESUME_CANDIDATE=${RESUME_CANDIDATE:-NONE}
EXPECTED_RESUME_CANDIDATE_SHA256=NONE
NUM_ROLLOUT_ITERATIONS=${NUM_ROLLOUT_ITERATIONS:-}
SAVE_FREQUENCY=${SAVE_FREQUENCY:-}
PRODUCTION_ACTION=${PRODUCTION_ACTION:-train}

fatal() { echo "FATAL: $*" >&2; exit 2; }
case "$DRY_RUN" in 0|1) ;; *) fatal "DRY_RUN must be 0 or 1" ;; esac
case "$RESUME_TRAINING" in 0|1) ;; *) fatal "RESUME_TRAINING must be 0 or 1" ;; esac
case "$PRODUCTION_ACTION" in train|validate) ;; *) fatal "PRODUCTION_ACTION must be train or validate" ;; esac
for timing_name in WANDB_WATCHER_POLL_S WANDB_WATCHER_READY_TIMEOUT_S \
                   WANDB_WATCHER_DRAIN_TIMEOUT_S; do
    timing_value=${!timing_name}
    [[ "$timing_value" =~ ^[1-9][0-9]*$ ]] || \
        fatal "$timing_name must be a positive integer"
done
[ "$WANDB_WATCHER_DRAIN_TIMEOUT_S" -le 270 ] || \
    fatal "WANDB_WATCHER_DRAIN_TIMEOUT_S must be at most 270 seconds"
for slurm_command in sbatch scontrol squeue sacct srun; do
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
        PACKED_PARTITION=batch PACKED_QOS=interactive IMAGE_DAGGER_DUMP_SIM_RGB=0 USE_WANDB=0
        NUM_ROLLOUT_ITERATIONS=1 SAVE_FREQUENCY=1
        ;;
    resume|resume_minimal)
        RUN_PROFILE=resume_minimal
        PACKED_NODES=1 GLOBAL_NUM_ENVS=1 ROLLOUT_STEPS=2
        TRAINER_RANKS=1 EXPECTED_RENDERERS=1 RUN_WALLTIME=01:30:00
        PACKED_PARTITION=batch PACKED_QOS=interactive IMAGE_DAGGER_DUMP_SIM_RGB=0 USE_WANDB=0
        NUM_ROLLOUT_ITERATIONS=2 SAVE_FREQUENCY=1
        ;;
    calibration|calibration_3view)
        RUN_PROFILE=calibration_3view
        PACKED_NODES=1 GLOBAL_NUM_ENVS=1 ROLLOUT_STEPS=2
        TRAINER_RANKS=1 EXPECTED_RENDERERS=1 RUN_WALLTIME=01:30:00
        PACKED_PARTITION=batch PACKED_QOS=interactive IMAGE_DAGGER_DUMP_SIM_RGB=1 USE_WANDB=0
        NUM_ROLLOUT_ITERATIONS=1 SAVE_FREQUENCY=1
        ;;
    32|smoke32|smoke_32x300)
        RUN_PROFILE=smoke_32x300
        PACKED_NODES=2 GLOBAL_NUM_ENVS=32 ROLLOUT_STEPS=300
        TRAINER_RANKS=1 EXPECTED_RENDERERS=7 RUN_WALLTIME=02:30:00
        PACKED_PARTITION=batch PACKED_QOS=interactive IMAGE_DAGGER_DUMP_SIM_RGB=0 USE_WANDB=1
        NUM_ROLLOUT_ITERATIONS=1 SAVE_FREQUENCY=1
        ;;
    256|smoke256|smoke_256x300)
        RUN_PROFILE=smoke_256x300
        PACKED_NODES=15 GLOBAL_NUM_ENVS=256 ROLLOUT_STEPS=300
        TRAINER_RANKS=8 EXPECTED_RENDERERS=52 RUN_WALLTIME=04:00:00
        PACKED_PARTITION=batch PACKED_QOS=normal IMAGE_DAGGER_DUMP_SIM_RGB=0 USE_WANDB=1
        NUM_ROLLOUT_ITERATIONS=1 SAVE_FREQUENCY=1
        ;;
    production|production_256x300)
        RUN_PROFILE=production_256x300
        PACKED_NODES=15 GLOBAL_NUM_ENVS=256 ROLLOUT_STEPS=300
        TRAINER_RANKS=8 EXPECTED_RENDERERS=52 RUN_WALLTIME=08:00:00
        PACKED_PARTITION=batch_long PACKED_QOS=normal IMAGE_DAGGER_DUMP_SIM_RGB=0 USE_WANDB=1
        NUM_ROLLOUT_ITERATIONS=${NUM_ROLLOUT_ITERATIONS:-500}
        SAVE_FREQUENCY=${SAVE_FREQUENCY:-5}
        [ "$NUM_ROLLOUT_ITERATIONS:$SAVE_FREQUENCY" = 500:5 ] || \
            fatal "production_256x300 absolute target/save interval are immutable (500:5)"
        ;;
    *) fatal "RUN_PROFILE must be minimal, resume, calibration, 32, 256, or production" ;;
esac

validate_sealed_production_source() {
    local expected_prefix canonical_root canonical_gr00t writable forbidden
    expected_prefix=$(realpath -m "$STAGE_ROOT/src_snapshots")/
    [ -d "$ROOT" ] && [ ! -L "$ROOT" ] || \
        fatal "production ROOT must be a real sealed source directory"
    canonical_root=$(realpath -e "$ROOT") || fatal "cannot resolve production ROOT"
    case "$canonical_root" in
        "$expected_prefix"fast_roborender_*_sealed) ;;
        *) fatal "production ROOT must be under ${expected_prefix}fast_roborender_*_sealed" ;;
    esac
    [ "$ROOT" = "$canonical_root" ] || \
        fatal "production ROOT must be supplied as its canonical sealed path"
    canonical_gr00t=$(realpath -e "$GR00T_ROOT") || fatal "cannot resolve GR00T_ROOT"
    [ "$canonical_gr00t" = "$canonical_root/GR00T-VisualSim2Real" ] || \
        fatal "production GR00T_ROOT must be exactly ROOT/GR00T-VisualSim2Real"
    [ "$GR00T_ROOT" = "$canonical_gr00t" ] || \
        fatal "production GR00T_ROOT must be supplied as its canonical path"
    [ ! -w "$canonical_root" ] && [ ! -w "$canonical_gr00t" ] || \
        fatal "production source snapshots must be read-only to the submitter"
    writable=$(find "$canonical_root" -xdev -perm /222 -print -quit) || \
        fatal "cannot audit sealed production source permissions"
    [ -z "$writable" ] || \
        fatal "sealed production source contains a writable inode: $writable"
    forbidden=$(find "$canonical_root" -xdev \
        \( -type d -name __pycache__ -o -type f \
           \( -name '*.pyc' -o -name '*.pyo' \) \) -print -quit) || \
        fatal "cannot audit sealed production source for executable bytecode"
    [ -z "$forbidden" ] || \
        fatal "sealed production source contains executable bytecode: $forbidden"
}
if [ "$RUN_PROFILE" = production_256x300 ]; then
    validate_sealed_production_source
fi

if [ "$RUN_PROFILE" != production_256x300 ]; then
    [ "$PRODUCTION_ACTION" = train ] || \
        fatal "PRODUCTION_ACTION=validate is restricted to production_256x300"
    [ "$RESUME_TRAINING:$CKPT:$EXPECTED_TRAIN_CHECKPOINT_SHA256:$RESUME_CANDIDATE" = \
      "0:none:NONE:NONE" ] || \
        fatal "checkpoint resume inputs are restricted to production_256x300"
fi

[ $((PACKED_NODES * 4)) -ge $((EXPECTED_RENDERERS + TRAINER_RANKS)) ] || \
    fatal "profile exceeds four GB200s per HSG node"
[ $((GLOBAL_NUM_ENVS % TRAINER_RANKS)) -eq 0 ] || fatal "environments do not divide trainer ranks"
[[ "$NUM_ROLLOUT_ITERATIONS" =~ ^[1-9][0-9]*$ ]] || \
    fatal "NUM_ROLLOUT_ITERATIONS must be a positive integer"
[[ "$SAVE_FREQUENCY" =~ ^[1-9][0-9]*$ ]] || \
    fatal "SAVE_FREQUENCY must be a positive integer"
[ $((NUM_ROLLOUT_ITERATIONS % SAVE_FREQUENCY)) -eq 0 ] || \
    fatal "absolute training target must be divisible by SAVE_FREQUENCY"

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
if [ "$RUN_PROFILE" = smoke_32x300 ] || [ "$RUN_PROFILE" = smoke_256x300 ] || \
   [ "$RUN_PROFILE" = production_256x300 ]; then
    [ -s "$CAMERA_CALIBRATION_ACCEPTANCE" ] || fatal "accepted HSG camera calibration is missing"
    CAMERA_CALIBRATION_ACCEPTANCE=$(realpath -e "$CAMERA_CALIBRATION_ACCEPTANCE")
    [ -f "$CAMERA_CALIBRATION_ACCEPTANCE" ] && [ ! -L "$CAMERA_CALIBRATION_ACCEPTANCE" ] || \
        fatal "accepted HSG camera calibration is unsafe"
    EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256=$(sha256sum "$CAMERA_CALIBRATION_ACCEPTANCE" | awk '{print $1}')
    python3 "$ROOT/scripts/scaling/camera_calibration_gate.py" verify \
        --acceptance "$CAMERA_CALIBRATION_ACCEPTANCE" \
        --expected-fast-roborender-revision "$FAST_REVISION" \
        --expected-gr00t-revision "$GR00T_REVISION" \
        --expected-acceptance-sha256 "$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256" >/dev/null
fi
PREREQUISITE_PROFILE_EXPECTED=NONE
if [ "$RUN_PROFILE" = smoke_256x300 ]; then
    PREREQUISITE_PROFILE_EXPECTED=smoke_32x300
elif [ "$RUN_PROFILE" = production_256x300 ]; then
    PREREQUISITE_PROFILE_EXPECTED=smoke_256x300
fi
if [ "$PREREQUISITE_PROFILE_EXPECTED" != NONE ]; then
    PREREQUISITE_PROFILE_VALIDATION=${REQUESTED_PREREQUISITE_PROFILE_VALIDATION:-$STAGE_ROOT/runtime/validation/${PREREQUISITE_PROFILE_EXPECTED}_profile_validation_manifest.txt}
    [ -s "$PREREQUISITE_PROFILE_VALIDATION" ] || \
        fatal "validated HSG $PREREQUISITE_PROFILE_EXPECTED predecessor is missing"
    PREREQUISITE_PROFILE_VALIDATION=$(realpath -e "$PREREQUISITE_PROFILE_VALIDATION")
    [ -f "$PREREQUISITE_PROFILE_VALIDATION" ] && \
    [ ! -L "$PREREQUISITE_PROFILE_VALIDATION" ] || \
        fatal "validated HSG predecessor resolved to an unsafe artifact"
    EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256=$(sha256sum "$PREREQUISITE_PROFILE_VALIDATION" | awk '{print $1}')
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
    # Keep the sanitized sbatch export total even when telemetry is disabled.
    # The batch entrypoint requires every exported contract field to be
    # nonempty before it branches on USE_WANDB; `null` is also Hydra's literal
    # disabled entity value.
    WANDB_ENTITY=null
    WANDB_ONLINE_VALIDATION=NONE
    EXPECTED_WANDB_ONLINE_VALIDATION_SHA256=NONE
fi

if [ "$DRY_RUN" -eq 1 ] && [ "$RUN_PROFILE" = production_256x300 ] && \
   [ "$PRODUCTION_ACTION" = train ] && [ "$RESUME_TRAINING" -eq 0 ]; then
    # A production dry run validates the entire submission path but never burns
    # the caller's requested logical RUN_ID.  Its clearly disposable root can
    # be audited or garbage-collected independently.
    RUN_ID=hsg_gb200_production_disposable_dryrun_$(date -u +%Y%m%dT%H%M%SZ)_$$
    RUN_ROOT=
    REQUESTED_RUN_ID=
    REQUESTED_RUN_ROOT=
fi
RUN_ID=${RUN_ID:-hsg_gb200_${RUN_PROFILE}_$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]{1,128}$ ]] || fatal "RUN_ID is malformed"
RUN_ROOT=${RUN_ROOT:-$STAGE_ROOT/runtime/$RUN_ID}
if [ "$RUN_PROFILE" = production_256x300 ]; then
    EXPECTED_RUN_ROOT=$STAGE_ROOT/runtime/$RUN_ID
    [ "$RUN_ROOT" = "$EXPECTED_RUN_ROOT" ] || \
        fatal "production RUN_ROOT must be exactly $EXPECTED_RUN_ROOT"
fi
TRAIN_RUN_DIR=$RUN_ROOT/train
RESUME_CHECKPOINT_STEP=0
CREATE_PRODUCTION_RUN_ROOT=0
if [ "$RUN_PROFILE" = production_256x300 ]; then
    if [ "$PRODUCTION_ACTION" = validate ]; then
        [ "$RESUME_TRAINING" -eq 0 ] || \
            fatal "validation-only recovery requires RESUME_TRAINING=0"
        [ -n "$REQUESTED_RUN_ID" ] || \
            fatal "validation-only recovery requires the original RUN_ID"
        [ -n "$REQUESTED_RUN_ROOT" ] || \
            fatal "validation-only recovery requires the original RUN_ROOT"
        [ -d "$RUN_ROOT" ] && [ ! -L "$RUN_ROOT" ] || \
            fatal "validation-only recovery RUN_ROOT is missing or unsafe: $RUN_ROOT"
        [ "$(realpath -e "$RUN_ROOT")" = "$(realpath -e "$STAGE_ROOT/runtime")/$RUN_ID" ] || \
            fatal "validation-only recovery RUN_ROOT canonical path changed"
        [ -d "$TRAIN_RUN_DIR" ] && [ ! -L "$TRAIN_RUN_DIR" ] || \
            fatal "validation-only recovery training directory is missing or unsafe"
        [ ! -e "$RUN_ROOT/RUN_DONE" ] || \
            fatal "completed production run does not need validation-only recovery"
        [ -f "$CKPT" ] && [ ! -L "$CKPT" ] || \
            fatal "validation-only recovery requires the exact step-500 checkpoint"
        CKPT=$(realpath -e "$CKPT")
        [ "$CKPT" = "$(realpath -e "$TRAIN_RUN_DIR")/model_step_000500.pt" ] || \
            fatal "validation-only recovery checkpoint must be model_step_000500.pt"
        RESUME_CHECKPOINT_STEP=500
        [[ "$EXPECTED_TRAIN_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
            fatal "validation-only recovery requires the exact checkpoint SHA-256"
        [ "$(sha256sum "$CKPT" | awk '{print $1}')" = "$EXPECTED_TRAIN_CHECKPOINT_SHA256" ] || \
            fatal "validation-only recovery checkpoint hash mismatch"
        for artifact in "$TRAIN_RUN_DIR/initial_policy.pt" "$TRAIN_RUN_DIR/meta.yaml" \
                        "$TRAIN_RUN_DIR/telemetry/WANDB_SIDECAR_RUN_ID"; do
            [ -s "$artifact" ] && [ ! -L "$artifact" ] || \
                fatal "validation-only recovery prerequisite is missing or unsafe: $artifact"
        done
    elif [ "$RESUME_TRAINING" -eq 0 ]; then
        [ "$CKPT:$EXPECTED_TRAIN_CHECKPOINT_SHA256:$RESUME_CANDIDATE" = none:NONE:NONE ] || \
            fatal "fresh production must use CKPT=none, checkpoint hash NONE, and no resume candidate"
        [ ! -e "$RUN_ROOT" ] || \
            fatal "fresh production refuses to reuse existing RUN_ROOT=$RUN_ROOT"
        CREATE_PRODUCTION_RUN_ROOT=1
    else
        [ -n "$REQUESTED_RUN_ID" ] || fatal "production resume requires the original RUN_ID"
        [ -n "$REQUESTED_RUN_ROOT" ] || fatal "production resume requires the original RUN_ROOT"
        [ -d "$RUN_ROOT" ] && [ ! -L "$RUN_ROOT" ] || \
            fatal "production resume RUN_ROOT is missing or unsafe: $RUN_ROOT"
        [ "$(realpath -e "$RUN_ROOT")" = "$(realpath -e "$STAGE_ROOT/runtime")/$RUN_ID" ] || \
            fatal "production resume RUN_ROOT canonical path changed"
        [ -d "$TRAIN_RUN_DIR" ] && [ ! -L "$TRAIN_RUN_DIR" ] || \
            fatal "production resume training directory is missing or unsafe"
        [ ! -e "$RUN_ROOT/RUN_DONE" ] || fatal "completed production run cannot be resumed"
        case "$CKPT" in none|null) fatal "production resume requires an explicit checkpoint" ;; esac
        [ -f "$CKPT" ] && [ ! -L "$CKPT" ] || \
            fatal "production resume checkpoint is missing or unsafe: $CKPT"
        CKPT=$(realpath -e "$CKPT")
        [ "$(dirname "$CKPT")" = "$(realpath -e "$TRAIN_RUN_DIR")" ] || \
            fatal "production resume checkpoint must be inside the original training directory"
        CHECKPOINT_BASENAME=$(basename "$CKPT")
        [[ "$CHECKPOINT_BASENAME" =~ ^model_step_([0-9]{6})\.pt$ ]] || \
            fatal "production resume requires a periodic model_step_NNNNNN.pt checkpoint"
        RESUME_CHECKPOINT_STEP=$((10#${BASH_REMATCH[1]}))
        [ "$RESUME_CHECKPOINT_STEP" -gt 0 ] && \
        [ "$RESUME_CHECKPOINT_STEP" -lt "$NUM_ROLLOUT_ITERATIONS" ] && \
        [ $((RESUME_CHECKPOINT_STEP % SAVE_FREQUENCY)) -eq 0 ] || \
            fatal "production resume checkpoint step violates the absolute target/save interval"
        [[ "$EXPECTED_TRAIN_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
            fatal "production resume requires the exact checkpoint SHA-256"
        [ "$(sha256sum "$CKPT" | awk '{print $1}')" = "$EXPECTED_TRAIN_CHECKPOINT_SHA256" ] || \
            fatal "production resume checkpoint hash mismatch"
        [ -s "$TRAIN_RUN_DIR/initial_policy.pt" ] || \
            fatal "production resume is missing the fresh-student baseline"
        [ -s "$TRAIN_RUN_DIR/meta.yaml" ] || \
            fatal "production resume is missing trainer W&B metadata"
        [ -s "$TRAIN_RUN_DIR/telemetry/WANDB_SIDECAR_RUN_ID" ] || \
            fatal "production resume is missing the media W&B run ID"
    fi
    if [ "$RESUME_TRAINING" -eq 1 ] || [ "$PRODUCTION_ACTION" = validate ]; then
        [ "$RESUME_CANDIDATE" != NONE ] && [ -f "$RESUME_CANDIDATE" ] && \
        [ ! -L "$RESUME_CANDIDATE" ] || \
            fatal "production recovery requires an explicit safe RESUME_CANDIDATE"
        RESUME_CANDIDATE=$(realpath -e "$RESUME_CANDIDATE")
        case "$RESUME_CANDIDATE" in
            "$RUN_ROOT"/attempts/*/resume_candidate.txt) ;;
            *) fatal "production resume candidate escaped the original attempt tree" ;;
        esac
        CANDIDATE_ATTEMPT_DIR=$(dirname "$RESUME_CANDIDATE")
        [ -s "$CANDIDATE_ATTEMPT_DIR/ATTEMPT_FAILED" ] && \
        [ ! -L "$CANDIDATE_ATTEMPT_DIR/ATTEMPT_FAILED" ] || \
            fatal "production resume candidate is not from a failed terminal attempt"
        LATEST_CHECKPOINT=$(find "$TRAIN_RUN_DIR" -maxdepth 1 -type f \
            -name 'model_step_[0-9][0-9][0-9][0-9][0-9][0-9].pt' \
            -printf '%f\n' | LC_ALL=C sort | tail -n 1)
        [ -n "$LATEST_CHECKPOINT" ] || fatal "production recovery found no durable checkpoint"
        [ "$CKPT" = "$TRAIN_RUN_DIR/$LATEST_CHECKPOINT" ] || \
            fatal "production recovery must use the latest durable periodic checkpoint"
    fi
    ATTEMPT_ID=${ATTEMPT_ID:-attempt_$(date -u +%Y%m%dT%H%M%SZ)_$$}
    [[ "$ATTEMPT_ID" =~ ^[A-Za-z0-9_-]{1,128}$ ]] || fatal "ATTEMPT_ID is malformed"
    if [ -e "$RUN_ROOT/attempts" ]; then
        [ -d "$RUN_ROOT/attempts" ] && [ ! -L "$RUN_ROOT/attempts" ] || \
            fatal "production attempts directory is unsafe"
    fi
    RUN_ATTEMPT_DIR=$RUN_ROOT/attempts/$ATTEMPT_ID
    [ ! -e "$RUN_ATTEMPT_DIR" ] || fatal "production ATTEMPT_ID already exists"
    FLEET_DIR=$RUN_ATTEMPT_DIR/fleet
    SUBMISSION_MANIFEST=$RUN_ATTEMPT_DIR/submission_manifest.txt
else
    ATTEMPT_ID=NONE
    RUN_ATTEMPT_DIR=$RUN_ROOT
    FLEET_DIR=$RUN_ROOT/fleet
    SUBMISSION_MANIFEST=$RUN_ROOT/submission_manifest.txt
fi
mkdir -p "$STAGE_ROOT/runtime" "$STAGE_ROOT/runtime/validation"
if [ "$CREATE_PRODUCTION_RUN_ROOT" -eq 1 ]; then
    mkdir "$RUN_ROOT" || fatal "could not exclusively create fresh production RUN_ROOT"
fi
mkdir -p "$FLEET_DIR/logs" "$TRAIN_RUN_DIR" "$STAGE_ROOT/runtime/validation"
EXPECTED_RUN_IDENTITY_SHA256=NONE
if [ "$RUN_PROFILE" = production_256x300 ]; then
    RUN_IDENTITY=$RUN_ROOT/run_identity.txt
    if [ "$RESUME_TRAINING" -eq 1 ] || [ "$PRODUCTION_ACTION" = validate ]; then
        [ -f "$RUN_IDENTITY" ] && [ ! -L "$RUN_IDENTITY" ] || \
            fatal "production resume run identity is missing or unsafe"
    fi
fi

python3 "$ROOT/scripts/scaling/hsg_gb200_smoke_gate.py" preflight \
    --profile "$RUN_PROFILE" --root "$ROOT" --gr00t-root "$GR00T_ROOT" \
    --fast-revision "$FAST_REVISION" --gr00t-revision "$GR00T_REVISION" \
    --renderer-container "$RENDERER_CONTAINER" --renderer-sha256 "$EXPECTED_RENDERER_CONTAINER_SHA256" \
    --trainer-container "$TRAINER_CONTAINER" --trainer-sha256 "$EXPECTED_TRAINER_CONTAINER_SHA256" \
    --image-root "$IMAGE3F_ROOT" --model-root "$ROBORENDER_MODEL_BASE" \
    --generated-root "$IMAGE3F_GENERATED_ROOT" --anchors "$ANCHORS" --anchor-min-files 2048 \
    --teacher "$TEACHER_CKPT" --teacher-sha256 "$EXPECTED_TEACHER_CKPT_SHA256" \
    --release-id "$EXPECTED_IMAGE3F_RELEASE_ID" --adapter-sha256 "$EXPECTED_IMAGE3F_ADAPTER_SHA256" \
    > "$RUN_ATTEMPT_DIR/preflight.json"
read -r EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256 EXPECTED_GR00T_SOURCE_TREE_SHA256 \
    EXPECTED_ASSET_CONTRACT_SHA256 EXPECTED_IMAGE3F_ENGINE_SHA256 \
    EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256 \
    EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256 < <(
python3 - "$RUN_ATTEMPT_DIR/preflight.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(
    payload["fast_roborender_source_tree_sha256"],
    payload["gr00t_source_tree_sha256"],
    payload["asset_contract_sha256"],
    payload["engine_sha256"],
    payload["source_snapshot_attestation_sha256"],
    payload["source_snapshot_inventory_sha256"],
)
PY
)
[[ "$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
    fatal "preflight returned a malformed fast_roborender source-tree hash"
[[ "$EXPECTED_GR00T_SOURCE_TREE_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
    fatal "preflight returned a malformed GR00T source-tree hash"
[[ "$EXPECTED_ASSET_CONTRACT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
    fatal "preflight returned a malformed immutable asset-contract hash"
[[ "$EXPECTED_IMAGE3F_ENGINE_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
    fatal "preflight returned a malformed TensorRT engine hash"
case "$RUN_PROFILE" in
    calibration_3view|smoke_32x300|smoke_256x300|production_256x300)
        [[ "$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
            fatal "preflight returned a malformed source snapshot attestation hash"
        [[ "$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
            fatal "preflight returned a malformed source snapshot inventory hash"
        ;;
    *)
        [ "$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256:$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256" = NONE:NONE ] || \
            fatal "non-sealed profile unexpectedly received source snapshot evidence"
        ;;
esac
if [ "$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256" != NONE ]; then
    python3 "$ROOT/scripts/scaling/camera_calibration_gate.py" verify \
        --acceptance "$CAMERA_CALIBRATION_ACCEPTANCE" \
        --expected-fast-roborender-revision "$FAST_REVISION" \
        --expected-gr00t-revision "$GR00T_REVISION" \
        --expected-fast-source-tree-sha256 "$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256" \
        --expected-gr00t-source-tree-sha256 "$EXPECTED_GR00T_SOURCE_TREE_SHA256" \
        --expected-source-snapshot-attestation-sha256 \
          "$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256" \
        --expected-source-snapshot-inventory-sha256 \
          "$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256" \
        --expected-asset-contract-sha256 "$EXPECTED_ASSET_CONTRACT_SHA256" \
        --expected-acceptance-sha256 "$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256" \
        >/dev/null || fatal "camera calibration acceptance is not bound to current source/assets"
fi
if [ "$PREREQUISITE_PROFILE_EXPECTED" != NONE ]; then
    python3 "$ROOT/scripts/scaling/profile_validation_gate.py" \
        --manifest "$PREREQUISITE_PROFILE_VALIDATION" \
        --expected-profile "$PREREQUISITE_PROFILE_EXPECTED" \
        --expected-fast-revision "$FAST_REVISION" \
        --expected-gr00t-revision "$GR00T_REVISION" \
        --expected-fast-source-tree-sha256 "$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256" \
        --expected-gr00t-source-tree-sha256 "$EXPECTED_GR00T_SOURCE_TREE_SHA256" \
        --expected-asset-contract-sha256 "$EXPECTED_ASSET_CONTRACT_SHA256" \
        --expected-wandb-email "$WANDB_EXPECTED_EMAIL" \
        --expected-wandb-entity "$WANDB_ENTITY" \
        --expected-manifest-sha256 "$EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256" \
        >/dev/null
fi

# Bind every immutable input that could change the student trajectory to the
# logical production run.  A later allocation may resume the same RUN_ID, but
# it must use byte-identical source trees, containers, model assets, scheduler
# topology, validation evidence, and personal W&B destination.
if [ "$RUN_PROFILE" = production_256x300 ]; then
    RUN_IDENTITY=$RUN_ROOT/run_identity.txt
    RUN_IDENTITY_MODE=create
    if [ "$RESUME_TRAINING" -eq 1 ] || [ "$PRODUCTION_ACTION" = validate ]; then
        RUN_IDENTITY_MODE=verify
    fi
    python3 - "$RUN_IDENTITY" "$RUN_IDENTITY_MODE" \
        "$RUN_ID" "$NUM_ROLLOUT_ITERATIONS" "$SAVE_FREQUENCY" \
        "$PACKED_NODES" "$GLOBAL_NUM_ENVS" "$ROLLOUT_STEPS" \
        "$TRAINER_RANKS" "$EXPECTED_RENDERERS" "$PACKED_PARTITION" "$PACKED_QOS" \
        "$WANDB_EXPECTED_EMAIL" "$WANDB_ENTITY" \
        "$EXPECTED_WANDB_ONLINE_VALIDATION_SHA256" \
        "$FAST_REVISION" "$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256" \
        "$GR00T_REVISION" "$EXPECTED_GR00T_SOURCE_TREE_SHA256" \
        "$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256" \
        "$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256" \
        "$EXPECTED_RENDERER_CONTAINER_SHA256" "$EXPECTED_TRAINER_CONTAINER_SHA256" \
        "$EXPECTED_RESNET18_SHA256" "$EXPECTED_TEACHER_CKPT_SHA256" \
        "$EXPECTED_IMAGE3F_RELEASE_ID" "$EXPECTED_IMAGE3F_ADAPTER_SHA256" \
        "$EXPECTED_ASSET_CONTRACT_SHA256" \
        "$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256" \
        "$PREREQUISITE_PROFILE_EXPECTED" \
        "$EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
mode = sys.argv[2]
keys = (
    "run_id",
    "absolute_target",
    "save_frequency",
    "nodes",
    "global_envs",
    "rollout_steps",
    "trainer_ranks",
    "renderer_tasks",
    "partition",
    "qos",
    "wandb_account_email",
    "wandb_entity",
    "wandb_online_validation_sha256",
    "fast_roborender_revision",
    "fast_roborender_source_tree_sha256",
    "gr00t_revision",
    "gr00t_source_tree_sha256",
    "source_snapshot_attestation_sha256",
    "source_snapshot_inventory_sha256",
    "renderer_container_sha256",
    "trainer_container_sha256",
    "resnet18_sha256",
    "teacher_checkpoint_sha256",
    "image3f_release_id",
    "image3f_adapter_sha256",
    "immutable_asset_contract_sha256",
    "camera_calibration_acceptance_sha256",
    "prerequisite_profile",
    "prerequisite_profile_validation_sha256",
)
values = sys.argv[3:]
if len(values) != len(keys):
    raise SystemExit("FATAL: internal production run identity argument mismatch")
expected = "\n".join(
    ["schema=fast_roborender.hsg_production_run_identity.v1"]
    + [f"{key}={value}" for key, value in zip(keys, values, strict=True)]
) + "\n"
if mode == "create":
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(expected)
    except FileExistsError as exc:
        raise SystemExit("FATAL: refusing to overwrite production run identity") from exc
elif mode == "verify":
    if path.read_text(encoding="utf-8") != expected:
        raise SystemExit(
            "FATAL: production resume immutable run identity does not match the original run"
        )
else:
    raise SystemExit("FATAL: invalid production run identity mode")
PY
    EXPECTED_RUN_IDENTITY_SHA256=$(sha256sum "$RUN_IDENTITY" | awk '{print $1}')
    if [ "$RESUME_TRAINING" -eq 1 ] || [ "$PRODUCTION_ACTION" = validate ]; then
        CANDIDATE_SOURCE_ATTEMPT=$(basename "$(dirname "$RESUME_CANDIDATE")")
        python3 - "$RESUME_CANDIDATE" "$RUN_ID" "$RUN_ROOT" \
            "$CANDIDATE_SOURCE_ATTEMPT" "$CKPT" "$RESUME_CHECKPOINT_STEP" \
            "$EXPECTED_TRAIN_CHECKPOINT_SHA256" "$NUM_ROLLOUT_ITERATIONS" \
            "$SAVE_FREQUENCY" "$EXPECTED_RUN_IDENTITY_SHA256" <<'PY'
import sys
from pathlib import Path

expected = [
    "schema=fast_roborender.production_resume_candidate.v1",
    "status=checkpoint_available",
    f"run_id={sys.argv[2]}",
    f"run_root={sys.argv[3]}",
    f"source_attempt_id={sys.argv[4]}",
    f"checkpoint={sys.argv[5]}",
    f"checkpoint_step={sys.argv[6]}",
    f"checkpoint_sha256={sys.argv[7]}",
    f"absolute_target={sys.argv[8]}",
    f"save_frequency={sys.argv[9]}",
    f"run_identity_sha256={sys.argv[10]}",
]
if Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() != expected:
    raise SystemExit("FATAL: production resume candidate does not bind the requested recovery")
PY
        EXPECTED_RESUME_CANDIDATE_SHA256=$(sha256sum "$RESUME_CANDIDATE" | awk '{print $1}')
    fi
fi

ACTIVE_ATTEMPT_DIR=NONE
ACTIVE_ATTEMPT_RESERVED=0
SUBMISSION_SUCCEEDED=0
cleanup_unsubmitted_reservation() {
    local rc=$?
    trap - EXIT
    if [ "$ACTIVE_ATTEMPT_RESERVED" -eq 1 ] && [ "$SUBMISSION_SUCCEEDED" -eq 0 ] && \
       [ -d "$ACTIVE_ATTEMPT_DIR" ] && [ ! -L "$ACTIVE_ATTEMPT_DIR" ]; then
        local lease archive
        lease=$ACTIVE_ATTEMPT_DIR/lease.txt
        if python3 - "$lease" "$RUN_ID" "$ATTEMPT_ID" <<'PY'
import sys
from pathlib import Path

fields = dict(
    line.split("=", 1)
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if "=" in line
)
if fields.get("run_id") != sys.argv[2] or fields.get("attempt_id") != sys.argv[3]:
    raise SystemExit(1)
PY
        then
            archive=$RUN_ATTEMPT_DIR/active_attempt_unsubmitted
            [ ! -e "$archive" ] && mv "$ACTIVE_ATTEMPT_DIR" "$archive" || true
        fi
    fi
    exit "$rc"
}

if [ "$RUN_PROFILE" = production_256x300 ]; then
    ACTIVE_ATTEMPT_DIR=$RUN_ROOT/.active_attempt
    STALE_ATTEMPT_DIR=$RUN_ROOT/stale_attempt_leases
    if [ -e "$STALE_ATTEMPT_DIR" ]; then
        [ -d "$STALE_ATTEMPT_DIR" ] && [ ! -L "$STALE_ATTEMPT_DIR" ] || \
            fatal "production stale-attempt archive is unsafe"
    else
        mkdir "$STALE_ATTEMPT_DIR"
    fi
    while ! mkdir "$ACTIVE_ATTEMPT_DIR" 2>/dev/null; do
        [ -d "$ACTIVE_ATTEMPT_DIR" ] && [ ! -L "$ACTIVE_ATTEMPT_DIR" ] || \
            fatal "production active-attempt reservation is unsafe"
        EXISTING_LEASE=$ACTIVE_ATTEMPT_DIR/lease.txt
        [ -f "$EXISTING_LEASE" ] && [ ! -L "$EXISTING_LEASE" ] || \
            fatal "production active-attempt lease is missing or unsafe"
        LEASE_SUMMARY=$(python3 - "$EXISTING_LEASE" "$RUN_ID" "$RUN_ROOT" <<'PY'
import re
import sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
expected_keys = (
    "schema", "state", "run_id", "attempt_id", "run_attempt_dir",
    "owner_host", "owner_pid", "created_epoch", "job_id",
)
if len(lines) != len(expected_keys):
    raise SystemExit("FATAL: active-attempt lease field count changed")
fields = {}
for key, line in zip(expected_keys, lines, strict=True):
    prefix = key + "="
    if not line.startswith(prefix):
        raise SystemExit("FATAL: active-attempt lease field order changed")
    fields[key] = line[len(prefix):]
if fields["schema"] != "fast_roborender.hsg_active_attempt.v1":
    raise SystemExit("FATAL: active-attempt lease schema changed")
if fields["state"] not in {"submitting", "submitted"}:
    raise SystemExit("FATAL: active-attempt lease state is invalid")
for key in ("run_id", "attempt_id"):
    if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", fields[key]) is None:
        raise SystemExit(f"FATAL: active-attempt lease {key} is malformed")
if fields["run_id"] != sys.argv[2]:
    raise SystemExit("FATAL: active-attempt lease belongs to another logical run")
if fields["run_attempt_dir"] != f'{sys.argv[3]}/attempts/{fields["attempt_id"]}':
    raise SystemExit("FATAL: active-attempt lease attempt path is inconsistent")
if re.fullmatch(r"[1-9][0-9]*", fields["owner_pid"]) is None:
    raise SystemExit("FATAL: active-attempt owner PID is malformed")
if re.fullmatch(r"[1-9][0-9]*", fields["created_epoch"]) is None:
    raise SystemExit("FATAL: active-attempt creation time is malformed")
if fields["state"] == "submitted":
    if re.fullmatch(r"[1-9][0-9]*", fields["job_id"]) is None:
        raise SystemExit("FATAL: submitted active-attempt job ID is malformed")
elif fields["job_id"] != "NONE":
    raise SystemExit("FATAL: submitting active-attempt unexpectedly has a job ID")
print("\t".join(fields[key] for key in (
    "state", "attempt_id", "owner_host", "owner_pid", "created_epoch", "job_id"
)))
PY
        ) || fatal "could not validate the existing production active-attempt lease"
        IFS=$'\t' read -r LEASE_STATE LEASE_ATTEMPT LEASE_HOST LEASE_PID \
            LEASE_CREATED LEASE_JOB_ID <<< "$LEASE_SUMMARY"
        LEASE_ACTIVE=0
        if [ "$LEASE_STATE" = submitted ]; then
            if ! SQUEUE_STATE=$("$SLURM_BIN_DIR/squeue" --noheader \
                    --jobs "$LEASE_JOB_ID" --format=%T 2>/dev/null); then
                fatal "squeue failed while validating active production lease $LEASE_JOB_ID"
            fi
            SQUEUE_STATE=$(printf '%s\n' "$SQUEUE_STATE" | awk 'NF {print; exit}')
            if [ -n "$SQUEUE_STATE" ]; then
                LEASE_ACTIVE=1
            else
                if ! SACCT_OUTPUT=$("$SLURM_BIN_DIR/sacct" -n -X \
                        -j "$LEASE_JOB_ID" --format=State --parsable2 2>/dev/null); then
                    fatal "sacct failed while validating active production lease $LEASE_JOB_ID"
                fi
                TERMINAL_STATE=$(printf '%s\n' "$SACCT_OUTPUT" | \
                    awk -F'|' 'NF && $1 != "" {sub(/[+ ].*/, "", $1); print $1; exit}')
                case "$TERMINAL_STATE" in
                    COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED) ;;
                    *) fatal "cannot prove prior production job $LEASE_JOB_ID is scheduler-terminal (state=${TERMINAL_STATE:-missing})" ;;
                esac
            fi
        else
            CURRENT_HOST=$(hostname -f 2>/dev/null || hostname)
            if [ "$LEASE_HOST" = "$CURRENT_HOST" ] && kill -0 "$LEASE_PID" 2>/dev/null; then
                LEASE_ACTIVE=1
            else
                fatal "ambiguous submitting lease $LEASE_ATTEMPT requires explicit operator recovery; a dead submitter does not prove sbatch failed"
            fi
        fi
        [ "$LEASE_ACTIVE" -eq 0 ] || \
            fatal "production run already has active attempt $LEASE_ATTEMPT"
        STALE_ARCHIVE=$STALE_ATTEMPT_DIR/${LEASE_ATTEMPT}_$(date -u +%Y%m%dT%H%M%SZ)_$$
        mv "$ACTIVE_ATTEMPT_DIR" "$STALE_ARCHIVE" || \
            fatal "could not atomically archive a stale active-attempt lease"
    done
    ACTIVE_ATTEMPT_RESERVED=1
    OWNER_HOST=$(hostname -f 2>/dev/null || hostname)
    CREATED_EPOCH=$(date +%s)
    ( set -o noclobber
      cat > "$ACTIVE_ATTEMPT_DIR/lease.txt" <<EOF
schema=fast_roborender.hsg_active_attempt.v1
state=submitting
run_id=$RUN_ID
attempt_id=$ATTEMPT_ID
run_attempt_dir=$RUN_ATTEMPT_DIR
owner_host=$OWNER_HOST
owner_pid=$$
created_epoch=$CREATED_EPOCH
job_id=NONE
EOF
    ) || fatal "could not create production active-attempt lease"
    trap cleanup_unsubmitted_reservation EXIT
fi

EXPORTS="PATH=$SLURM_BIN_DIR:/usr/local/bin:/usr/bin:/bin,SLURM_BIN_DIR=$SLURM_BIN_DIR,HOME=$STAGE_ROOT/runtime/home,TORCH_HOME=$TORCH_HOME,RESNET18_WEIGHTS=$RESNET18_WEIGHTS,EXPECTED_RESNET18_SHA256=$EXPECTED_RESNET18_SHA256,IMAGE_DAGGER_FFMPEG=$IMAGE_DAGGER_FFMPEG,STAGE_ROOT=$STAGE_ROOT,ROOT=$ROOT,GR00T_ROOT=$GR00T_ROOT,RUN_PROFILE=$RUN_PROFILE,PRODUCTION_ACTION=$PRODUCTION_ACTION,RUN_ID=$RUN_ID,RUN_ROOT=$RUN_ROOT,RUN_ATTEMPT_DIR=$RUN_ATTEMPT_DIR,ATTEMPT_ID=$ATTEMPT_ID,EXPECTED_RUN_IDENTITY_SHA256=$EXPECTED_RUN_IDENTITY_SHA256,FLEET_DIR=$FLEET_DIR,TRAIN_RUN_DIR=$TRAIN_RUN_DIR,PACKED_NODES=$PACKED_NODES,PACKED_GPUS_PER_NODE=4,PACKED_PARTITION=$PACKED_PARTITION,PACKED_QOS=$PACKED_QOS,GLOBAL_NUM_ENVS=$GLOBAL_NUM_ENVS,ROLLOUT_STEPS=$ROLLOUT_STEPS,TRAINER_RANKS=$TRAINER_RANKS,EXPECTED_RENDERERS=$EXPECTED_RENDERERS,NUM_ROLLOUT_ITERATIONS=$NUM_ROLLOUT_ITERATIONS,SAVE_FREQUENCY=$SAVE_FREQUENCY,RESUME_TRAINING=$RESUME_TRAINING,RESUME_CHECKPOINT_STEP=$RESUME_CHECKPOINT_STEP,CKPT=$CKPT,EXPECTED_TRAIN_CHECKPOINT_SHA256=$EXPECTED_TRAIN_CHECKPOINT_SHA256,RESUME_CANDIDATE=$RESUME_CANDIDATE,EXPECTED_RESUME_CANDIDATE_SHA256=$EXPECTED_RESUME_CANDIDATE_SHA256,RENDERER_CONTAINER=$RENDERER_CONTAINER,EXPECTED_RENDERER_CONTAINER_SHA256=$EXPECTED_RENDERER_CONTAINER_SHA256,TRAINER_CONTAINER=$TRAINER_CONTAINER,EXPECTED_TRAINER_CONTAINER_SHA256=$EXPECTED_TRAINER_CONTAINER_SHA256,IMAGE3F_ROOT=$IMAGE3F_ROOT,ROBORENDER_MODEL_BASE=$ROBORENDER_MODEL_BASE,IMAGE3F_GENERATED_ROOT=$IMAGE3F_GENERATED_ROOT,ANCHORS=$ANCHORS,TEACHER_CKPT=$TEACHER_CKPT,EXPECTED_TEACHER_CKPT_SHA256=$EXPECTED_TEACHER_CKPT_SHA256,EXPECTED_FAST_ROBORENDER_REVISION=$FAST_REVISION,EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256=$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256,EXPECTED_GR00T_REVISION=$GR00T_REVISION,EXPECTED_GR00T_SOURCE_TREE_SHA256=$EXPECTED_GR00T_SOURCE_TREE_SHA256,EXPECTED_IMAGE3F_RELEASE_ID=$EXPECTED_IMAGE3F_RELEASE_ID,EXPECTED_IMAGE3F_ADAPTER_SHA256=$EXPECTED_IMAGE3F_ADAPTER_SHA256,EXPECTED_ASSET_CONTRACT_SHA256=$EXPECTED_ASSET_CONTRACT_SHA256,EXPECTED_IMAGE3F_ENGINE_SHA256=$EXPECTED_IMAGE3F_ENGINE_SHA256,EXPECTED_IMAGE3F_INFERENCE_STEPS=5,EXPECTED_IMAGE3F_TEACACHE_THRESHOLD=0.3,EXPECTED_IMAGE3F_CFG_SCALE=1.0,EXPECTED_IMAGE3F_CUDA_GRAPH=disabled,IMAGE_DAGGER_DUMP_SIM_RGB=$IMAGE_DAGGER_DUMP_SIM_RGB,USE_WANDB=$USE_WANDB,WANDB_EXPECTED_EMAIL=$WANDB_EXPECTED_EMAIL,WANDB_ENTITY=$WANDB_ENTITY,WANDB_ONLINE_VALIDATION=$WANDB_ONLINE_VALIDATION,EXPECTED_WANDB_ONLINE_VALIDATION_SHA256=$EXPECTED_WANDB_ONLINE_VALIDATION_SHA256,WANDB_WATCHER_POLL_S=$WANDB_WATCHER_POLL_S,WANDB_WATCHER_READY_TIMEOUT_S=$WANDB_WATCHER_READY_TIMEOUT_S,WANDB_WATCHER_DRAIN_TIMEOUT_S=$WANDB_WATCHER_DRAIN_TIMEOUT_S,CAMERA_CALIBRATION_ACCEPTANCE=$CAMERA_CALIBRATION_ACCEPTANCE,EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256=$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256,PREREQUISITE_PROFILE_EXPECTED=$PREREQUISITE_PROFILE_EXPECTED,PREREQUISITE_PROFILE_VALIDATION=$PREREQUISITE_PROFILE_VALIDATION,EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256=$EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256"
EXPORTS="$EXPORTS,EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256=$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256,EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256=$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256"
CMD=("$SLURM_BIN_DIR/sbatch" --parsable --account=llmservice_nemotron_ultra --partition="$PACKED_PARTITION" --qos="$PACKED_QOS"
    --nodes="$PACKED_NODES" --gpus-per-node=4 --cpus-per-gpu=32 --mem=800G
    --time="$RUN_WALLTIME" --signal=B:USR1@300
    --job-name="rr_${RUN_PROFILE}_gb200"
    --output="$FLEET_DIR/logs/packed_%j.out" --error="$FLEET_DIR/logs/packed_%j.err"
    --export="$EXPORTS" "$ROOT/scripts/scaling/hsg_gb200_packed_smoke.sbatch")

( set -o noclobber
  cat > "$SUBMISSION_MANIFEST" <<EOF
status=planned
disposable_dry_run=$DRY_RUN
run_profile=$RUN_PROFILE
run_id=$RUN_ID
run_root=$RUN_ROOT
run_attempt_dir=$RUN_ATTEMPT_DIR
production_action=$PRODUCTION_ACTION
run_identity_sha256=$EXPECTED_RUN_IDENTITY_SHA256
nodes=$PACKED_NODES
gpus_per_node=4
qos=$PACKED_QOS
partition=$PACKED_PARTITION
renderer_tasks=$EXPECTED_RENDERERS
trainer_tasks=$TRAINER_RANKS
global_envs=$GLOBAL_NUM_ENVS
rollout_steps=$ROLLOUT_STEPS
absolute_target=$NUM_ROLLOUT_ITERATIONS
save_frequency=$SAVE_FREQUENCY
resume_training=$RESUME_TRAINING
resume_checkpoint=$CKPT
resume_checkpoint_step=$RESUME_CHECKPOINT_STEP
resume_checkpoint_sha256=$EXPECTED_TRAIN_CHECKPOINT_SHA256
resume_candidate=$RESUME_CANDIDATE
resume_candidate_sha256=$EXPECTED_RESUME_CANDIDATE_SHA256
renderer_container=$RENDERER_CONTAINER
renderer_container_sha256=$EXPECTED_RENDERER_CONTAINER_SHA256
trainer_container=$TRAINER_CONTAINER
trainer_container_sha256=$EXPECTED_TRAINER_CONTAINER_SHA256
resnet18_weights=$RESNET18_WEIGHTS
resnet18_weights_sha256=$EXPECTED_RESNET18_SHA256
image_dagger_ffmpeg=$IMAGE_DAGGER_FFMPEG
fast_roborender_revision=$FAST_REVISION
fast_roborender_source_tree_sha256=$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256
gr00t_revision=$GR00T_REVISION
gr00t_source_tree_sha256=$EXPECTED_GR00T_SOURCE_TREE_SHA256
source_snapshot_attestation_sha256=$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256
source_snapshot_inventory_sha256=$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256
immutable_asset_contract_sha256=$EXPECTED_ASSET_CONTRACT_SHA256
image3f_engine_sha256=$EXPECTED_IMAGE3F_ENGINE_SHA256
wandb_online_validation=$WANDB_ONLINE_VALIDATION
wandb_online_validation_sha256=$EXPECTED_WANDB_ONLINE_VALIDATION_SHA256
wandb_watcher_poll_s=$WANDB_WATCHER_POLL_S
wandb_watcher_ready_timeout_s=$WANDB_WATCHER_READY_TIMEOUT_S
wandb_watcher_drain_timeout_s=$WANDB_WATCHER_DRAIN_TIMEOUT_S
slurm_bin_dir=$SLURM_BIN_DIR
camera_calibration_acceptance_sha256=$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256
prerequisite_profile_validation_sha256=$EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256
prerequisite_profile_expected=$PREREQUISITE_PROFILE_EXPECTED
EOF
) || fatal "refusing to overwrite submission manifest"

if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN:'; printf ' %q' "${CMD[@]}"; printf '\n'
    exit 0
fi
# Once sbatch is invoked its outcome can be ambiguous (for example, a network
# failure after controller acceptance).  Retain the lease for explicit
# operator recovery on every failure from this point forward.
SUBMISSION_SUCCEEDED=1
[ "$RUN_PROFILE" != production_256x300 ] || : > "$RUN_ATTEMPT_DIR/SBATCH_INVOKED"
if ! JOB_ID=$("${CMD[@]}"); then
    fatal "sbatch failed after invocation; production lease is retained for explicit recovery"
fi
JOB_ID=${JOB_ID%%;*}
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || fatal "sbatch returned malformed job id: $JOB_ID"
if [ "$RUN_PROFILE" = production_256x300 ]; then
    python3 - "$ACTIVE_ATTEMPT_DIR/lease.txt" "$RUN_ID" "$ATTEMPT_ID" "$JOB_ID" <<'PY'
import os
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines()
fields = dict(line.split("=", 1) for line in lines if "=" in line)
expected = {
    "schema": "fast_roborender.hsg_active_attempt.v1",
    "state": "submitting",
    "run_id": sys.argv[2],
    "attempt_id": sys.argv[3],
    "job_id": "NONE",
}
for key, value in expected.items():
    if fields.get(key) != value:
        raise SystemExit(f"FATAL: active-attempt lease {key} changed before job binding")
fields["state"] = "submitted"
fields["job_id"] = sys.argv[4]
ordered = (
    "schema", "state", "run_id", "attempt_id", "run_attempt_dir",
    "owner_host", "owner_pid", "created_epoch", "job_id",
)
payload = "\n".join(f"{key}={fields[key]}" for key in ordered) + "\n"
descriptor, temporary = tempfile.mkstemp(prefix=".lease.txt.", dir=path.parent)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
fi
printf 'status=submitted\njob_id=%s\n' "$JOB_ID" >> "$SUBMISSION_MANIFEST"
echo "HSG GB200 $RUN_PROFILE job: $JOB_ID"
