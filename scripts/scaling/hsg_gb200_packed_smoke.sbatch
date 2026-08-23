#!/bin/bash
# HSG packed validation/production: four GB200s/node with distinct renderer
# and Isaac/GR00T images.  Scheduler resources are supplied by the wrapper.
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

required=(STAGE_ROOT ROOT GR00T_ROOT RUN_PROFILE PRODUCTION_ACTION RUN_ID RUN_ROOT
          RUN_ATTEMPT_DIR ATTEMPT_ID
          EXPECTED_RUN_IDENTITY_SHA256
          FLEET_DIR TRAIN_RUN_DIR
          SLURM_BIN_DIR
          PACKED_NODES PACKED_GPUS_PER_NODE PACKED_PARTITION GLOBAL_NUM_ENVS ROLLOUT_STEPS TRAINER_RANKS
          NUM_ROLLOUT_ITERATIONS SAVE_FREQUENCY RESUME_TRAINING
          RESUME_CHECKPOINT_STEP CKPT EXPECTED_TRAIN_CHECKPOINT_SHA256
          RESUME_CANDIDATE EXPECTED_RESUME_CANDIDATE_SHA256
          EXPECTED_RENDERERS RENDERER_CONTAINER TRAINER_CONTAINER
          EXPECTED_RENDERER_CONTAINER_SHA256 EXPECTED_TRAINER_CONTAINER_SHA256
          IMAGE3F_ROOT
          ROBORENDER_MODEL_BASE IMAGE3F_GENERATED_ROOT ANCHORS TEACHER_CKPT
          TORCH_HOME RESNET18_WEIGHTS EXPECTED_RESNET18_SHA256
          IMAGE_DAGGER_FFMPEG
          PACKED_QOS EXPECTED_GR00T_SOURCE_TREE_SHA256
          EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256
          EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256
          EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256
          WANDB_ONLINE_VALIDATION EXPECTED_WANDB_ONLINE_VALIDATION_SHA256
          WANDB_WATCHER_POLL_S WANDB_WATCHER_READY_TIMEOUT_S
          WANDB_WATCHER_DRAIN_TIMEOUT_S WANDB_EXPECTED_EMAIL WANDB_ENTITY
          EXPECTED_TEACHER_CKPT_SHA256 EXPECTED_FAST_ROBORENDER_REVISION
          EXPECTED_GR00T_REVISION EXPECTED_IMAGE3F_RELEASE_ID
          EXPECTED_IMAGE3F_ADAPTER_SHA256 EXPECTED_ASSET_CONTRACT_SHA256
          EXPECTED_IMAGE3F_ENGINE_SHA256
          IMAGE_DAGGER_DUMP_SIM_RGB USE_WANDB
          CAMERA_CALIBRATION_ACCEPTANCE CAMERA_CALIBRATION_RUN_ID
          EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256
          CAMERA_CALIBRATION_MANIFEST_DESTINATION
          PREREQUISITE_PROFILE_EXPECTED PREREQUISITE_PROFILE_VALIDATION
          EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256)
for name in "${required[@]}"; do
    [ -n "${!name:-}" ] || { echo "FATAL: missing sanitized launch variable $name" >&2; exit 2; }
done
for slurm_command in scontrol srun; do
    [ -x "$SLURM_BIN_DIR/$slurm_command" ] || {
        echo "FATAL: missing HSG Slurm command: $SLURM_BIN_DIR/$slurm_command" >&2
        exit 2
    }
done
[ -x /usr/bin/timeout ] || { echo "FATAL: /usr/bin/timeout is required for bounded recovery" >&2; exit 2; }

# The submit wrapper deliberately gives sbatch a sanitized --export list.
# Without overriding Slurm's inherited export policy, child srun steps only
# receive that original list and miss variables computed inside this script
# (for example EP_LIST, LOCAL_NUM_ENVS, and BASE_RENDER_SEED).  The batch
# environment is already sanitized above, so propagate its complete current
# state to every containerized step.
export SLURM_EXPORT_ENV=ALL

fatal() { echo "FATAL: $*" >&2; exit 2; }

validate_calibration_run_directories() {
    local runtime_dir expected_run_root run_directory canonical
    [ "$RUN_PROFILE" = calibration_3view ] || return 0
    runtime_dir=$STAGE_ROOT/runtime
    expected_run_root=$runtime_dir/$RUN_ID
    [ -d "$runtime_dir" ] && [ ! -L "$runtime_dir" ] || \
        fatal "calibration runtime directory is missing or unsafe"
    canonical=$(realpath -e "$runtime_dir") || \
        fatal "cannot resolve calibration runtime directory"
    [ "$canonical" = "$runtime_dir" ] || \
        fatal "calibration runtime directory must be canonical"
    [ "$RUN_ROOT" = "$expected_run_root" ] || \
        fatal "calibration RUN_ROOT must be exactly $expected_run_root"
    [ "$RUN_ATTEMPT_DIR" = "$RUN_ROOT" ] || \
        fatal "calibration RUN_ATTEMPT_DIR must equal RUN_ROOT"
    [ "$FLEET_DIR" = "$RUN_ROOT/fleet" ] || \
        fatal "calibration FLEET_DIR must be an exact RUN_ROOT child"
    [ "$TRAIN_RUN_DIR" = "$RUN_ROOT/train" ] || \
        fatal "calibration TRAIN_RUN_DIR must be an exact RUN_ROOT child"
    for run_directory in "$RUN_ROOT" "$FLEET_DIR" "$TRAIN_RUN_DIR"; do
        [ -d "$run_directory" ] && [ ! -L "$run_directory" ] || \
            fatal "calibration run directory is missing or unsafe: $run_directory"
        canonical=$(realpath -e "$run_directory") || \
            fatal "cannot resolve calibration run directory: $run_directory"
        [ "$canonical" = "$run_directory" ] || \
            fatal "calibration run directory must be canonical: $run_directory"
    done
}
validate_calibration_run_directories

case "$RUN_PROFILE" in
    smoke_32x300|smoke_256x300|production_256x300)
        [[ "$CAMERA_CALIBRATION_RUN_ID" =~ ^[A-Za-z0-9_-]{1,128}$ ]] || \
            fatal "CAMERA_CALIBRATION_RUN_ID is malformed"
        [ "$CAMERA_CALIBRATION_ACCEPTANCE" = \
          "$STAGE_ROOT/runtime/validation/camera_calibration_acceptance_${CAMERA_CALIBRATION_RUN_ID}.json" ] || \
            fatal "camera calibration acceptance escaped its exact versioned path"
        ;;
    *)
        [ "$CAMERA_CALIBRATION_ACCEPTANCE:$CAMERA_CALIBRATION_RUN_ID:$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256" = \
          NONE:NONE:NONE ] || fatal "$RUN_PROFILE cannot consume a camera calibration acceptance"
        ;;
esac

validate_camera_calibration_manifest_destination() {
    local validation_dir expected_destination canonical_validation_dir
    validation_dir=$STAGE_ROOT/runtime/validation
    if [ "$RUN_PROFILE" = calibration_3view ]; then
        expected_destination=$validation_dir/camera_calibration_manifest_${RUN_ID}.json
        [ "$CAMERA_CALIBRATION_MANIFEST_DESTINATION" = "$expected_destination" ] || \
            fatal "calibration manifest destination does not exactly bind RUN_ID=$RUN_ID"
        [ -d "$validation_dir" ] && [ ! -L "$validation_dir" ] || \
            fatal "calibration validation directory is missing or unsafe"
        canonical_validation_dir=$(realpath -e "$validation_dir") || \
            fatal "cannot resolve calibration validation directory"
        [ "$canonical_validation_dir" = "$validation_dir" ] || \
            fatal "calibration validation directory must be canonical"
        [ "$(dirname -- "$CAMERA_CALIBRATION_MANIFEST_DESTINATION")" = \
          "$validation_dir" ] || \
            fatal "calibration manifest destination must be a direct validation child"
        [ "$(basename -- "$CAMERA_CALIBRATION_MANIFEST_DESTINATION")" = \
          "camera_calibration_manifest_${RUN_ID}.json" ] || \
            fatal "calibration manifest destination filename changed"
        [ ! -L "$CAMERA_CALIBRATION_MANIFEST_DESTINATION" ] || \
            fatal "calibration manifest destination cannot be a symlink"
        [ ! -e "$CAMERA_CALIBRATION_MANIFEST_DESTINATION" ] || \
        [ -f "$CAMERA_CALIBRATION_MANIFEST_DESTINATION" ] || \
            fatal "existing calibration manifest destination is not a regular file"
    else
        [ "$CAMERA_CALIBRATION_MANIFEST_DESTINATION" = NONE ] || \
            fatal "$RUN_PROFILE must use CAMERA_CALIBRATION_MANIFEST_DESTINATION=NONE"
    fi
}
validate_camera_calibration_manifest_destination

[ "$PACKED_GPUS_PER_NODE" -eq 4 ] || fatal "HSG profile requires four GB200s per node"
[ "${SLURM_NNODES:-0}" -eq "$PACKED_NODES" ] || fatal "allocated node count changed"
[ "${SLURM_JOB_PARTITION:-}" = "$PACKED_PARTITION" ] || \
    fatal "allocated partition changed: ${SLURM_JOB_PARTITION:-unset} != $PACKED_PARTITION"
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
        [ "$PRODUCTION_ACTION:$ATTEMPT_ID" = train:NONE ] || fatal "non-production action contract changed"
        [ "$RESUME_CANDIDATE:$EXPECTED_RESUME_CANDIDATE_SHA256" = NONE:NONE ] || fatal "non-production resume candidate changed"
        EXPECTED_CONTRACT="1:1:2:1:1:0:0:interactive:batch"
        EXPECTED_TRAINING_CONTRACT="1:1:0:0:none:NONE:train"
        ;;
    resume_minimal)
        [ "$PRODUCTION_ACTION:$ATTEMPT_ID" = train:NONE ] || fatal "non-production action contract changed"
        [ "$RESUME_CANDIDATE:$EXPECTED_RESUME_CANDIDATE_SHA256" = NONE:NONE ] || fatal "non-production resume candidate changed"
        EXPECTED_CONTRACT="1:1:2:1:1:0:0:interactive:batch"
        EXPECTED_TRAINING_CONTRACT="2:1:0:0:none:NONE:train"
        ;;
    calibration_3view)
        [ "$PRODUCTION_ACTION:$ATTEMPT_ID" = train:NONE ] || fatal "non-production action contract changed"
        [ "$RESUME_CANDIDATE:$EXPECTED_RESUME_CANDIDATE_SHA256" = NONE:NONE ] || fatal "non-production resume candidate changed"
        EXPECTED_CONTRACT="1:1:2:1:1:1:0:interactive:batch"
        EXPECTED_TRAINING_CONTRACT="1:1:0:0:none:NONE:train"
        ;;
    smoke_32x300)
        [ "$PRODUCTION_ACTION:$ATTEMPT_ID" = train:NONE ] || fatal "non-production action contract changed"
        [ "$RESUME_CANDIDATE:$EXPECTED_RESUME_CANDIDATE_SHA256" = NONE:NONE ] || fatal "non-production resume candidate changed"
        EXPECTED_CONTRACT="2:32:300:1:7:0:1:interactive:batch"
        EXPECTED_TRAINING_CONTRACT="1:1:0:0:none:NONE:train"
        ;;
    smoke_256x300)
        [ "$PRODUCTION_ACTION:$ATTEMPT_ID" = train:NONE ] || fatal "non-production action contract changed"
        [ "$RESUME_CANDIDATE:$EXPECTED_RESUME_CANDIDATE_SHA256" = NONE:NONE ] || fatal "non-production resume candidate changed"
        EXPECTED_CONTRACT="15:256:300:8:52:0:1:normal:batch"
        EXPECTED_TRAINING_CONTRACT="1:1:0:0:none:NONE:train"
        ;;
    production_256x300)
        EXPECTED_CONTRACT="15:256:300:8:52:0:1:normal:batch_long"
        [[ "$ATTEMPT_ID" =~ ^[A-Za-z0-9_-]{1,128}$ ]] || \
            fatal "production attempt ID is malformed"
        case "$PRODUCTION_ACTION" in
            train)
                EXPECTED_TRAINING_CONTRACT="500:5:$RESUME_TRAINING:$RESUME_CHECKPOINT_STEP:$CKPT:$EXPECTED_TRAIN_CHECKPOINT_SHA256:train"
                [ "$RESUME_TRAINING" = 0 ] || [ "$RESUME_TRAINING" = 1 ] || \
                    fatal "production RESUME_TRAINING must be 0 or 1"
                if [ "$RESUME_TRAINING" = 0 ]; then
                    [ "$RESUME_CHECKPOINT_STEP:$CKPT:$EXPECTED_TRAIN_CHECKPOINT_SHA256" = 0:none:NONE ] || \
                        fatal "fresh production student contract changed"
                    [ "$RESUME_CANDIDATE:$EXPECTED_RESUME_CANDIDATE_SHA256" = NONE:NONE ] || \
                        fatal "fresh production unexpectedly received a resume candidate"
                else
                    [ "$RESUME_CHECKPOINT_STEP" -gt 0 ] && \
                    [ "$RESUME_CHECKPOINT_STEP" -lt 500 ] && \
                    [ $((RESUME_CHECKPOINT_STEP % 5)) -eq 0 ] || \
                        fatal "production resume checkpoint step changed"
                    [[ "$EXPECTED_TRAIN_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
                        fatal "production resume checkpoint hash is malformed"
                    [[ "$EXPECTED_RESUME_CANDIDATE_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
                        fatal "production resume candidate hash is malformed"
                fi
                ;;
            validate)
                EXPECTED_TRAINING_CONTRACT="500:5:0:500:$CKPT:$EXPECTED_TRAIN_CHECKPOINT_SHA256:validate"
                [ "$(basename "$CKPT")" = model_step_000500.pt ] || \
                    fatal "validation-only recovery checkpoint changed"
                [[ "$EXPECTED_TRAIN_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
                    fatal "validation-only checkpoint hash is malformed"
                [[ "$EXPECTED_RESUME_CANDIDATE_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
                    fatal "validation-only resume candidate hash is malformed"
                ;;
            *) fatal "production action must be train or validate" ;;
        esac
        ;;
    *) fatal "unsupported HSG profile $RUN_PROFILE" ;;
esac
ACTUAL_CONTRACT="$PACKED_NODES:$GLOBAL_NUM_ENVS:$ROLLOUT_STEPS:$TRAINER_RANKS:$EXPECTED_RENDERERS:$IMAGE_DAGGER_DUMP_SIM_RGB:$USE_WANDB:$PACKED_QOS:$PACKED_PARTITION"
[ "$ACTUAL_CONTRACT" = "$EXPECTED_CONTRACT" ] || fatal "$RUN_PROFILE contract changed: $ACTUAL_CONTRACT"
ACTUAL_TRAINING_CONTRACT="$NUM_ROLLOUT_ITERATIONS:$SAVE_FREQUENCY:$RESUME_TRAINING:$RESUME_CHECKPOINT_STEP:$CKPT:$EXPECTED_TRAIN_CHECKPOINT_SHA256:$PRODUCTION_ACTION"
[ "$ACTUAL_TRAINING_CONTRACT" = "$EXPECTED_TRAINING_CONTRACT" ] || \
    fatal "$RUN_PROFILE training contract changed"

if [ "$RUN_PROFILE" = production_256x300 ]; then
    EXPECTED_SNAPSHOT_PREFIX=$(realpath -m "$STAGE_ROOT/src_snapshots")/
    [ -d "$ROOT" ] && [ ! -L "$ROOT" ] || \
        fatal "production ROOT is not a real sealed source directory"
    CANONICAL_ROOT=$(realpath -e "$ROOT") || fatal "cannot resolve production ROOT"
    case "$CANONICAL_ROOT" in
        "$EXPECTED_SNAPSHOT_PREFIX"fast_roborender_*_sealed) ;;
        *) fatal "allocated production ROOT escaped the sealed snapshot directory" ;;
    esac
    [ "$ROOT" = "$CANONICAL_ROOT" ] || fatal "allocated production ROOT is not canonical"
    CANONICAL_GR00T=$(realpath -e "$GR00T_ROOT") || fatal "cannot resolve GR00T_ROOT"
    [ "$CANONICAL_GR00T" = "$CANONICAL_ROOT/GR00T-VisualSim2Real" ] && \
    [ "$GR00T_ROOT" = "$CANONICAL_GR00T" ] || \
        fatal "allocated GR00T_ROOT is not the exact nested sealed tree"
    [ ! -w "$CANONICAL_ROOT" ] && [ ! -w "$CANONICAL_GR00T" ] || \
        fatal "allocated production source snapshots are writable"
    WRITABLE_SOURCE=$(find "$CANONICAL_ROOT" -xdev -perm /222 -print -quit) || \
        fatal "cannot audit allocated source permissions"
    [ -z "$WRITABLE_SOURCE" ] || \
        fatal "allocated sealed source contains a writable inode: $WRITABLE_SOURCE"
    FORBIDDEN_SOURCE=$(find "$CANONICAL_ROOT" -xdev \
        \( -type d -name __pycache__ -o -type f \
           \( -name '*.pyc' -o -name '*.pyo' \) \) -print -quit) || \
        fatal "cannot audit allocated source for executable bytecode"
    [ -z "$FORBIDDEN_SOURCE" ] || \
        fatal "allocated sealed source contains executable bytecode: $FORBIDDEN_SOURCE"
fi

for file in "$RENDERER_CONTAINER" "$TRAINER_CONTAINER" "$TEACHER_CKPT" \
            "$ROOT/.source_revision" "$GR00T_ROOT/.source_revision"; do
    [ -s "$file" ] || fatal "required file missing: $file"
done
[ "$(sha256sum "$RENDERER_CONTAINER" | awk '{print $1}')" = \
  "$EXPECTED_RENDERER_CONTAINER_SHA256" ] || \
    fatal "renderer container changed after submission"
[ "$(sha256sum "$TRAINER_CONTAINER" | awk '{print $1}')" = \
  "$EXPECTED_TRAINER_CONTAINER_SHA256" ] || \
    fatal "trainer container changed after submission"
[ "$(tr -d '\r\n' < "$ROOT/.source_revision")" = "$EXPECTED_FAST_ROBORENDER_REVISION" ] || \
    fatal "fast_roborender source revision changed after submission"
[ "$(tr -d '\r\n' < "$GR00T_ROOT/.source_revision")" = "$EXPECTED_GR00T_REVISION" ] || \
    fatal "GR00T source revision changed after submission"
ACTUAL_FAST_ROBORENDER_SOURCE_TREE_SHA256=$(python3 "$ROOT/scripts/scaling/hsg_gb200_smoke_gate.py" \
    tree-hash --root "$ROOT")
[ "$ACTUAL_FAST_ROBORENDER_SOURCE_TREE_SHA256" = \
  "$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256" ] || \
    fatal "fast_roborender source tree changed after submission"
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

# Re-run the complete content-addressed asset gate after the queue wait.  The
# model, TensorRT engine, foundation shards, and 2,048-anchor pool live outside
# the sealed source trees and therefore need their own runtime attestation.
RUNTIME_PREFLIGHT=$RUN_ATTEMPT_DIR/runtime_preflight.json
python3 "$ROOT/scripts/scaling/hsg_gb200_smoke_gate.py" preflight \
    --profile "$RUN_PROFILE" --root "$ROOT" --gr00t-root "$GR00T_ROOT" \
    --fast-revision "$EXPECTED_FAST_ROBORENDER_REVISION" \
    --gr00t-revision "$EXPECTED_GR00T_REVISION" \
    --renderer-container "$RENDERER_CONTAINER" \
    --renderer-sha256 "$EXPECTED_RENDERER_CONTAINER_SHA256" \
    --trainer-container "$TRAINER_CONTAINER" \
    --trainer-sha256 "$EXPECTED_TRAINER_CONTAINER_SHA256" \
    --image-root "$IMAGE3F_ROOT" --model-root "$ROBORENDER_MODEL_BASE" \
    --generated-root "$IMAGE3F_GENERATED_ROOT" --anchors "$ANCHORS" \
    --anchor-min-files 2048 --teacher "$TEACHER_CKPT" \
    --teacher-sha256 "$EXPECTED_TEACHER_CKPT_SHA256" \
    --release-id "$EXPECTED_IMAGE3F_RELEASE_ID" \
    --adapter-sha256 "$EXPECTED_IMAGE3F_ADAPTER_SHA256" > "$RUNTIME_PREFLIGHT"
read -r RUNTIME_ASSET_CONTRACT_SHA256 RUNTIME_ENGINE_SHA256 \
    RUNTIME_SOURCE_SNAPSHOT_ATTESTATION_SHA256 \
    RUNTIME_SOURCE_SNAPSHOT_INVENTORY_SHA256 < <(python3 - "$RUNTIME_PREFLIGHT" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    payload.get("asset_contract_sha256", ""),
    payload.get("engine_sha256", ""),
    payload.get("source_snapshot_attestation_sha256", ""),
    payload.get("source_snapshot_inventory_sha256", ""),
)
PY
)
[ "$RUNTIME_ASSET_CONTRACT_SHA256" = "$EXPECTED_ASSET_CONTRACT_SHA256" ] || \
    fatal "immutable model/engine/foundation/anchor asset contract changed after submission"
[ "$RUNTIME_ENGINE_SHA256" = "$EXPECTED_IMAGE3F_ENGINE_SHA256" ] || \
    fatal "TensorRT engine changed after submission"
[ "$RUNTIME_SOURCE_SNAPSHOT_ATTESTATION_SHA256" = \
  "$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256" ] || \
    fatal "source snapshot attestation changed after submission"
[ "$RUNTIME_SOURCE_SNAPSHOT_INVENTORY_SHA256" = \
  "$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256" ] || \
    fatal "source snapshot inventory changed after submission"

# The stage root must remain writable for run artifacts, so overlay every
# immutable source/model input at its exact original path as read-only.  This
# prevents code that bypasses the convenience aliases from mutating campaign
# inputs through the broad STAGE_ROOT mount.
ASSET_ROOT=$STAGE_ROOT/assets
[ -d "$ASSET_ROOT" ] && [ ! -L "$ASSET_ROOT" ] || \
    fatal "HSG immutable asset root is missing or unsafe"
IMMUTABLE_RO_MOUNTS="$ROOT:$ROOT:ro,$GR00T_ROOT:$GR00T_ROOT:ro,$ASSET_ROOT:$ASSET_ROOT:ro,$IMAGE3F_ROOT:$IMAGE3F_ROOT:ro,$ROBORENDER_MODEL_BASE:$ROBORENDER_MODEL_BASE:ro,$IMAGE3F_GENERATED_ROOT:$IMAGE3F_GENERATED_ROOT:ro,$ANCHORS:$ANCHORS:ro,$TEACHER_CKPT:$TEACHER_CKPT:ro,$RESNET18_WEIGHTS:$RESNET18_WEIGHTS:ro"

case "$RUN_PROFILE" in
    smoke_32x300|smoke_256x300|production_256x300)
        [ -f "$CAMERA_CALIBRATION_ACCEPTANCE" ] && \
        [ ! -L "$CAMERA_CALIBRATION_ACCEPTANCE" ] || \
            fatal "camera calibration acceptance is missing or unsafe"
        [ "$(sha256sum "$CAMERA_CALIBRATION_ACCEPTANCE" | awk '{print $1}')" = \
          "$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256" ] || \
            fatal "camera calibration acceptance changed after submission"
        python3 "$ROOT/scripts/scaling/camera_calibration_gate.py" verify \
            --acceptance "$CAMERA_CALIBRATION_ACCEPTANCE" \
            --expected-run-id "$CAMERA_CALIBRATION_RUN_ID" \
            --expected-fast-roborender-revision "$EXPECTED_FAST_ROBORENDER_REVISION" \
            --expected-gr00t-revision "$EXPECTED_GR00T_REVISION" \
            --expected-fast-source-tree-sha256 "$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256" \
            --expected-gr00t-source-tree-sha256 "$EXPECTED_GR00T_SOURCE_TREE_SHA256" \
            --expected-source-snapshot-attestation-sha256 \
              "$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256" \
            --expected-source-snapshot-inventory-sha256 \
              "$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256" \
            --expected-asset-contract-sha256 "$EXPECTED_ASSET_CONTRACT_SHA256" \
            --expected-acceptance-sha256 "$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256" \
            >/dev/null || fatal "camera calibration acceptance validation failed"
        ;;
esac
case "$RUN_PROFILE:$PREREQUISITE_PROFILE_EXPECTED" in
    smoke_256x300:smoke_32x300|production_256x300:smoke_256x300)
        [ -f "$PREREQUISITE_PROFILE_VALIDATION" ] && \
        [ ! -L "$PREREQUISITE_PROFILE_VALIDATION" ] || \
            fatal "predecessor profile validation is missing or unsafe"
        [ "$(sha256sum "$PREREQUISITE_PROFILE_VALIDATION" | awk '{print $1}')" = \
          "$EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256" ] || \
            fatal "predecessor profile validation changed after submission"
        python3 "$ROOT/scripts/scaling/profile_validation_gate.py" \
            --manifest "$PREREQUISITE_PROFILE_VALIDATION" \
            --expected-profile "$PREREQUISITE_PROFILE_EXPECTED" \
            --expected-fast-revision "$EXPECTED_FAST_ROBORENDER_REVISION" \
            --expected-gr00t-revision "$EXPECTED_GR00T_REVISION" \
            --expected-fast-source-tree-sha256 "$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256" \
            --expected-gr00t-source-tree-sha256 "$EXPECTED_GR00T_SOURCE_TREE_SHA256" \
            --expected-source-snapshot-attestation-sha256 \
              "$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256" \
            --expected-source-snapshot-inventory-sha256 \
              "$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256" \
            --expected-asset-contract-sha256 "$EXPECTED_ASSET_CONTRACT_SHA256" \
            --expected-wandb-email "$WANDB_EXPECTED_EMAIL" \
            --expected-wandb-entity "$WANDB_ENTITY" \
            --expected-manifest-sha256 "$EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256" \
            >/dev/null || fatal "predecessor profile validation failed"
        ;;
    *:NONE)
        [ "$PREREQUISITE_PROFILE_VALIDATION:$EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256" = NONE:NONE ] || \
            fatal "unexpected predecessor validation supplied"
        ;;
    *) fatal "invalid predecessor transition $PREREQUISITE_PROFILE_EXPECTED -> $RUN_PROFILE" ;;
esac

if [ "$RUN_PROFILE" = production_256x300 ]; then
    [ "$RUN_ROOT" = "$STAGE_ROOT/runtime/$RUN_ID" ] || \
        fatal "production RUN_ROOT escaped the pinned runtime directory"
    case "$RUN_ATTEMPT_DIR" in "$RUN_ROOT"/attempts/*) ;; *)
        fatal "production attempt directory escaped RUN_ROOT" ;;
    esac
    [ "$(realpath -e "$RUN_ATTEMPT_DIR")" = "$RUN_ATTEMPT_DIR" ] || \
        fatal "production attempt directory is redirected"
    RUN_IDENTITY=$RUN_ROOT/run_identity.txt
    [ -f "$RUN_IDENTITY" ] && [ ! -L "$RUN_IDENTITY" ] || \
        fatal "production run identity is missing or unsafe"
    [ "$(sha256sum "$RUN_IDENTITY" | awk '{print $1}')" = \
      "$EXPECTED_RUN_IDENTITY_SHA256" ] || \
        fatal "production run identity changed after submission"
    python3 - "$RUN_IDENTITY" \
        "$RUN_ID" "$NUM_ROLLOUT_ITERATIONS" "$SAVE_FREQUENCY" \
        "$PACKED_NODES" "$GLOBAL_NUM_ENVS" "$ROLLOUT_STEPS" \
        "$TRAINER_RANKS" "$EXPECTED_RENDERERS" "$PACKED_PARTITION" "$PACKED_QOS" \
        "$WANDB_EXPECTED_EMAIL" "$WANDB_ENTITY" \
        "$EXPECTED_WANDB_ONLINE_VALIDATION_SHA256" \
        "$EXPECTED_FAST_ROBORENDER_REVISION" \
        "$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256" \
        "$EXPECTED_GR00T_REVISION" "$EXPECTED_GR00T_SOURCE_TREE_SHA256" \
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
values = sys.argv[2:]
if len(values) != len(keys):
    raise SystemExit("FATAL: internal production run identity argument mismatch")
expected = "\n".join(
    ["schema=fast_roborender.hsg_production_run_identity.v1"]
    + [f"{key}={value}" for key, value in zip(keys, values, strict=True)]
) + "\n"
if Path(sys.argv[1]).read_text(encoding="utf-8") != expected:
    raise SystemExit("FATAL: production run identity field set changed")
PY
    if [ "$RESUME_TRAINING" = 1 ] || [ "$PRODUCTION_ACTION" = validate ]; then
        [ -f "$RESUME_CANDIDATE" ] && [ ! -L "$RESUME_CANDIDATE" ] || \
            fatal "production resume candidate is missing or unsafe"
        [ "$(sha256sum "$RESUME_CANDIDATE" | awk '{print $1}')" = \
          "$EXPECTED_RESUME_CANDIDATE_SHA256" ] || \
            fatal "production resume candidate changed after submission"
        case "$RESUME_CANDIDATE" in
            "$RUN_ROOT"/attempts/*/resume_candidate.txt) ;;
            *) fatal "production resume candidate escaped the original attempt tree" ;;
        esac
        CANDIDATE_SOURCE_ATTEMPT=$(basename "$(dirname "$RESUME_CANDIDATE")")
        [ -f "$(dirname "$RESUME_CANDIDATE")/ATTEMPT_FAILED" ] && \
        [ ! -L "$(dirname "$RESUME_CANDIDATE")/ATTEMPT_FAILED" ] || \
            fatal "production resume candidate is not from a failed attempt"
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
    raise SystemExit("FATAL: production resume candidate field set changed")
PY
        LATEST_CHECKPOINT=$(find "$TRAIN_RUN_DIR" -maxdepth 1 -type f \
            -name 'model_step_[0-9][0-9][0-9][0-9][0-9][0-9].pt' \
            -printf '%f\n' | LC_ALL=C sort | tail -n 1)
        [ "$CKPT" = "$TRAIN_RUN_DIR/$LATEST_CHECKPOINT" ] || \
            fatal "production recovery checkpoint is no longer the latest durable checkpoint"
    fi
    if [ "$PRODUCTION_ACTION" = validate ]; then
        [ "$CKPT" = "$TRAIN_RUN_DIR/model_step_000500.pt" ] || \
            fatal "validation-only checkpoint left the reused training directory"
        [ -f "$CKPT" ] && [ ! -L "$CKPT" ] || \
            fatal "validation-only checkpoint is missing or unsafe"
        [ "$(sha256sum "$CKPT" | awk '{print $1}')" = "$EXPECTED_TRAIN_CHECKPOINT_SHA256" ] || \
            fatal "validation-only checkpoint changed after submission"
        for artifact in "$TRAIN_RUN_DIR/initial_policy.pt" "$TRAIN_RUN_DIR/meta.yaml" \
                        "$TRAIN_RUN_DIR/telemetry/WANDB_SIDECAR_RUN_ID"; do
            [ -f "$artifact" ] && [ ! -L "$artifact" ] || \
                fatal "validation-only prerequisite is missing or unsafe: $artifact"
        done
    elif [ "$RESUME_TRAINING" = 0 ]; then
        [ ! -e "$TRAIN_RUN_DIR/initial_policy.pt" ] || \
            fatal "fresh production refuses an existing student baseline"
        [ -z "$(find "$TRAIN_RUN_DIR" -maxdepth 1 -type f -name 'model_step_*.pt' -print -quit)" ] || \
            fatal "fresh production refuses existing trained checkpoints"
    else
        [ -f "$CKPT" ] && [ ! -L "$CKPT" ] || \
            fatal "production resume checkpoint is missing or unsafe"
        [ "$(realpath -e "$(dirname "$CKPT")")" = "$(realpath -e "$TRAIN_RUN_DIR")" ] || \
            fatal "production resume checkpoint left the reused training directory"
        [ "$(sha256sum "$CKPT" | awk '{print $1}')" = "$EXPECTED_TRAIN_CHECKPOINT_SHA256" ] || \
            fatal "production resume checkpoint changed after submission"
        [ -f "$TRAIN_RUN_DIR/meta.yaml" ] && [ ! -L "$TRAIN_RUN_DIR/meta.yaml" ] || \
            fatal "production resume trainer W&B metadata is missing or unsafe"
        [ -f "$TRAIN_RUN_DIR/telemetry/WANDB_SIDECAR_RUN_ID" ] && \
        [ ! -L "$TRAIN_RUN_DIR/telemetry/WANDB_SIDECAR_RUN_ID" ] || \
            fatal "production resume media W&B ID is missing or unsafe"
    fi
    ACTIVE_ATTEMPT_DIR=$RUN_ROOT/.active_attempt
    ACTIVE_LEASE_DEADLINE=$((SECONDS + 120))
    while :; do
        [ -d "$ACTIVE_ATTEMPT_DIR" ] && [ ! -L "$ACTIVE_ATTEMPT_DIR" ] || \
            fatal "production active-attempt reservation is missing or unsafe"
        ACTIVE_LEASE=$ACTIVE_ATTEMPT_DIR/lease.txt
        [ -f "$ACTIVE_LEASE" ] && [ ! -L "$ACTIVE_LEASE" ] || \
            fatal "production active-attempt lease is missing or unsafe"
        ACTIVE_LEASE_BINDING=$(python3 - "$ACTIVE_LEASE" "$RUN_ID" "$ATTEMPT_ID" \
            "$RUN_ATTEMPT_DIR" <<'PY'
import re
import sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
keys = (
    "schema", "state", "run_id", "attempt_id", "run_attempt_dir",
    "owner_host", "owner_pid", "created_epoch", "job_id",
)
if len(lines) != len(keys):
    raise SystemExit("FATAL: active-attempt lease field count changed")
fields = {}
for key, line in zip(keys, lines, strict=True):
    prefix = key + "="
    if not line.startswith(prefix):
        raise SystemExit("FATAL: active-attempt lease field order changed")
    fields[key] = line[len(prefix):]
expected = {
    "schema": "fast_roborender.hsg_active_attempt.v1",
    "run_id": sys.argv[2],
    "attempt_id": sys.argv[3],
    "run_attempt_dir": sys.argv[4],
}
for key, value in expected.items():
    if fields.get(key) != value:
        raise SystemExit(f"FATAL: active-attempt lease {key} mismatch")
if fields["state"] not in {"submitting", "submitted"}:
    raise SystemExit("FATAL: active-attempt lease state is invalid")
if fields["state"] == "submitting" and fields["job_id"] != "NONE":
    raise SystemExit("FATAL: unbound active-attempt lease has a job ID")
if fields["state"] == "submitted" and re.fullmatch(r"[1-9][0-9]*", fields["job_id"]) is None:
    raise SystemExit("FATAL: bound active-attempt job ID is malformed")
print(f'{fields["state"]}\t{fields["job_id"]}')
PY
        ) || fatal "production active-attempt lease validation failed"
        IFS=$'\t' read -r ACTIVE_LEASE_STATE ACTIVE_LEASE_JOB <<< "$ACTIVE_LEASE_BINDING"
        if [ "$ACTIVE_LEASE_STATE" = submitted ]; then
            [ "$ACTIVE_LEASE_JOB" = "$SLURM_JOB_ID" ] || \
                fatal "production active-attempt lease is bound to another Slurm job"
            break
        fi
        [ "$SECONDS" -lt "$ACTIVE_LEASE_DEADLINE" ] || \
            fatal "timed out waiting for submission to bind the active-attempt lease"
        sleep 1
    done
else
    [ "$EXPECTED_RUN_IDENTITY_SHA256" = NONE ] || \
        fatal "non-production profile received a run identity hash"
fi

mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[ "${#ALLOCATED_NODES[@]}" -eq "$PACKED_NODES" ] || fatal "cannot resolve allocated nodes"
TRAINER_START=$((PACKED_NODES - TRAINER_RANKS))
[ "$TRAINER_START" -ge 0 ] || fatal "trainer ranks exceed allocated nodes"
TRAINER_NODES=("${ALLOCATED_NODES[@]:TRAINER_START:TRAINER_RANKS}")
TRAINER_NODELIST=$(IFS=,; echo "${TRAINER_NODES[*]}")
MASTER_ADDR=${TRAINER_NODES[0]}
MASTER_PORT=$((15000 + SLURM_JOB_ID % 20000))
LOCAL_NUM_ENVS=$((GLOBAL_NUM_ENVS / TRAINER_RANKS))
RENDERER_PLACEMENT_PLAN=$RUN_ATTEMPT_DIR/renderer_placement_plan.json
python3 - "$RENDERER_PLACEMENT_PLAN" "$EXPECTED_RENDERERS" \
    "${ALLOCATED_NODES[@]}" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

destination = Path(sys.argv[1])
renderer_count = int(sys.argv[2])
nodes = sys.argv[3:]
base, remainder = divmod(renderer_count, len(nodes))
host_counts = {
    node: base + (index < remainder)
    for index, node in enumerate(nodes)
    if base + (index < remainder) > 0
}
payload = {
    "schema": "fast_roborender.renderer_placement_plan.v1",
    "renderer_count": renderer_count,
    "host_counts": host_counts,
}
destination.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
PREFLIGHT_ARTIFACT_DIR=$RUN_ROOT
TRAIN_LOG_DIR=$TRAIN_RUN_DIR/slurm
TRAIN_HYDRA_DIR=$TRAIN_RUN_DIR/.hydra
TRAIN_OUTPUT_DIR=$TRAIN_RUN_DIR/output
SCALE_PLAN_PATH=$TRAIN_RUN_DIR/scale_plan.json
WANDB_WATCHER_LOG=$TRAIN_RUN_DIR/wandb_media_watcher.log
POSTRUN_GATE_PATH=$TRAIN_RUN_DIR/postrun_gate.json
if [ "$RUN_PROFILE" = production_256x300 ]; then
    PREFLIGHT_ARTIFACT_DIR=$RUN_ATTEMPT_DIR
    TRAIN_LOG_DIR=$RUN_ATTEMPT_DIR/trainer_logs
    TRAIN_HYDRA_DIR=$RUN_ATTEMPT_DIR/hydra
    TRAIN_OUTPUT_DIR=$RUN_ATTEMPT_DIR/output
    SCALE_PLAN_PATH=$RUN_ATTEMPT_DIR/scale_plan.json
    WANDB_WATCHER_LOG=$RUN_ATTEMPT_DIR/wandb_media_watcher.log
    POSTRUN_GATE_PATH=$RUN_ATTEMPT_DIR/postrun_gate.json
fi

# Validate the exact encoder in the sealed trainer before starting renderer
# servers or importing Isaac/GR00T.  IMAGE_DAGGER_FFMPEG is carried through the
# sanitized sbatch environment and is also passed explicitly at this boundary,
# so a missing Slurm export cannot silently fall back to PATH.
FFMPEG_PREFLIGHT="$PREFLIGHT_ARTIFACT_DIR/trainer_ffmpeg_preflight.txt"
srun --nodes=1 --ntasks=1 --nodelist="$MASTER_ADDR" \
    --gpus=0 --cpus-per-task=2 --mem=4G --exact --exclusive \
    --container-image="$TRAINER_CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$IMMUTABLE_RO_MOUNTS" \
    env IMAGE_DAGGER_FFMPEG="$IMAGE_DAGGER_FFMPEG" \
    /usr/local/bin/roborender-trainer-sm100 bash \
    "$ROOT/scripts/scaling/check_image_dagger_ffmpeg.sh" \
    | tee "$FFMPEG_PREFLIGHT"
grep -Fx "IMAGE_DAGGER_FFMPEG=$IMAGE_DAGGER_FFMPEG" "$FFMPEG_PREFLIGHT" >/dev/null || \
    fatal "trainer FFmpeg preflight did not attest the selected encoder"

mkdir -p "$FLEET_DIR/endpoints" "$FLEET_DIR/queues" "$FLEET_DIR/logs" \
         "$TRAIN_RUN_DIR/slurm" "$TRAIN_RUN_DIR/output" "$TRAIN_RUN_DIR/telemetry" \
         "$TRAIN_LOG_DIR" "$TRAIN_HYDRA_DIR" "$TRAIN_OUTPUT_DIR" \
         "$STAGE_ROOT/runtime/home" "$STAGE_ROOT/runtime/cache" "$STAGE_ROOT/runtime/tmp"

[ -d "$RUN_ATTEMPT_DIR" ] && [ ! -L "$RUN_ATTEMPT_DIR" ] || \
    fatal "run attempt directory is missing or unsafe"
[ -d "$FLEET_DIR" ] && [ ! -L "$FLEET_DIR" ] || \
    fatal "renderer fleet directory is missing or unsafe"
RUN_ATTEMPT_CANONICAL=$(realpath -e "$RUN_ATTEMPT_DIR")
FLEET_CANONICAL=$(realpath -e "$FLEET_DIR")
[ "$FLEET_CANONICAL" = "$RUN_ATTEMPT_CANONICAL/fleet" ] || \
    fatal "renderer fleet directory is not attempt-scoped"
FLEET_STOP_FILE=$FLEET_CANONICAL/STOP
RENDERER_CLEAN_STOP_EVIDENCE=$RUN_ATTEMPT_CANONICAL/renderer_clean_stop.json
[ ! -e "$FLEET_STOP_FILE" ] && [ ! -L "$FLEET_STOP_FILE" ] || \
    fatal "attempt-scoped renderer STOP sentinel already exists or is unsafe"
[ ! -e "$RENDERER_CLEAN_STOP_EVIDENCE" ] && \
[ ! -L "$RENDERER_CLEAN_STOP_EVIDENCE" ] || \
    fatal "attempt-scoped renderer clean-stop evidence already exists or is unsafe"

if [ "$USE_WANDB" -eq 1 ]; then
    [ -s "$WANDB_ONLINE_VALIDATION" ] || fatal "W&B online validation disappeared after submission"
    [ "$(sha256sum "$WANDB_ONLINE_VALIDATION" | awk '{print $1}')" = \
      "$EXPECTED_WANDB_ONLINE_VALIDATION_SHA256" ] || \
        fatal "W&B online validation changed after submission"
    ACCEPTED_WANDB_ENTITY=$(python3 - "$WANDB_ONLINE_VALIDATION" "$WANDB_EXPECTED_EMAIL" "$WANDB_ENTITY" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "schema": "fast_roborender.wandb_online_smoke.v2",
    "status": "PASS",
    "account_email": sys.argv[2],
    "entity": sys.argv[3],
    "authorization_scope": "team",
    "team_member": True,
    "entity_is_team": True,
    "personal": False,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"W&B validation {key} mismatch")
required = {"validation/scalar", "validation/image", "validation/video"}
if not required.issubset(payload.get("observed_keys", [])):
    raise SystemExit("W&B validation media readback is incomplete")
print(payload["entity"])
PY
    ) || fatal "W&B online validation is invalid"
    [ -n "$WANDB_ENTITY" ] && [ "$WANDB_ENTITY" = "$ACCEPTED_WANDB_ENTITY" ] || \
        fatal "W&B entity is not bound to the accepted authorized-team write"
    WANDB_PREFLIGHT=(python "$ROOT/scripts/scaling/wandb_preflight.py"
        --expected-email "$WANDB_EXPECTED_EMAIL" --entity "$WANDB_ENTITY" \
        --authorization-scope team --format shell)
    WANDB_PREFLIGHT_OUTPUT=$(srun --nodes=1 --ntasks=1 --nodelist="$MASTER_ADDR" \
        --gpus=0 --cpus-per-task=2 --mem=4G --exact --exclusive \
        --container-image="$TRAINER_CONTAINER" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$IMMUTABLE_RO_MOUNTS,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
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
CLEANUP_DEADLINE=0
IN_CLEANUP=0
TERMINATION_GRACE_S=20
COOPERATIVE_STOP_GRACE_S=90
CLEANUP_TOTAL_BUDGET_S=240

terminate_pid_bounded() {
    local pid=$1 label=$2 deadline now forced_termination=0
    [ -n "$pid" ] || return 0
    kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
    kill -TERM "$pid" 2>/dev/null || true
    deadline=$((SECONDS + TERMINATION_GRACE_S))
    if [ "$CLEANUP_DEADLINE" -gt 0 ] && [ "$deadline" -gt "$CLEANUP_DEADLINE" ]; then
        deadline=$CLEANUP_DEADLINE
    fi
    while kill -0 "$pid" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: $label ignored TERM; sending KILL after ${TERMINATION_GRACE_S}s" >&2
        forced_termination=1
        kill -KILL "$pid" 2>/dev/null || true
        now=$((SECONDS + 5))
        if [ "$CLEANUP_DEADLINE" -gt 0 ] && [ "$now" -gt "$CLEANUP_DEADLINE" ]; then
            now=$CLEANUP_DEADLINE
        fi
        while kill -0 "$pid" 2>/dev/null && [ "$SECONDS" -lt "$now" ]; do
            sleep 1
        done
    fi
    if kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: $label remains live after bounded TERM/KILL" >&2
        return 73
    fi
    wait "$pid" 2>/dev/null || true
    [ "$forced_termination" -eq 0 ] || return 74
}

stop_renderer_fleet_cleanly() {
    local pid=$1 deadline renderer_rc endpoint_count ready_count ack_count
    [ -n "$pid" ] || {
        echo "ERROR: renderer fleet PID is missing before STOP" >&2
        return 74
    }
    if ! kill -0 "$pid" 2>/dev/null; then
        if wait "$pid" 2>/dev/null; then renderer_rc=0; else renderer_rc=$?; fi
        echo "ERROR: renderer srun exited before STOP request (rc=$renderer_rc)" >&2
        return 74
    fi
    [ -n "$FLEET_STOP_FILE" ] && [ -n "$RENDERER_CLEAN_STOP_EVIDENCE" ] || {
        echo "ERROR: renderer clean-stop paths are unset" >&2
        return 74
    }
    [ ! -e "$FLEET_STOP_FILE" ] && [ ! -L "$FLEET_STOP_FILE" ] || {
        echo "ERROR: renderer STOP sentinel already exists or is unsafe" >&2
        return 74
    }
    [ ! -e "$RENDERER_CLEAN_STOP_EVIDENCE" ] && \
    [ ! -L "$RENDERER_CLEAN_STOP_EVIDENCE" ] || {
        echo "ERROR: renderer clean-stop evidence already exists or is unsafe" >&2
        return 74
    }
    python3 - "$FLEET_STOP_FILE" "$RUN_ID" "$ATTEMPT_ID" "$SLURM_JOB_ID" <<'PY' || return 74
import json
import os
import sys
import tempfile
from pathlib import Path

destination = Path(sys.argv[1])
if destination.exists() or destination.is_symlink():
    raise SystemExit("renderer STOP sentinel already exists or is unsafe")
payload = {
    "schema": "fast_roborender.renderer_stop_request.v1",
    "run_id": sys.argv[2],
    "attempt_id": sys.argv[3],
    "job_id": sys.argv[4],
}
descriptor, temporary = tempfile.mkstemp(
    prefix=f".{destination.name}.", dir=destination.parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
    deadline=$((SECONDS + COOPERATIVE_STOP_GRACE_S))
    while kill -0 "$pid" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: renderer srun did not honor cooperative STOP within ${COOPERATIVE_STOP_GRACE_S}s" >&2
        return 74
    fi
    if wait "$pid"; then renderer_rc=0; else renderer_rc=$?; fi
    [ "$renderer_rc" -eq 0 ] || {
        echo "ERROR: renderer srun exited nonzero after STOP (rc=$renderer_rc)" >&2
        return 74
    }
    endpoint_count=$(find "$FLEET_DIR/endpoints" -mindepth 1 -maxdepth 1 -printf . | wc -c)
    ready_count=$(find "$FLEET_DIR/queues" -name READY -printf . | wc -c)
    ack_count=$(find "$FLEET_DIR/stop_acks" -mindepth 1 -maxdepth 1 -printf . | wc -c)
    [ "$endpoint_count" -eq 0 ] && [ "$ready_count" -eq 0 ] || {
        echo "ERROR: renderer STOP left endpoint=$endpoint_count READY=$ready_count markers" >&2
        return 74
    }
    [ "$ack_count" -eq "$EXPECTED_RENDERERS" ] || {
        echo "ERROR: renderer STOP ACK count $ack_count != $EXPECTED_RENDERERS" >&2
        return 74
    }
    python3 "$ROOT/scripts/scaling/hsg_gb200_smoke_gate.py" renderer-clean-stop \
        --profile "$RUN_PROFILE" --run-id "$RUN_ID" --attempt-id "$ATTEMPT_ID" \
        --job-id "$SLURM_JOB_ID" --fleet-dir "$FLEET_DIR" \
        --evidence "$RENDERER_CLEAN_STOP_EVIDENCE" \
        --expected-renderers "$EXPECTED_RENDERERS" \
        --renderer-srun-exit-code "$renderer_rc" \
        --srun-was-live-before-stop >/dev/null || return 74
    [ -s "$RENDERER_CLEAN_STOP_EVIDENCE" ] && \
    [ ! -L "$RENDERER_CLEAN_STOP_EVIDENCE" ] || return 74
}

drain_wandb_watcher() {
    [ "$USE_WANDB" -eq 1 ] || return 0
    [ -n "$WANDB_WATCHER_PID" ] || return 0

    : > "$TRAIN_RUN_DIR/WANDB_UPLOAD_STOP"
    local drain_budget=$WANDB_WATCHER_DRAIN_TIMEOUT_S
    if [ "$IN_CLEANUP" -eq 1 ] && [ "$drain_budget" -gt 90 ]; then
        drain_budget=90
    fi
    local deadline=$((SECONDS + drain_budget))
    if [ "$CLEANUP_DEADLINE" -gt 0 ] && [ "$deadline" -gt $((CLEANUP_DEADLINE - 30)) ]; then
        deadline=$((CLEANUP_DEADLINE - 30))
    fi
    while kill -0 "$WANDB_WATCHER_PID" 2>/dev/null && \
          [ "$SECONDS" -lt "$deadline" ]; do
        sleep 1
    done
    if kill -0 "$WANDB_WATCHER_PID" 2>/dev/null; then
        echo "ERROR: W&B media watcher did not finish its bounded final drain" >&2
        terminate_pid_bounded "$WANDB_WATCHER_PID" "W&B media watcher" || true
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

write_resume_candidate() {
    [ "$RUN_PROFILE" = production_256x300 ] || return 0
    local basename checkpoint step digest temporary hash_budget remaining
    basename=$(find "$TRAIN_RUN_DIR" -maxdepth 1 -type f \
        -name 'model_step_[0-9][0-9][0-9][0-9][0-9][0-9].pt' \
        -printf '%f\n' | LC_ALL=C sort | tail -n 1)
    [ -n "$basename" ] || return 3
    if [[ ! "$basename" =~ ^model_step_([0-9]{6})\.pt$ ]]; then
        echo "ERROR: latest production checkpoint name is malformed" >&2
        return 71
    fi
    step=$((10#${BASH_REMATCH[1]}))
    if [ "$step" -le 0 ] || [ "$step" -gt "$NUM_ROLLOUT_ITERATIONS" ] || \
       [ $((step % SAVE_FREQUENCY)) -ne 0 ]; then
        echo "ERROR: latest production checkpoint violates the save contract" >&2
        return 71
    fi
    checkpoint=$TRAIN_RUN_DIR/$basename
    if [ ! -f "$checkpoint" ] || [ -L "$checkpoint" ]; then
        echo "ERROR: latest production checkpoint is unsafe" >&2
        return 71
    fi
    hash_budget=45
    if [ "$CLEANUP_DEADLINE" -gt 0 ]; then
        remaining=$((CLEANUP_DEADLINE - SECONDS - 20))
        [ "$remaining" -gt 0 ] || return 71
        [ "$hash_budget" -le "$remaining" ] || hash_budget=$remaining
    fi
    digest=$(/usr/bin/timeout --signal=TERM "${hash_budget}s" \
        sha256sum "$checkpoint" | awk '{print $1}') || return 71
    temporary=$RUN_ATTEMPT_DIR/.resume_candidate.txt.$$
    cat > "$temporary" <<EOF
schema=fast_roborender.production_resume_candidate.v1
status=checkpoint_available
run_id=$RUN_ID
run_root=$RUN_ROOT
source_attempt_id=$ATTEMPT_ID
checkpoint=$checkpoint
checkpoint_step=$step
checkpoint_sha256=$digest
absolute_target=$NUM_ROLLOUT_ITERATIONS
save_frequency=$SAVE_FREQUENCY
run_identity_sha256=$EXPECTED_RUN_IDENTITY_SHA256
EOF
    mv "$temporary" "$RUN_ATTEMPT_DIR/resume_candidate.txt"
}

release_active_attempt() {
    [ "$RUN_PROFILE" = production_256x300 ] || return 0
    local lease destination
    lease=$RUN_ROOT/.active_attempt/lease.txt
    [ -f "$lease" ] && [ ! -L "$lease" ] || {
        echo "ERROR: production active-attempt lease disappeared before terminal release" >&2
        return 72
    }
    python3 - "$lease" "$RUN_ID" "$ATTEMPT_ID" "$SLURM_JOB_ID" <<'PY'
import sys
from pathlib import Path

fields = dict(
    line.split("=", 1)
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if "=" in line
)
expected = {
    "schema": "fast_roborender.hsg_active_attempt.v1",
    "state": "submitted",
    "run_id": sys.argv[2],
    "attempt_id": sys.argv[3],
    "job_id": sys.argv[4],
}
for key, value in expected.items():
    if fields.get(key) != value:
        raise SystemExit(f"active-attempt terminal release {key} mismatch")
PY
    destination=$RUN_ATTEMPT_DIR/active_attempt_terminal
    [ ! -e "$destination" ] || {
        echo "ERROR: production active-attempt terminal archive already exists" >&2
        return 72
    }
    mv "$RUN_ROOT/.active_attempt" "$destination" || return 72
}

cleanup() {
    local rc=$?
    local watcher_rc=0
    local process_rc=0
    local candidate_status=0
    trap - EXIT TERM INT USR1
    IN_CLEANUP=1
    if [ "$CLEANUP_DEADLINE" -eq 0 ]; then
        CLEANUP_DEADLINE=$((SECONDS + CLEANUP_TOTAL_BUDGET_S))
    fi
    # Publish the failure state and best durable checkpoint before waiting on
    # any process or network service.  A scheduler hard-kill can then never
    # erase the resume/validation-only recovery path.
    if [ "$rc" -ne 0 ] && [ "$RUN_PROFILE" = production_256x300 ]; then
        printf '%s\n' "$rc" > "$RUN_ATTEMPT_DIR/ATTEMPT_FAILED"
        if write_resume_candidate; then
            : > "$RUN_ATTEMPT_DIR/RESUME_CANDIDATE_READY"
        else
            candidate_status=$?
            if [ "$candidate_status" -eq 3 ]; then
                : > "$RUN_ATTEMPT_DIR/NO_DURABLE_CHECKPOINT"
            else
                : > "$RUN_ATTEMPT_DIR/RESUME_CANDIDATE_FAILED"
            fi
        fi
    fi
    for pid in "$TRAIN_STEP_PID" "$RENDER_STEP_PID"; do
        if ! terminate_pid_bounded "$pid" "Slurm job step"; then
            process_rc=73
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
    if [ "$rc" -eq 0 ] && [ "$process_rc" -ne 0 ]; then
        rc=$process_rc
    fi
    if [ "$RUN_PROFILE" = production_256x300 ]; then
        if write_resume_candidate; then
            : > "$RUN_ATTEMPT_DIR/RESUME_CANDIDATE_READY"
        else
            candidate_status=$?
            if [ "$candidate_status" -eq 3 ]; then
                : > "$RUN_ATTEMPT_DIR/NO_DURABLE_CHECKPOINT"
            else
                : > "$RUN_ATTEMPT_DIR/RESUME_CANDIDATE_FAILED"
                [ "$rc" -ne 0 ] || rc=71
            fi
        fi
    fi
    if ! release_active_attempt && [ "$rc" -eq 0 ]; then
        rc=72
    fi
    if [ "$rc" -eq 0 ]; then
        : > "$RUN_ATTEMPT_DIR/ATTEMPT_DONE"
        : > "$RUN_ROOT/RUN_DONE"
    elif [ "$RUN_PROFILE" = production_256x300 ]; then
        printf '%s\n' "$rc" > "$RUN_ATTEMPT_DIR/ATTEMPT_FAILED"
    else
        printf '%s\n' "$rc" > "$RUN_ROOT/RUN_FAILED"
    fi
    exit "$rc"
}
pre_timeout() {
    local candidate_status=0
    : > "$RUN_ATTEMPT_DIR/PRE_TIMEOUT_WARNING"
    CLEANUP_DEADLINE=$((SECONDS + CLEANUP_TOTAL_BUDGET_S))
    if [ "$RUN_PROFILE" = production_256x300 ]; then
        printf '124\n' > "$RUN_ATTEMPT_DIR/ATTEMPT_FAILED"
    fi
    [ -z "$TRAIN_STEP_PID" ] || kill -TERM "$TRAIN_STEP_PID" 2>/dev/null || true
    if [ "$RUN_PROFILE" = production_256x300 ]; then
        if write_resume_candidate; then
            : > "$RUN_ATTEMPT_DIR/RESUME_CANDIDATE_READY"
        else
            candidate_status=$?
            if [ "$candidate_status" -eq 3 ]; then
                : > "$RUN_ATTEMPT_DIR/NO_DURABLE_CHECKPOINT"
            else
                : > "$RUN_ATTEMPT_DIR/RESUME_CANDIDATE_FAILED"
            fi
        fi
    fi
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
# This digest vector is derived inside the allocation from the trusted
# pre-resume snapshot.  Never accept an inherited value for fresh or
# validation-only launches.
unset EXPECTED_IMAGE_DAGGER_RUNTIME_SHA256S

if [ "$RUN_PROFILE" = production_256x300 ] && [ "$RESUME_TRAINING" = 1 ]; then
    PRE_RESUME_SNAPSHOT=$RUN_ATTEMPT_DIR/pre_resume_snapshot.json
    srun --nodes=1 --ntasks=1 --nodelist="$MASTER_ADDR" \
        --gpus=0 --cpus-per-task=2 --mem=4G --exact --exclusive \
        --container-image="$TRAINER_CONTAINER" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$IMMUTABLE_RO_MOUNTS,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
        --container-workdir=/opt/GR00T-VisualSim2Real \
        /usr/local/bin/roborender-trainer-sm100 python \
        "$ROOT/scripts/scaling/resume_continuity_gate.py" snapshot \
        --checkpoint "$CKPT" --run-dir "$TRAIN_RUN_DIR" \
        --expected-world-size "$TRAINER_RANKS" --wandb-mode required \
        --expected-checkpoint-sha256 "$EXPECTED_TRAIN_CHECKPOINT_SHA256" \
        --output "$PRE_RESUME_SNAPSHOT"
    EXPECTED_IMAGE_DAGGER_RUNTIME_SHA256S=$(python3 - \
        "$PRE_RESUME_SNAPSHOT" "$RESUME_CHECKPOINT_STEP" \
        "$NUM_ROLLOUT_ITERATIONS" "$EXPECTED_TRAIN_CHECKPOINT_SHA256" \
        "$TRAINER_RANKS" <<'PY'
import json
import re
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "schema": "fast_roborender.resume_continuity.snapshot.v1",
    "checkpoint_step": int(sys.argv[2]),
    "absolute_target": int(sys.argv[3]),
    "checkpoint_sha256": sys.argv[4],
    "wandb_required": True,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"FATAL: production resume snapshot {key} changed")
for key in ("trainer_wandb_run_id", "sidecar_wandb_run_id"):
    if not payload.get(key):
        raise SystemExit(f"FATAL: production resume snapshot has no {key}")
world_size = int(sys.argv[5])
ranks = payload.get("runtime_ranks")
if not isinstance(ranks, list) or len(ranks) != world_size:
    raise SystemExit("FATAL: production resume snapshot has incomplete runtime ranks")
digests = []
for expected_rank, record in enumerate(ranks):
    if not isinstance(record, dict) or type(record.get("rank")) is not int:
        raise SystemExit("FATAL: production resume runtime rank record is malformed")
    if record["rank"] != expected_rank:
        raise SystemExit("FATAL: production resume runtime ranks are not ordered")
    digest = record.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise SystemExit("FATAL: production resume runtime rank hash is malformed")
    digests.append(digest)
print(",".join(digests))
PY
    )
    export EXPECTED_IMAGE_DAGGER_RUNTIME_SHA256S
fi

EP_LIST="[]"
srun --nodes="$PACKED_NODES" --ntasks="$EXPECTED_RENDERERS" --ntasks-per-node=4 \
    --distribution=cyclic --gpus="$EXPECTED_RENDERERS" --gpus-per-task=1 \
    --cpus-per-task=8 --mem=512G \
    --exact --exclusive --gpu-bind=closest --cpu-bind=cores --kill-on-bad-exit=1 \
    --container-image="$RENDERER_CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$IMMUTABLE_RO_MOUNTS,$ROBORENDER_MODEL_BASE:/models:ro,$IMAGE3F_GENERATED_ROOT:/opt/roborender/assets/generated:ro" \
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
    --expected-engine-sha256 "$EXPECTED_IMAGE3F_ENGINE_SHA256" \
    --expected-inference-steps 5 --expected-teacache-threshold 0.3 \
    --expected-cfg-scale 1.0 --expected-cuda-graph disabled \
    --expected-output-views head_color,hand_left,hand_right \
    --expected-output-layout vertical --expected-output-tile-shape 240,416,3 \
    --expected-output-stacked-shape 720,416,3 --require-tensorrt-engine \
    --expected-host-counts "$RENDERER_PLACEMENT_PLAN" \
    --write-placement-report "$TRAIN_RUN_DIR/validation/renderer_placement.json" \
    --write-manifest "$FLEET_DIR/selected_endpoints.txt" \
    --write-report "$FLEET_DIR/renderer_capabilities.json" --hydra-list)

TRAIN_ABSOLUTE_TARGET=$NUM_ROLLOUT_ITERATIONS
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
python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --format json > "$SCALE_PLAN_PATH"

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
# These paths are selected inside this batch process, after sbatch's sanitized
# environment is created.  The trainer command runs in a nested `srun` shell,
# so they must be environment variables rather than shell-only assignments.
export TRAIN_HYDRA_DIR TRAIN_OUTPUT_DIR
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
export IMAGE_DAGGER_EXPECT_ENFORCE_TEACHER=false
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
if [ "$RUN_PROFILE" = production_256x300 ]; then
    PROJ=roborender_dagger_hsg_gb200_production
fi
export EXP="${RUN_PROFILE}_${RUN_ID}"
export TRAIN_ABSOLUTE_TARGET

WANDB_WATCHER_READY="$TRAIN_RUN_DIR/telemetry/WANDB_WATCHER_READY.json"
if [ "$USE_WANDB" -eq 1 ]; then
    # Preserve the durable sidecar run ID and upload-state files across a
    # restarted process.  Only per-process lifecycle markers are disposable.
    rm -f "$WANDB_WATCHER_READY" "$TRAIN_RUN_DIR/WANDB_UPLOAD_STOP"
    srun --nodes=1 --ntasks=1 --nodelist="$MASTER_ADDR" \
        --gpus=0 --cpus-per-task=2 --mem=4G --exact --exclusive \
        --container-image="$TRAINER_CONTAINER" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$IMMUTABLE_RO_MOUNTS,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
        --container-workdir=/opt/GR00T-VisualSim2Real \
        env HOME="$STAGE_ROOT/runtime/home" WANDB_MODE="$WANDB_MODE" \
        /usr/local/bin/roborender-trainer-sm100 python \
        "$ROOT/scripts/scaling/wandb_run_watcher.py" \
        --run-dir "$TRAIN_RUN_DIR" --project "$PROJ" --name "${EXP}-media" \
        --expected-email "$WANDB_EXPECTED_EMAIL" --entity "$WANDB_ENTITY" \
        --authorization-scope team \
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
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$IMMUTABLE_RO_MOUNTS,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
        --container-workdir=/opt/GR00T-VisualSim2Real \
        --output="$TRAIN_LOG_DIR/${phase}_rank_%t.out" \
        --error="$TRAIN_LOG_DIR/${phase}_rank_%t.err" \
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
                algo.config.enforce_teacher_rollout=false algo.config.ratio_teacher_rollout=0.0 \
                algo.config.value_loss_coef=0.0 algo.config.entropy_coef=0.0 \
                algo.config.videogen.rollout_action_source=beta \
                ++algo.config.videogen.beta_schedule=linear ++algo.config.videogen.beta_start=1.0 \
                ++algo.config.videogen.beta_end=0.0 ++algo.config.videogen.beta_decay_iters=500 \
                algo.trl.per_device_train_batch_size=1 \
                algo.trl.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
                algo.trl.num_total_batches="$TRAIN_ABSOLUTE_TARGET" algo.trl.total_episodes=null \
                callbacks.model_save.save_frequency="$SAVE_FREQUENCY" callbacks.model_save.save_initial_checkpoint=true \
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
                hydra.run.dir="$TRAIN_HYDRA_DIR/${TRAIN_PHASE}_rank_${SLURM_PROCID}" \
                output_dir="$TRAIN_OUTPUT_DIR/rank_${SLURM_PROCID}"
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
            terminate_pid_bounded "$TRAIN_STEP_PID" "trainer after W&B failure" || true
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
    local checkpoint=$1 output=$2 expected_world_size=${3:-1} wandb_mode=${4:-disabled}
    local expected_checkpoint_sha256=${5:-NONE}
    srun --nodes=1 --ntasks=1 --nodelist="$MASTER_ADDR" \
        --gpus=0 --cpus-per-task=2 --mem=4G --exact --exclusive \
        --container-image="$TRAINER_CONTAINER" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$IMMUTABLE_RO_MOUNTS,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
        --container-workdir=/opt/GR00T-VisualSim2Real \
        /usr/local/bin/roborender-trainer-sm100 python \
        "$ROOT/scripts/scaling/resume_continuity_gate.py" snapshot \
        --checkpoint "$checkpoint" --run-dir "$TRAIN_RUN_DIR" \
        --expected-world-size "$expected_world_size" --wandb-mode "$wandb_mode" \
        --expected-checkpoint-sha256 "$expected_checkpoint_sha256" \
        --output "$output"
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
elif [ "$RUN_PROFILE" = production_256x300 ]; then
    if [ "$PRODUCTION_ACTION" = validate ]; then
        : # Finalize the already durable step-500 checkpoint without retraining.
    elif [ "$RESUME_TRAINING" = 1 ]; then
        run_train_phase production_resume "$CKPT" 1 0
    else
        run_train_phase production_fresh null 0 0
    fi
else
    run_train_phase single null 0 0
fi

if [ -n "$RENDER_STEP_PID" ]; then
    stop_renderer_fleet_cleanly "$RENDER_STEP_PID" || \
        fatal "renderer fleet did not complete the cooperative clean-stop contract"
fi
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
    WANDB_TRAINING_READBACK=$TRAIN_RUN_DIR/validation/wandb_training_readback.json
    srun --nodes=1 --ntasks=1 --nodelist="$MASTER_ADDR" \
        --gpus=0 --cpus-per-task=2 --mem=4G --exact --exclusive \
        --container-image="$TRAINER_CONTAINER" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$IMMUTABLE_RO_MOUNTS,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
        --container-workdir=/opt/GR00T-VisualSim2Real \
        env HOME="$STAGE_ROOT/runtime/home" WANDB_MODE=online \
        /usr/local/bin/roborender-trainer-sm100 python \
        "$ROOT/scripts/scaling/wandb_training_readback.py" \
        --run-dir "$TRAIN_RUN_DIR" --project "$PROJ" --entity "$WANDB_ENTITY" \
        --expected-email "$WANDB_EXPECTED_EMAIL" --authorization-scope team \
        --rollout-steps "$ROLLOUT_STEPS" \
        --resume-training "$RESUME_TRAINING" \
        --resume-checkpoint-step "$([ "$RESUME_TRAINING" -eq 1 ] && \
            echo "$RESUME_CHECKPOINT_STEP" || echo 0)" \
        --timeout-s 180 \
        --output "$WANDB_TRAINING_READBACK"
fi

if [ "$RUN_PROFILE" = production_256x300 ]; then
    FINAL_PRODUCTION_CHECKPOINT=$TRAIN_RUN_DIR/model_step_000500.pt
    [ -f "$FINAL_PRODUCTION_CHECKPOINT" ] && [ ! -L "$FINAL_PRODUCTION_CHECKPOINT" ] || \
        fatal "final production checkpoint is missing or unsafe"
    FINAL_PRODUCTION_CHECKPOINT_SHA256=$(sha256sum "$FINAL_PRODUCTION_CHECKPOINT" | awk '{print $1}')
    [[ "$FINAL_PRODUCTION_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
        fatal "could not pin the final production checkpoint SHA-256"
    if [ "$PRODUCTION_ACTION" = validate ]; then
        [ "$FINAL_PRODUCTION_CHECKPOINT_SHA256" = "$EXPECTED_TRAIN_CHECKPOINT_SHA256" ] || \
            fatal "validate-only final checkpoint changed before final deserialization"
    fi
    FINAL_CHECKPOINT_SNAPSHOT=$RUN_ATTEMPT_DIR/final_checkpoint_snapshot.json
    resume_snapshot "$FINAL_PRODUCTION_CHECKPOINT" \
        "$FINAL_CHECKPOINT_SNAPSHOT" "$TRAINER_RANKS" required \
        "$FINAL_PRODUCTION_CHECKPOINT_SHA256"
    if [ "$RESUME_TRAINING" = 1 ]; then
        RESUME_CONTINUITY_REPORT=$RUN_ATTEMPT_DIR/resume_continuity_report.json
        python3 "$ROOT/scripts/scaling/resume_continuity_gate.py" verify \
            --before "$PRE_RESUME_SNAPSHOT" --after "$FINAL_CHECKPOINT_SNAPSHOT" \
            --expected-before-step "$RESUME_CHECKPOINT_STEP" --expected-after-step 500 \
            --rollout-steps "$ROLLOUT_STEPS" --expected-world-size "$TRAINER_RANKS" \
            --wandb-mode required --output "$RESUME_CONTINUITY_REPORT"
    fi
fi

POLICY_CHANGE_GATE=$TRAIN_RUN_DIR/validation/checkpoint_policy_change.json
INITIAL_POLICY_CHECKPOINT=$TRAIN_RUN_DIR/initial_policy.pt
FINAL_POLICY_CHECKPOINT=$TRAIN_RUN_DIR/model_step_$(printf '%06d' "$TRAIN_ABSOLUTE_TARGET").pt
[ -f "$INITIAL_POLICY_CHECKPOINT" ] && [ ! -L "$INITIAL_POLICY_CHECKPOINT" ] || \
    fatal "initial policy checkpoint is missing or unsafe"
[ -f "$FINAL_POLICY_CHECKPOINT" ] && [ ! -L "$FINAL_POLICY_CHECKPOINT" ] || \
    fatal "final policy checkpoint is missing or unsafe"
INITIAL_POLICY_CHECKPOINT_SHA256=$(sha256sum "$INITIAL_POLICY_CHECKPOINT" | awk '{print $1}')
FINAL_POLICY_CHECKPOINT_SHA256=$(sha256sum "$FINAL_POLICY_CHECKPOINT" | awk '{print $1}')
[[ "$INITIAL_POLICY_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
    fatal "could not pin the initial policy checkpoint SHA-256"
[[ "$FINAL_POLICY_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
    fatal "could not pin the final policy checkpoint SHA-256"
if [ "$RUN_PROFILE" = production_256x300 ]; then
    [ "$FINAL_POLICY_CHECKPOINT_SHA256" = "$FINAL_PRODUCTION_CHECKPOINT_SHA256" ] || \
        fatal "final checkpoint changed between resumability and policy gates"
fi
srun --nodes=1 --ntasks=1 --nodelist="$MASTER_ADDR" \
    --gpus=0 --cpus-per-task=2 --mem=4G --exact --exclusive \
    --container-image="$TRAINER_CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$IMMUTABLE_RO_MOUNTS,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
    --container-workdir=/opt/GR00T-VisualSim2Real \
    /usr/local/bin/roborender-trainer-sm100 python \
    "$ROOT/scripts/scaling/checkpoint_policy_gate.py" \
    --initial "$INITIAL_POLICY_CHECKPOINT" \
    --final "$FINAL_POLICY_CHECKPOINT" \
    --final-step "$TRAIN_ABSOLUTE_TARGET" \
    --expected-initial-sha256 "$INITIAL_POLICY_CHECKPOINT_SHA256" \
    --expected-final-sha256 "$FINAL_POLICY_CHECKPOINT_SHA256" \
    --output "$POLICY_CHANGE_GATE"

# The sources were mounted read-only into every container step, but another
# host could still mutate a shared Lustre snapshot.  Re-attest both trees at
# the result boundary so postrun/RUN_DONE can never bless a mid-run change.
FINAL_FAST_SOURCE_TREE_SHA256=$(python3 \
    "$ROOT/scripts/scaling/hsg_gb200_smoke_gate.py" tree-hash --root "$ROOT")
[ "$FINAL_FAST_SOURCE_TREE_SHA256" = "$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256" ] || \
    fatal "fast_roborender source tree changed during the allocation"
FINAL_GR00T_SOURCE_TREE_SHA256=$(python3 \
    "$ROOT/scripts/scaling/hsg_gb200_smoke_gate.py" tree-hash --root "$GR00T_ROOT")
[ "$FINAL_GR00T_SOURCE_TREE_SHA256" = "$EXPECTED_GR00T_SOURCE_TREE_SHA256" ] || \
    fatal "GR00T source tree changed during the allocation"

FINAL_ASSET_PREFLIGHT=$RUN_ATTEMPT_DIR/final_asset_preflight.json
python3 "$ROOT/scripts/scaling/hsg_gb200_smoke_gate.py" preflight \
    --profile "$RUN_PROFILE" --root "$ROOT" --gr00t-root "$GR00T_ROOT" \
    --fast-revision "$EXPECTED_FAST_ROBORENDER_REVISION" \
    --gr00t-revision "$EXPECTED_GR00T_REVISION" \
    --renderer-container "$RENDERER_CONTAINER" \
    --renderer-sha256 "$EXPECTED_RENDERER_CONTAINER_SHA256" \
    --trainer-container "$TRAINER_CONTAINER" \
    --trainer-sha256 "$EXPECTED_TRAINER_CONTAINER_SHA256" \
    --image-root "$IMAGE3F_ROOT" --model-root "$ROBORENDER_MODEL_BASE" \
    --generated-root "$IMAGE3F_GENERATED_ROOT" --anchors "$ANCHORS" \
    --anchor-min-files 2048 --teacher "$TEACHER_CKPT" \
    --teacher-sha256 "$EXPECTED_TEACHER_CKPT_SHA256" \
    --release-id "$EXPECTED_IMAGE3F_RELEASE_ID" \
    --adapter-sha256 "$EXPECTED_IMAGE3F_ADAPTER_SHA256" > "$FINAL_ASSET_PREFLIGHT"
read -r FINAL_ASSET_CONTRACT_SHA256 FINAL_ENGINE_SHA256 \
    FINAL_PREFLIGHT_FAST_TREE_SHA256 FINAL_PREFLIGHT_GR00T_TREE_SHA256 \
    FINAL_SOURCE_SNAPSHOT_ATTESTATION_SHA256 \
    FINAL_SOURCE_SNAPSHOT_INVENTORY_SHA256 < <(
python3 - "$FINAL_ASSET_PREFLIGHT" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    payload.get("asset_contract_sha256", ""),
    payload.get("engine_sha256", ""),
    payload.get("fast_roborender_source_tree_sha256", ""),
    payload.get("gr00t_source_tree_sha256", ""),
    payload.get("source_snapshot_attestation_sha256", ""),
    payload.get("source_snapshot_inventory_sha256", ""),
)
PY
)
[ "$FINAL_ASSET_CONTRACT_SHA256" = "$EXPECTED_ASSET_CONTRACT_SHA256" ] || \
    fatal "immutable asset contract changed during the allocation"
[ "$FINAL_ENGINE_SHA256" = "$EXPECTED_IMAGE3F_ENGINE_SHA256" ] || \
    fatal "TensorRT engine changed during the allocation"
[ "$FINAL_PREFLIGHT_FAST_TREE_SHA256" = "$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256" ] || \
    fatal "final asset preflight observed a changed fast_roborender source tree"
[ "$FINAL_PREFLIGHT_GR00T_TREE_SHA256" = "$EXPECTED_GR00T_SOURCE_TREE_SHA256" ] || \
    fatal "final asset preflight observed a changed GR00T source tree"
[ "$FINAL_SOURCE_SNAPSHOT_ATTESTATION_SHA256" = \
  "$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256" ] || \
    fatal "source snapshot attestation changed during the allocation"
[ "$FINAL_SOURCE_SNAPSHOT_INVENTORY_SHA256" = \
  "$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256" ] || \
    fatal "source snapshot inventory changed during the allocation"
[ "$(sha256sum "$RESNET18_WEIGHTS" | awk '{print $1}')" = "$EXPECTED_RESNET18_SHA256" ] || \
    fatal "pinned ResNet18 weights changed during the allocation"

python3 "$ROOT/scripts/scaling/hsg_gb200_smoke_gate.py" postrun \
    --profile "$RUN_PROFILE" --run-id "$RUN_ID" --run-dir "$TRAIN_RUN_DIR" \
    --global-envs "$GLOBAL_NUM_ENVS" --rollout-steps "$ROLLOUT_STEPS" \
    --final-step "$TRAIN_ABSOLUTE_TARGET" \
    --world-size "$TRAINER_RANKS" --fast-revision "$EXPECTED_FAST_ROBORENDER_REVISION" \
    --gr00t-revision "$EXPECTED_GR00T_REVISION" \
    --fast-source-tree-sha256 "$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256" \
    --gr00t-source-tree-sha256 "$EXPECTED_GR00T_SOURCE_TREE_SHA256" \
    --source-snapshot-attestation-sha256 \
      "$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256" \
    --source-snapshot-inventory-sha256 \
      "$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256" \
    --asset-contract-sha256 "$EXPECTED_ASSET_CONTRACT_SHA256" \
    --renderer-clean-stop-evidence "$RENDERER_CLEAN_STOP_EVIDENCE" \
    --wandb-expected-email "$WANDB_EXPECTED_EMAIL" --wandb-entity "$WANDB_ENTITY" \
    > "$POSTRUN_GATE_PATH"

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
        # Recheck immediately before publishing so a queued allocation cannot
        # silently redirect the run-scoped evidence after submission.
        validate_camera_calibration_manifest_destination
        python3 "$ROOT/scripts/scaling/camera_calibration_gate.py" publish \
            --manifest "$TECHNICAL" \
            --destination "$CAMERA_CALIBRATION_MANIFEST_DESTINATION" \
            --validation-directory "$STAGE_ROOT/runtime/validation" \
            --expected-run-id "$RUN_ID" \
            --expected-fast-roborender-revision \
              "$EXPECTED_FAST_ROBORENDER_REVISION" \
            --expected-gr00t-revision "$EXPECTED_GR00T_REVISION" \
            --expected-fast-source-tree-sha256 \
              "$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256" \
            --expected-gr00t-source-tree-sha256 \
              "$EXPECTED_GR00T_SOURCE_TREE_SHA256" \
            --expected-source-snapshot-attestation-sha256 \
              "$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256" \
            --expected-source-snapshot-inventory-sha256 \
              "$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256" \
            --expected-asset-contract-sha256 \
              "$EXPECTED_ASSET_CONTRACT_SHA256" >/dev/null
        echo "Calibration is technically valid; explicit visual acceptance is still required."
        ;;
    smoke_32x300|smoke_256x300)
        PROFILE_MANIFEST="$TRAIN_RUN_DIR/profile_validation_manifest.txt"
        python3 "$ROOT/scripts/scaling/profile_validation_gate.py" \
            --manifest "$PROFILE_MANIFEST" --expected-profile "$RUN_PROFILE" \
            --expected-fast-revision "$EXPECTED_FAST_ROBORENDER_REVISION" \
            --expected-gr00t-revision "$EXPECTED_GR00T_REVISION" \
            --expected-fast-source-tree-sha256 "$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256" \
            --expected-gr00t-source-tree-sha256 "$EXPECTED_GR00T_SOURCE_TREE_SHA256" \
            --expected-source-snapshot-attestation-sha256 \
              "$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256" \
            --expected-source-snapshot-inventory-sha256 \
              "$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256" \
            --expected-asset-contract-sha256 "$EXPECTED_ASSET_CONTRACT_SHA256" \
            --expected-wandb-email "$WANDB_EXPECTED_EMAIL" \
            --expected-wandb-entity "$WANDB_ENTITY" >/dev/null
        publish_symlink_once "$PROFILE_MANIFEST" \
            "$STAGE_ROOT/runtime/validation/${RUN_PROFILE}_profile_validation_manifest.txt"
        ;;
esac
echo "HSG_GB200_DAGGER_SMOKE_PASS profile=$RUN_PROFILE"
