#!/bin/bash
# Run one independent native-RGB BC actor on its single visible H200.
set -Eeuo pipefail

SAFE_ACTOR_PATH=/usr/sbin:/usr/bin:/sbin:/bin
[[ ${PATH:-} == "$SAFE_ACTOR_PATH" ]] || {
    echo "native H200 actor rank: child PATH was not sanitized by the fleet" >&2
    exit 2
}
[[ -z ${LD_LIBRARY_PATH:-} && -z ${LD_PRELOAD:-} && -z ${BASH_ENV:-} ]] || {
    echo "native H200 actor rank: inherited loader/shell injection state" >&2
    exit 2
}
[[ ! -v CUDA_VISIBLE_DEVICES && ! -v NVIDIA_VISIBLE_DEVICES ]] || {
    echo "native H200 actor rank: inherited CUDA/NVIDIA visibility remapping" >&2
    exit 2
}
export PATH=$SAFE_ACTOR_PATH
unset -f dirname pwd realpath mount mountpoint umount findmnt setpriv tail \
    sha256sum awk stat mkdir chmod touch mktemp env python3 2>/dev/null || true

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
GR00T_PYTHON=${GR00T_PYTHON:?set GR00T_PYTHON to the attested GR00T Python}
export GR00T_ROOT
WORKSPACE_ROOT=/workspace
DAGGER_STORAGE_ROOT=/workspace/project/roborender/data
CHECKPOINT_ROOT=/workspace/project/roborender/checkpoints
MOTION_ROOT=$GR00T_ROOT/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775
VALIDATION_ROOT=$DAGGER_STORAGE_ROOT/validation
TEST_ROOT=$DAGGER_STORAGE_ROOT/test
PUBLIC_SPLIT_DETAILS=(
    "$ROOT/configs/round_dagger/native_h200_group_registry.json"
    "$ROOT/configs/round_dagger/native_h200_group_registry_audit.json"
    "$ROOT/configs/round_dagger/native_h200_validation_panel.json"
    "$ROOT/configs/round_dagger/native_h200_validation_panel_audit.json"
    "$ROOT/configs/round_dagger/production_heldout_motion_plan_8x32.json"
    "$ROOT/configs/round_dagger/native_h200_bc_dagger4_protocol.v1.json"
    "$ROOT/configs/round_dagger/production_paired_scenarios_2x8.json"
    "$GR00T_ROOT/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775_heldout_256_v1.json"
)

fatal() { echo "native H200 actor rank: $*" >&2; exit 2; }
required=(
    ROUND_DAGGER_ATTEMPT_ID ROUND_DAGGER_ACTOR_RANK
    ROUND_DAGGER_ACTOR_WORLD_SIZE
    ROUND_DAGGER_ENVS_PER_GPU ROUND_DAGGER_STEPS_PER_ENVIRONMENT
    ROUND_DAGGER_SATURATION_PHASE ROUND_DAGGER_SATURATION_WAVE_INDEX
    ROUND_DAGGER_NATIVE_PROBE_ONLY ROUND_DAGGER_OBSERVATION_SOURCE
    ROUND_DAGGER_ACTOR_ROLE ROUND_DAGGER_PURPOSE ROUND_DAGGER_BETA
    ROUND_DAGGER_BETA_SHA256 ROUND_DAGGER_CONFIG_SHA256
    ROUND_DAGGER_CAMPAIGN_ROOT ROUND_DAGGER_CAMPAIGN_ID ROUND_DAGGER_MACRO_ROUND
    ROUND_DAGGER_TRAIN_PARTITION ROUND_DAGGER_TRAIN_PARTITION_SHA256
    ROUND_DAGGER_TRAIN_MOTION_ROOT
    ROUND_DAGGER_SPLIT_SHA256 ROUND_DAGGER_TRAIN_GROUP_IDS_JSON
    ROUND_DAGGER_POLICY_CHECKPOINT ROUND_DAGGER_POLICY_SHA256
    ROUND_DAGGER_RUNTIME_POLICY_SHA256 ROUND_DAGGER_TEACHER_CHECKPOINT
    ROUND_DAGGER_TEACHER_SHA256 ROUND_DAGGER_SIMULATOR_BASE_SEED
    ROUND_DAGGER_GENERATION_ROOT ROUND_DAGGER_GENERATION_ID
    ROUND_DAGGER_OBSERVATION_RUNTIME_SHA256 ROUND_DAGGER_ISAAC_RUNTIME_SHA256
    ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256
    ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256 ROUND_DAGGER_GR00T_REVISION
    ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256 ROUND_DAGGER_GPU_NAME
    ROUND_DAGGER_COMPUTE_CAPABILITY ROUND_DAGGER_GPU_COUNT
    ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON
    ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256
    ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON
    ROUND_DAGGER_ASSIGNED_PHYSICAL_INDEX
    ROUND_DAGGER_ASSIGNED_GPU_UUID
    ROUND_DAGGER_GPU_DEVICE_NAMESPACE_JSON
    ROUND_DAGGER_GPU_DEVICE_NAMESPACE_SHA256
    ROUND_DAGGER_NVIDIA_DEVICE ROUND_DAGGER_DRM_CARD_DEVICE
    ROUND_DAGGER_DRM_RENDER_DEVICE
    ROUND_DAGGER_HOST_RUNTIME_ATTESTATION_SHA256
    ROUND_DAGGER_FLEET_RELEASE_FILE
    ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST
    ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST
    ROUND_DAGGER_ACTOR_SOURCE_INVENTORY ROUND_DAGGER_GR00T_SOURCE_INVENTORY
    ROUND_DAGGER_ISAAC_SETUP_SCRIPT ROUND_DAGGER_CUDNN_LIBRARY_DIR
    ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT
    ROUND_DAGGER_ISAAC_RUNTIME_ROOT ROUND_DAGGER_PYTHON_RUNTIME_ROOT
    ROUND_DAGGER_ISAACLAB_SOURCE_ROOT ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT
    ROUND_DAGGER_SYSTEM_LIBRARY_DIR ROBORENDER_HEADLESS_EXPERIENCE
    OMNI_KIT_ACCEPT_EULA OMNI_KIT_ALLOW_ROOT ACCEPT_EULA PRIVACY_CONSENT
)
for name in "${required[@]}"; do
    [[ -n ${!name:-} ]] || fatal "missing required environment: $name"
done
[[ $ROUND_DAGGER_ATTEMPT_ID =~ ^[a-z0-9][a-z0-9-]{0,63}$ ]] || \
    fatal "attempt ID is malformed"

[[ $OMNI_KIT_ACCEPT_EULA == YES && $OMNI_KIT_ALLOW_ROOT == 1 && \
   $ACCEPT_EULA == Y && $PRIVACY_CONSENT == Y ]] || \
    fatal "Isaac EULA/root/privacy runtime contract differs"
[[ $ROUND_DAGGER_ISAAC_SETUP_SCRIPT == /* && \
   $ROUND_DAGGER_ISAAC_SETUP_SCRIPT == "$ROOT/scripts/round_dagger/setup_native_h200_isaac_runtime.sh" && \
   -f $ROUND_DAGGER_ISAAC_SETUP_SCRIPT && ! -L $ROUND_DAGGER_ISAAC_SETUP_SCRIPT ]] || \
    fatal "Isaac runtime setup script is missing or unsafe"
[[ $ROBORENDER_HEADLESS_EXPERIENCE == /* && \
   $ROBORENDER_HEADLESS_EXPERIENCE == *.kit && \
   -f $ROBORENDER_HEADLESS_EXPERIENCE && ! -L $ROBORENDER_HEADLESS_EXPERIENCE ]] || \
    fatal "qualified headless experience is missing or unsafe"
[[ $ROUND_DAGGER_CUDNN_LIBRARY_DIR == /* && \
   -d $ROUND_DAGGER_CUDNN_LIBRARY_DIR && ! -L $ROUND_DAGGER_CUDNN_LIBRARY_DIR ]] || \
    fatal "cuDNN library directory is missing or unsafe"
compgen -G "$ROUND_DAGGER_CUDNN_LIBRARY_DIR/libcudnn.so*" >/dev/null || \
    fatal "cuDNN library directory contains no libcudnn"
[[ ! -e $ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT/sitecustomize.py ]] || \
    fatal "qualified dependency root must not contain sitecustomize.py"
if compgen -G "$ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT/*.pth" >/dev/null || \
   compgen -G "$ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT/*editable*" >/dev/null; then
    fatal "qualified dependency root contains path-injection metadata"
fi
private_input_dir=$ROUND_DAGGER_GENERATION_ROOT/runtime/private-inputs/attempts/$ROUND_DAGGER_ATTEMPT_ID/$ROUND_DAGGER_SATURATION_PHASE/wave-$(printf '%03d' "$ROUND_DAGGER_SATURATION_WAVE_INDEX")/gpu-$(printf '%02d' "$ROUND_DAGGER_ACTOR_RANK")
private_train_partition=$private_input_dir/train_motion_partition.v1.json
checkpoint_input_dir=$CHECKPOINT_ROOT
runtime_contract_dir=$ROUND_DAGGER_GENERATION_ROOT/runtime-contracts
native_config=$GR00T_ROOT/gr00t/rl/config/exp/loco_manip/wsdpt_round_actor_native_3view.yaml

# No actor-controlled Python runs until this process is inside a private mount
# and PID namespace.  The mount stage masks the *whole* shared workspace and
# canonical data tree, then bind-restores only the current source checkout, the
# qualified Isaac runtime, BC collection's writable output, and train-only input.
# This makes sibling worktrees, every Git object store, validation, and sealed
# test material unreachable rather than relying on path-name deny lists.
mount_stage=${ROUND_DAGGER_PRIVATE_MOUNT_STAGE:-entry}
case "$mount_stage" in
    entry)
        for tool in unshare mount mountpoint umount findmnt setpriv tail; do
            command -v "$tool" >/dev/null || fatal "required confinement tool is missing: $tool"
        done
        [[ $ROUND_DAGGER_CAMPAIGN_ROOT == /workspace/project/roborender/data && \
           $ROUND_DAGGER_GENERATION_ROOT == "$ROUND_DAGGER_CAMPAIGN_ROOT/BC/isaac/collection" ]] || \
            fatal "generation storage identity differs before confinement"
        [[ $ROUND_DAGGER_SATURATION_PHASE =~ ^(smoke|probe|collect)$ && \
           $ROUND_DAGGER_SATURATION_WAVE_INDEX =~ ^(0|[1-9][0-9]*)$ && \
           $ROUND_DAGGER_ACTOR_RANK =~ ^[0-7]$ ]] || \
            fatal "rank namespace identity is malformed"
        [[ $ROUND_DAGGER_ASSIGNED_GPU_UUID == GPU-* && \
           $ROUND_DAGGER_ASSIGNED_GPU_UUID != *,* && \
           $ROUND_DAGGER_NVIDIA_DEVICE =~ ^/dev/nvidia[0-9]+$ && \
           $ROUND_DAGGER_DRM_CARD_DEVICE =~ ^/dev/dri/card[0-9]+$ && \
           $ROUND_DAGGER_DRM_RENDER_DEVICE =~ ^/dev/dri/renderD[0-9]+$ ]] || \
            fatal "rank-private GPU device mapping is malformed"
        for gpu_device in "$ROUND_DAGGER_NVIDIA_DEVICE" \
            "$ROUND_DAGGER_DRM_CARD_DEVICE" "$ROUND_DAGGER_DRM_RENDER_DEVICE" \
            /dev/nvidiactl /dev/nvidia-modeset /dev/nvidia-uvm \
            /dev/nvidia-uvm-tools; do
            [[ -c $gpu_device && ! -L $gpu_device && \
               $(realpath -e "$gpu_device") == "$gpu_device" ]] || \
                fatal "GPU device is missing or unsafe: $gpu_device"
        done
        [[ $ROUND_DAGGER_TRAIN_MOTION_ROOT == /* && \
           -d $ROUND_DAGGER_TRAIN_MOTION_ROOT && \
           ! -L $ROUND_DAGGER_TRAIN_MOTION_ROOT ]] || \
            fatal "train-only motion root is missing or unsafe"
        [[ $(realpath -e "$ROUND_DAGGER_TRAIN_MOTION_ROOT") == \
           "$ROUND_DAGGER_TRAIN_MOTION_ROOT" ]] || \
            fatal "train-only motion root must be canonical"
        case "$ROUND_DAGGER_TRAIN_MOTION_ROOT/" in
            "$MOTION_ROOT/"*) fatal "train-only root must be outside the full motion root" ;;
        esac
        [[ -d $MOTION_ROOT && ! -L $MOTION_ROOT ]] || \
            fatal "full motion mount target is missing or unsafe"
        for immutable_root in "$ROOT" "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT" \
            "$ROUND_DAGGER_ISAAC_RUNTIME_ROOT" \
            "$ROUND_DAGGER_PYTHON_RUNTIME_ROOT" "$ROUND_DAGGER_CUDNN_LIBRARY_DIR" \
            "$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT" \
            "$ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT" \
            "$ROUND_DAGGER_SYSTEM_LIBRARY_DIR"; do
            [[ $immutable_root == /* && -d $immutable_root && ! -L $immutable_root && \
               $(realpath -e "$immutable_root") == "$immutable_root" ]] || \
                fatal "immutable runtime root is missing or unsafe: $immutable_root"
        done
        case "$ROOT/" in "$WORKSPACE_ROOT/"*) ;; *) fatal "actor source is outside /workspace" ;; esac
        case "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/" in
            "$WORKSPACE_ROOT/"*) ;;
            *) fatal "qualified runtime is outside /workspace" ;;
        esac
        [[ $ROUND_DAGGER_ISAAC_RUNTIME_ROOT == \
               "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/runtime/isaacsim-5.1" && \
           $ROUND_DAGGER_PYTHON_RUNTIME_ROOT == \
               "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT" && \
           $ROUND_DAGGER_ISAACLAB_SOURCE_ROOT == \
               "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/runtime/IsaacLab/source" && \
           $ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT == \
               "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/runtime/actor-python-site-packages" && \
           $ROUND_DAGGER_SYSTEM_LIBRARY_DIR == \
               "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/packages/root/usr/lib/x86_64-linux-gnu" && \
           $ROUND_DAGGER_CUDNN_LIBRARY_DIR == \
               "$ROUND_DAGGER_ISAAC_RUNTIME_ROOT/lib/python3.11/site-packages/nvidia/cudnn/lib" && \
           $GR00T_PYTHON == "$ROUND_DAGGER_ISAAC_RUNTIME_ROOT/bin/python" && \
           $ROBORENDER_HEADLESS_EXPERIENCE == \
               "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/runtime/IsaacLab/apps/phc.isaaclab.python.headless.rendering.kit" ]] || \
            fatal "qualified runtime topology differs"
        for runtime_path in "$ROUND_DAGGER_ISAAC_RUNTIME_ROOT" \
            "$ROUND_DAGGER_PYTHON_RUNTIME_ROOT" "$ROUND_DAGGER_CUDNN_LIBRARY_DIR" \
            "$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT" \
            "$ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT" \
            "$ROUND_DAGGER_SYSTEM_LIBRARY_DIR" "$ROBORENDER_HEADLESS_EXPERIENCE"; do
            case "$runtime_path/" in
                "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/"*) ;;
                *) fatal "runtime input leaves the qualified runtime root: $runtime_path" ;;
            esac
        done
        [[ $GR00T_PYTHON == /* && -L $GR00T_PYTHON && -x $GR00T_PYTHON ]] || \
            fatal "qualified GR00T Python must be an executable absolute symlink"
        gr00t_python_link_target=$(readlink "$GR00T_PYTHON")
        [[ $gr00t_python_link_target == /* && \
           $(realpath -e "$GR00T_PYTHON") == "$gr00t_python_link_target" && \
           $gr00t_python_link_target == \
               "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/runtime/python/cpython-3.11.13-linux-x86_64-gnu/bin/python3.11" && \
           -f $gr00t_python_link_target && ! -L $gr00t_python_link_target ]] || \
            fatal "qualified GR00T Python link must name one absolute regular target"
        case "$GR00T_PYTHON/" in
            "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/"*) ;;
            *) fatal "qualified GR00T Python link leaves runtime root" ;;
        esac
        case "$gr00t_python_link_target/" in
            "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/"*) ;;
            *) fatal "qualified GR00T Python target leaves runtime root" ;;
        esac
        [[ -d $ROUND_DAGGER_GENERATION_ROOT && ! -L $ROUND_DAGGER_GENERATION_ROOT && \
           -d $checkpoint_input_dir && ! -L $checkpoint_input_dir ]] || \
            fatal "generation/checkpoint input root is missing or unsafe"
        case "$ROUND_DAGGER_TRAIN_MOTION_ROOT/" in
            "$DAGGER_STORAGE_ROOT/"*) ;;
            *) fatal "train-only motion materialization leaves canonical data storage" ;;
        esac
        pinned_inputs=(
            "$ROUND_DAGGER_POLICY_CHECKPOINT" "$ROUND_DAGGER_TEACHER_CHECKPOINT"
            "$ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST"
            "$ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST"
            "$ROUND_DAGGER_ACTOR_SOURCE_INVENTORY"
            "$ROUND_DAGGER_GR00T_SOURCE_INVENTORY" "$native_config"
        )
        for pinned in "${pinned_inputs[@]}"; do
            [[ $pinned == /* && -f $pinned && ! -L $pinned && \
               $(realpath -e "$pinned") == "$pinned" ]] || \
                fatal "pinned actor input is missing or unsafe: $pinned"
        done
        for pinned in "${pinned_inputs[@]:0:2}"; do
            case "$pinned/" in
                "$checkpoint_input_dir/"*) ;;
                *) fatal "model input leaves canonical checkpoint root: $pinned" ;;
            esac
        done
        for pinned in "${pinned_inputs[@]:2:4}"; do
            case "$pinned/" in
                "$runtime_contract_dir/"*) ;;
                *) fatal "runtime provenance leaves generation runtime-contract root: $pinned" ;;
            esac
        done
        [[ -d $VALIDATION_ROOT && ! -L $VALIDATION_ROOT && \
           -d $TEST_ROOT && ! -L $TEST_ROOT ]] || \
            fatal "held-out masking targets are missing or unsafe"
        [[ $ROUND_DAGGER_TRAIN_PARTITION == /* && \
           -f $ROUND_DAGGER_TRAIN_PARTITION && ! -L $ROUND_DAGGER_TRAIN_PARTITION && \
           $(sha256sum "$ROUND_DAGGER_TRAIN_PARTITION" | awk '{print $1}') == \
               "$ROUND_DAGGER_TRAIN_PARTITION_SHA256" ]] || \
            fatal "original train partition is missing, unsafe, or digest-mismatched"
        mkdir -p "$private_input_dir"
        chmod 0700 "$private_input_dir"
        if [[ ! -e $private_train_partition ]]; then
            (umask 077; : > "$private_train_partition")
        fi
        [[ -f $private_train_partition && ! -L $private_train_partition && \
           ! -s $private_train_partition ]] || \
            fatal "private train-partition mount target is unsafe"
        for sensitive in "${PUBLIC_SPLIT_DETAILS[@]}"; do
            [[ -f $sensitive && ! -L $sensitive ]] || \
                fatal "split-detail masking target is missing or unsafe: $sensitive"
        done
        parent_pid_namespace=$(readlink /proc/self/ns/pid)
        exec unshare --mount --propagation private --pid --fork --kill-child \
            --mount-proc=/proc -- \
            env ROUND_DAGGER_PRIVATE_MOUNT_STAGE=mount \
                ROUND_DAGGER_PARENT_PID_NAMESPACE="$parent_pid_namespace" \
                ROUND_DAGGER_ORIGINAL_TRAIN_PARTITION="$ROUND_DAGGER_TRAIN_PARTITION" \
                "$BASH" "$0" "$@"
        ;;
    mount)
        [[ ${ROUND_DAGGER_PARENT_PID_NAMESPACE:-} == pid:* && \
           $(readlink /proc/self/ns/pid) != "$ROUND_DAGGER_PARENT_PID_NAMESPACE" ]] || \
            fatal "private PID namespace was not established"
        [[ ${ROUND_DAGGER_ORIGINAL_TRAIN_PARTITION:-} == /* && \
           -f $ROUND_DAGGER_ORIGINAL_TRAIN_PARTITION && \
           ! -L $ROUND_DAGGER_ORIGINAL_TRAIN_PARTITION ]] || \
            fatal "original train partition was not retained safely"
        jail_handles=$(mktemp -d "/run/round-dagger-actor-${ROUND_DAGGER_SATURATION_PHASE}-w${ROUND_DAGGER_SATURATION_WAVE_INDEX}-r${ROUND_DAGGER_ACTOR_RANK}.XXXXXX")
        chmod 0700 "$jail_handles"
        mkdir -m 0700 "$jail_handles/source" "$jail_handles/runtime" \
            "$jail_handles/generation" "$jail_handles/train-motions" \
            "$jail_handles/checkpoints" "$jail_handles/pins" "$jail_handles/devices"
        mount --bind "$ROOT" "$jail_handles/source"
        mount --bind "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT" "$jail_handles/runtime"
        mount --bind "$ROUND_DAGGER_GENERATION_ROOT" "$jail_handles/generation"
        mount --bind "$ROUND_DAGGER_TRAIN_MOTION_ROOT" "$jail_handles/train-motions"
        mount --bind "$checkpoint_input_dir" "$jail_handles/checkpoints"
        pinned_inputs=(
            "$ROUND_DAGGER_POLICY_CHECKPOINT" "$ROUND_DAGGER_TEACHER_CHECKPOINT"
            "$ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST"
            "$ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST"
            "$ROUND_DAGGER_ACTOR_SOURCE_INVENTORY"
            "$ROUND_DAGGER_GR00T_SOURCE_INVENTORY" "$native_config"
        )
        for index in "${!pinned_inputs[@]}"; do
            touch "$jail_handles/pins/$index"
            chmod 0600 "$jail_handles/pins/$index"
            mount --bind "${pinned_inputs[$index]}" "$jail_handles/pins/$index"
        done
        touch "$jail_handles/train-partition"
        chmod 0600 "$jail_handles/train-partition"
        mount --bind "$ROUND_DAGGER_ORIGINAL_TRAIN_PARTITION" \
            "$jail_handles/train-partition"
        for device_handle in nvidia drm-card drm-render; do
            touch "$jail_handles/devices/$device_handle"
            chmod 0600 "$jail_handles/devices/$device_handle"
        done
        mount --bind "$ROUND_DAGGER_NVIDIA_DEVICE" \
            "$jail_handles/devices/nvidia"
        mount --bind "$ROUND_DAGGER_DRM_CARD_DEVICE" \
            "$jail_handles/devices/drm-card"
        mount --bind "$ROUND_DAGGER_DRM_RENDER_DEVICE" \
            "$jail_handles/devices/drm-render"

        # Empty /workspace hides sibling worktrees and agent run histories.  Only
        # this attested checkout and the one qualified runtime are restored.
        cd /
        unset OLDPWD
        mount -t tmpfs -o size=1m,mode=0755,nosuid,nodev \
            native-actor-workspace "$WORKSPACE_ROOT"
        mkdir -p "$ROOT" "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT" "$checkpoint_input_dir"
        mount --bind "$jail_handles/source" "$ROOT"
        mount -o remount,bind,ro,nosuid,nodev "$ROOT"
        mount --bind "$jail_handles/runtime" "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT"
        mount -o remount,bind,ro,nosuid,nodev "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT"

        # Likewise expose only gen-000 output and the exact train-only motions
        # below the otherwise empty project storage mount.
        mount -t tmpfs -o size=1m,mode=0755,nosuid,nodev \
            native-actor-storage "$DAGGER_STORAGE_ROOT"
        mkdir -p "$ROUND_DAGGER_GENERATION_ROOT" "$ROUND_DAGGER_TRAIN_MOTION_ROOT" \
            "$VALIDATION_ROOT" "$TEST_ROOT"
        mount --bind "$jail_handles/generation" "$ROUND_DAGGER_GENERATION_ROOT"
        mount --bind "$jail_handles/train-motions" "$ROUND_DAGGER_TRAIN_MOTION_ROOT"
        mount -o remount,bind,ro,nosuid,nodev,noexec "$ROUND_DAGGER_TRAIN_MOTION_ROOT"

        # The checkpoint directory is immutable in the actor view.  Individual
        # model/provenance/config files are additionally bind-pinned so a path
        # replacement cannot race the post-jail hash verification.
        mount --bind "$jail_handles/checkpoints" "$checkpoint_input_dir"
        mount -o remount,bind,ro,nosuid,nodev,noexec "$checkpoint_input_dir"
        for index in "${!pinned_inputs[@]}"; do
            mount --bind "$jail_handles/pins/$index" "${pinned_inputs[$index]}"
            mount -o remount,bind,ro,nosuid,nodev,noexec "${pinned_inputs[$index]}"
        done
        mount --bind "$jail_handles/train-partition" "$private_train_partition"
        mount -o remount,bind,ro,nosuid,nodev,noexec "$private_train_partition"
        mount --bind "$ROUND_DAGGER_TRAIN_MOTION_ROOT" "$MOTION_ROOT"
        mount -o remount,bind,ro,nosuid,nodev,noexec "$MOTION_ROOT"

        mount -t tmpfs -o size=4k,mode=000,ro,nosuid,nodev,noexec \
            native-validation "$VALIDATION_ROOT"
        mount -t tmpfs -o size=4k,mode=000,ro,nosuid,nodev,noexec \
            native-test "$TEST_ROOT"
        for git_store in "$ROOT/.git" "$GR00T_ROOT/.git"; do
            [[ -d $git_store && ! -L $git_store ]] || \
                fatal "expected Git store is missing before mask: $git_store"
            mount -t tmpfs -o size=4k,mode=000,ro,nosuid,nodev,noexec \
                native-git-store "$git_store"
        done
        for sensitive in "${PUBLIC_SPLIT_DETAILS[@]}"; do
            mount --bind /dev/null "$sensitive"
            mount -o remount,bind,ro,nosuid,nodev,noexec "$sensitive"
        done

        # Build a true per-rank GPU device view.  CUDA ordinals and NVIDIA
        # device minors are unrelated on this host (for example physical rank 1
        # is /dev/nvidia5), so the controller resolves all three device nodes
        # from the assigned UUID/PCI function before launch.  Preserve common
        # NVIDIA control devices, mask every physical GPU node, restore only the
        # assigned one, and replace /dev/dri with its matching card/render pair.
        physical_nvidia_devices=(/dev/nvidia[0-9]*)
        [[ ${#physical_nvidia_devices[@]} == 8 ]] || \
            fatal "host must expose exactly eight physical NVIDIA device nodes"
        assigned_nvidia_count=0
        for gpu_device in "${physical_nvidia_devices[@]}"; do
            [[ $gpu_device =~ ^/dev/nvidia[0-9]+$ && -c $gpu_device && \
               ! -L $gpu_device ]] || \
                fatal "physical NVIDIA device inventory is malformed: $gpu_device"
            if [[ $gpu_device == "$ROUND_DAGGER_NVIDIA_DEVICE" ]]; then
                assigned_nvidia_count=$((assigned_nvidia_count + 1))
            fi
            mount --bind /dev/null "$gpu_device"
        done
        [[ $assigned_nvidia_count == 1 ]] || \
            fatal "assigned NVIDIA device is absent from the physical inventory"
        mount --bind "$jail_handles/devices/nvidia" "$ROUND_DAGGER_NVIDIA_DEVICE"

        mount -t tmpfs -o size=64k,mode=0755,nosuid,noexec \
            native-actor-dri /dev/dri
        private_drm_card=/dev/dri/${ROUND_DAGGER_DRM_CARD_DEVICE##*/}
        private_drm_render=/dev/dri/${ROUND_DAGGER_DRM_RENDER_DEVICE##*/}
        touch "$private_drm_card" "$private_drm_render"
        chmod 0600 "$private_drm_card" "$private_drm_render"
        mount --bind "$jail_handles/devices/drm-card" "$private_drm_card"
        mount --bind "$jail_handles/devices/drm-render" "$private_drm_render"
        mount -o remount,ro,nosuid,noexec /dev/dri

        mount -o remount,ro,nosuid,nodev "$WORKSPACE_ROOT"
        mount -o remount,ro,nosuid,nodev "$DAGGER_STORAGE_ROOT"
        # Remove every alternate bind alias before dropping mount privilege.
        for index in "${!pinned_inputs[@]}"; do
            umount "$jail_handles/pins/$index"
        done
        umount "$jail_handles/train-partition"
        umount "$jail_handles/train-motions"
        umount "$jail_handles/generation"
        umount "$jail_handles/runtime"
        umount "$jail_handles/source"
        umount "$jail_handles/devices/drm-render"
        umount "$jail_handles/devices/drm-card"
        umount "$jail_handles/devices/nvidia"
        mount -t tmpfs -o size=4k,mode=000,ro,nosuid,nodev,noexec \
            native-jail-handles "$jail_handles"
        exec setpriv --no-new-privs \
            --bounding-set=-sys_admin,-dac_override,-dac_read_search,-mknod \
            --inh-caps=-all --ambient-caps=-all \
            env ROUND_DAGGER_PRIVATE_MOUNT_STAGE=ready \
                ROUND_DAGGER_JAIL_HANDLE_ROOT="$jail_handles" \
                ROUND_DAGGER_TRAIN_PARTITION="$private_train_partition" \
                "$BASH" "$0" "$@"
        ;;
    ready)
        [[ ${ROUND_DAGGER_PARENT_PID_NAMESPACE:-} == pid:* && \
           $(readlink /proc/self/ns/pid) != "$ROUND_DAGGER_PARENT_PID_NAMESPACE" ]] || \
            fatal "private PID namespace was not retained"
        # `findmnt -T` reports every stacked mount at a target on this host;
        # the last row is the effective topmost mount inside our namespace.
        [[ $(findmnt -n -o SOURCE -T "$WORKSPACE_ROOT" | tail -n 1) == native-actor-workspace && \
           $(findmnt -n -o SOURCE -T "$DAGGER_STORAGE_ROOT" | tail -n 1) == native-actor-storage ]] || \
            fatal "broad workspace/storage masks are absent"
        for immutable_root in "$ROOT" "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT" \
            "$ROUND_DAGGER_ISAAC_RUNTIME_ROOT" \
            "$ROUND_DAGGER_PYTHON_RUNTIME_ROOT" "$ROUND_DAGGER_CUDNN_LIBRARY_DIR" \
            "$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT" \
            "$ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT" \
            "$ROUND_DAGGER_SYSTEM_LIBRARY_DIR"; do
            immutable_options=$(findmnt -n -o OPTIONS -T "$immutable_root" | tail -n 1)
            [[ ,$immutable_options, == *,ro,* ]] || \
                fatal "source/runtime bind mount is not read-only: $immutable_root"
        done
        mountpoint -q "$MOTION_ROOT" || fatal "train-only bind mount is absent"
        mountpoint -q "$private_train_partition" || \
            fatal "private train-partition bind mount is absent"
        motion_options=$(findmnt -n -o OPTIONS -T "$MOTION_ROOT" | tail -n 1)
        [[ ,$motion_options, == *,ro,* ]] || \
            fatal "train-only bind mount is not read-only"
        checkpoint_options=$(findmnt -n -o OPTIONS -T "$checkpoint_input_dir" | tail -n 1)
        [[ ,$checkpoint_options, == *,ro,* ]] || \
            fatal "checkpoint input tree is not read-only"
        pinned_inputs=(
            "$ROUND_DAGGER_POLICY_CHECKPOINT" "$ROUND_DAGGER_TEACHER_CHECKPOINT"
            "$ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST"
            "$ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST"
            "$ROUND_DAGGER_ACTOR_SOURCE_INVENTORY"
            "$ROUND_DAGGER_GR00T_SOURCE_INVENTORY" "$native_config"
        )
        for pinned in "${pinned_inputs[@]}"; do
            mountpoint -q "$pinned" || fatal "actor input is not bind-pinned: $pinned"
            pinned_options=$(findmnt -n -o OPTIONS -T "$pinned" | tail -n 1)
            [[ ,$pinned_options, == *,ro,* ]] || \
                fatal "bind-pinned actor input is not read-only: $pinned"
        done
        [[ $(findmnt -n -o FSTYPE -T "$VALIDATION_ROOT" | tail -n 1) == tmpfs && \
           $(findmnt -n -o FSTYPE -T "$TEST_ROOT" | tail -n 1) == tmpfs ]] || \
            fatal "held-out trees are not masked"
        [[ -n ${ROUND_DAGGER_JAIL_HANDLE_ROOT:-} && \
           $(findmnt -n -o FSTYPE -T "$ROUND_DAGGER_JAIL_HANDLE_ROOT" | tail -n 1) == tmpfs && \
           ! -r $ROUND_DAGGER_JAIL_HANDLE_ROOT && ! -x $ROUND_DAGGER_JAIL_HANDLE_ROOT ]] || \
            fatal "temporary source handles remain reachable"
        [[ ! -e /workspace/codex/roborender-agent/.git && \
           ! -e /workspace/codex/roborender-agent/workspaces ]] || \
            fatal "a sibling worktree or Git store escaped the broad mask"
        for git_store in "$ROOT/.git" "$GR00T_ROOT/.git"; do
            [[ $(findmnt -n -o FSTYPE -T "$git_store" | tail -n 1) == tmpfs && \
               ! -r $git_store && ! -x $git_store ]] || \
                fatal "Git store remains reachable: $git_store"
        done
        for sensitive in "${PUBLIC_SPLIT_DETAILS[@]}"; do
            [[ $(stat -Lc '%t:%T' "$sensitive") == 1:3 ]] || \
                fatal "public split detail is not masked: $sensitive"
        done
        [[ ! -v CUDA_VISIBLE_DEVICES && ! -v NVIDIA_VISIBLE_DEVICES ]] || \
            fatal "CUDA/NVIDIA visibility remapping survived confinement"
        [[ $(findmnt -n -o SOURCE -T /dev/dri | tail -n 1) == native-actor-dri ]] || \
            fatal "rank-private DRM tmpfs is absent"
        private_drm_entries=(/dev/dri/*)
        [[ ${#private_drm_entries[@]} == 2 ]] || \
            fatal "rank-private DRM view must contain exactly two nodes"
        for expected_drm in "$ROUND_DAGGER_DRM_CARD_DEVICE" \
            "$ROUND_DAGGER_DRM_RENDER_DEVICE"; do
            private_drm=/dev/dri/${expected_drm##*/}
            [[ -c $private_drm && ! -L $private_drm ]] && \
                mountpoint -q "$private_drm" || \
                fatal "rank-private DRM node differs: $private_drm"
        done
        physical_nvidia_devices=(/dev/nvidia[0-9]*)
        [[ ${#physical_nvidia_devices[@]} == 8 ]] || \
            fatal "rank-private NVIDIA inventory changed"
        for gpu_device in "${physical_nvidia_devices[@]}"; do
            if [[ $gpu_device == "$ROUND_DAGGER_NVIDIA_DEVICE" ]]; then
                [[ -c $gpu_device && \
                   $(stat -Lc '%t:%T' "$gpu_device") != 1:3 ]] || \
                    fatal "assigned NVIDIA device is not exposed"
            else
                [[ $(stat -Lc '%t:%T' "$gpu_device") == 1:3 ]] || \
                    fatal "unassigned NVIDIA device remains exposed: $gpu_device"
            fi
        done
        for control_device in /dev/nvidiactl /dev/nvidia-modeset \
            /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
            [[ -c $control_device && ! -L $control_device ]] || \
                fatal "common NVIDIA control device was not preserved: $control_device"
        done
        no_new_privs=$(awk '/^NoNewPrivs:/ {print $2}' /proc/self/status)
        cap_eff=$(awk '/^CapEff:/ {print $2}' /proc/self/status)
        cap_bnd=$(awk '/^CapBnd:/ {print $2}' /proc/self/status)
        [[ $no_new_privs == 1 ]] || fatal "no-new-privs was not applied"
        for caps in "$cap_eff" "$cap_bnd"; do
            (( (16#$caps & ((1 << 1) | (1 << 2) | (1 << 21) | (1 << 27))) == 0 )) || \
                fatal "sys_admin/DAC/mknod capabilities remain after confinement"
        done
        ;;
    *) fatal "invalid private mount stage" ;;
esac

[[ $ROUND_DAGGER_OBSERVATION_SOURCE == native ]] || fatal "observation source must be native"
[[ $ROUND_DAGGER_ACTOR_ROLE == bc_collect && $ROUND_DAGGER_PURPOSE == train ]] ||
    fatal "BC rank requires role=bc_collect and purpose=train"
[[ $ROUND_DAGGER_BETA == 1.0 ]] || fatal "BC rank requires beta=1.0"
[[ $ROUND_DAGGER_GENERATION_ID == gen-000-bc ]] || fatal "generation must be gen-000-bc"
[[ $ROUND_DAGGER_CAMPAIGN_ROOT == /workspace/project/roborender/data ]] || fatal "campaign root differs"
[[ $ROUND_DAGGER_GENERATION_ROOT == "$ROUND_DAGGER_CAMPAIGN_ROOT/BC/isaac/collection" ]] || fatal "generation root differs"
[[ $ROUND_DAGGER_CAMPAIGN_ID == roborender-h200-v001 && $ROUND_DAGGER_MACRO_ROUND == 0 ]] || fatal "campaign identity differs"
[[ $ROUND_DAGGER_ACTOR_WORLD_SIZE == 8 ]] || fatal "actor world size must be 8"
[[ $ROUND_DAGGER_GPU_COUNT == 8 && $ROUND_DAGGER_GPU_NAME == 'NVIDIA H200' && \
   $ROUND_DAGGER_COMPUTE_CAPABILITY == 9.0 ]] || fatal "runtime identity must be exact 8xH200"
[[ $ROUND_DAGGER_ACTOR_RANK =~ ^[0-7]$ && \
   $ROUND_DAGGER_ASSIGNED_PHYSICAL_INDEX == "$ROUND_DAGGER_ACTOR_RANK" ]] || \
    fatal "actor rank must map one-to-one onto H200 indices 0..7"
[[ $ROUND_DAGGER_ENVS_PER_GPU =~ ^(1|32)$ ]] || fatal "unapproved env count"
[[ $ROUND_DAGGER_SATURATION_WAVE_INDEX =~ ^(0|[1-9][0-9]*)$ ]] || fatal "invalid wave index"
[[ $ROUND_DAGGER_SIMULATOR_BASE_SEED =~ ^(0|[1-9][0-9]*)$ ]] || fatal "invalid base seed"
case "$ROUND_DAGGER_SATURATION_PHASE:$ROUND_DAGGER_STEPS_PER_ENVIRONMENT:$ROUND_DAGGER_NATIVE_PROBE_ONLY:$ROUND_DAGGER_ENVS_PER_GPU" in
    smoke:16:1:1|probe:128:1:32|collect:512:0:32) ;;
    *) fatal "phase, steps, and probe-only flag disagree" ;;
esac
[[ ! -v CUDA_VISIBLE_DEVICES && ! -v NVIDIA_VISIBLE_DEVICES ]] || \
    fatal "native H200 rank forbids CUDA/NVIDIA visibility remapping"
[[ $GR00T_ROOT == "$ROOT/GR00T-VisualSim2Real" && \
   -x $GR00T_PYTHON && -d $GR00T_ROOT && ! -L $GR00T_ROOT ]] || \
    fatal "GR00T runtime/source topology differs"
[[ -d $MOTION_ROOT && ! -L $MOTION_ROOT ]] || fatal "33-demo motion root is missing or unsafe"
for pair in \
    "$ROUND_DAGGER_TRAIN_PARTITION:$ROUND_DAGGER_TRAIN_PARTITION_SHA256" \
    "$ROUND_DAGGER_POLICY_CHECKPOINT:$ROUND_DAGGER_POLICY_SHA256" \
    "$ROUND_DAGGER_TEACHER_CHECKPOINT:$ROUND_DAGGER_TEACHER_SHA256" \
    "$ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST:$ROUND_DAGGER_ISAAC_RUNTIME_SHA256" \
    "$ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST:$ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256" \
    "$ROUND_DAGGER_ACTOR_SOURCE_INVENTORY:$ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256" \
    "$ROUND_DAGGER_GR00T_SOURCE_INVENTORY:$ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256"; do
    path=${pair%%:*}; digest=${pair#*:}
    [[ $path == /* && -f $path && ! -L $path ]] || fatal "input file is missing or unsafe: $path"
    [[ $digest =~ ^[0-9a-f]{64}$ ]] || fatal "input SHA-256 is malformed"
    [[ $(sha256sum "$path" | awk '{print $1}') == "$digest" ]] || fatal "input SHA-256 mismatch: $path"
done

PYTHONPATH="$ROOT" python3 - \
    "$ROUND_DAGGER_ACTOR_SOURCE_INVENTORY" \
    "$ROUND_DAGGER_GR00T_SOURCE_INVENTORY" \
    "$ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST" \
    "$ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST" \
    "$ROOT" "$GR00T_ROOT" "$GR00T_PYTHON" \
    "$ROUND_DAGGER_ISAAC_RUNTIME_ROOT" "$ROUND_DAGGER_PYTHON_RUNTIME_ROOT" \
    "$ROUND_DAGGER_ISAAC_SETUP_SCRIPT" "$ROBORENDER_HEADLESS_EXPERIENCE" \
    "$ROUND_DAGGER_CUDNN_LIBRARY_DIR" \
    "$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT" \
    "$ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT" \
    "$ROUND_DAGGER_SYSTEM_LIBRARY_DIR" <<'PY'
import sys
from pathlib import Path

from round_dagger.native_actor_file_inventory import verify_native_actor_bundle

(
    actor_manifest,
    gr00t_manifest,
    isaac_manifest,
    python_manifest,
    actor_root,
    gr00t_root,
    gr00t_python,
    isaac_root,
    python_root,
    isaac_setup,
    headless_experience,
    cudnn_root,
    isaaclab_source_root,
    dependency_root,
    system_library_root,
) = map(Path, sys.argv[1:])
verify_native_actor_bundle(
    actor_manifest=actor_manifest,
    gr00t_manifest=gr00t_manifest,
    isaac_manifest=isaac_manifest,
    python_manifest=python_manifest,
    actor_root=actor_root,
    gr00t_root=gr00t_root,
    gr00t_python=gr00t_python,
    isaac_root=isaac_root,
    python_root=python_root,
    isaac_setup=isaac_setup,
    headless_experience=headless_experience,
    cudnn_root=cudnn_root,
    isaaclab_source_root=isaaclab_source_root,
    dependency_root=dependency_root,
    system_library_root=system_library_root,
    full_runtime_trees=False,
)
PY
[[ $ROUND_DAGGER_SPLIT_SHA256 =~ ^[0-9a-f]{64}$ && \
   $ROUND_DAGGER_RUNTIME_POLICY_SHA256 =~ ^[0-9a-f]{64}$ ]] || fatal "runtime/split SHA-256 is malformed"
for name in ROUND_DAGGER_OBSERVATION_RUNTIME_SHA256 ROUND_DAGGER_ISAAC_RUNTIME_SHA256 \
    ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256 ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256 \
    ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256 ROUND_DAGGER_BETA_SHA256 \
    ROUND_DAGGER_CONFIG_SHA256 ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256 \
    ROUND_DAGGER_GPU_DEVICE_NAMESPACE_SHA256 \
    ROUND_DAGGER_HOST_RUNTIME_ATTESTATION_SHA256; do
    [[ ${!name} =~ ^[0-9a-f]{64}$ ]] || fatal "$name is malformed"
done
[[ $ROUND_DAGGER_GR00T_REVISION =~ ^[0-9a-f]{40}$ ]] || fatal "GR00T revision is malformed"
native_config=$GR00T_ROOT/gr00t/rl/config/exp/loco_manip/wsdpt_round_actor_native_3view.yaml
[[ -f $native_config && ! -L $native_config && \
   $(sha256sum "$native_config" | awk '{print $1}') == "$ROUND_DAGGER_CONFIG_SHA256" ]] || \
    fatal "native actor config SHA-256 mismatch"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import json
import os
from pathlib import Path
import stat
from round_dagger.artifacts import canonical_json_sha256, validate_physical_gpu_inventory
from round_dagger.native_actor_saturation import (
    COMMON_NVIDIA_CONTROL_DEVICES,
    GPU_DEVICE_NAMESPACE_SCHEMA,
    native_actor_gpu_namespace_marker,
)

def device_record(path):
    candidate = Path(path)
    metadata = candidate.lstat()
    if candidate.is_symlink() or not stat.S_ISCHR(metadata.st_mode):
        raise SystemExit(f"GPU namespace path is not a direct character device: {path}")
    return {
        "path": str(candidate),
        "major": os.major(metadata.st_rdev),
        "minor": os.minor(metadata.st_rdev),
    }

inventory = validate_physical_gpu_inventory(
    json.loads(os.environ["ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON"])
)
if canonical_json_sha256(inventory) != os.environ["ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256"]:
    raise SystemExit("physical GPU inventory SHA-256 mismatch")
if canonical_json_sha256({"beta": 1.0}) != os.environ["ROUND_DAGGER_BETA_SHA256"]:
    raise SystemExit("BC beta SHA-256 mismatch")
host_attestation = {
    "schema": "fast_roborender.round_dagger.native_actor_host_attestation.v1",
    "manifests": {
        "actor_source": os.environ["ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256"],
        "gr00t_source": os.environ["ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256"],
        "isaac_runtime": os.environ["ROUND_DAGGER_ISAAC_RUNTIME_SHA256"],
        "python_runtime": os.environ["ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256"],
    },
}
if canonical_json_sha256(host_attestation) != os.environ[
    "ROUND_DAGGER_HOST_RUNTIME_ATTESTATION_SHA256"
]:
    raise SystemExit("host runtime attestation binding differs")
rank = int(os.environ["ROUND_DAGGER_ACTOR_RANK"])
physical_index = int(os.environ["ROUND_DAGGER_ASSIGNED_PHYSICAL_INDEX"])
expected = json.loads(os.environ["ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON"])
if (
    physical_index != 0
    or expected != inventory[physical_index]
    or expected["rank"] != physical_index
    or expected["physical_index"] != physical_index
):
    raise SystemExit("rank physical GPU mapping differs from fleet inventory")
if "CUDA_VISIBLE_DEVICES" in os.environ or "NVIDIA_VISIBLE_DEVICES" in os.environ:
    raise SystemExit("CUDA/NVIDIA visibility remapping survived confinement")
if os.environ["ROUND_DAGGER_ASSIGNED_GPU_UUID"] != expected["uuid"]:
    raise SystemExit("assigned H200 UUID differs from the physical inventory")
gpu_namespace = json.loads(os.environ["ROUND_DAGGER_GPU_DEVICE_NAMESPACE_JSON"])
if canonical_json_sha256(gpu_namespace) != os.environ[
    "ROUND_DAGGER_GPU_DEVICE_NAMESPACE_SHA256"
]:
    raise SystemExit("GPU device namespace SHA-256 mismatch")
expected_namespace_keys = {
    "schema", "rank", "physical_index", "uuid", "pci_bus_id", "pci_function",
    "nvidia_device", "drm_card_device", "drm_render_device",
    "common_control_devices",
}
if (
    not isinstance(gpu_namespace, dict)
    or set(gpu_namespace) != expected_namespace_keys
    or gpu_namespace["schema"] != GPU_DEVICE_NAMESPACE_SCHEMA
    or gpu_namespace["rank"] != physical_index
    or gpu_namespace["physical_index"] != physical_index
    or gpu_namespace["uuid"] != expected["uuid"]
    or gpu_namespace["pci_bus_id"] != expected["pci_bus_id"]
):
    raise SystemExit("GPU device namespace identity differs")
for field, environment_name in (
    ("nvidia_device", "ROUND_DAGGER_NVIDIA_DEVICE"),
    ("drm_card_device", "ROUND_DAGGER_DRM_CARD_DEVICE"),
    ("drm_render_device", "ROUND_DAGGER_DRM_RENDER_DEVICE"),
):
    if (
        gpu_namespace[field]["path"] != os.environ[environment_name]
        or device_record(os.environ[environment_name]) != gpu_namespace[field]
    ):
        raise SystemExit(f"rank-private device identity differs: {field}")
control_paths = list(COMMON_NVIDIA_CONTROL_DEVICES)
if [row["path"] for row in gpu_namespace["common_control_devices"]] != control_paths:
    raise SystemExit("common NVIDIA control-device set differs")
if [device_record(path) for path in control_paths] != gpu_namespace[
    "common_control_devices"
]:
    raise SystemExit("common NVIDIA control-device identity differs")
if sorted(entry.name for entry in Path("/dev/dri").iterdir()) != sorted(
    (
        Path(os.environ["ROUND_DAGGER_DRM_CARD_DEVICE"]).name,
        Path(os.environ["ROUND_DAGGER_DRM_RENDER_DEVICE"]).name,
    )
):
    raise SystemExit("rank-private DRM view contains an unassigned node")
null_record = device_record("/dev/null")
for candidate in sorted(Path("/dev").glob("nvidia[0-9]*")):
    observed = device_record(candidate)
    if str(candidate) == os.environ["ROUND_DAGGER_NVIDIA_DEVICE"]:
        if observed != gpu_namespace["nvidia_device"]:
            raise SystemExit("assigned NVIDIA device identity differs")
    elif observed["major"] != null_record["major"] or observed["minor"] != null_record["minor"]:
        raise SystemExit(f"unassigned NVIDIA device remains exposed: {candidate}")
print(native_actor_gpu_namespace_marker(rank=rank, gpu_namespace=gpu_namespace))
runtime = {
    "schema": "fast_roborender.round_dagger.native_h200_observation_runtime.v1",
    "observation_source": "native",
    "isaac_runtime_sha256": os.environ["ROUND_DAGGER_ISAAC_RUNTIME_SHA256"],
    "actor_python_freeze_sha256": os.environ["ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256"],
    "config_sha256": os.environ["ROUND_DAGGER_CONFIG_SHA256"],
    "actor_source_inventory_sha256": os.environ["ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256"],
    "gr00t_revision": os.environ["ROUND_DAGGER_GR00T_REVISION"],
    "gr00t_source_tree_sha256": os.environ["ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256"],
    "gpu_name": os.environ["ROUND_DAGGER_GPU_NAME"],
    "compute_capability": os.environ["ROUND_DAGGER_COMPUTE_CAPABILITY"],
    "gpu_count": int(os.environ["ROUND_DAGGER_GPU_COUNT"]),
    "physical_gpu_inventory": inventory,
    "physical_gpu_inventory_sha256": os.environ[
        "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256"
    ],
}
if canonical_json_sha256(runtime) != os.environ["ROUND_DAGGER_OBSERVATION_RUNTIME_SHA256"]:
    raise SystemExit("native observation runtime SHA-256 mismatch")
PY

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$MOTION_ROOT" "$ROUND_DAGGER_TRAIN_PARTITION" <<'PY'
import hashlib, json, os, stat, sys
from pathlib import Path

root, manifest = map(Path, sys.argv[1:])
document = json.loads(manifest.read_text(encoding="utf-8"))
expected = {row["relative_path"]: row for row in document["groups"]}
actual = {}
for entry in os.scandir(root):
    info = entry.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or entry.name.startswith("."):
        raise SystemExit(f"train-only mount contains unsafe entry: {entry.name}")
    actual[entry.name] = (Path(entry.path), info.st_size)
if set(actual) != set(expected):
    raise SystemExit("train-only mounted inventory differs from partition")
for name, (path, size) in actual.items():
    row = expected[name]
    if size != row["size_bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["motion_sha256"]:
        raise SystemExit(f"train-only mounted motion differs: {name}")
PY

mapfile -t ROTATION < <(
    python3 - "$ROUND_DAGGER_TRAIN_GROUP_IDS_JSON" \
        "${ROUND_DAGGER_GROUP_ROTATION_JSON:-}" "$ROUND_DAGGER_ACTOR_RANK" \
        "$ROUND_DAGGER_ENVS_PER_GPU" "$ROUND_DAGGER_SATURATION_PHASE" \
        "$ROUND_DAGGER_ACTOR_WORLD_SIZE" <<'PY'
import json, sys
groups_raw, rotation_raw, rank_raw, envs_raw, phase, world_raw = sys.argv[1:]
groups = json.loads(groups_raw)
if not isinstance(groups, list) or not groups or any(type(item) is not str or not item for item in groups):
    raise SystemExit("invalid ordered train group IDs")
rank, envs, world = int(rank_raw), int(envs_raw), int(world_raw)
if world != 1 or rank not in range(world):
    raise SystemExit("virtual actor topology differs")
if phase == "collect":
    if not rotation_raw:
        raise SystemExit("collection requires ROUND_DAGGER_GROUP_ROTATION_JSON")
    rotation = json.loads(rotation_raw)
    if not isinstance(rotation, dict) or set(rotation) != {"start_group_offset", "rank_group_offsets", "counts"}:
        raise SystemExit("group rotation fields differ")
    offsets = rotation["rank_group_offsets"]
    if not isinstance(offsets, list) or len(offsets) != world or any(type(v) is not int or v < 0 for v in offsets):
        raise SystemExit("rank group offsets differ")
    offset = offsets[rank]
else:
    offset = (rank * envs) % len(groups)
print(json.dumps(groups, separators=(",", ":")))
print(offset)
PY
)
[[ ${#ROTATION[@]} == 2 && ${ROTATION[1]} =~ ^(0|[1-9][0-9]*)$ ]] || fatal "motion rotation is invalid"
GROUP_IDS_JSON=${ROTATION[0]}
MOTION_ASSIGNMENT_OFFSET=${ROTATION[1]}

if [[ $ROUND_DAGGER_SATURATION_PHASE == collect ]]; then
    required_collect=(
        ROUND_DAGGER_COHORT_DIR ROUND_DAGGER_COHORT_INDEX
        ROUND_DAGGER_ROLLOUT_ID
        ROUND_DAGGER_GLOBAL_ENV_OFFSET ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON
        ROUND_DAGGER_GROUP_ROTATION_JSON
        ROUND_DAGGER_ACTOR_SEED_CONTRACT_JSON
        ROUND_DAGGER_ACTOR_SEED_CONTRACT_SHA256
    )
    for name in "${required_collect[@]}"; do
        [[ -n ${!name:-} ]] || fatal "collection is missing $name"
    done
    expected_cohort=$ROUND_DAGGER_GENERATION_ROOT/conditioning/waves/wave-$(printf '%03d' "$ROUND_DAGGER_SATURATION_WAVE_INDEX")
    [[ $ROUND_DAGGER_COHORT_DIR == "$expected_cohort" ]] || fatal "cohort path differs from exact wave namespace"
    mapfile -t SEED_BINDING < <(
        PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import json, os
from round_dagger.artifacts import canonical_json_sha256
from round_dagger.seed_policy import rank_seed_assignment, validate_actor_seed_contract

rank = int(os.environ["ROUND_DAGGER_ACTOR_RANK"])
wave = int(os.environ["ROUND_DAGGER_SATURATION_WAVE_INDEX"])
envs = int(os.environ["ROUND_DAGGER_ENVS_PER_GPU"])
world = int(os.environ["ROUND_DAGGER_ACTOR_WORLD_SIZE"])
offset = int(os.environ["ROUND_DAGGER_GLOBAL_ENV_OFFSET"])
if world != 1 or rank not in range(world):
    raise SystemExit("virtual actor topology differs")
if int(os.environ["ROUND_DAGGER_COHORT_INDEX"]) != wave or offset != wave * world * envs:
    raise SystemExit("wave/cohort/global environment identity differs")
contract = json.loads(os.environ["ROUND_DAGGER_ACTOR_SEED_CONTRACT_JSON"])
if canonical_json_sha256(contract) != os.environ["ROUND_DAGGER_ACTOR_SEED_CONTRACT_SHA256"]:
    raise SystemExit("actor seed contract SHA-256 mismatch")
contract = validate_actor_seed_contract(
    contract,
    campaign_id=os.environ["ROUND_DAGGER_CAMPAIGN_ID"],
    macro_round=int(os.environ["ROUND_DAGGER_MACRO_ROUND"]),
    cohort_index=wave,
    expected_ranks=list(range(world)),
    expected_global_env_ids=list(range(offset, offset + world * envs)),
)
assignment = rank_seed_assignment(contract, rank)
if int(os.environ["ROUND_DAGGER_SIMULATOR_BASE_SEED"]) != contract["simulator_base_seed"]:
    raise SystemExit("simulator base seed differs from immutable wave contract")
print(contract["simulator_base_seed"])
print(assignment["simulator_seed"])
PY
    )
    [[ ${#SEED_BINDING[@]} == 2 ]] || fatal "actor seed binding is incomplete"
    wave_seed=${SEED_BINDING[0]}
    effective_seed=${SEED_BINDING[1]}
else
    collection_only=(
        ROUND_DAGGER_COHORT_DIR ROUND_DAGGER_COHORT_INDEX
        ROUND_DAGGER_ROLLOUT_ID ROUND_DAGGER_GLOBAL_ENV_OFFSET
        ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON
        ROUND_DAGGER_GROUP_ROTATION_JSON
        ROUND_DAGGER_ACTOR_SEED_CONTRACT_JSON
        ROUND_DAGGER_ACTOR_SEED_CONTRACT_SHA256
    )
    for name in "${collection_only[@]}"; do
        [[ ! -v $name ]] || fatal "probe/smoke inherited collection-only state: $name"
    done
    # The fixed 128-environment production probe and smoke use a common
    # rank-local derivation that never depends on the attempt/log namespace.
    wave_seed=$ROUND_DAGGER_SIMULATOR_BASE_SEED
    effective_seed=$((10#$wave_seed + 10#$ROUND_DAGGER_ACTOR_RANK))
fi

export ROUND_DAGGER_EFFECTIVE_SIMULATOR_SEED=$effective_seed
run_dir=$ROUND_DAGGER_GENERATION_ROOT/runtime/attempts/$ROUND_DAGGER_ATTEMPT_ID/$ROUND_DAGGER_SATURATION_PHASE/wave-$(printf '%03d' "$ROUND_DAGGER_SATURATION_WAVE_INDEX")/gpu-$(printf '%02d' "$ROUND_DAGGER_ACTOR_RANK")
mkdir -p "$run_dir/hydra" "$run_dir/output"
process_state=$run_dir/process-state
cache_identity=$ROUND_DAGGER_ISAAC_RUNTIME_SHA256-$ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256
physical_cache=$ROUND_DAGGER_GENERATION_ROOT/runtime/physical-cache/gpu-$(printf '%02d' "$ROUND_DAGGER_ASSIGNED_PHYSICAL_INDEX")/$ROUND_DAGGER_ASSIGNED_GPU_UUID/$cache_identity
mkdir -p "$process_state"/{xdg-runtime,tmp,kit-logs,crash} \
    "$physical_cache"/{home,xdg-cache,xdg-config,xdg-data,hf,matplotlib,torch,kit-portable,warp,cuda}
chmod 700 "$process_state" "$process_state"/*
chmod 700 "$physical_cache" "$physical_cache"/*
# Mutable caches are keyed by physical index, UUID, and both complete runtime
# hashes. Cohorts never overlap on one admitted GPU, and the controller proves
# zero-memory/process quiescence before reuse, so smoke/probe warming is reused
# without sharing mutable state concurrently or across hardware/runtimes.
export XDG_CACHE_HOME=$physical_cache/xdg-cache
export HOME=$physical_cache/home
export XDG_CONFIG_HOME=$physical_cache/xdg-config
export XDG_DATA_HOME=$physical_cache/xdg-data
export XDG_RUNTIME_DIR=$process_state/xdg-runtime
export HF_HOME=$physical_cache/hf
export MPLCONFIGDIR=$physical_cache/matplotlib
export TORCH_HOME=$physical_cache/torch
export TMPDIR=$process_state/tmp
export WARP_CACHE_PATH=$physical_cache/warp
export CUDA_CACHE_PATH=$physical_cache/cuda
trainer_success_marker=$process_state/trainer-body.success
[[ ! -e $trainer_success_marker && ! -L $trainer_success_marker ]] || \
    fatal "trainer success marker exists before launch"
export ROUND_DAGGER_TRAINER_SUCCESS_MARKER=$trainer_success_marker
readonly ROUND_DAGGER_TRAINER_SUCCESS_MARKER
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
# Isaac's graceful close can spend many minutes draining Replicator after the
# trainer has already published its immutable success proof.  Native actor
# ranks are isolated processes with no reusable Kit state, so use Isaac's
# supported immediate process shutdown path.  The success marker below remains
# the fail-closed proof that the trainer body completed before shutdown.
export ROUND_DAGGER_SIMULATION_APP_SKIP_CLEANUP=1
readonly ROUND_DAGGER_SIMULATION_APP_SKIP_CLEANUP
unset PYTHONPATH
# The attested setup reconstructs exact Python and library search paths from
# the single QROOT bundle; it cannot append ambient or legacy-worktree paths.
for protected_name in "${required[@]}" ROOT GR00T_ROOT GR00T_PYTHON; do
    case "$protected_name" in
        OMNI_KIT_ACCEPT_EULA|OMNI_KIT_ALLOW_ROOT|ACCEPT_EULA|PRIVACY_CONSENT)
            # The attested setup script re-exports these exact consent values.
            # Freeze them immediately after sourcing instead.
            continue
            ;;
    esac
    readonly "$protected_name"
done
# Rehydrate the exact self-contained Isaac environment inside confinement.
set +u
# shellcheck source=/dev/null
source "$ROUND_DAGGER_ISAAC_SETUP_SCRIPT"
set -Eeuo pipefail
readonly OMNI_KIT_ACCEPT_EULA OMNI_KIT_ALLOW_ROOT ACCEPT_EULA PRIVACY_CONSENT
expected_ld_library_path=$ROUND_DAGGER_CUDNN_LIBRARY_DIR:$ROUND_DAGGER_SYSTEM_LIBRARY_DIR:/usr/lib/x86_64-linux-gnu
[[ $LD_LIBRARY_PATH == "$expected_ld_library_path" ]] || \
    fatal "Isaac setup produced a non-canonical library search path"
expected_pythonpath=$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT/isaaclab:$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT/isaaclab_assets:$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT/isaaclab_contrib:$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT/isaaclab_mimic:$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT/isaaclab_rl:$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT/isaaclab_tasks:$ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT:$ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT/cmeel.prefix/lib/python3.11/site-packages:$ROOT:$GR00T_ROOT
[[ $PYTHONPATH == "$expected_pythonpath" ]] || \
    fatal "Isaac setup produced a non-canonical Python search path"
export ROUND_DAGGER_EXPECT_POLICY_VIEW_COUNT=3
export ROUND_DAGGER_LOCAL_GPU_INDEX=0
export EXPECTED_TRAIN_CHECKPOINT_SHA256="$ROUND_DAGGER_POLICY_SHA256"
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK MASTER_ADDR MASTER_PORT

"$GR00T_PYTHON" - <<'PY'
import importlib
import os
from pathlib import Path
import sys

unsafe = [
    entry for entry in sys.path
    if "/workspaces/" in entry or "rollout_gen" in entry or "/bootstrap" in entry
]
if unsafe:
    raise SystemExit(f"legacy worktree/bootstrap path entered sys.path: {unsafe}")

expected = {
    "hydra": Path(os.environ["ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT"]),
    "omegaconf": Path(os.environ["ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT"]),
    "isaaclab": Path(os.environ["ROUND_DAGGER_ISAACLAB_SOURCE_ROOT"]) / "isaaclab",
    "isaaclab_rl": Path(os.environ["ROUND_DAGGER_ISAACLAB_SOURCE_ROOT"]) / "isaaclab_rl",
    "gr00t": Path(os.environ["GR00T_ROOT"]),
}
for name, root in expected.items():
    module = importlib.import_module(name)
    origin = getattr(module, "__file__", None)
    if origin is None:
        raise SystemExit(f"qualified module has no regular origin: {name}")
    try:
        Path(origin).resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"qualified module resolved outside its sealed root: {name}") from exc

from isaaclab.app import AppLauncher  # noqa: F401
importlib.import_module("gr00t.rl.train_agent_trl")
print("NATIVE_ACTOR_IMPORT_ROOTS_OK")
PY

"$GR00T_PYTHON" - <<'PY'
import json
import os
import subprocess
import torch

expected = json.loads(os.environ["ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON"])
inventory = json.loads(os.environ["ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON"])
rank = int(os.environ["ROUND_DAGGER_ACTOR_RANK"])
physical_index = int(os.environ["ROUND_DAGGER_ASSIGNED_PHYSICAL_INDEX"])
if (
    len(inventory) != 8
    or physical_index != 0
    or expected != inventory[physical_index]
    or expected["rank"] != physical_index
    or expected["physical_index"] != physical_index
):
    raise SystemExit("rank assignment differs from the attested H200 fleet")
if "CUDA_VISIBLE_DEVICES" in os.environ or "NVIDIA_VISIBLE_DEVICES" in os.environ:
    raise SystemExit("CUDA/NVIDIA visibility remapping survived runtime setup")
if os.environ["ROUND_DAGGER_ASSIGNED_GPU_UUID"] != expected["uuid"]:
    raise SystemExit("assigned H200 UUID differs from the attested fleet")
if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
    raise SystemExit("rank must see exactly one logical CUDA device zero")
properties = torch.cuda.get_device_properties(0)
raw_uuid = getattr(properties, "uuid", None)
if isinstance(raw_uuid, bytes):
    raw_uuid = raw_uuid.decode("ascii")
actual_uuid = str(raw_uuid)
if not actual_uuid.startswith("GPU-"):
    actual_uuid = "GPU-" + actual_uuid
actual_uuid = actual_uuid.lower().replace("gpu-", "GPU-", 1)
if (
    properties.name != expected["name"]
    or f"{properties.major}.{properties.minor}" != expected["compute_capability"]
    or actual_uuid != expected["uuid"]
    or int(properties.total_memory) < 140_000 * 1024 * 1024
):
    raise SystemExit("logical CUDA device zero differs from the assigned H200")
query = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,compute_cap,memory.total,mig.mode.current",
        "--format=csv,noheader,nounits",
    ],
    check=True, capture_output=True, text=True, timeout=15,
)
rows = [row for row in query.stdout.splitlines() if row.strip()]
if len(rows) != 1:
    raise SystemExit("rank-private NVML view must contain exactly one GPU")
tokens = [token.strip() for token in rows[0].split(",")]
if len(tokens) != 7 or tokens[0] != "0":
    raise SystemExit("rank-private NVML device is not local index zero")
observed = {
    "rank": physical_index,
    "uuid": tokens[1].lower().replace("gpu-", "GPU-", 1),
    "pci_bus_id": tokens[2].upper(),
    "name": tokens[3],
    "compute_capability": tokens[4],
    "total_memory_mib": int(tokens[5]),
    "mig_mode": tokens[6],
}
expected_projection = {
    key: expected[key]
    for key in (
        "rank", "uuid", "pci_bus_id", "name", "compute_capability",
        "total_memory_mib", "mig_mode",
    )
}
if observed != expected_projection:
    raise SystemExit("NVML CUDA device differs from attested rank mapping")
print(
    f"NATIVE_ACTOR_GPU_OK rank={rank} uuid={expected['uuid']} "
    f"pci={expected['pci_bus_id']} logical_cuda=0 local_nvml=0 "
    "visible_devices=1 cuda_visible_devices=unset"
)
PY

command=(
    "$GR00T_PYTHON" "$ROOT/scripts/round_dagger/actor_training_entrypoint.py"
    "$GR00T_ROOT/gr00t/rl/train_agent_trl.py"
    "--kit_args=--portable --portable-root $physical_cache/kit-portable --/log/file=$process_state/kit-logs/kit.log --/crashreporter/dumpDir=$process_state/crash --/crashreporter/enabled=false --/crashreporter/gatherUserStory=0"
    +exp=loco_manip/wsdpt_round_actor_native_3view
    seed="$wave_seed" num_envs="$ROUND_DAGGER_ENVS_PER_GPU"
    # Allocate production-sized rollout storage during smoke/probes. The native
    # trainer executes only the requested 16/128 steps in no-replay modes, so the
    # measured peak already includes the exact 512-step collection allocation.
    algo.config.num_steps_per_env=512
    algo.config.num_learning_epochs=0 algo.config.rollout_with_teacher_num_steps=0
    algo.config.enforce_teacher_rollout=false
    algo.config.actor.backbone.vision_module.module_config_dict.layer_config.pretrained=false
    algo.config.native_dagger.role=bc_collect algo.config.native_dagger.purpose=train
    algo.config.native_dagger.observation_source=native
    algo.config.native_dagger.rollout_action_source=beta
    algo.config.native_dagger.beta_schedule=constant
    algo.config.native_dagger.beta_start=1.0 algo.config.native_dagger.beta_end=1.0
    algo.config.native_dagger.beta_decay_iters=1
    algo.config.native_dagger.split_manifest_path="$ROUND_DAGGER_TRAIN_PARTITION"
    algo.config.native_dagger.split_manifest_sha256="$ROUND_DAGGER_TRAIN_PARTITION_SHA256"
    algo.config.native_dagger.split_sha256="$ROUND_DAGGER_SPLIT_SHA256"
    algo.config.native_dagger.group_ids="$GROUP_IDS_JSON"
    algo.config.native_dagger.panel_sha256=null
    env.config.reset_from_dataset.use_motion_file_dir=true
    env.config.reset_from_dataset.motion_file_dir="$MOTION_ROOT"
    env.config.reset_from_dataset.motion_split_manifest="$ROUND_DAGGER_TRAIN_PARTITION"
    env.config.reset_from_dataset.motion_split_manifest_sha256="$ROUND_DAGGER_TRAIN_PARTITION_SHA256"
    env.config.reset_from_dataset.motion_split_role=train
    env.config.reset_from_dataset.motion_split_group_ids="$GROUP_IDS_JSON"
    env.config.reset_from_dataset.motion_split_verify_content=true
    env.config.reset_from_dataset.motion_split_offset=0
    env.config.reset_from_dataset.motion_split_limit=null
    env.config.reset_from_dataset.motion_assignment_offset="$MOTION_ASSIGNMENT_OFFSET"
    checkpoint="$ROUND_DAGGER_POLICY_CHECKPOINT" resume=false
    teacher_actor_path="$ROUND_DAGGER_TEACHER_CHECKPOINT"
    simulator.config.cameras.policy_views='[head_color,hand_left,hand_right]'
    policy_view_count=3 env.config.native_policy_view_count=3
    simulator.config.render_results=false headless=true use_wandb=false
    experiment_name="gen-000-bc-${ROUND_DAGGER_SATURATION_PHASE}-w${ROUND_DAGGER_SATURATION_WAVE_INDEX}-g${ROUND_DAGGER_ACTOR_RANK}"
    experiment_dir="$run_dir" hydra.run.dir="$run_dir/hydra" output_dir="$run_dir/output"
)

cd "$GR00T_ROOT"
echo "NATIVE_ACTOR_SEED rank=$ROUND_DAGGER_ACTOR_RANK effective_seed=$ROUND_DAGGER_EFFECTIVE_SIMULATOR_SEED"
"${command[@]}"
expected_trainer_success="NATIVE_ACTOR_TRAINER_BODY_OK rank=$ROUND_DAGGER_ACTOR_RANK phase=$ROUND_DAGGER_SATURATION_PHASE"
[[ -f $trainer_success_marker && ! -L $trainer_success_marker && \
   $(stat -c '%a' -- "$trainer_success_marker") == 400 && \
   $(stat -c '%s' -- "$trainer_success_marker") == $((${#expected_trainer_success} + 1)) ]] || \
    fatal "trainer did not publish one immutable body-success proof"
mapfile -t trainer_success_lines < "$trainer_success_marker"
[[ ${#trainer_success_lines[@]} == 1 && \
   ${trainer_success_lines[0]} == "$expected_trainer_success" ]] || \
    fatal "trainer body-success proof differs from this rank and phase"
echo "$expected_trainer_success"
kit_log=$process_state/kit-logs/kit.log
[[ -f $kit_log && ! -L $kit_log ]] || \
    fatal "Kit did not publish one safe per-rank log"
"$GR00T_PYTHON" - "$kit_log" "$physical_cache/kit-portable" \
    "$ROUND_DAGGER_ACTOR_RANK" <<'PY'
import json
import os
from pathlib import Path
import sys

from round_dagger.native_actor_saturation import verify_native_actor_kit_log

kit_log = Path(sys.argv[1])
expected_portable_root = Path(sys.argv[2])
rank = int(sys.argv[3])
print(
    verify_native_actor_kit_log(
        text=kit_log.read_text(encoding="utf-8", errors="replace"),
        rank=rank,
        expected_portable_root=expected_portable_root,
        expected_physical_gpu=json.loads(
            os.environ["ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON"]
        ),
    )
)
PY
echo "NATIVE_ACTOR_RANK_OK rank=$ROUND_DAGGER_ACTOR_RANK phase=$ROUND_DAGGER_SATURATION_PHASE"
