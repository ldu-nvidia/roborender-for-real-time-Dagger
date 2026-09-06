#!/bin/bash
# Run one rank of a fixed, labeled validation cohort on one visible H200.
set -Eeuo pipefail

# Reject loader/shell state before invoking even one external utility.  The
# fleet removes every BASH_FUNC_* entry; clearing functions here is a second
# boundary for direct invocations.
SAFE_ACTOR_PATH=/usr/sbin:/usr/bin:/sbin:/bin
[[ ${PATH:-} == "$SAFE_ACTOR_PATH" ]] || {
    echo "native H200 validation rank: child PATH was not sanitized by the fleet" >&2
    exit 2
}
[[ -z ${LD_LIBRARY_PATH:-} && -z ${LD_PRELOAD:-} && -z ${BASH_ENV:-} ]] || {
    echo "native H200 validation rank: inherited loader/shell injection state" >&2
    exit 2
}
export PATH=$SAFE_ACTOR_PATH SAFE_ACTOR_PATH
while IFS= read -r inherited_function; do
    builtin unset -f -- "$inherited_function"
done < <(builtin compgen -A function)
unset inherited_function
readonly PATH SAFE_ACTOR_PATH

fatal() { echo "native H200 validation rank: $*" >&2; exit 2; }

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
export ROOT
GR00T_ROOT=${GR00T_ROOT:?set GR00T_ROOT}
GR00T_PYTHON=${GR00T_PYTHON:?set GR00T_PYTHON}

required=(
    ROUND_DAGGER_ACTOR_RANK ROUND_DAGGER_ACTOR_WORLD_SIZE
    ROUND_DAGGER_ENVS_PER_GPU ROUND_DAGGER_STEPS_PER_ENVIRONMENT
    ROUND_DAGGER_ENVS_PER_SHARD ROUND_DAGGER_ACTOR_ROLE ROUND_DAGGER_PURPOSE
    ROUND_DAGGER_BETA ROUND_DAGGER_BETA_SHA256 ROUND_DAGGER_CAMPAIGN_ID
    ROUND_DAGGER_MACRO_ROUND ROUND_DAGGER_COHORT_INDEX ROUND_DAGGER_ROLLOUT_ID
    ROUND_DAGGER_COHORT_DIR ROUND_DAGGER_GLOBAL_ENV_OFFSET
    ROUND_DAGGER_VALIDATION_ROOT ROUND_DAGGER_VALIDATION_REPLAY_ROOT
    ROUND_DAGGER_VALIDATION_INPUT_ROOT ROUND_DAGGER_VALIDATION_TEST_ROOT
    ROUND_DAGGER_VALIDATION_CAMPAIGN_ROOT
    ROUND_DAGGER_VALIDATION_SOURCE_MOTION_ROOT ROUND_DAGGER_VALIDATION_MOTION_ROOT
    ROUND_DAGGER_VALIDATION_PARTITION ROUND_DAGGER_VALIDATION_PARTITION_SHA256
    ROUND_DAGGER_VALIDATION_REGISTRY ROUND_DAGGER_VALIDATION_REGISTRY_SHA256
    ROUND_DAGGER_SPLIT_SHA256 ROUND_DAGGER_VALIDATION_GROUP_IDS_JSON
    ROUND_DAGGER_VALIDATION_ASSIGNMENT ROUND_DAGGER_VALIDATION_ASSIGNMENT_SHA256
    ROUND_DAGGER_VALIDATION_COLLECTION_ID
    ROUND_DAGGER_VALIDATION_SOURCE_GENERATION_ID
    ROUND_DAGGER_VALIDATION_SOURCE_MACRO_ROUND
    ROUND_DAGGER_VALIDATION_PHYSICAL_GPU ROUND_DAGGER_VALIDATION_PAIRED_SEED
    ROUND_DAGGER_VALIDATION_HYDRA_BASE_SEED
    ROUND_DAGGER_EFFECTIVE_SIMULATOR_SEED
    ROUND_DAGGER_VALIDATION_MOTION_ASSIGNMENT_OFFSET
    ROUND_DAGGER_VALIDATION_RUN_DIR
    ROUND_DAGGER_POLICY_CHECKPOINT ROUND_DAGGER_POLICY_SHA256
    ROUND_DAGGER_POLICY_SPEC ROUND_DAGGER_POLICY_SPEC_SHA256
    ROUND_DAGGER_RUNTIME_POLICY_SHA256 ROUND_DAGGER_TEACHER_CHECKPOINT
    ROUND_DAGGER_TEACHER_SHA256 ROUND_DAGGER_CONFIG_SHA256
    ROUND_DAGGER_OBSERVATION_SOURCE ROUND_DAGGER_OBSERVATION_RUNTIME_SHA256
    ROUND_DAGGER_ISAAC_RUNTIME_SHA256 ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256
    ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256 ROUND_DAGGER_GR00T_REVISION
    ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256 ROUND_DAGGER_GPU_NAME
    ROUND_DAGGER_COMPUTE_CAPABILITY ROUND_DAGGER_GPU_COUNT
    ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON
    ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256
    ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON
    ROUND_DAGGER_HOST_RUNTIME_ATTESTATION_SHA256
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

# These variables activate the train-only gen-000 filename contract in the
# exporter.  A validation process must never inherit even one of them.
for name in \
    ROUND_DAGGER_CAMPAIGN_ROOT ROUND_DAGGER_GENERATION_ROOT \
    ROUND_DAGGER_GENERATION_ID ROUND_DAGGER_SATURATION_PHASE \
    ROUND_DAGGER_SATURATION_WAVE_INDEX ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON; do
    [[ -z ${!name:-} ]] || fatal "validation inherited forbidden BC naming: $name"
done

[[ $ROUND_DAGGER_ACTOR_WORLD_SIZE == 2 ]] || fatal "cohort world size must be two"
[[ $ROUND_DAGGER_ACTOR_RANK =~ ^[01]$ ]] || fatal "cohort rank must be zero or one"
[[ $ROUND_DAGGER_ENVS_PER_GPU == 32 ]] || fatal "each rank must own 32 environments"
[[ $ROUND_DAGGER_STEPS_PER_ENVIRONMENT == 512 ]] || fatal "each environment must contribute 512 steps"
[[ $ROUND_DAGGER_ENVS_PER_SHARD == 4 ]] || fatal "validation shard size must be four environments"
[[ $ROUND_DAGGER_ACTOR_ROLE == validation_collect && $ROUND_DAGGER_PURPOSE == validation ]] || \
    fatal "validation requires role=validation_collect and purpose=validation"
[[ $ROUND_DAGGER_BETA == 1.0 ]] || fatal "validation collection requires beta=1"
[[ $ROUND_DAGGER_CAMPAIGN_ID == roborender-h200-v001 ]] || fatal "campaign identity differs"
[[ $ROUND_DAGGER_VALIDATION_ROOT == /workspace/project/roborender/data/validation/heldout ]] || \
    fatal "validation output root differs"
[[ $ROUND_DAGGER_VALIDATION_TEST_ROOT == /workspace/project/roborender/data/test ]] || \
    fatal "test mask root differs"
[[ $ROUND_DAGGER_GPU_COUNT == 8 && $ROUND_DAGGER_GPU_NAME == 'NVIDIA H200' && \
   $ROUND_DAGGER_COMPUTE_CAPABILITY == 9.0 ]] || fatal "runtime is not exact 8x H200"
[[ $ROUND_DAGGER_VALIDATION_PHYSICAL_GPU =~ ^[0-7]$ ]] || fatal "physical GPU is outside [0,7]"
[[ ${CUDA_VISIBLE_DEVICES:-} == GPU-* ]] || \
    fatal "rank must be launched by its assigned physical GPU UUID"
[[ $OMNI_KIT_ACCEPT_EULA == YES && $OMNI_KIT_ALLOW_ROOT == 1 && \
   $ACCEPT_EULA == Y && $PRIVACY_CONSENT == Y ]] || \
    fatal "Isaac EULA/root/privacy runtime contract differs"
[[ $ROUND_DAGGER_VALIDATION_HYDRA_BASE_SEED =~ ^[0-9]+$ && \
   $ROUND_DAGGER_EFFECTIVE_SIMULATOR_SEED =~ ^[0-9]+$ && \
   $ROUND_DAGGER_VALIDATION_PAIRED_SEED == "$ROUND_DAGGER_EFFECTIVE_SIMULATOR_SEED" ]] || \
    fatal "paired simulator seed binding differs"
[[ $((10#$ROUND_DAGGER_VALIDATION_HYDRA_BASE_SEED + 10#$ROUND_DAGGER_ACTOR_RANK)) == \
   10#$ROUND_DAGGER_EFFECTIVE_SIMULATOR_SEED ]] || \
    fatal "train_agent_trl rank offset would change the paired seed"

for pair in \
    "$ROUND_DAGGER_VALIDATION_PARTITION:$ROUND_DAGGER_VALIDATION_PARTITION_SHA256" \
    "$ROUND_DAGGER_VALIDATION_REGISTRY:$ROUND_DAGGER_VALIDATION_REGISTRY_SHA256" \
    "$ROUND_DAGGER_VALIDATION_ASSIGNMENT:$ROUND_DAGGER_VALIDATION_ASSIGNMENT_SHA256" \
    "$ROUND_DAGGER_POLICY_CHECKPOINT:$ROUND_DAGGER_POLICY_SHA256" \
    "$ROUND_DAGGER_POLICY_SPEC:$ROUND_DAGGER_POLICY_SPEC_SHA256" \
    "$ROUND_DAGGER_TEACHER_CHECKPOINT:$ROUND_DAGGER_TEACHER_SHA256" \
    "$ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST:$ROUND_DAGGER_ISAAC_RUNTIME_SHA256" \
    "$ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST:$ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256" \
    "$ROUND_DAGGER_ACTOR_SOURCE_INVENTORY:$ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256" \
    "$ROUND_DAGGER_GR00T_SOURCE_INVENTORY:$ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256"; do
    path=${pair%%:*}
    digest=${pair#*:}
    [[ $path == /* && -f $path && ! -L $path && $digest =~ ^[0-9a-f]{64}$ ]] || \
        fatal "unsafe or malformed pinned input: $path"
    [[ $(sha256sum "$path" | awk '{print $1}') == "$digest" ]] || \
        fatal "pinned input digest mismatch: $path"
done
[[ $ROUND_DAGGER_PYTHON_RUNTIME_ROOT == "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT" && \
   $ROUND_DAGGER_ISAAC_RUNTIME_ROOT == \
       "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/runtime/isaacsim-5.1" && \
   $ROUND_DAGGER_ISAACLAB_SOURCE_ROOT == \
       "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/runtime/IsaacLab/source" && \
   $ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT == \
       "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/runtime/actor-python-site-packages" && \
   $ROUND_DAGGER_SYSTEM_LIBRARY_DIR == \
       "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/packages/root/usr/lib/x86_64-linux-gnu" && \
   $ROUND_DAGGER_CUDNN_LIBRARY_DIR == \
       "$ROUND_DAGGER_ISAAC_RUNTIME_ROOT/lib/python3.11/site-packages/nvidia/cudnn/lib" && \
   $ROBORENDER_HEADLESS_EXPERIENCE == \
       "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/runtime/IsaacLab/apps/phc.isaaclab.python.headless.rendering.kit" ]] || \
    fatal "qualified validation runtime topology differs"
gr00t_python_link_text=$(readlink "$GR00T_PYTHON") || \
    fatal "qualified GR00T Python is not a readable symlink"
gr00t_python_target=$(realpath -e "$GR00T_PYTHON") || \
    fatal "qualified GR00T Python target is unavailable"
[[ -x $GR00T_PYTHON && -L $GR00T_PYTHON && \
   $GR00T_PYTHON == "$ROUND_DAGGER_ISAAC_RUNTIME_ROOT/bin/python" && \
   $gr00t_python_link_text == /* && \
   $gr00t_python_link_text == "$gr00t_python_target" && \
   $gr00t_python_target == \
       "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT/runtime/python/cpython-3.11.13-linux-x86_64-gnu/bin/python3.11" && \
   -f $gr00t_python_target && \
   -d $GR00T_ROOT && ! -L $GR00T_ROOT ]] || \
    fatal "GR00T runtime is unsafe"
[[ -d $ROUND_DAGGER_VALIDATION_MOTION_ROOT && ! -L $ROUND_DAGGER_VALIDATION_MOTION_ROOT ]] || \
    fatal "validation-only motion root is unsafe"
[[ -d $ROUND_DAGGER_VALIDATION_SOURCE_MOTION_ROOT && ! -L $ROUND_DAGGER_VALIDATION_SOURCE_MOTION_ROOT ]] || \
    fatal "source all-motion root is unsafe"
[[ -d $ROUND_DAGGER_VALIDATION_TEST_ROOT && ! -L $ROUND_DAGGER_VALIDATION_TEST_ROOT ]] || \
    fatal "sealed test root is unsafe"
for path in "$ROOT" "$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT" \
    "$ROUND_DAGGER_ISAAC_RUNTIME_ROOT" \
    "$ROUND_DAGGER_PYTHON_RUNTIME_ROOT" "$ROUND_DAGGER_CUDNN_LIBRARY_DIR" \
    "$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT" \
    "$ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT" \
    "$ROUND_DAGGER_SYSTEM_LIBRARY_DIR"; do
    [[ $path == /* && -d $path && ! -L $path && $(realpath -e "$path") == "$path" ]] || \
        fatal "immutable source/runtime root is unsafe: $path"
done
[[ ! -e $ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT/sitecustomize.py ]] || \
    fatal "qualified dependency root contains sitecustomize.py"
if compgen -G "$ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT/*.pth" >/dev/null || \
   compgen -G "$ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT/*editable*" >/dev/null; then
    fatal "qualified dependency root contains path-injection metadata"
fi
[[ $GR00T_ROOT == "$ROOT/GR00T-VisualSim2Real" ]] || \
    fatal "GR00T source must be the repository checkout"
[[ $ROUND_DAGGER_ISAAC_SETUP_SCRIPT == /* && -f $ROUND_DAGGER_ISAAC_SETUP_SCRIPT && \
   ! -L $ROUND_DAGGER_ISAAC_SETUP_SCRIPT ]] || fatal "Isaac setup script is unsafe"
[[ $ROBORENDER_HEADLESS_EXPERIENCE == /* && -f $ROBORENDER_HEADLESS_EXPERIENCE && \
   ! -L $ROBORENDER_HEADLESS_EXPERIENCE ]] || fatal "headless experience is unsafe"
compgen -G "$ROUND_DAGGER_CUDNN_LIBRARY_DIR/libcudnn.so*" >/dev/null || \
    fatal "cuDNN library directory contains no libcudnn"
for utility in unshare mount umount findmnt setpriv mktemp nvidia-smi readlink tail; do
    command -v "$utility" >/dev/null 2>&1 || fatal "missing isolation utility: $utility"
done
[[ $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=name --format=csv,noheader | \
    sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//') == 'NVIDIA H200' ]] || \
    fatal "visible device is not exactly one NVIDIA H200"

run_dir=$ROUND_DAGGER_VALIDATION_RUN_DIR
[[ $run_dir == "$ROUND_DAGGER_VALIDATION_ROOT/replay/collections/$ROUND_DAGGER_VALIDATION_COLLECTION_ID/runtime/gpu-$(printf '%02d' "$ROUND_DAGGER_VALIDATION_PHYSICAL_GPU")" ]] || \
    fatal "rank runtime path differs"
[[ -d $run_dir && ! -L $run_dir ]] || fatal "rank runtime directory was not prepared create-only"
[[ -z $(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit) ]] || \
    fatal "rank runtime directory is not empty"
mkdir "$run_dir/hydra" "$run_dir/output" "$run_dir/process-state"
process_state=$run_dir/process-state
mkdir "$process_state/xdg-cache" "$process_state/xdg-config" \
    "$process_state/xdg-data" "$process_state/xdg-runtime" \
    "$process_state/hf" "$process_state/matplotlib" \
    "$process_state/torch" "$process_state/tmp"
chmod 700 "$run_dir" "$run_dir"/* "$process_state"/*
export XDG_CACHE_HOME=$process_state/xdg-cache
export XDG_CONFIG_HOME=$process_state/xdg-config
export XDG_DATA_HOME=$process_state/xdg-data
export XDG_RUNTIME_DIR=$process_state/xdg-runtime
export HF_HOME=$process_state/hf
export MPLCONFIGDIR=$process_state/matplotlib
export TORCH_HOME=$process_state/torch
export TMPDIR=$process_state/tmp
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
export ROUND_DAGGER_EXPECT_POLICY_VIEW_COUNT=3
export ROUND_DAGGER_LOCAL_GPU_INDEX=0
export EXPECTED_TRAIN_CHECKPOINT_SHA256="$ROUND_DAGGER_POLICY_SHA256"
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK MASTER_ADDR MASTER_PORT

native_config=$GR00T_ROOT/gr00t/rl/config/exp/loco_manip/wsdpt_round_actor_native_3view.yaml
[[ -f $native_config && ! -L $native_config && \
   $(sha256sum "$native_config" | awk '{print $1}') == "$ROUND_DAGGER_CONFIG_SHA256" ]] || \
    fatal "native actor config SHA-256 mismatch"

command=(
    "$GR00T_PYTHON" "$ROOT/scripts/round_dagger/actor_training_entrypoint.py"
    "$GR00T_ROOT/gr00t/rl/train_agent_trl.py"
    +exp=loco_manip/wsdpt_round_actor_native_3view
    seed="$ROUND_DAGGER_VALIDATION_HYDRA_BASE_SEED"
    num_envs=32
    algo.config.num_steps_per_env=512
    algo.config.num_learning_epochs=0
    algo.config.rollout_with_teacher_num_steps=0
    algo.config.enforce_teacher_rollout=false
    algo.config.native_dagger.role=validation_collect
    algo.config.native_dagger.purpose=validation
    algo.config.native_dagger.observation_source=native
    algo.config.native_dagger.rollout_action_source=beta
    algo.config.native_dagger.beta_schedule=constant
    algo.config.native_dagger.beta_start=1.0
    algo.config.native_dagger.beta_end=1.0
    algo.config.native_dagger.beta_decay_iters=1
    algo.config.native_dagger.split_manifest_path="$ROUND_DAGGER_VALIDATION_PARTITION"
    algo.config.native_dagger.split_manifest_sha256="$ROUND_DAGGER_VALIDATION_PARTITION_SHA256"
    algo.config.native_dagger.split_sha256="$ROUND_DAGGER_SPLIT_SHA256"
    algo.config.native_dagger.group_ids="$ROUND_DAGGER_VALIDATION_GROUP_IDS_JSON"
    algo.config.native_dagger.panel_sha256="$ROUND_DAGGER_VALIDATION_REGISTRY_SHA256"
    # The complete checkpoint is loaded strictly below.  Never let the ResNet
    # constructor perform an implicit network fetch or alter the initialization
    # before those checkpoint bytes are applied.
    algo.config.actor.backbone.vision_module.module_config_dict.layer_config.pretrained=false
    env.config.reset_from_dataset.use_motion_file_dir=true
    env.config.reset_from_dataset.motion_file_dir="$ROUND_DAGGER_VALIDATION_MOTION_ROOT"
    env.config.reset_from_dataset.motion_split_manifest="$ROUND_DAGGER_VALIDATION_PARTITION"
    env.config.reset_from_dataset.motion_split_manifest_sha256="$ROUND_DAGGER_VALIDATION_PARTITION_SHA256"
    env.config.reset_from_dataset.motion_split_role=validation
    env.config.reset_from_dataset.motion_split_group_ids="$ROUND_DAGGER_VALIDATION_GROUP_IDS_JSON"
    env.config.reset_from_dataset.motion_split_verify_content=true
    env.config.reset_from_dataset.motion_split_offset=0
    env.config.reset_from_dataset.motion_split_limit=null
    env.config.reset_from_dataset.motion_assignment_offset="$ROUND_DAGGER_VALIDATION_MOTION_ASSIGNMENT_OFFSET"
    checkpoint="$ROUND_DAGGER_POLICY_CHECKPOINT"
    resume=false
    teacher_actor_path="$ROUND_DAGGER_TEACHER_CHECKPOINT"
    simulator.config.cameras.policy_views='[head_color,hand_left,hand_right]'
    policy_view_count=3
    env.config.native_policy_view_count=3
    simulator.config.render_results=false
    headless=true
    use_wandb=false
    experiment_name="$ROUND_DAGGER_VALIDATION_COLLECTION_ID-gpu-$ROUND_DAGGER_VALIDATION_PHYSICAL_GPU"
    experiment_dir="$run_dir"
    hydra.run.dir="$run_dir/hydra"
    output_dir="$run_dir/output"
)

replay_roots=(
    "$ROUND_DAGGER_VALIDATION_CAMPAIGN_ROOT/BC/replay"
    "$ROUND_DAGGER_VALIDATION_CAMPAIGN_ROOT/dagger_1/replay"
    "$ROUND_DAGGER_VALIDATION_CAMPAIGN_ROOT/dagger_2/replay"
    "$ROUND_DAGGER_VALIDATION_CAMPAIGN_ROOT/dagger_3/replay"
    "$ROUND_DAGGER_VALIDATION_CAMPAIGN_ROOT/dagger_4/replay"
)
for path in "${replay_roots[@]}"; do
    [[ -d $path && ! -L $path ]] || fatal "training replay mask target is unsafe: $path"
done

# The simulator and exporter live in a private mount namespace.  Validation
# inputs become read-only; the sealed test set, the source all-motion directory,
# and every training replay directory become mode-000 bind masks.  Capabilities
# that could undo those mounts or bypass permissions are then dropped.
parent_pid_namespace=$(readlink /proc/self/ns/pid)
unshare --mount --propagation private --pid --fork --kill-child \
    --mount-proc=/proc -- \
    env ROUND_DAGGER_PARENT_PID_NAMESPACE="$parent_pid_namespace" \
    /bin/bash -Eeuo pipefail -s -- \
    "$ROUND_DAGGER_VALIDATION_INPUT_ROOT" \
    "$ROUND_DAGGER_VALIDATION_TEST_ROOT" \
    "$ROUND_DAGGER_VALIDATION_SOURCE_MOTION_ROOT" \
    "${replay_roots[@]}" -- "${command[@]}" <<'NAMESPACE'
inputs=$1
shift
[[ ${ROUND_DAGGER_PARENT_PID_NAMESPACE:-} == pid:* && \
   $(readlink /proc/self/ns/pid) != "$ROUND_DAGGER_PARENT_PID_NAMESPACE" ]] || {
    echo "private validation PID namespace was not established" >&2
    exit 2
}
[[ $(findmnt -fn -o FSTYPE -T /proc) == proc ]] || {
    echo "private validation procfs was not established" >&2
    exit 2
}
targets=()
while [[ $# -gt 0 && $1 != -- ]]; do
    targets+=("$1")
    shift
done
[[ $# -gt 0 && $1 == -- ]] || { echo "missing namespace command separator" >&2; exit 2; }
shift
command=("$@")
mounts=()
jail_stage=
cleanup() {
    local index
    if [[ -n ${jail_stage:-} ]]; then
        umount "$jail_stage/source" 2>/dev/null || true
        umount "$jail_stage/qualified" 2>/dev/null || true
        rmdir "$jail_stage/source" "$jail_stage/qualified" \
            "$jail_stage" 2>/dev/null || true
    fi
    for ((index=${#mounts[@]}-1; index>=0; index--)); do
        umount "${mounts[index]}" 2>/dev/null || true
    done
}
trap cleanup EXIT

# Replace /workspace with a tiny private filesystem containing only this
# attested checkout and the complete qualified QROOT.  This hides sibling
# worktrees and all other mutable workspace paths.  The two Git stores are
# separately masked below so history/objects cannot be used as alternate code.
workspace_root=/workspace
qualified_root=$ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT
[[ -d $workspace_root && ! -L $workspace_root && \
   $(realpath -e "$workspace_root") == "$workspace_root" && \
   $ROOT == "$workspace_root"/* && $qualified_root == "$workspace_root"/* && \
   -d $qualified_root && ! -L $qualified_root && \
   $(realpath -e "$qualified_root") == "$qualified_root" ]] || {
    echo "workspace/QROOT topology is unsafe for validation jail" >&2
    exit 2
}
case "$ROOT/" in "$qualified_root/"*) echo "actor source overlaps QROOT" >&2; exit 2 ;; esac
case "$qualified_root/" in "$ROOT/"*) echo "QROOT overlaps actor source" >&2; exit 2 ;; esac
for pinned_manifest in \
    "$ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST" \
    "$ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST" \
    "$ROUND_DAGGER_ACTOR_SOURCE_INVENTORY" \
    "$ROUND_DAGGER_GR00T_SOURCE_INVENTORY"; do
    case "$pinned_manifest/" in
        "$workspace_root/"*)
            case "$pinned_manifest/" in
                "$ROOT/"*) ;;
                "$qualified_root/"*) ;;
                *) echo "pinned manifest is hidden workspace state: $pinned_manifest" >&2; exit 2 ;;
            esac
            ;;
    esac
done
jail_stage=$(mktemp -d /tmp/round-dagger-validation-workspace.XXXXXX)
mkdir "$jail_stage/source" "$jail_stage/qualified"
mount --bind "$ROOT" "$jail_stage/source"
mount -o remount,bind,ro,nosuid,nodev "$jail_stage/source"
mount --bind "$qualified_root" "$jail_stage/qualified"
mount -o remount,bind,ro,nosuid,nodev "$jail_stage/qualified"
cd /
mount -t tmpfs -o size=64k,mode=0755,nosuid,nodev,noexec \
    native-validation-workspace-jail "$workspace_root"
mounts+=("$workspace_root")
mkdir -p "$(dirname "$ROOT")" "$(dirname "$qualified_root")"
mkdir "$ROOT" "$qualified_root"
mount --bind "$jail_stage/source" "$ROOT"
mount -o remount,bind,ro,nosuid,nodev "$ROOT"
mounts+=("$ROOT")
mount --bind "$jail_stage/qualified" "$qualified_root"
mount -o remount,bind,ro,nosuid,nodev "$qualified_root"
mounts+=("$qualified_root")
umount "$jail_stage/source"
umount "$jail_stage/qualified"
rmdir "$jail_stage/source" "$jail_stage/qualified" "$jail_stage"
jail_stage=
mount -o remount,ro,nosuid,nodev,noexec "$workspace_root"
workspace_mount_fstype=$(findmnt -n -o FSTYPE -M "$workspace_root" | tail -n 1)
[[ $workspace_mount_fstype == tmpfs && \
   $(findmnt -fn -o TARGET --target "$ROOT") == "$ROOT" && \
   $(findmnt -fn -o TARGET --target "$qualified_root") == "$qualified_root" && \
   ,$(findmnt -fn -o OPTIONS -T "$ROOT"), == *,ro,* && \
   ,$(findmnt -fn -o OPTIONS -T "$qualified_root"), == *,ro,* ]] || {
    echo "validation workspace/QROOT jail is not active" >&2
    exit 2
}
[[ -d $ROOT/.git && ! -L $ROOT/.git && \
   -d $GR00T_ROOT/.git && ! -L $GR00T_ROOT/.git ]] || {
    echo "validation Git-store mask targets are unsafe" >&2
    exit 2
}
targets+=("$ROOT/.git" "$GR00T_ROOT/.git")
cd "$ROOT"

# Bind every byte-bearing input used after authorization as a read-only
# mountpoint.  In particular, directory write permission cannot be used to
# rename a pinned checkpoint between the outer hash check and model loading.
readonly_targets=(
    "$inputs"
    "$ROUND_DAGGER_ISAAC_RUNTIME_ROOT"
    "$ROUND_DAGGER_PYTHON_RUNTIME_ROOT"
    "$ROUND_DAGGER_CUDNN_LIBRARY_DIR"
    "$ROUND_DAGGER_VALIDATION_PARTITION"
    "$ROUND_DAGGER_VALIDATION_REGISTRY"
    "$ROUND_DAGGER_VALIDATION_ASSIGNMENT"
    "$ROUND_DAGGER_POLICY_CHECKPOINT"
    "$ROUND_DAGGER_POLICY_SPEC"
    "$ROUND_DAGGER_TEACHER_CHECKPOINT"
    "$ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST"
    "$ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST"
    "$ROUND_DAGGER_ACTOR_SOURCE_INVENTORY"
    "$ROUND_DAGGER_GR00T_SOURCE_INVENTORY"
    "$GR00T_ROOT/gr00t/rl/config/exp/loco_manip/wsdpt_round_actor_native_3view.yaml"
)
declare -A readonly_seen=()
for readonly_target in "${readonly_targets[@]}"; do
    [[ -n ${readonly_seen[$readonly_target]:-} ]] && continue
    mount --bind "$readonly_target" "$readonly_target"
    mount -o remount,bind,ro,nosuid,nodev "$readonly_target"
    mounts+=("$readonly_target")
    readonly_seen[$readonly_target]=1
done
for target in "${targets[@]}"; do
    mask=$(mktemp -d /tmp/round-dagger-validation-mask.XXXXXX)
    chmod 000 "$mask"
    mount --bind "$mask" "$target"
    mounts+=("$target")
    rmdir "$mask"
    mount -o remount,bind,ro,nosuid,nodev,noexec "$target"
    [[ $(findmnt -fn -o TARGET --target "$target") == "$target" ]] || {
        echo "validation bind mask is not active: $target" >&2
        exit 2
    }
done
setpriv --no-new-privs \
    --bounding-set=-sys_admin,-dac_override,-dac_read_search \
    --inh-caps=-all --ambient-caps=-all -- \
    /bin/bash -Eeuo pipefail -s -- "$inputs" "${targets[@]}" -- "${command[@]}" <<'CAPDROP'
inputs=$1
shift
[[ ${ROUND_DAGGER_PARENT_PID_NAMESPACE:-} == pid:* && \
   $(readlink /proc/self/ns/pid) != "$ROUND_DAGGER_PARENT_PID_NAMESPACE" ]] || {
    echo "private validation PID namespace was not retained" >&2
    exit 2
}
[[ $(findmnt -fn -o FSTYPE -T /proc) == proc ]] || {
    echo "private validation procfs was not retained" >&2
    exit 2
}
targets=()
while [[ $# -gt 0 && $1 != -- ]]; do
    targets+=("$1")
    shift
done
shift
[[ $PATH == "$SAFE_ACTOR_PATH" ]] || {
    echo "validation executable PATH differs inside the jail" >&2
    exit 2
}
readonly PATH
[[ -r $inputs && -x $inputs ]] || { echo "validation inputs are inaccessible" >&2; exit 2; }
for target in "${targets[@]}"; do
    [[ ! -r $target && ! -x $target ]] || {
        echo "masked root remains accessible: $target" >&2
        exit 2
    }
done
while IFS= read -r protected_name; do
    readonly "$protected_name"
done < <(compgen -v ROUND_DAGGER_)
readonly CUDA_VISIBLE_DEVICES ROOT GR00T_ROOT GR00T_PYTHON
set +u
# shellcheck source=/dev/null
source "$ROUND_DAGGER_ISAAC_SETUP_SCRIPT"
set -Eeuo pipefail
expected_library_path=$ROUND_DAGGER_CUDNN_LIBRARY_DIR:$ROUND_DAGGER_SYSTEM_LIBRARY_DIR:/usr/lib/x86_64-linux-gnu
[[ $LD_LIBRARY_PATH == "$expected_library_path" ]] || {
    echo "Isaac setup produced a non-canonical library search path" >&2
    exit 2
}
expected_pythonpath=$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT/isaaclab:$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT/isaaclab_assets:$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT/isaaclab_contrib:$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT/isaaclab_mimic:$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT/isaaclab_rl:$ROUND_DAGGER_ISAACLAB_SOURCE_ROOT/isaaclab_tasks:$ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT:$ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT/cmeel.prefix/lib/python3.11/site-packages:$ROOT:$GR00T_ROOT
[[ $PYTHONPATH == "$expected_pythonpath" ]] || {
    echo "Isaac setup produced a non-canonical Python search path" >&2
    exit 2
}
readonly expected_library_path expected_pythonpath
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
        raise SystemExit(
            f"qualified module resolved outside its sealed root: {name}"
        ) from exc
from isaaclab.app import AppLauncher  # noqa: F401
importlib.import_module("gr00t.rl.train_agent_trl")
print("NATIVE_VALIDATION_IMPORT_ROOTS_OK")
PY
"$GR00T_PYTHON" - <<'PY'
import json
import os
import subprocess

import torch

expected = json.loads(os.environ["ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON"])
physical_gpu = int(os.environ["ROUND_DAGGER_VALIDATION_PHYSICAL_GPU"])
if expected["rank"] != physical_gpu or expected["physical_index"] != physical_gpu:
    raise SystemExit("validation physical GPU assignment differs from inventory")
if os.environ.get("CUDA_VISIBLE_DEVICES") != expected["uuid"]:
    raise SystemExit("CUDA_VISIBLE_DEVICES is not the attested physical UUID")
if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
    raise SystemExit("validation rank does not see exactly one logical CUDA device 0")
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
    raise SystemExit("Torch CUDA device differs from attested physical H200")
query = subprocess.run(
    [
        "nvidia-smi", "-i", expected["uuid"],
        "--query-gpu=index,uuid,pci.bus_id,name,compute_cap,memory.total,mig.mode.current",
        "--format=csv,noheader,nounits",
    ],
    check=True, capture_output=True, text=True, timeout=15,
)
lines = [line for line in query.stdout.splitlines() if line.strip()]
if len(lines) != 1:
    raise SystemExit("NVML query did not return exactly the assigned H200")
tokens = [token.strip() for token in lines[0].split(",")]
if len(tokens) != 7:
    raise SystemExit("NVML assigned-H200 inventory row is malformed")
observed = {
    "rank": physical_gpu,
    "physical_index": int(tokens[0]),
    "uuid": tokens[1].lower().replace("gpu-", "GPU-", 1),
    "pci_bus_id": tokens[2].upper(),
    "name": tokens[3],
    "compute_capability": tokens[4],
    "total_memory_mib": int(tokens[5]),
    "mig_mode": tokens[6],
}
if observed != expected:
    raise SystemExit("NVML CUDA device differs from attested validation mapping")
print(
    f"NATIVE_VALIDATION_GPU_OK physical_gpu={physical_gpu} "
    f"uuid={expected['uuid']} pci={expected['pci_bus_id']}"
)
PY
PYTHONPATH="$ROOT" \
    python3 -m round_dagger.native_h200_validation rank-preflight >/dev/null
PYTHONPATH="$ROOT" \
    python3 -m round_dagger.native_h200_validation verify-runtime \
        --critical-only >/dev/null
"$@"
PYTHONPATH="$ROOT" \
    python3 -m round_dagger.native_h200_validation verify-runtime \
        --critical-only >/dev/null
CAPDROP
NAMESPACE

echo "NATIVE_VALIDATION_RANK_OK physical_gpu=$ROUND_DAGGER_VALIDATION_PHYSICAL_GPU paired_seed=$ROUND_DAGGER_VALIDATION_PAIRED_SEED"
