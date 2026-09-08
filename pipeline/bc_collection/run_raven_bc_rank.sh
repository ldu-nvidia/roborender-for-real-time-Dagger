#!/usr/bin/env bash
# Launch one independent H200 shard of Raven's exact BC collector.
set -Eeuo pipefail

if (( $# != 6 )); then
    echo "usage: $0 GPU RANK WORLD_SIZE ENVS_PER_RANK OUTPUT_DIR RUN_TAG" >&2
    exit 2
fi

gpu=$1
rank=$2
world_size=$3
envs_per_rank=$4
output_dir=$5
run_tag=$6

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
GR00T=$ROOT/pipeline/GR00T-VisualSim2Real
RUNTIME=$ROOT/runtime/isaac_actor/qualified_h200_v1
PYTHON=$RUNTIME/runtime/isaacsim-5.1/bin/python
ACCELERATE=$RUNTIME/runtime/isaacsim-5.1/bin/accelerate
TEACHER=$ROOT/checkpoints/teachers/robot_policy/g1_privileged_action_teacher_wsdpt_step044500.pt
WALK=$ROOT/checkpoints/teachers/locomotion/g1_locomotion_walk_controller.pt
STAND=$ROOT/checkpoints/teachers/locomotion/g1_locomotion_stand_controller.pt

for file in "$PYTHON" "$ACCELERATE" "$TEACHER" "$WALK" "$STAND"; do
    [[ -e $file ]] || { echo "missing required input: $file" >&2; exit 2; }
done
[[ $gpu =~ ^[0-9]+$ && $rank =~ ^[0-9]+$ && $world_size =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $envs_per_rank =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $run_tag =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
mkdir -p "$output_dir"
rank_tag=$(printf '%s_r%02d' "$run_tag" "$rank")
[[ -z $(find "$output_dir" -maxdepth 1 -type f -name "chunk_${rank_tag}_*.npz" -print -quit) ]] || {
    echo "rank/run output already exists in: $output_dir" >&2
    exit 2
}

# Kit/PhysX does not support CUDA_VISIBLE_DEVICES ordinal remapping reliably on
# this bare-metal host. Enter a private mount namespace, retain only the
# assigned NVIDIA character device and its matching DRM card/render nodes, and
# let every subsystem use its resulting process-local cuda:0.
if [[ ${ROBORENDER_BC_PRIVATE_GPU_READY:-0} != 1 ]]; then
    exec unshare --mount --propagation private \
        env ROBORENDER_BC_PRIVATE_GPU_READY=1 "$0" "$@"
fi
unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
gpu_uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
gpu_pci=$(nvidia-smi -i "$gpu" --query-gpu=pci.bus_id --format=csv,noheader \
    | tr '[:upper:]' '[:lower:]' | sed 's/^00000000:/0000:/' | tr -d '[:space:]')
gpu_minor=$(awk -F: '/Device Minor/ {gsub(/[[:space:]]/, "", $2); print $2}' \
    "/proc/driver/nvidia/gpus/$gpu_pci/information")
[[ $gpu_minor =~ ^[0-9]+$ ]] || {
    echo "cannot resolve NVIDIA device minor for physical index $gpu" >&2
    exit 2
}
nvidia_device=/dev/nvidia${gpu_minor}
drm_card=
drm_render=
for sysdev in /sys/class/drm/card[0-9]* /sys/class/drm/renderD[0-9]*; do
    [[ -e $sysdev/device ]] || continue
    [[ ${sysdev##*/} != *-* ]] || continue
    [[ $(basename "$(readlink -f "$sysdev/device")") == "$gpu_pci" ]] || continue
    case ${sysdev##*/} in
        card*) drm_card=/dev/dri/${sysdev##*/} ;;
        renderD*) drm_render=/dev/dri/${sysdev##*/} ;;
    esac
done
for device in "$nvidia_device" "$drm_card" "$drm_render"; do
    [[ -n $device && -c $device && ! -L $device ]] || {
        echo "cannot resolve private GPU device for physical index $gpu: $device" >&2
        exit 2
    }
done
handles=$output_dir/.gpu-device-handles
mkdir -p "$handles"
touch "$handles/nvidia" "$handles/card" "$handles/render"
mount --bind "$nvidia_device" "$handles/nvidia"
mount --bind "$drm_card" "$handles/card"
mount --bind "$drm_render" "$handles/render"
for device in /dev/nvidia[0-9]*; do
    [[ -c $device && ! -L $device ]] || exit 2
    mount --bind /dev/null "$device"
done
mount --bind "$handles/nvidia" "$nvidia_device"
mount -t tmpfs -o size=64k,mode=0755,nosuid,noexec roborender-bc-dri /dev/dri
private_card=/dev/dri/${drm_card##*/}
private_render=/dev/dri/${drm_render##*/}
touch "$private_card" "$private_render"
mount --bind "$handles/card" "$private_card"
mount --bind "$handles/render" "$private_render"

export ROBORENDER_BC_RANK=$rank
export ROBORENDER_BC_WORLD_SIZE=$world_size
export ROBORENDER_BC_RUN_TAG=$run_tag
export ROBORENDER_BC_ISAAC_OUTPUT_DIR=$output_dir
# Raven uses SLURM_JOB_ID as the chunk prefix. A rank-unique local value is a
# filename/topology relocation only and prevents eight independent processes
# from overwriting one another.
export SLURM_JOB_ID=$rank_tag
export ROBORENDER_HEADLESS_EXPERIENCE=$RUNTIME/runtime/IsaacLab/apps/phc.isaaclab.python.headless.rendering.kit

# Each Isaac/Kit process needs private mutable state. Sharing Kit's database,
# shader caches, log, or crash directory across eight simultaneous processes
# can race during Vulkan initialization and produce ERROR_DEVICE_LOST. These
# paths are operational isolation only; no simulation or dataset parameter is
# changed.
runtime_state=$output_dir/.rank-runtime
cache_state=$output_dir/.rank-cache
mkdir -p \
    "$runtime_state"/{xdg-runtime,tmp,kit-logs,crash} \
    "$cache_state"/{xdg-cache,xdg-config,xdg-data,hf,matplotlib,torch,warp,cuda} \
    "$cache_state"/{kit-cache,root-ov,root-nvidia}
chmod 700 "$runtime_state" "$runtime_state"/* "$cache_state" "$cache_state"/*
export XDG_CACHE_HOME=$cache_state/xdg-cache
export XDG_CONFIG_HOME=$cache_state/xdg-config
export XDG_DATA_HOME=$cache_state/xdg-data
export XDG_RUNTIME_DIR=$runtime_state/xdg-runtime
export HF_HOME=$cache_state/hf
export MPLCONFIGDIR=$cache_state/matplotlib
export TORCH_HOME=$cache_state/torch
export TMPDIR=$runtime_state/tmp
export WARP_CACHE_PATH=$cache_state/warp
export CUDA_CACHE_PATH=$cache_state/cuda

# Kit also writes to cache/log directories rooted at its installation and at
# the effective user's standard Omniverse locations. Give each rank its own
# writable copy/bind inside this mount namespace. A bind is used because the
# workspace's virtiofs does not support overlay mounts.
kit_root=$RUNTIME/runtime/isaacsim-5.1/lib/python3.11/site-packages/isaacsim/kit
cp -a --reflink=auto "$kit_root/cache/." "$cache_state/kit-cache/"
mount --bind "$cache_state/kit-cache" "$kit_root/cache"
mount --bind "$runtime_state/kit-logs" "$kit_root/logs"
mount --bind "$cache_state/root-ov" /root/.cache/ov
mount --bind "$cache_state/root-nvidia" /root/.nvidia-omniverse

# Reuse only the qualified H200 Python/Isaac library environment. This does
# not alter Raven's environment, camera, rollout, or dataset parameters.
source "$ROOT/pipeline/common/setup_h200_isaac_runtime.sh"

EXPECTED_GPU_UUID=$gpu_uuid "$PYTHON" - <<'PY'
import os
import torch

if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
    raise SystemExit("private BC collector must see exactly one logical CUDA device zero")
value = getattr(torch.cuda.get_device_properties(0), "uuid", "")
if isinstance(value, bytes):
    value = value.decode("ascii")
actual = str(value)
if not actual.startswith("GPU-"):
    actual = "GPU-" + actual
if actual.lower() != os.environ["EXPECTED_GPU_UUID"].lower():
    raise SystemExit(
        f"private BC collector GPU UUID mismatch: {actual} != "
        f"{os.environ['EXPECTED_GPU_UUID']}"
    )
print(f"ROBORENDER_BC_PRIVATE_GPU_READY uuid={actual}", flush=True)
PY
if [[ ${ROBORENDER_BC_DEVICE_PROBE_ONLY:-0} == 1 ]]; then
    exit 0
fi

cd "$GR00T"
exec "$ACCELERATE" launch --num_processes 1 \
    gr00t/rl/train_agent_trl.py \
    +exp=loco_manip/wsdpt_videogen_collect \
    "num_envs=$envs_per_rank" \
    "seed=$rank" \
    "teacher_actor_path=$TEACHER" \
    "algo.config.homie_walk_model_path=$WALK" \
    "algo.config.homie_stand_model_path=$STAND" \
    "algo.config.videogen.collect_dir=$output_dir" \
    "algo.config.videogen.dump_dir=null" \
    "algo.config.num_learning_epochs=0" \
    headless=True \
    use_wandb=false \
    "experiment_name=raven_bc_collect_${run_tag}_r$(printf '%02d' "$rank")" \
    project_name=roborender_bc_collection
