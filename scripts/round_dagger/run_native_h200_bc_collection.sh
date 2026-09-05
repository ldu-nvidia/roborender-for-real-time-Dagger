#!/bin/bash
# Launch gen-000 BC collection on admitted GPU 0 of the native 8xH200 node.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
CAMPAIGN=/workspace/project/roborender/data
GENERATION=$CAMPAIGN/BC/collection
CHECKPOINT_INPUT=$GENERATION/checkpoints/input
RUNTIME_INPUT=$CHECKPOINT_INPUT/gen-000-bc__runtime-v008
QROOT=/workspace/codex/roborender-agent/runs/h200_rollout_qualification_20260903T201136Z
PARTITION=$CAMPAIGN/BC/contracts/train_motion_partition.v1.json
MOTIONS=$CAMPAIGN/BC/inputs/train-motion-files
CONFIG=$ROOT/GR00T-VisualSim2Real/gr00t/rl/config/exp/loco_manip/wsdpt_round_actor_native_3view.yaml
POLICY=$CHECKPOINT_INPUT/gen-000-bc__bootstrap-v000/checkpoint.pt
TEACHER=$CHECKPOINT_INPUT/gen-000-bc__teacher-v000/teacher.pt

fatal() { echo "native H200 BC controller failed: $*" >&2; exit 2; }

mode=${1:-}
attempt_id=${2:-}
case "$mode" in
    smoke|production) ;;
    *) echo "usage: $0 smoke|production ATTEMPT-ID [--gpu-indices 0]" >&2; exit 2 ;;
esac
[[ $attempt_id =~ ^[a-z0-9][a-z0-9-]{0,63}$ ]] || {
    echo "ATTEMPT-ID must be lowercase alphanumeric/hyphen and at most 64 bytes" >&2
    exit 2
}
if (( $# != 2 )); then
    [[ $# == 4 && ${3:-} == --gpu-indices && ${4:-} == 0 ]] || \
        fatal "current BC topology admits exactly --gpu-indices 0"
fi

# Every smoke and production attempt shares generation-local rank caches and
# conditioning namespaces. Hold one kernel lease on the canonical generation
# directory across the final exec; the kernel releases it on every process exit
# and cannot leave stale PID or mutable lock-file state behind.
[[ -d $GENERATION && ! -L $GENERATION && \
   $(realpath -e -- "$GENERATION") == "$GENERATION" ]] || \
    fatal "generation lock target is missing or unsafe"
command -v flock >/dev/null || fatal "flock is unavailable"
exec {CONTROLLER_LOCK_FD}<"$GENERATION" || fatal "cannot open generation lock target"
[[ $(readlink -f -- "/proc/$$/fd/$CONTROLLER_LOCK_FD") == "$GENERATION" ]] || \
    fatal "generation lock descriptor changed identity"
flock -n "$CONTROLLER_LOCK_FD" || \
    fatal "another smoke/production controller is active for gen-000-bc"

sha_file() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
    local path=$1 expected=$2
    [[ -f $path && ! -L $path && $(sha_file "$path") == "$expected" ]] || {
        echo "immutable input is missing or differs: $path" >&2
        exit 2
    }
}

require_hash "$POLICY" 720a975812f6083192234d650033511c821b6902da2ef7734d87943f6c0a359a
require_hash "$TEACHER" 943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38
require_hash "$PARTITION" 771566684701664322fb824a73837b51446c7572cd506ed540a266a0277969da
[[ $(git -C "$ROOT/GR00T-VisualSim2Real" rev-parse HEAD) == \
   9dd746d7fe86cbc09f20546695ab46bec609a3ea ]] || {
    echo "GR00T revision differs" >&2
    exit 2
}

export GR00T_PYTHON=$QROOT/runtime/isaacsim-5.1/bin/python
export ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT=$QROOT
export ROUND_DAGGER_ISAAC_RUNTIME_ROOT=$QROOT/runtime/isaacsim-5.1
export ROUND_DAGGER_PYTHON_RUNTIME_ROOT=$QROOT
export ROUND_DAGGER_ISAACLAB_SOURCE_ROOT=$QROOT/runtime/IsaacLab/source
export ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT=$QROOT/runtime/actor-python-site-packages
export ROUND_DAGGER_SYSTEM_LIBRARY_DIR=$QROOT/packages/root/usr/lib/x86_64-linux-gnu
export ROUND_DAGGER_CUDNN_LIBRARY_DIR=$QROOT/runtime/isaacsim-5.1/lib/python3.11/site-packages/nvidia/cudnn/lib
export ROUND_DAGGER_ISAAC_SETUP_SCRIPT=$ROOT/scripts/round_dagger/setup_native_h200_isaac_runtime.sh
export ROBORENDER_HEADLESS_EXPERIENCE=$QROOT/runtime/IsaacLab/apps/phc.isaaclab.python.headless.rendering.kit

export ROUND_DAGGER_POLICY_CHECKPOINT=$POLICY
export ROUND_DAGGER_POLICY_SHA256=720a975812f6083192234d650033511c821b6902da2ef7734d87943f6c0a359a
export ROUND_DAGGER_RUNTIME_POLICY_SHA256=cd6f70aa9d589e810b7ab22999941930742a94621870457aadd068d2394deb9a
export ROUND_DAGGER_TEACHER_CHECKPOINT=$TEACHER
export ROUND_DAGGER_TEACHER_SHA256=943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38
export ROUND_DAGGER_BETA_SHA256=cd0f4b65443e05862b89dd761a0752958aa4902eba8f79b63002b12a6d1d0e15
export ROUND_DAGGER_CONFIG_SHA256=$(sha_file "$CONFIG")
export ROUND_DAGGER_GR00T_REVISION=9dd746d7fe86cbc09f20546695ab46bec609a3ea

export ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST=$RUNTIME_INPUT/isaac-runtime-inventory.v1.json
export ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST=$RUNTIME_INPUT/actor-python-inventory.v1.json
export ROUND_DAGGER_ACTOR_SOURCE_INVENTORY=$RUNTIME_INPUT/actor-source-inventory.v1.json
export ROUND_DAGGER_GR00T_SOURCE_INVENTORY=$RUNTIME_INPUT/gr00t-source-inventory.v1.json
export ROUND_DAGGER_ISAAC_RUNTIME_SHA256=$(sha_file "$ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST")
export ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256=$(sha_file "$ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST")
export ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256=$(sha_file "$ROUND_DAGGER_ACTOR_SOURCE_INVENTORY")
export ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256=$(sha_file "$ROUND_DAGGER_GR00T_SOURCE_INVENTORY")

export ROUND_DAGGER_GPU_NAME='NVIDIA H200'
export ROUND_DAGGER_COMPUTE_CAPABILITY=9.0
export ROUND_DAGGER_GPU_COUNT=8
mapfile -t physical_binding < <(
    PYTHONPATH=$ROOT /usr/bin/python3 - <<'PY'
import json
from round_dagger.artifacts import canonical_json_sha256
from round_dagger.native_actor_saturation import query_physical_gpu_inventory

inventory = query_physical_gpu_inventory()
print(json.dumps(inventory, sort_keys=True, separators=(",", ":")))
print(canonical_json_sha256(inventory))
PY
)
[[ ${#physical_binding[@]} == 2 ]] || {
    echo "physical H200 inventory binding failed" >&2
    exit 2
}
export ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON=${physical_binding[0]}
export ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256=${physical_binding[1]}
export ROUND_DAGGER_OBSERVATION_RUNTIME_SHA256=$(
    PYTHONPATH=$ROOT /usr/bin/python3 - <<'PY'
import json
import os
from round_dagger.artifacts import canonical_json_sha256

document = {
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
    "physical_gpu_inventory": json.loads(os.environ["ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON"]),
    "physical_gpu_inventory_sha256": os.environ["ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256"],
}
print(canonical_json_sha256(document))
PY
)

export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

arguments=(
    "$ROOT/scripts/round_dagger/run_native_h200_actor_fleet.py"
    --campaign-root "$CAMPAIGN"
    --train-partition "$PARTITION"
    --train-partition-sha256 771566684701664322fb824a73837b51446c7572cd506ed540a266a0277969da
    --train-motion-root "$MOTIONS"
    --environments-per-shard 4
    --attempt-id "$attempt_id"
    --gpu-indices 0
)
if [[ $mode == smoke ]]; then
    arguments+=(--smoke-only)
fi
exec /usr/bin/python3 "${arguments[@]}"
