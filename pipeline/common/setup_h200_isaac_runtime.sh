#!/usr/bin/env bash
# Source the single qualified local Isaac Sim 5.1 / IsaacLab H200 runtime.

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
ROBORENDER_ASSET_ROOT=${ROBORENDER_ASSET_ROOT:-$PROJECT_ROOT}
[[ $ROBORENDER_ASSET_ROOT == /* ]] || { echo "ROBORENDER_ASSET_ROOT must be absolute" >&2; return 2; }
GR00T_ROOT=${GR00T_ROOT:-$ROBORENDER_ASSET_ROOT/pipeline/GR00T-VisualSim2Real}
H200_RUNTIME_ROOT=${H200_RUNTIME_ROOT:-$ROBORENDER_ASSET_ROOT/runtime/isaac_actor/qualified_h200_v1}
ISAACLAB_SOURCE=$H200_RUNTIME_ROOT/runtime/IsaacLab/source
PYTHON_DEPS=$H200_RUNTIME_ROOT/runtime/actor-python-site-packages
SYSTEM_LIBS=$H200_RUNTIME_ROOT/packages/root/usr/lib/x86_64-linux-gnu
CUDNN_LIBS=$H200_RUNTIME_ROOT/runtime/isaacsim-5.1/lib/python3.11/site-packages/nvidia/cudnn/lib

for directory in \
    "$GR00T_ROOT" \
    "$ISAACLAB_SOURCE/isaaclab" \
    "$ISAACLAB_SOURCE/isaaclab_assets" \
    "$ISAACLAB_SOURCE/isaaclab_contrib" \
    "$ISAACLAB_SOURCE/isaaclab_mimic" \
    "$ISAACLAB_SOURCE/isaaclab_rl" \
    "$ISAACLAB_SOURCE/isaaclab_tasks" \
    "$PYTHON_DEPS" \
    "$PYTHON_DEPS/cmeel.prefix/lib/python3.11/site-packages" \
    "$SYSTEM_LIBS" \
    "$CUDNN_LIBS"; do
    [[ -d $directory ]] || { echo "qualified H200 runtime component missing: $directory" >&2; return 2; }
done
[[ -f $CUDNN_LIBS/libcudnn.so.9 ]] || { echo "qualified cuDNN is missing" >&2; return 2; }

export PYTHONPATH="$ISAACLAB_SOURCE/isaaclab:$ISAACLAB_SOURCE/isaaclab_assets:$ISAACLAB_SOURCE/isaaclab_contrib:$ISAACLAB_SOURCE/isaaclab_mimic:$ISAACLAB_SOURCE/isaaclab_rl:$ISAACLAB_SOURCE/isaaclab_tasks:$PYTHON_DEPS:$PYTHON_DEPS/cmeel.prefix/lib/python3.11/site-packages:$GR00T_ROOT"
export LD_LIBRARY_PATH="$CUDNN_LIBS:$SYSTEM_LIBS:/usr/lib/x86_64-linux-gnu"
export ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1
export HYDRA_FULL_ERROR=1 LOGURU_LEVEL=INFO PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1
export GR00T_PYTHON=${GR00T_PYTHON:-$H200_RUNTIME_ROOT/runtime/isaacsim-5.1/bin/python}
