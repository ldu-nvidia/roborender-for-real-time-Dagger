#!/usr/bin/env bash
# Exact local Isaac Sim 5.1 runtime selected by the native-H200 qualification.
# This file is sourced by the confined actor rank launcher.

set -Eeuo pipefail

ROUND_DAGGER_RUNTIME_ROOT=/workspace/codex/roborender-agent/runs/h200_rollout_qualification_20260903T201136Z
ROUND_DAGGER_SOURCE_ROOT=/workspace/project/roborender/pipeline
ROUND_DAGGER_GR00T_ROOT=$ROUND_DAGGER_SOURCE_ROOT/GR00T-VisualSim2Real
ROUND_DAGGER_CUDNN_ROOT=$ROUND_DAGGER_RUNTIME_ROOT/runtime/isaacsim-5.1/lib/python3.11/site-packages/nvidia/cudnn/lib
QUALIFIED_SYSTEM_LIBRARY_ROOT=$ROUND_DAGGER_RUNTIME_ROOT/packages/root/usr/lib/x86_64-linux-gnu
QUALIFIED_ISAACLAB_SOURCE_ROOT=$ROUND_DAGGER_RUNTIME_ROOT/runtime/IsaacLab/source
QUALIFIED_PYTHON_DEPENDENCY_ROOT=$ROUND_DAGGER_RUNTIME_ROOT/runtime/actor-python-site-packages

[[ -d $ROUND_DAGGER_CUDNN_ROOT && -f $ROUND_DAGGER_CUDNN_ROOT/libcudnn.so.9 ]] || {
    echo "qualified cuDNN runtime is missing" >&2
    return 2
}
[[ -d $ROUND_DAGGER_SOURCE_ROOT && -d $ROUND_DAGGER_GR00T_ROOT ]] || {
    echo "native H200 source roots are missing" >&2
    return 2
}
for package_root in \
    "$QUALIFIED_ISAACLAB_SOURCE_ROOT/isaaclab" \
    "$QUALIFIED_ISAACLAB_SOURCE_ROOT/isaaclab_assets" \
    "$QUALIFIED_ISAACLAB_SOURCE_ROOT/isaaclab_contrib" \
    "$QUALIFIED_ISAACLAB_SOURCE_ROOT/isaaclab_mimic" \
    "$QUALIFIED_ISAACLAB_SOURCE_ROOT/isaaclab_rl" \
    "$QUALIFIED_ISAACLAB_SOURCE_ROOT/isaaclab_tasks" \
    "$QUALIFIED_PYTHON_DEPENDENCY_ROOT" \
    "$QUALIFIED_PYTHON_DEPENDENCY_ROOT/cmeel.prefix/lib/python3.11/site-packages"; do
    [[ -d $package_root && ! -L $package_root ]] || {
        echo "qualified Python package root is missing or unsafe: $package_root" >&2
        return 2
    }
done
[[ -d $QUALIFIED_SYSTEM_LIBRARY_ROOT && ! -L $QUALIFIED_SYSTEM_LIBRARY_ROOT ]] || {
    echo "qualified system library root is missing or unsafe" >&2
    return 2
}
[[ ${ROUND_DAGGER_ISAACLAB_SOURCE_ROOT:-} == "$QUALIFIED_ISAACLAB_SOURCE_ROOT" && \
   ${ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT:-} == "$QUALIFIED_PYTHON_DEPENDENCY_ROOT" && \
   ${ROUND_DAGGER_SYSTEM_LIBRARY_DIR:-} == "$QUALIFIED_SYSTEM_LIBRARY_ROOT" ]] || {
    echo "qualified actor runtime roots differ from setup" >&2
    return 2
}

# This is intentionally exact: the qualified bundle is self-contained and must
# never recover the obsolete retained worktree through bootstrap/sitecustomize
# or ambient Python/library search paths.
export PYTHONPATH="$QUALIFIED_ISAACLAB_SOURCE_ROOT/isaaclab:$QUALIFIED_ISAACLAB_SOURCE_ROOT/isaaclab_assets:$QUALIFIED_ISAACLAB_SOURCE_ROOT/isaaclab_contrib:$QUALIFIED_ISAACLAB_SOURCE_ROOT/isaaclab_mimic:$QUALIFIED_ISAACLAB_SOURCE_ROOT/isaaclab_rl:$QUALIFIED_ISAACLAB_SOURCE_ROOT/isaaclab_tasks:$QUALIFIED_PYTHON_DEPENDENCY_ROOT:$QUALIFIED_PYTHON_DEPENDENCY_ROOT/cmeel.prefix/lib/python3.11/site-packages:$ROUND_DAGGER_SOURCE_ROOT:$ROUND_DAGGER_GR00T_ROOT"
export LD_LIBRARY_PATH="$ROUND_DAGGER_CUDNN_ROOT:$QUALIFIED_SYSTEM_LIBRARY_ROOT:/usr/lib/x86_64-linux-gnu"
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export HYDRA_FULL_ERROR=1
export LOGURU_LEVEL=INFO
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
