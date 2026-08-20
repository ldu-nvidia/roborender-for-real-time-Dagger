#!/bin/bash
set -Eeuo pipefail

export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
export OMNI_KIT_ALLOW_ROOT=${OMNI_KIT_ALLOW_ROOT:-1}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${PYTHONPATH:-}
export ISAAC_PATH=/isaac-sim
export CARB_APP_PATH=$ISAAC_PATH/kit
export EXP_PATH=$ISAAC_PATH/apps
# Isaac Sim's official python.sh preloads libcarb, while the ARM64 policy stack
# also needs libgomp early enough to avoid static-TLS exhaustion.
export LD_PRELOAD="$CARB_APP_PATH/libcarb.so:/lib/aarch64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source /isaac-sim/setup_python_env.sh
export PATH=/opt/viral-sm100/bin:$PATH
export GR00T_ROOT=${GR00T_ROOT:-/opt/GR00T-VisualSim2Real}
export ISAACLAB_ROOT=${ISAACLAB_ROOT:-/opt/IsaacLab}

if [ "$#" -eq 0 ]; then
    set -- python /opt/trainer_import_gate.py
fi
exec "$@"
