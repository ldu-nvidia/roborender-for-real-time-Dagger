#!/bin/bash
set -Eeuo pipefail

export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
export OMNI_KIT_ALLOW_ROOT=${OMNI_KIT_ALLOW_ROOT:-1}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${PYTHONPATH:-}
export ISAAC_PATH=/isaac-sim
export CARB_APP_PATH=$ISAAC_PATH/kit
export EXP_PATH=$ISAAC_PATH/apps
export LD_PRELOAD="$CARB_APP_PATH/libcarb.so:/lib/x86_64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export ROBORENDER_EXPECTED_MACHINE=x86_64
export ROBORENDER_EXPECTED_CC=9,0
export ROBORENDER_EXPECTED_GR00T_REVISION=${ROBORENDER_EXPECTED_GR00T_REVISION:-fe1603a6c6ee63cacb3a4d79a00054cc2fcd2d6f}
export TORCH_HOME=${TORCH_HOME:-/opt/torch-cache}
source /isaac-sim/setup_python_env.sh
export PATH=/opt/viral-h100/bin:$PATH
export GR00T_ROOT=${GR00T_ROOT:-/opt/GR00T-VisualSim2Real}
export ISAACLAB_ROOT=${ISAACLAB_ROOT:-/opt/IsaacLab}

if [ "$#" -eq 0 ]; then
    set -- python /opt/trainer_import_gate.py
fi
exec "$@"
