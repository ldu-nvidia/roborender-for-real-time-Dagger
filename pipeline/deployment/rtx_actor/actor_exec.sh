#!/usr/bin/env bash
set -Eeuo pipefail
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
export OMNI_KIT_ALLOW_ROOT=${OMNI_KIT_ALLOW_ROOT:-1}
export ACCEPT_EULA=${ACCEPT_EULA:-Y} PRIVACY_CONSENT=${PRIVACY_CONSENT:-Y}
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=${PYTHONPATH:-}
export ISAAC_PATH=/isaac-sim CARB_APP_PATH=/isaac-sim/kit EXP_PATH=/isaac-sim/apps
export LD_PRELOAD="$CARB_APP_PATH/libcarb.so:/lib/x86_64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} TORCH_HOME=${TORCH_HOME:-/root/.cache/torch}
source /isaac-sim/setup_python_env.sh
export PATH=/opt/round-dagger-rtx/bin:$PATH
if [[ $# -eq 0 ]]; then
  set -- python -c 'import isaaclab, torch; print(torch.__version__, isaaclab.__file__)'
fi
exec "$@"
