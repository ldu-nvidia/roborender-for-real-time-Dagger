#!/bin/bash
set -Eeuo pipefail

export ROBORENDER_ROOT=${ROBORENDER_ROOT:-/opt/roborender}
export ROBORENDER_MODEL_BASE=${ROBORENDER_MODEL_BASE:-/models}
export ROBORENDER_OUTPUTS=${ROBORENDER_OUTPUTS:-/outputs}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-$ROBORENDER_MODEL_BASE}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-true}
export PYTHONDONTWRITEBYTECODE=1
export PATH=/opt/roborender-venv/bin:$PATH

if [ "$#" -eq 0 ]; then
    set -- python /opt/roborender/bin/doctor.py --stage package --fast
fi
exec "$@"
