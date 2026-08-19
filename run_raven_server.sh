#!/bin/bash
# Launch the Raven video-gen render server (inside the roborender venv).
# Usage: ./run_raven_server.sh [--port 18913] [--host 0.0.0.0]
set -Eeuo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROBORENDER_ACCEL_ROOT=${ROBORENDER_ACCEL_ROOT:-$ROBORENDER_STAGE_ROOT/assets/roborender-acceleration}
ROBORENDER_ACCEL_PYTHON=${ROBORENDER_ACCEL_PYTHON:-$ROBORENDER_ACCEL_ROOT/.venv/bin/python}
[ -d "$ROBORENDER_ACCEL_ROOT" ] || {
    echo "FATAL: legacy Raven acceleration bundle is not staged: $ROBORENDER_ACCEL_ROOT" >&2
    exit 2
}
[ -x "$ROBORENDER_ACCEL_PYTHON" ] || {
    echo "FATAL: legacy Raven Python is not staged: $ROBORENDER_ACCEL_PYTHON" >&2
    exit 2
}
cd "$ROBORENDER_ACCEL_ROOT"
export RAVEN_ROOT=$PWD
export RAVEN_ASSETS=$PWD/assets
export RAVEN_OUTPUTS=$PWD/outputs
export PYTHONPATH=$PWD/vendor/sim2realgen
export DIFFSYNTH_MODEL_BASE_PATH=$PWD/assets/models
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TMPDIR=${TMPDIR:-$ROBORENDER_STAGE_ROOT/runtime/tmp}
exec "$ROBORENDER_ACCEL_PYTHON" runtime/raven_render_server.py "$@"
