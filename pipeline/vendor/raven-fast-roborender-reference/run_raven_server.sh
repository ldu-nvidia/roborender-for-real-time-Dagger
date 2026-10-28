#!/bin/bash
# Launch the Raven video-gen render server (inside the roborender venv).
# Usage: ./run_raven_server.sh [--port 18913] [--host 0.0.0.0]
set -Eeuo pipefail
cd /svl/u/ravenh/fast_roborender/roborender-acceleration
export RAVEN_ROOT=$PWD
export RAVEN_ASSETS=$PWD/assets
export RAVEN_OUTPUTS=$PWD/outputs
export PYTHONPATH=$PWD/vendor/sim2realgen
export DIFFSYNTH_MODEL_BASE_PATH=$PWD/assets/models
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TMPDIR=/svl/u/ravenh/tmp
source .venv/bin/activate
exec python runtime/raven_render_server.py "$@"
