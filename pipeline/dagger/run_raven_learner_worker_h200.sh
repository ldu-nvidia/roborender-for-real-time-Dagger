#!/usr/bin/env bash
# Fixed worker executable used by the H200 learner service.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
source "$ROOT/pipeline/common/setup_h200_isaac_runtime.sh"
export PYTHONPATH=$ROOT:$ROOT/pipeline/GR00T-VisualSim2Real${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
exec "$GR00T_PYTHON" -m pipeline.dagger.services.raven_learner_worker "$@"
