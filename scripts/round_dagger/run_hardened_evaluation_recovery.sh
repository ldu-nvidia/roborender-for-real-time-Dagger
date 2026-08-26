#!/usr/bin/env bash
# Logic-free entry into the trusted Python held-candidate recovery runner.
set -Eeuo pipefail

SOURCE_ROOT=${ROUND_DAGGER_EVALUATION_SOURCE_ROOT:?missing ROUND_DAGGER_EVALUATION_SOURCE_ROOT}
CONTROL_PYTHON=${ROUND_DAGGER_CONTROL_PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/GR00T-VisualSim2Real"
exec "$CONTROL_PYTHON" -m round_dagger.evaluation_recovery run
