#!/usr/bin/env bash
# Inference-only validation on the already-allocated, run-owned H200 pod.
set -Eeuo pipefail
[[ $# == 10 && $1 == --run-id && $3 == --asset-root && $5 == --netrc-file && $7 == --request && $9 == --result ]] || {
    echo "explicit allocation identity, assets, credentials and request/result paths required" >&2; exit 2;
}
[[ ${ROBORENDER_BETA_RUN_ID:-} == "$2" ]] || { echo "owned allocation identity differs" >&2; exit 2; }
ROBORENDER_ASSET_ROOT=$4
NETRC=$6
MSE_REQUEST=$8
MSE_RESULT=${10}
export ROBORENDER_ASSET_ROOT NETRC
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
source "$ROOT/pipeline/common/setup_h200_isaac_runtime.sh"
export PYTHONPATH="$ROOT:$GR00T_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$GR00T_PYTHON" -m pipeline.dagger.heldout_mse_worker --request "$MSE_REQUEST" --result "$MSE_RESULT" --device cuda:0
