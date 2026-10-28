#!/usr/bin/env bash
# Launch one persistent Raven-equivalent Image3F service on an assigned H200.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
PYTHON=${IMAGE3F_PYTHON:-$ROOT/renderer/envs/image3f-py313-cu130-trt10133/venv/bin/python}
GPU=${1:?usage: $0 GPU_INDEX_OR_UUID ENDPOINT_FILE [PORT]}
ENDPOINT=${2:?usage: $0 GPU_INDEX_OR_UUID ENDPOINT_FILE [PORT]}
PORT=${3:-18500}

[[ $ENDPOINT == /* ]] || { echo "ENDPOINT_FILE must be absolute" >&2; exit 2; }
[[ $PORT =~ ^[0-9]+$ && $PORT -ge 1024 && $PORT -le 65535 ]] || {
    echo "PORT must be 1024..65535" >&2
    exit 2
}
[[ -x $PYTHON ]] || { echo "Image3F Python missing: $PYTHON" >&2; exit 2; }

if [[ ${IMAGE3F_SKIP_ASSET_VALIDATION:-false} != true ]]; then
    "$PYTHON" "$ROOT/pipeline/dagger/validate_raven_image3f_contract.py" --full-assets
fi
QUEUE=${IMAGE3F_QUEUE:-$ROOT/runtime/image3f_queue}
mkdir -p "$(dirname "$ENDPOINT")" "$QUEUE/inbox" "$QUEUE/done"

export CUDA_VISIBLE_DEVICES=$GPU
export IMAGE3F_ROOT=$ROOT/renderer/online_image3f/runtime
export IMAGE3F_QUEUE=$QUEUE
export IMAGE3F_PORT=$PORT
export ENDPOINT_FILE=$ENDPOINT
export ROBORENDER_MODEL_BASE=$ROOT/renderer/models
export DIFFSYNTH_MODEL_BASE_PATH=$ROBORENDER_MODEL_BASE
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

exec "$PYTHON" "$ROOT/pipeline/dagger/image3f_server.py"
