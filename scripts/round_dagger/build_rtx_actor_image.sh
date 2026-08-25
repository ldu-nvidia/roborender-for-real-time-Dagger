#!/usr/bin/env bash
# Build the reproducible Isaac/GR00T dependency layer for local RTX actors.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
IMAGE=${ROUND_DAGGER_RTX_IMAGE:-roborender-round-actor:isaaclab2.3.0-local}
BASE=nvcr.io/nvidia/isaac-lab:2.3.0

docker image inspect "$BASE" >/dev/null 2>&1 || {
    echo "missing pinned base image: $BASE" >&2
    exit 2
}

docker build \
    --pull=false \
    --file "$ROOT/containers/rtx/Dockerfile.round-actor" \
    --tag "$IMAGE" \
    "$ROOT/containers/rtx"

docker run --rm \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    "$IMAGE" python -c \
    'import accelerate, datasets, loguru, onnxruntime, os, plotly, torch, torchvision, trl; assert os.environ["EXP_PATH"] == "/isaac-sim/apps"; assert torch.__version__.startswith("2.7.0+"); assert torchvision.__version__.startswith("0.22.0+"); print("ROUND_DAGGER_RTX_DEPENDENCIES_OK")'

docker image inspect "$IMAGE" --format \
    'ROUND_DAGGER_RTX_IMAGE_OK id={{.Id}} size={{.Size}}'
