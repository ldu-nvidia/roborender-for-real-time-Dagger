#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
IMAGE=${ROBORENDER_REBUILT_IMAGE:-roborender-round-actor:isaaclab2.3.0-beta-rebuild-20261008}
BASE=nvcr.io/nvidia/isaac-lab@sha256:f1eae1f847cc77e00738d919138d78dde9a12589da9a83b4909555a4abeddb20
docker image inspect "$BASE" >/dev/null 2>&1 || docker pull "$BASE"
docker build --pull=false --progress=plain --build-arg "ISAAC_BASE=$BASE" -t "$IMAGE" "$ROOT"
docker image inspect "$IMAGE" --format 'REBUILT_IMAGE id={{.Id}} size={{.Size}}'
docker run --rm --network none "$IMAGE" python -c \
  'import sys, torch, torchvision, numpy; print({"python":sys.version,"torch":torch.__version__,"torchvision":torchvision.__version__,"numpy":numpy.__version__}); assert torch.__version__.startswith("2.7.0+"); assert torchvision.__version__.startswith("0.22.0+"); assert numpy.__version__ == "1.26.0"'
echo 'CPU dependency probe passed. Native GPU/Isaac qualification is still required.'
