#!/bin/bash
# Re-run one staged batch payload through OCI's srun-only Pyxis interface.
# The cluster does not expose container options on sbatch itself.
set -Eeuo pipefail

TARGET=${1:?usage: reexec_in_oci_container.sh /absolute/staged/payload.sbatch}
: "${ROOT:?ROOT must identify the staged fast_roborender checkout}"
: "${ROBORENDER_STAGE_ROOT:?ROBORENDER_STAGE_ROOT is required}"
OCI_CONTAINER_IMAGE=${OCI_CONTAINER_IMAGE:-nvcr.io#nvidia/isaac-sim:5.1.0}

[ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ] || {
    echo "FATAL: recursive OCI container re-exec for $TARGET" >&2
    exit 2
}
TARGET=$(realpath -e "$TARGET")
case "$TARGET" in
    "$ROOT"/*) ;;
    *) echo "FATAL: container payload is outside the pinned checkout: $TARGET" >&2; exit 2 ;;
esac

exec srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 \
    --container-image="$OCI_CONTAINER_IMAGE" \
    --container-mounts="$ROBORENDER_STAGE_ROOT:$ROBORENDER_STAGE_ROOT" \
    env ROBORENDER_CONTAINERIZED=1 bash "$TARGET"
