#!/bin/bash
# Submit the isolated two-process checkpoint-continuity smoke profile.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[ -z "${RUN_PROFILE:-}" ] || [ "$RUN_PROFILE" = resume_minimal ] || {
    echo "FATAL: this entrypoint only submits RUN_PROFILE=resume_minimal" >&2
    exit 2
}
export RUN_PROFILE=resume_minimal
exec "$SCRIPT_DIR/submit_hsg_gb200_smoke.sh" "$@"
