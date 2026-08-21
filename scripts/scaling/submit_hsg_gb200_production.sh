#!/bin/bash
# Submit or resume the fail-closed 15-node HSG GB200 production run.
set -Eeuo pipefail

case "${RUN_PROFILE:-production_256x300}" in
    production|production_256x300) ;;
    *)
        echo "FATAL: this entrypoint only submits RUN_PROFILE=production_256x300" >&2
        exit 2
        ;;
esac
export RUN_PROFILE=production_256x300

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/submit_hsg_gb200_smoke.sh"
