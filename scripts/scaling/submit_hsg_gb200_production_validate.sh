#!/bin/bash
# Retry only the W&B drain, continuity snapshot, and postrun gates for a
# production run whose exact step-500 checkpoint is already durable.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RUN_PROFILE=production_256x300
export PRODUCTION_ACTION=validate
export RESUME_TRAINING=0
exec bash "$SCRIPT_DIR/submit_hsg_gb200_smoke.sh" "$@"
