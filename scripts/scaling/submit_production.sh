#!/bin/bash
# Compatibility entry point for the production DAgger launch.
#
# The only supported production topology is the packed OCI allocation. Keeping
# this wrapper prevents older operator commands from silently submitting the
# retired multi-job topology.
set -Eeuo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAGE_ROOT=${STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
OCI_ROOT="$STAGE_ROOT/src/fast_roborender"

if [ -n "${ROOT:-}" ] && [ "$ROOT" != "$OCI_ROOT" ]; then
    echo "FATAL: production ROOT must be the OCI staged checkout: $OCI_ROOT" >&2
    exit 2
fi
if [ -n "${GR00T_ROOT:-}" ] && [ "$GR00T_ROOT" != "$OCI_ROOT/GR00T-VisualSim2Real" ]; then
    echo "FATAL: production GR00T_ROOT must be inside the OCI staged checkout" >&2
    exit 2
fi

export STAGE_ROOT
export ROOT="$OCI_ROOT"
export GR00T_ROOT="$OCI_ROOT/GR00T-VisualSim2Real"

echo "Production uses the packed OCI 8-node/64-H100 launcher."
exec bash "$SCRIPT_ROOT/scripts/scaling/submit_oci_packed.sh"
