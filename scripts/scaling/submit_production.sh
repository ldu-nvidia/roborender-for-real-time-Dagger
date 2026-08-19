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
export RUN_PROFILE=production_256x300
export PACKED_NODES=8
export PACKED_GPUS_PER_NODE=8
export GLOBAL_NUM_ENVS=256
export ROLLOUT_STEPS=300
export TRAINER_NODES=8
export ENVS_PER_RENDERER=5
export EXPECTED_RENDERERS=52
export PACKED_PARTITION=grizzly
export PACKED_CPUS_PER_GPU=10
export PACKED_MEM=768G
export IMAGE_DAGGER_DUMP_SIM_RGB=0
export IMAGE_DAGGER_DUMP_MAX_STEPS=0
export IMAGE_DAGGER_DUMP_MAX_ENVS=0
export IMAGE_DAGGER_VERIFY_DDP_SYNC=1
export IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS=1

echo "Production uses the packed OCI 8-node/64-H100 launcher."
exec bash "$SCRIPT_ROOT/scripts/scaling/submit_oci_packed.sh"
