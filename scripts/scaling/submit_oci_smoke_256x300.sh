#!/bin/bash
# Full-topology bounded validation: 52 renderers + eight synchronized trainers,
# one 300-step rollout/update, then fail-closed checkpoint and DDP hash gates.
set -Eeuo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

export RUN_PROFILE=smoke_256x300
export PACKED_NODES=8
export PACKED_GPUS_PER_NODE=8
export PACKED_PARTITION=grizzly
export PACKED_CPUS_PER_GPU=10
export PACKED_MEM=768G
export GLOBAL_NUM_ENVS=256
export ROLLOUT_STEPS=300
export TRAINER_NODES=8
export ENVS_PER_RENDERER=5
export EXPECTED_RENDERERS=52
export NUM_ROLLOUT_ITERATIONS=1
export SAVE_FREQUENCY=1
export RENDERER_CPUS_PER_TASK=8
export RENDERER_MEM_PER_NODE=512G
export TRAINER_CPUS_PER_TASK=16
export TRAINER_MEM_PER_NODE=128G
export IMAGE_DAGGER_DUMP_SIM_RGB=0
export IMAGE_DAGGER_DUMP_MAX_STEPS=0
export IMAGE_DAGGER_DUMP_MAX_ENVS=0
export IMAGE_DAGGER_VERIFY_DDP_SYNC=1
export IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS=1
export RUN_WALLTIME=${RUN_WALLTIME:-04:00:00}
export PROJ=roborender_dagger_3view_validation
export RUN_ID=${RUN_ID:-dagger_3view_smoke_256x300_$(date +%Y%m%d_%H%M%S)}
export EXP=$RUN_ID
export VG_PRIMARY_VIEW=head_color
export VG_POLICY_VIEWS=head_color:hand_left:hand_right
export VG_POLICY_VIEW_COUNT=3
export VG_POLICY_VIEW_LAYOUT=vertical
export VG_POLICY_IMAGE_SHAPE=324x192x3

exec bash "$SCRIPT_ROOT/scripts/scaling/submit_oci_packed.sh"
