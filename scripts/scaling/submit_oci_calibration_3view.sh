#!/bin/bash
# Minimal camera-calibration gate: one env, two simulation steps, three views.
# The two GPUs host one Image3F renderer and one single-rank trainer.
set -Eeuo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

export RUN_PROFILE=calibration_3view
export PACKED_NODES=1
export PACKED_GPUS_PER_NODE=2
export PACKED_PARTITION=backfill_block1
export PACKED_CPUS_PER_GPU=16
export PACKED_MEM=256G
export GLOBAL_NUM_ENVS=1
export ROLLOUT_STEPS=2
export TRAINER_NODES=1
export ENVS_PER_RENDERER=5
export EXPECTED_RENDERERS=1
export NUM_ROLLOUT_ITERATIONS=1
export SAVE_FREQUENCY=1
export RENDERER_CPUS_PER_TASK=8
export RENDERER_MEM_PER_NODE=64G
export TRAINER_CPUS_PER_TASK=16
export TRAINER_MEM_PER_NODE=128G
export IMAGE_DAGGER_DUMP_SIM_RGB=1
export IMAGE_DAGGER_DUMP_MAX_STEPS=2
export IMAGE_DAGGER_DUMP_MAX_ENVS=1
export IMAGE_DAGGER_VERIFY_DDP_SYNC=0
export IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS=1
export RUN_WALLTIME=${RUN_WALLTIME:-01:00:00}
export PROJ=roborender_dagger_3view_calibration
export RUN_ID=${RUN_ID:-dagger_3view_calibration_$(date +%Y%m%d_%H%M%S)}
export EXP=$RUN_ID
export VG_PRIMARY_VIEW=head_color
export VG_POLICY_VIEWS=head_color:hand_left:hand_right
export VG_POLICY_VIEW_COUNT=3
export VG_POLICY_VIEW_LAYOUT=vertical
export VG_POLICY_IMAGE_SHAPE=324x192x3

exec bash "$SCRIPT_ROOT/scripts/scaling/submit_oci_packed.sh"
