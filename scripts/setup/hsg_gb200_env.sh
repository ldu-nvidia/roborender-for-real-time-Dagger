#!/bin/bash
# Isolated HSG/GB200 (ARM64, sm100) defaults. This profile deliberately does
# not inherit paths from any other deployment site.

export ROBORENDER_STAGE_ROOT="${ROBORENDER_STAGE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ldu/roborender_gb200_sm100}"
export HSG_RENDERER_CONTAINER_IMAGE="${HSG_RENDERER_CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:25.09-py3}"
export HSG_ISAAC_CONTAINER_IMAGE="${HSG_ISAAC_CONTAINER_IMAGE:-nvcr.io#nvidia/isaac-sim:5.1.0}"
export HSG_RENDERER_CONTAINER_INDEX_DIGEST="${HSG_RENDERER_CONTAINER_INDEX_DIGEST:-sha256:f3b4a33fd60d5e0358287bb44d57e91eda45e8f1681fef1bfd6c969371417130}"
export HSG_ISAAC_CONTAINER_INDEX_DIGEST="${HSG_ISAAC_CONTAINER_INDEX_DIGEST:-sha256:f3563cb2ba0c18af0b2fb321360dcb73a917b899f879e3213623d6bee484fa54}"
export ROOT="${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}"
export ROBORENDER_ROOT="${ROBORENDER_ROOT:-$ROOT}"
export GR00T_ROOT="${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}"
export IMAGE3F_ROOT="${IMAGE3F_ROOT:-$ROBORENDER_STAGE_ROOT/assets/roborender_image_model_accelerated}"
export IMAGE3F_ENV="${IMAGE3F_ENV:-$ROBORENDER_STAGE_ROOT/runtime/envs/image3f-sm100}"
export IMAGE3F_PYTHON="${IMAGE3F_PYTHON:-$IMAGE3F_ENV/bin/python}"
export ROBORENDER_MODEL_BASE="${ROBORENDER_MODEL_BASE:-$ROBORENDER_STAGE_ROOT/assets/models}"
export ISAACLAB_ROOT="${ISAACLAB_ROOT:-$ROBORENDER_STAGE_ROOT/src/IsaacLab}"
export VIRAL_ENV="${VIRAL_ENV:-$ROBORENDER_STAGE_ROOT/runtime/envs/viral-sm100}"
export VIRAL_PYTHON="${VIRAL_PYTHON:-$VIRAL_ENV/bin/python}"
export HOME_OVERRIDE="${HOME_OVERRIDE:-$ROBORENDER_STAGE_ROOT/runtime/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROBORENDER_STAGE_ROOT/runtime/cache/xdg}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$ROBORENDER_STAGE_ROOT/runtime/cache/share}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROBORENDER_STAGE_ROOT/runtime/cache/pip}"
export TORCH_HOME="${TORCH_HOME:-$ROBORENDER_STAGE_ROOT/runtime/cache/torch}"
export TMPDIR="${TMPDIR:-$ROBORENDER_STAGE_ROOT/runtime/tmp}"
export PYTHONDONTWRITEBYTECODE=1
