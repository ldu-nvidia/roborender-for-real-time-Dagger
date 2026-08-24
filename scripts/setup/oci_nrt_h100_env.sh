#!/bin/bash
# Isolated OCI-NRT/H100 (x86_64, SM90) defaults. Callers may override paths.

export ROBORENDER_STAGE_ROOT="${ROBORENDER_STAGE_ROOT:-/lustre/fs1/portfolios/general/projects/general_cs_infra__ngcsre/users/ldu/roborender_h100_build}"
export OCI_H100_BASE_IMAGE="${OCI_H100_BASE_IMAGE:-nvcr.io#nvidia/isaac-sim:5.1.0}"
export ROOT="${ROOT:?ROOT must identify an immutable staged fast_roborender snapshot}"
export GR00T_ROOT="${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}"
export IMAGE3F_ROOT="${IMAGE3F_ROOT:-$ROBORENDER_STAGE_ROOT/assets/roborender_image_model_accelerated}"
export ROBORENDER_MODEL_BASE="${ROBORENDER_MODEL_BASE:-$ROBORENDER_STAGE_ROOT/assets/models}"
export HOME_OVERRIDE="${HOME_OVERRIDE:-$ROBORENDER_STAGE_ROOT/runtime/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROBORENDER_STAGE_ROOT/runtime/cache/xdg}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$ROBORENDER_STAGE_ROOT/runtime/cache/share}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROBORENDER_STAGE_ROOT/runtime/cache/pip}"
export TORCH_HOME="${TORCH_HOME:-$ROBORENDER_STAGE_ROOT/runtime/cache/torch}"
export TMPDIR="${TMPDIR:-$ROBORENDER_STAGE_ROOT/runtime/tmp}"
export PYTHONDONTWRITEBYTECODE=1
