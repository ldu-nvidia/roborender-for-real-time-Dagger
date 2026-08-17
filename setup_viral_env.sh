#!/bin/bash
set -e
export HOME=/sailhome/ravenh
export CONDA_PKGS_DIRS=/svl/u/ravenh/.conda_pkgs
export TMPDIR=/svl/u/ravenh/tmp PIP_CACHE_DIR=/svl/u/ravenh/.pip_cache
CONDA=/svl/u/ravenh/miniconda3/bin/conda
$CONDA create -y -n viral python=3.11
PY=/svl/u/ravenh/miniconda3/envs/viral/bin/python
$PY -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install isaacsim==5.1.0.0 isaacsim-rl==5.1.0.0
cd /svl/u/ravenh/fast_roborender
[ -d IsaacLab ] || git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git
$PY -m pip install setuptools poetry-core flatdict
$PY -m pip install --no-build-isolation -e IsaacLab/source/isaaclab
$PY -m pip install --no-build-isolation -e IsaacLab/source/isaaclab_assets -e IsaacLab/source/isaaclab_tasks -e "IsaacLab/source/isaaclab_rl[all]"
$PY -m pip install numpy==1.26.0
cd GR00T-VisualSim2Real
$PY -m pip install -e .
echo VIRAL_ENV_DONE
