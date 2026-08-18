#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CONDA=${CONDA:-$(command -v conda || true)}
VIRAL_ENV_NAME=${VIRAL_ENV_NAME:-viral}
CACHE_ROOT=${ROBORENDER_CACHE_ROOT:-$ROOT/.cache}
[ -n "$CONDA" ] && [ -x "$CONDA" ] || {
    echo "FATAL: conda is not executable; set CONDA=/path/to/conda" >&2
    exit 2
}
if [ -n "${ROBORENDER_HOME:-}" ]; then
    export HOME=$ROBORENDER_HOME
fi
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$CACHE_ROOT/conda_pkgs}
export TMPDIR=${TMPDIR:-$CACHE_ROOT/tmp}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$CACHE_ROOT/pip}
mkdir -p "$CONDA_PKGS_DIRS" "$TMPDIR" "$PIP_CACHE_DIR"

"$CONDA" create -y -n "$VIRAL_ENV_NAME" python=3.11
CONDA_BASE=$("$CONDA" info --base)
PY=${VIRAL_PY:-$CONDA_BASE/envs/$VIRAL_ENV_NAME/bin/python}
[ -x "$PY" ] || { echo "FATAL: environment Python is not executable: $PY" >&2; exit 2; }
"$PY" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install isaacsim==5.1.0.0 isaacsim-rl==5.1.0.0
cd "$ROOT"
[ -d IsaacLab ] || git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git
"$PY" -m pip install setuptools poetry-core flatdict
"$PY" -m pip install --no-build-isolation -e IsaacLab/source/isaaclab
"$PY" -m pip install --no-build-isolation -e IsaacLab/source/isaaclab_assets -e IsaacLab/source/isaaclab_tasks -e "IsaacLab/source/isaaclab_rl[all]"
"$PY" -m pip install numpy==1.26.0
cd GR00T-VisualSim2Real
"$PY" -m pip install -e .
echo VIRAL_ENV_DONE
