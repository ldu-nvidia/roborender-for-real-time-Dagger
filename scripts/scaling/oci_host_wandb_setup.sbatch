#!/bin/bash
# Build the CPU-only W&B sidecar environment with the OCI login-node Python.
# The packed coordinator runs on the host, while Isaac/GR00T ranks run in a
# container, so this environment must not inherit a container-only interpreter.
#SBATCH --account=general_cs_infra
#SBATCH --partition=backfill_block1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=rr_wandb_env
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/wandb_host_setup_%j.log
set -Eeuo pipefail

STAGE_ROOT=${STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
WANDB_ENV=${WANDB_ENV:-$STAGE_ROOT/runtime/envs/wandb-host}
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3}
WANDB_VERSION=${WANDB_VERSION:-0.21.1}
NUMPY_VERSION=${NUMPY_VERSION:-1.24.4}
READY_MARKER=$WANDB_ENV/.oci_environment_ready
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$STAGE_ROOT/runtime/cache/pip}
export TMPDIR=${TMPDIR:-$STAGE_ROOT/runtime/tmp}

[ -x "$HOST_PYTHON" ] || { echo "FATAL: host Python missing: $HOST_PYTHON" >&2; exit 2; }
mkdir -p "$(dirname "$WANDB_ENV")" "$PIP_CACHE_DIR" "$TMPDIR"
if [ ! -x "$WANDB_ENV/bin/python" ]; then
    "$HOST_PYTHON" -m venv "$WANDB_ENV"
fi
"$WANDB_ENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$WANDB_ENV/bin/python" -m pip install \
    "wandb==$WANDB_VERSION" "numpy==$NUMPY_VERSION" pillow
"$WANDB_ENV/bin/python" - "$WANDB_VERSION" "$NUMPY_VERSION" <<'PY'
import sys
import numpy
import wandb

expected_wandb, expected_numpy = sys.argv[1:]
if wandb.__version__ != expected_wandb:
    raise SystemExit(f"expected wandb {expected_wandb}, found {wandb.__version__}")
if numpy.__version__ != expected_numpy:
    raise SystemExit(f"expected numpy {expected_numpy}, found {numpy.__version__}")
print(f"wandb={wandb.__version__} numpy={numpy.__version__} python={sys.executable}")
PY
{
    echo "python=$HOST_PYTHON"
    echo "wandb=$WANDB_VERSION"
    echo "numpy=$NUMPY_VERSION"
    echo "context=oci-host"
} > "$READY_MARKER"
echo "W&B host environment ready: $WANDB_ENV"
