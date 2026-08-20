#!/bin/bash
# Prove whether the exact Isaac Sim image used by the DAgger trainer provides
# an ARM64 userspace and can see SM100 on HSG.
#SBATCH --account=llmservice_nemotron_ultra
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=rr_gb200_isaac_probe
set -Eeuo pipefail

STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ldu/roborender_gb200_sm100}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
export ROBORENDER_STAGE_ROOT=$STAGE_ROOT ROOT
source "$ROOT/scripts/setup/hsg_gb200_env.sh"
mkdir -p "$STAGE_ROOT/validation" "$HOME_OVERRIDE" "$XDG_CACHE_HOME" "$TMPDIR"

REPORT="$STAGE_ROOT/validation/isaac_base_probe.txt"
if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    exec srun --nodes=1 --ntasks=1 --gpus-per-task=4 --kill-on-bad-exit=1 \
        --container-image="$HSG_ISAAC_CONTAINER_IMAGE" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT" \
        --container-workdir="$ROOT" \
        env ROBORENDER_CONTAINERIZED=1 \
            ROBORENDER_STAGE_ROOT="$STAGE_ROOT" ROOT="$ROOT" \
            HOME_OVERRIDE="$HOME_OVERRIDE" XDG_CACHE_HOME="$XDG_CACHE_HOME" \
            TMPDIR="$TMPDIR" bash "$ROOT/scripts/scaling/hsg_gb200_isaac_probe.sbatch"
fi

ISAAC_PYTHON=${ISAAC_PYTHON:-/isaac-sim/python.sh}
[ -x "$ISAAC_PYTHON" ] || { echo "FATAL: missing $ISAAC_PYTHON" | tee "$REPORT"; exit 2; }
"$ISAAC_PYTHON" - <<'PY' | tee "$REPORT"
import json
import importlib.metadata
import platform
import sys

import isaacsim
import torch

try:
    import triton
except ImportError:
    triton = None

triton_distribution = next(
    (
        distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name", "").lower() == "triton"
    ),
    None,
)

capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
report = {
    "architecture": platform.machine(),
    "python": sys.version,
    "isaacsim": getattr(isaacsim, "__version__", "import-ok"),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "triton_import": getattr(triton, "__version__", None),
    "triton_distribution": triton_distribution,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "compute_capability": capability,
}
print(json.dumps(report, indent=2, sort_keys=True))
if platform.machine() not in {"aarch64", "arm64"}:
    raise SystemExit("expected ARM64 container userspace")
if not torch.cuda.is_available() or capability != (10, 0):
    raise SystemExit(f"expected an SM100 GB200, found {report}")
PY
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader | tee -a "$REPORT"
echo "HSG_GB200_ISAAC_BASE_PASS" | tee -a "$REPORT"
