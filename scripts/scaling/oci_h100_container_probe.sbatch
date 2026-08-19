#!/bin/bash
# One-GPU OCI smoke gate for the exact Isaac Sim container used by DAgger.
#SBATCH --account=general_cs_infra
#SBATCH --partition=backfill_block1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=rr_isaac51_probe
# Pyxis uses REGISTRY#IMAGE syntax; a slash here is interpreted as Docker Hub.
set -Eeuo pipefail

STAGE_ROOT=${STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-$STAGE_ROOT}
source "$ROOT/scripts/setup/oci_env.sh"
if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    exec bash "$ROOT/scripts/setup/reexec_in_oci_container.sh" \
        "$ROOT/scripts/scaling/oci_h100_container_probe.sbatch"
fi

ISAAC_PYTHON=${ISAAC_PYTHON:-/isaac-sim/python.sh}
[ -x "$ISAAC_PYTHON" ] || {
    echo "FATAL: Isaac Sim Python launcher is not executable: $ISAAC_PYTHON" >&2
    exit 2
}
"$ISAAC_PYTHON" - <<'PY'
import platform
import sys

print(f"python={sys.version}")
print(f"platform={platform.platform()}")
try:
    import isaacsim
except Exception as exc:
    raise SystemExit(f"isaacsim import failed: {exc}")
print(f"isaacsim={getattr(isaacsim, '__version__', 'import-ok')}")
PY
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
