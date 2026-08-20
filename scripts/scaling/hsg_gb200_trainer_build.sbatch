#!/bin/bash
# Build and save the ARM64 Isaac Sim 5.1 + pinned IsaacLab + GR00T trainer.
#SBATCH --account=llmservice_nemotron_ultra
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --mem=230G
#SBATCH --time=03:00:00
#SBATCH --job-name=rr_gb200_trainer_build
set -Eeuo pipefail

STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ldu/roborender_gb200_sm100}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
export ROBORENDER_STAGE_ROOT=$STAGE_ROOT ROOT
source "$ROOT/scripts/setup/hsg_gb200_env.sh"

GR00T_STAGE=${GR00T_STAGE:-$ROOT/GR00T-VisualSim2Real}
EXPECTED_GR00T_REVISION=${EXPECTED_GR00T_REVISION:-b4d17449d70eb3fb7caa7ea9366ca59bb0589cdd}
ISAACLAB_REVISION=${ISAACLAB_REVISION:-37ddf626871758333d6ed89cf64ad702aef127d0}
CONTAINER_ROOT=${CONTAINER_ROOT:-$STAGE_ROOT/containers}
CONTAINER_OUT=${CONTAINER_OUT:-$CONTAINER_ROOT/roborender-isaac-gr00t-arm64-sm100-isaac5.1-r5.sqsh}
PARTIAL="$CONTAINER_OUT.partial.${SLURM_JOB_ID:-$$}"
IMPORT_REPORT=${IMPORT_REPORT:-$STAGE_ROOT/validation/trainer_build_imports.json}
HYDRA_REPORT=${HYDRA_REPORT:-$STAGE_ROOT/validation/production_hydra_contract_gb200.json}
export GR00T_STAGE EXPECTED_GR00T_REVISION ISAACLAB_REVISION IMPORT_REPORT HYDRA_REPORT

mkdir -p "$CONTAINER_ROOT" "$STAGE_ROOT/validation" "$HOME_OVERRIDE" \
    "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TMPDIR"
[ -d "$GR00T_STAGE/gr00t" ] || { echo "FATAL: staged GR00T source missing: $GR00T_STAGE" >&2; exit 2; }
[ "$(cat "$GR00T_STAGE/.source_revision")" = "$EXPECTED_GR00T_REVISION" ] || {
    echo "FATAL: staged GR00T revision mismatch" >&2; exit 2;
}
[ ! -e "$CONTAINER_OUT" ] || { echo "FATAL: refusing to overwrite $CONTAINER_OUT" >&2; exit 2; }
[ ! -e "$PARTIAL" ] || { echo "FATAL: stale partial output exists: $PARTIAL" >&2; exit 2; }

if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    set +e
    srun --nodes=1 --ntasks=1 --gpus-per-task=4 --kill-on-bad-exit=1 \
        --container-image="$HSG_ISAAC_CONTAINER_IMAGE" \
        --container-writable --container-remap-root \
        --container-save="$PARTIAL" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT" \
        --container-workdir="$ROOT" \
        env ROBORENDER_CONTAINERIZED=1 ROBORENDER_STAGE_ROOT="$STAGE_ROOT" \
            ROOT="$ROOT" GR00T_STAGE="$GR00T_STAGE" \
            EXPECTED_GR00T_REVISION="$EXPECTED_GR00T_REVISION" \
            ISAACLAB_REVISION="$ISAACLAB_REVISION" IMPORT_REPORT="$IMPORT_REPORT" \
            HYDRA_REPORT="$HYDRA_REPORT" HOME="$HOME_OVERRIDE" \
            HOME_OVERRIDE="$HOME_OVERRIDE" XDG_CACHE_HOME="$XDG_CACHE_HOME" \
            PIP_CACHE_DIR="$PIP_CACHE_DIR" TORCH_HOME="$TORCH_HOME" \
            TMPDIR="$TMPDIR" OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 \
            bash "$ROOT/scripts/scaling/hsg_gb200_trainer_build.sbatch"
    BUILD_RC=$?
    set -e
    [ "$BUILD_RC" -eq 0 ] || { echo "FATAL: trainer image build failed: $BUILD_RC" >&2; exit "$BUILD_RC"; }
    [ -s "$PARTIAL" ] || { echo "FATAL: Pyxis did not save $PARTIAL" >&2; exit 2; }
    mv "$PARTIAL" "$CONTAINER_OUT"
    sha256sum "$CONTAINER_OUT" | tee "$CONTAINER_OUT.sha256"
    echo "HSG_GB200_TRAINER_CONTAINER_READY=$CONTAINER_OUT"
    exit 0
fi

export HOME=${HOME:-$HOME_OVERRIDE}
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${PYTHONPATH:-}
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export PIP_CONSTRAINT=/opt/trainer-constraints.txt

if ! command -v gcc >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends build-essential ca-certificates git
    apt-get clean
    find /var/lib/apt/lists -mindepth 1 -delete
fi

if [ ! -d /opt/GR00T-VisualSim2Real/gr00t ]; then
    cp -a "$GR00T_STAGE" /opt/GR00T-VisualSim2Real
fi
[ "$(cat /opt/GR00T-VisualSim2Real/.source_revision)" = "$EXPECTED_GR00T_REVISION" ] || {
    echo "FATAL: container GR00T revision mismatch" >&2; exit 2;
}
if [ ! -s /opt/IsaacLab/.source_revision ]; then
    [ ! -e /opt/IsaacLab ] || { echo "FATAL: incomplete /opt/IsaacLab exists" >&2; exit 2; }
    /isaac-sim/python.sh - /opt/IsaacLab "$ISAACLAB_REVISION" <<'PY'
from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

root = Path(sys.argv[1])
revision = sys.argv[2]
root.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="isaaclab_", dir=root.parent) as temp_name:
    temp = Path(temp_name)
    archive = temp / "source.tar.gz"
    urllib.request.urlretrieve(
        f"https://codeload.github.com/isaac-sim/IsaacLab/tar.gz/{revision}", archive
    )
    unpack = temp / "unpack"
    unpack.mkdir()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (unpack / member.name).resolve()
            if unpack.resolve() not in target.parents and target != unpack.resolve():
                raise SystemExit(f"unsafe archive member: {member.name}")
        handle.extractall(unpack)
    children = [path for path in unpack.iterdir() if path.is_dir()]
    if len(children) != 1 or not (children[0] / "source/isaaclab/pyproject.toml").is_file():
        raise SystemExit("malformed IsaacLab archive")
    os.replace(children[0], root)
(root / ".source_revision").write_text(revision + "\n", encoding="utf-8")
PY
fi
[ "$(cat /opt/IsaacLab/.source_revision)" = "$ISAACLAB_REVISION" ] || {
    echo "FATAL: container IsaacLab revision mismatch" >&2; exit 2;
}

if [ ! -x /opt/viral-sm100/bin/python ]; then
    /isaac-sim/python.sh -m venv --system-site-packages /opt/viral-sm100
fi
PY=/opt/viral-sm100/bin/python
install -m 0444 "$ROOT/containers/gb200/trainer-constraints.txt" "$PIP_CONSTRAINT"
"$PY" -m pip install --upgrade pip setuptools "wheel==0.45.1" \
    poetry-core "flatdict==4.0.1" "packaging==23.2"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source /isaac-sim/setup_python_env.sh
"$PY" - <<'PY'
import torch
import torchvision

print(f"isaac_base_torch={torch.__version__} cuda={torch.version.cuda}")
print(f"isaac_base_torchvision={torchvision.__version__}")
if not torch.__version__.startswith("2.7.0") or torch.version.cuda != "12.8":
    raise SystemExit("unexpected Isaac 5.1 torch/CUDA base runtime")
if not torchvision.__version__.startswith("0.22.0"):
    raise SystemExit("unexpected Isaac 5.1 torchvision base runtime")
PY
# NVIDIA's Isaac image contains the correct Torch 2.7 CUDA 12.8 build, but its
# embedded package set does not expose the matching Triton wheel to pip's
# resolver.  Install the official ARM64 wheel before resolving IsaacLab so pip
# never attempts to replace Torch or its CUDA ABI.
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    triton==3.3.0
# The ARM64 quadprog source archive contains a .pyx rather than generated C;
# build isolation is deliberately disabled to preserve Isaac's ABI, so provide
# its build dependency explicitly in the sealed environment.
"$PY" -m pip install "Cython==3.2.9"
"$PY" -m pip install "toml==0.10.2"
"$PY" -m pip install --no-build-isolation -e /opt/IsaacLab/source/isaaclab
"$PY" -m pip install --no-build-isolation \
    -e /opt/IsaacLab/source/isaaclab_contrib \
    -e /opt/IsaacLab/source/isaaclab_assets \
    -e /opt/IsaacLab/source/isaaclab_tasks \
    -e '/opt/IsaacLab/source/isaaclab_rl[all]'
"$PY" -m pip install numpy==1.26.0
"$PY" -m pip install -e /opt/GR00T-VisualSim2Real
"$PY" -m pip install docstring-parser==0.16 PyJWT==2.10.1

install -m 0555 "$ROOT/containers/gb200/trainer_exec.sh" /usr/local/bin/roborender-trainer-sm100
install -m 0444 "$ROOT/containers/gb200/trainer_import_gate.py" /opt/trainer_import_gate.py
install -m 0444 "$ROOT/containers/gb200/isaac_camera_smoke.py" /opt/isaac_camera_smoke.py

"$PY" /opt/trainer_import_gate.py | tee "$IMPORT_REPORT"
"$PY" "$ROOT/scripts/scaling/validate_production_hydra.py" \
    --gr00t-root /opt/GR00T-VisualSim2Real \
    --expected-revision "$EXPECTED_GR00T_REVISION" \
    --output "$HYDRA_REPORT"
"$PY" - <<'PY'
from torchvision.models import ResNet18_Weights, resnet18
resnet18(weights=ResNet18_Weights.DEFAULT)
print("resnet18_imagenet_cache=ready")
PY
PIP_CHECK_REPORT="$STAGE_ROOT/validation/trainer_pip_check.txt"
set +e
"$PY" -m pip check >"$PIP_CHECK_REPORT" 2>&1
PIP_CHECK_RC=$?
set -e
if [ "$PIP_CHECK_RC" -ne 0 ]; then
    "$PY" - "$PIP_CHECK_REPORT" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

report = Path(sys.argv[1])
lines = [line.strip() for line in report.read_text(encoding="utf-8").splitlines() if line.strip()]
allowed = (
    re.compile(r'^nvidia-srl-(?:usd|usd-to-urdf) [^ ]+ requires usd-core, which is not installed\.$'),
    re.compile(r'^nvidia-srl-usd-to-urdf [^ ]+ has requirement lxml<5\.0\.0,>=4\.9\.2, but you have lxml 5\.4\.0\.$'),
    re.compile(r'^onnx [^ ]+ has requirement typing_extensions>=4\.15\.0, but you have typing-extensions 4\.12\.2\.$'),
    re.compile(r'^isaaclab [^ ]+ has requirement starlette==0\.49\.1, but you have starlette 0\.45\.3\.$'),
    re.compile(r'^wandb [^ ]+ has requirement click>=8\.2\.0, but you have click 8\.1\.7\.$'),
)
unexpected = [line for line in lines if not any(pattern.fullmatch(line) for pattern in allowed)]
print(report.read_text(encoding="utf-8"), end="")
if unexpected:
    raise SystemExit(f"unexpected pip metadata conflicts: {unexpected}")
print("HSG_GB200_TRAINER_PIP_METADATA_ALLOWLIST_PASS")
PY
else
    cat "$PIP_CHECK_REPORT"
fi
echo "HSG_GB200_TRAINER_BUILD_CONTENT_PASS"
