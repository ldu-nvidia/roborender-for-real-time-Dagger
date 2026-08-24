#!/bin/bash
# Build and save Isaac Sim 5.1 + pinned IsaacLab + current GR00T for H100.
#SBATCH --account=general_cs_infra
#SBATCH --partition=batch_block1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=224G
#SBATCH --time=03:30:00
#SBATCH --job-name=rr_h100_trainer_build
set -Eeuo pipefail

STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs1/portfolios/general/projects/general_cs_infra__ngcsre/users/ldu/roborender_h100_build}
ROOT=${ROOT:?submit with ROOT set to the immutable staged source snapshot}
export ROBORENDER_STAGE_ROOT=$STAGE_ROOT ROOT
source "$ROOT/scripts/setup/oci_nrt_h100_env.sh"

GR00T_STAGE=${GR00T_STAGE:-$ROOT/GR00T-VisualSim2Real}
EXPECTED_GR00T_REVISION=${EXPECTED_GR00T_REVISION:-fe1603a6c6ee63cacb3a4d79a00054cc2fcd2d6f}
ISAACLAB_REVISION=${ISAACLAB_REVISION:-37ddf626871758333d6ed89cf64ad702aef127d0}
CONTAINER_ROOT=${CONTAINER_ROOT:-$STAGE_ROOT/containers}
CONTAINER_OUT=${CONTAINER_OUT:-$CONTAINER_ROOT/roborender-isaac-gr00t-x86_64-sm90-isaac5.1.sqsh}
PARTIAL="$CONTAINER_OUT.partial.${SLURM_JOB_ID:-$$}"
IMPORT_REPORT=${IMPORT_REPORT:-$STAGE_ROOT/validation/h100_trainer_build_imports.json}
HYDRA_REPORT=${HYDRA_REPORT:-$STAGE_ROOT/validation/h100_production_hydra_contract.json}
export GR00T_STAGE EXPECTED_GR00T_REVISION ISAACLAB_REVISION IMPORT_REPORT HYDRA_REPORT

mkdir -p "$CONTAINER_ROOT" "$STAGE_ROOT/validation" "$HOME_OVERRIDE" \
    "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
[ -d "$GR00T_STAGE/gr00t" ] || { echo "FATAL: staged GR00T source missing: $GR00T_STAGE" >&2; exit 2; }
[ "$(cat "$GR00T_STAGE/.source_revision")" = "$EXPECTED_GR00T_REVISION" ] || {
    echo "FATAL: staged GR00T revision mismatch" >&2; exit 2;
}
[ ! -e "$CONTAINER_OUT" ] || { echo "FATAL: refusing to overwrite $CONTAINER_OUT" >&2; exit 2; }
[ ! -e "$PARTIAL" ] || { echo "FATAL: stale partial output exists: $PARTIAL" >&2; exit 2; }

if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    set +e
    srun --nodes=1 --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 \
        --container-image="$OCI_H100_BASE_IMAGE" \
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
            PIP_CACHE_DIR="$PIP_CACHE_DIR" TMPDIR="$TMPDIR" \
            OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 \
            bash "$ROOT/scripts/scaling/oci_nrt_h100_trainer_build.sbatch"
    BUILD_RC=$?
    set -e
    [ "$BUILD_RC" -eq 0 ] || { echo "FATAL: trainer image build failed: $BUILD_RC" >&2; exit "$BUILD_RC"; }
    [ -s "$PARTIAL" ] || { echo "FATAL: Pyxis did not save $PARTIAL" >&2; exit 2; }
    mv "$PARTIAL" "$CONTAINER_OUT"
    sha256sum "$CONTAINER_OUT" | tee "$CONTAINER_OUT.sha256"
    echo "OCI_NRT_H100_TRAINER_CONTAINER_READY=$CONTAINER_OUT"
    exit 0
fi

export HOME=${HOME:-$HOME_OVERRIDE}
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${PYTHONPATH:-}
export LD_PRELOAD="/lib/x86_64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export PIP_CONSTRAINT=/opt/trainer-constraints.txt
export TORCH_HOME=/opt/torch-cache
mkdir -p "$TORCH_HOME"

if ! command -v gcc >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends build-essential ca-certificates git
    apt-get clean
    find /var/lib/apt/lists -mindepth 1 -delete
fi

cp -a "$GR00T_STAGE" /opt/GR00T-VisualSim2Real
[ "$(cat /opt/GR00T-VisualSim2Real/.source_revision)" = "$EXPECTED_GR00T_REVISION" ] || {
    echo "FATAL: container GR00T revision mismatch" >&2; exit 2;
}

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
    required = children[0] / "source/isaaclab/pyproject.toml" if len(children) == 1 else None
    if len(children) != 1 or required is None or not required.is_file():
        raise SystemExit("malformed IsaacLab archive")
    os.replace(children[0], root)
(root / ".source_revision").write_text(revision + "\n", encoding="utf-8")
PY
[ "$(cat /opt/IsaacLab/.source_revision)" = "$ISAACLAB_REVISION" ] || {
    echo "FATAL: container IsaacLab revision mismatch" >&2; exit 2;
}

/isaac-sim/python.sh -m venv --system-site-packages /opt/viral-h100
PY=/opt/viral-h100/bin/python
install -m 0444 "$ROOT/containers/h100/trainer-constraints.txt" "$PIP_CONSTRAINT"
"$PY" -m pip install --upgrade pip==26.1.2 setuptools==80.9.0 \
    "wheel==0.45.1" poetry-core "packaging==23.2"
"$PY" -m pip install --no-build-isolation "flatdict==4.0.1"
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
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu128 triton==3.3.0
"$PY" -m pip install "Cython==3.2.9" "toml==0.10.2"
"$PY" -m pip install --no-build-isolation -e /opt/IsaacLab/source/isaaclab
"$PY" -m pip install --no-build-isolation \
    -e /opt/IsaacLab/source/isaaclab_contrib \
    -e /opt/IsaacLab/source/isaaclab_assets \
    -e /opt/IsaacLab/source/isaaclab_tasks \
    -e '/opt/IsaacLab/source/isaaclab_rl[all]'
"$PY" -m pip install numpy==1.26.0
"$PY" -m pip install -e /opt/GR00T-VisualSim2Real
"$PY" -m pip install docstring-parser==0.16 PyJWT==2.10.1

install -m 0555 "$ROOT/containers/h100/trainer_exec.sh" /usr/local/bin/roborender-trainer-sm90
install -m 0444 "$ROOT/containers/gb200/trainer_import_gate.py" /opt/trainer_import_gate.py
install -m 0444 "$ROOT/containers/gb200/isaac_camera_smoke.py" /opt/isaac_camera_smoke.py

ROBORENDER_EXPECTED_MACHINE=x86_64 ROBORENDER_EXPECTED_CC=9,0 \
ROBORENDER_EXPECTED_GR00T_REVISION="$EXPECTED_GR00T_REVISION" \
    "$PY" /opt/trainer_import_gate.py | tee "$IMPORT_REPORT"
grep -q ROBORENDER_TRAINER_IMPORT_GATE_PASS "$IMPORT_REPORT"
"$PY" "$ROOT/scripts/scaling/validate_production_hydra.py" \
    --gr00t-root /opt/GR00T-VisualSim2Real \
    --expected-revision "$EXPECTED_GR00T_REVISION" \
    --output "$HYDRA_REPORT"
"$PY" - <<'PY'
from torchvision.models import ResNet18_Weights, resnet18
resnet18(weights=ResNet18_Weights.DEFAULT)
print("resnet18_imagenet_cache=ready")
PY
echo "OCI_NRT_H100_TRAINER_BUILD_CONTENT_PASS"
