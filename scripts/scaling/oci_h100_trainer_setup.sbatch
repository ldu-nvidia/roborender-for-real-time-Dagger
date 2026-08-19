#!/bin/bash
# One-time Isaac 5.1 / IsaacLab / GR00T trainer environment on shared OCI Lustre.
#SBATCH --account=general_cs_infra
#SBATCH --partition=backfill_block1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=224G
#SBATCH --time=04:00:00
#SBATCH --job-name=rr_viral_setup
set -Eeuo pipefail

STAGE_ROOT=${STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-$STAGE_ROOT}
source "$ROOT/scripts/setup/oci_env.sh"
if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    exec bash "$ROOT/scripts/setup/reexec_in_oci_container.sh" \
        "$ROOT/scripts/scaling/oci_h100_trainer_setup.sbatch"
fi
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
EXPECTED_GR00T_REVISION=${EXPECTED_GR00T_REVISION:-b4d17449d70eb3fb7caa7ea9366ca59bb0589cdd}
ISAACLAB_ROOT=${ISAACLAB_ROOT:-$STAGE_ROOT/src/IsaacLab}
ISAACLAB_REVISION=${ISAACLAB_REVISION:-37ddf626871758333d6ed89cf64ad702aef127d0}
VIRAL_ENV=${VIRAL_ENV:-$STAGE_ROOT/runtime/envs/viral}
BASE_PYTHON=${BASE_PYTHON:-/isaac-sim/python.sh}
PY=${PY:-$VIRAL_ENV/bin/python}
ACTIVATE_WRAPPER=${ACTIVATE_WRAPPER:-$STAGE_ROOT/runtime/envs/activate_viral_isaac.sh}
HYDRA_CONTRACT_REPORT=${HYDRA_CONTRACT_REPORT:-$STAGE_ROOT/runtime/envs/production_hydra_contract.json}
export HOME=${HOME_OVERRIDE:-$STAGE_ROOT/runtime/home}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$STAGE_ROOT/runtime/cache/xdg}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$STAGE_ROOT/runtime/cache/pip}
export TORCH_HOME=${TORCH_HOME:-$STAGE_ROOT/runtime/cache/torch}
export TMPDIR=${TMPDIR:-$STAGE_ROOT/runtime/tmp}
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TMPDIR" "$(dirname "$VIRAL_ENV")"

[ -d "$GR00T_ROOT" ] || { echo "FATAL: integrated GR00T source missing: $GR00T_ROOT" >&2; exit 2; }
[ -x "$BASE_PYTHON" ] || { echo "FATAL: Isaac Python missing: $BASE_PYTHON" >&2; exit 2; }

# Resolve a normal Git checkout's HEAD without depending on a git binary in
# the Isaac container. Both loose and packed refs are supported.
GR00T_REVISION=$("$BASE_PYTHON" - "$GR00T_ROOT" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1])
git_dir = repo / ".git"
if git_dir.is_file():
    line = git_dir.read_text(encoding="utf-8").strip()
    if not line.startswith("gitdir: "):
        raise SystemExit("FATAL: malformed GR00T .git file")
    git_dir = (repo / line.removeprefix("gitdir: ")).resolve()
head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
if not head.startswith("ref: "):
    print(head)
    raise SystemExit(0)
ref = head.removeprefix("ref: ")
loose = git_dir / ref
if loose.is_file():
    print(loose.read_text(encoding="utf-8").strip())
    raise SystemExit(0)
packed = git_dir / "packed-refs"
if packed.is_file():
    for line in packed.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "^")):
            revision, name = line.split(" ", 1)
            if name == ref:
                print(revision)
                raise SystemExit(0)
raise SystemExit(f"FATAL: cannot resolve GR00T HEAD ref {ref}")
PY
)
[ "$GR00T_REVISION" = "$EXPECTED_GR00T_REVISION" ] || {
    echo "FATAL: GR00T revision $GR00T_REVISION != $EXPECTED_GR00T_REVISION" >&2
    exit 2
}

# The Isaac container intentionally has no git binary. Download the pinned
# upstream source archive with the standard library and record its immutable
# revision instead of silently following a branch head.
"$BASE_PYTHON" - "$ISAACLAB_ROOT" "$ISAACLAB_REVISION" <<'PY'
from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

root = Path(sys.argv[1])
revision = sys.argv[2]
marker = root / ".oci_source_revision"
required = root / "source/isaaclab/pyproject.toml"
if marker.is_file() and marker.read_text(encoding="utf-8").strip() == revision and required.is_file():
    print(f"IsaacLab source already staged at {revision}")
    raise SystemExit(0)
if root.exists() and any(root.iterdir()):
    raise SystemExit(
        f"FATAL: unverified IsaacLab source exists at {root}; stage revision {revision} explicitly"
    )
root.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="isaaclab_stage_", dir=root.parent) as temp_name:
    temp = Path(temp_name)
    archive = temp / "source.tar.gz"
    url = f"https://codeload.github.com/isaac-sim/IsaacLab/tar.gz/{revision}"
    print(f"Downloading pinned IsaacLab revision {revision}")
    urllib.request.urlretrieve(url, archive)
    unpack = temp / "unpack"
    unpack.mkdir()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (unpack / member.name).resolve()
            if unpack.resolve() not in target.parents and target != unpack.resolve():
                raise SystemExit(f"FATAL: unsafe archive member: {member.name}")
        handle.extractall(unpack)
    children = [path for path in unpack.iterdir() if path.is_dir()]
    if len(children) != 1 or not (children[0] / "source/isaaclab/pyproject.toml").is_file():
        raise SystemExit("FATAL: malformed IsaacLab source archive")
    if root.exists():
        root.rmdir()
    os.replace(children[0], root)
marker.write_text(revision + "\n", encoding="utf-8")
print(f"IsaacLab source staged at {root}")
PY

if [ ! -x "$PY" ]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$VIRAL_ENV"
fi
"$PY" -m pip install --upgrade pip setuptools wheel poetry-core flatdict
"$PY" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install --no-build-isolation -e "$ISAACLAB_ROOT/source/isaaclab"
"$PY" -m pip install --no-build-isolation \
    -e "$ISAACLAB_ROOT/source/isaaclab_assets" \
    -e "$ISAACLAB_ROOT/source/isaaclab_tasks" \
    -e "$ISAACLAB_ROOT/source/isaaclab_rl[all]"
"$PY" -m pip install numpy==1.26.0
"$PY" -m pip install -e "$GR00T_ROOT"
"$PY" "$ROOT/scripts/scaling/validate_production_hydra.py" \
    --gr00t-root "$GR00T_ROOT" \
    --expected-revision "$GR00T_REVISION" \
    --output "$HYDRA_CONTRACT_REPORT"
# Fresh policy initialization uses ImageNet ResNet-18 weights. Populate the
# shared cache once so eight ranks never race on a launch-time download.
"$PY" - <<'PY'
from torchvision.models import ResNet18_Weights, resnet18
resnet18(weights=ResNet18_Weights.DEFAULT)
print("resnet18_imagenet_cache=ready")
PY

cat > "$ACTIVATE_WRAPPER" <<EOF
#!/bin/bash
set -Eeuo pipefail
source /isaac-sim/setup_python_env.sh
source "$VIRAL_ENV/bin/activate"
EOF
chmod 0755 "$ACTIVATE_WRAPPER"

source "$ACTIVATE_WRAPPER"
python - <<'PY'
import isaaclab
import torch
import gr00t
import wandb
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"isaaclab={isaaclab.__file__}")
print(f"gr00t={gr00t.__version__}")
print(f"wandb={wandb.__version__}")
PY
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
cat > "$STAGE_ROOT/runtime/envs/trainer_environment_manifest.txt" <<EOF
isaac_container=nvcr.io#nvidia/isaac-sim:5.1.0
isaaclab_revision=$(cat "$ISAACLAB_ROOT/.oci_source_revision")
gr00t_revision=$GR00T_REVISION
python=$PY
activation=$ACTIVATE_WRAPPER
torch_home=$TORCH_HOME
production_hydra_contract=$HYDRA_CONTRACT_REPORT
production_hydra_contract_sha256=$(sha256sum "$HYDRA_CONTRACT_REPORT" | awk '{print $1}')
EOF
echo "VIRAL_H100_SETUP_DONE"
