#!/bin/bash
# One-time Isaac 5.1 / IsaacLab / GR00T trainer environment on shared OCI Lustre.
#SBATCH --account=general_cs_infra
#SBATCH --partition=backfill_block1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --job-name=rr_viral_setup
#SBATCH --container-image="nvcr.io#nvidia/isaac-sim:5.1.0"
#SBATCH --container-mounts=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale:/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale
set -Eeuo pipefail

STAGE_ROOT=${STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
ISAACLAB_ROOT=${ISAACLAB_ROOT:-$STAGE_ROOT/src/IsaacLab}
VIRAL_ENV=${VIRAL_ENV:-$STAGE_ROOT/runtime/envs/viral}
BASE_PYTHON=${BASE_PYTHON:-/isaac-sim/python.sh}
PY=${PY:-$VIRAL_ENV/bin/python}
ACTIVATE_WRAPPER=${ACTIVATE_WRAPPER:-$STAGE_ROOT/runtime/envs/activate_viral_isaac.sh}
export HOME=${HOME_OVERRIDE:-$STAGE_ROOT/runtime/home}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$STAGE_ROOT/runtime/cache/xdg}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$STAGE_ROOT/runtime/cache/pip}
export TORCH_HOME=${TORCH_HOME:-$STAGE_ROOT/runtime/cache/torch}
export TMPDIR=${TMPDIR:-$STAGE_ROOT/runtime/tmp}
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TMPDIR" "$(dirname "$VIRAL_ENV")"

[ -d "$GR00T_ROOT" ] || { echo "FATAL: integrated GR00T source missing: $GR00T_ROOT" >&2; exit 2; }
[ -x "$BASE_PYTHON" ] || { echo "FATAL: Isaac Python missing: $BASE_PYTHON" >&2; exit 2; }

if [ ! -d "$ISAACLAB_ROOT/.git" ]; then
    git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_ROOT"
fi
if [ -n "${ISAACLAB_REVISION:-}" ]; then
    [ "$(git -C "$ISAACLAB_ROOT" rev-parse HEAD)" = "$ISAACLAB_REVISION" ] || {
        echo "FATAL: IsaacLab revision does not match ISAACLAB_REVISION" >&2
        exit 2
    }
fi

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
isaaclab_revision=$(git -C "$ISAACLAB_ROOT" rev-parse HEAD)
gr00t_revision=$(git -C "$GR00T_ROOT" rev-parse HEAD)
python=$PY
activation=$ACTIVATE_WRAPPER
torch_home=$TORCH_HOME
EOF
echo "VIRAL_H100_SETUP_DONE"
