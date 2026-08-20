#!/bin/bash
# Validate the sealed Isaac/GR00T image, including native Kit camera output.
#SBATCH --account=llmservice_nemotron_ultra
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --mem=230G
#SBATCH --time=00:45:00
#SBATCH --job-name=rr_gb200_trainer_validate
set -Eeuo pipefail

STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ldu/roborender_gb200_sm100}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
CONTAINER=${CONTAINER:-$STAGE_ROOT/containers/roborender-isaac-gr00t-arm64-sm100-isaac5.1-r5.sqsh}
OUTPUT=${OUTPUT:-$STAGE_ROOT/validation/isaac_camera_smoke_r5}
SMOKE_SCRIPT=${SMOKE_SCRIPT:-/opt/isaac_camera_smoke.py}
export STAGE_ROOT ROOT CONTAINER OUTPUT SMOKE_SCRIPT
[ -s "$CONTAINER" ] || { echo "FATAL: trainer container missing: $CONTAINER" >&2; exit 2; }
mkdir -p "$OUTPUT" "$STAGE_ROOT/runtime/home"

if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    exec srun --nodes=1 --ntasks=1 --gpus-per-task=4 --kill-on-bad-exit=1 \
        --container-image="$CONTAINER" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT" \
        --container-workdir=/opt/GR00T-VisualSim2Real \
        env ROBORENDER_CONTAINERIZED=1 STAGE_ROOT="$STAGE_ROOT" ROOT="$ROOT" \
            CONTAINER="$CONTAINER" OUTPUT="$OUTPUT" \
            SMOKE_SCRIPT="$SMOKE_SCRIPT" \
            HOME="$STAGE_ROOT/runtime/home" OMNI_KIT_ACCEPT_EULA=YES \
            OMNI_KIT_ALLOW_ROOT=1 \
            bash "$ROOT/scripts/scaling/hsg_gb200_trainer_validate.sbatch"
fi

export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export PYTHONPATH=${PYTHONPATH:-}
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source /isaac-sim/setup_python_env.sh
CHECKER=
for candidate in /isaac-sim/isaac-sim.compatibility_check.sh /isaac-sim/compatibility_check.sh; do
    if [ -x "$candidate" ]; then CHECKER=$candidate; break; fi
done
[ -n "$CHECKER" ] || { echo "FATAL: Isaac compatibility checker missing" >&2; exit 2; }
timeout 10m "$CHECKER" --/app/quitAfter=10 --no-window \
    >"$OUTPUT/compatibility_check.log" 2>&1
timeout 15m /usr/local/bin/roborender-trainer-sm100 \
    python /opt/trainer_import_gate.py | tee "$OUTPUT/sealed_imports.json"
timeout 15m /usr/local/bin/roborender-trainer-sm100 \
    python "$SMOKE_SCRIPT" --output-dir "$OUTPUT" \
    | tee "$OUTPUT/camera_stdout.log"
test -s "$OUTPUT/isaac_rgb.png"
test -s "$OUTPUT/isaac_depth.npy"
grep -q '"status": "PASS"' "$OUTPUT/isaac_camera_smoke.json"
echo "HSG_GB200_TRAINER_SEALED_PASS"
