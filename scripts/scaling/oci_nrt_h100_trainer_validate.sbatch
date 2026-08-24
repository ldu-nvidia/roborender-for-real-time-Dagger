#!/bin/bash
# Validate the sealed H100 Isaac/GR00T image, including PhysX and RGB/depth.
#SBATCH --account=general_cs_infra
#SBATCH --partition=batch_block1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=224G
#SBATCH --time=00:50:00
#SBATCH --job-name=rr_h100_trainer_validate
set -Eeuo pipefail

STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs1/portfolios/general/projects/general_cs_infra__ngcsre/users/ldu/roborender_h100_build}
ROOT=${ROOT:?submit with ROOT set to the immutable staged source snapshot}
CONTAINER=${CONTAINER:-$STAGE_ROOT/containers/roborender-isaac-gr00t-x86_64-sm90-isaac5.1.sqsh}
OUTPUT=${OUTPUT:-$STAGE_ROOT/validation/h100_isaac_camera_smoke}
export STAGE_ROOT ROOT CONTAINER OUTPUT
[ -s "$CONTAINER" ] || { echo "FATAL: trainer container missing: $CONTAINER" >&2; exit 2; }
mkdir -p "$OUTPUT" "$STAGE_ROOT/runtime/home"

if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    exec srun --nodes=1 --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 \
        --container-image="$CONTAINER" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT" \
        --container-workdir=/opt/GR00T-VisualSim2Real \
        env ROBORENDER_CONTAINERIZED=1 STAGE_ROOT="$STAGE_ROOT" ROOT="$ROOT" \
            CONTAINER="$CONTAINER" OUTPUT="$OUTPUT" \
            HOME="$STAGE_ROOT/runtime/home" OMNI_KIT_ACCEPT_EULA=YES \
            OMNI_KIT_ALLOW_ROOT=1 \
            bash "$ROOT/scripts/scaling/oci_nrt_h100_trainer_validate.sbatch"
fi

export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export PYTHONPATH=${PYTHONPATH:-}
export LD_PRELOAD="/lib/x86_64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source /isaac-sim/setup_python_env.sh
CHECKER=
for candidate in /isaac-sim/isaac-sim.compatibility_check.sh /isaac-sim/compatibility_check.sh; do
    if [ -x "$candidate" ]; then CHECKER=$candidate; break; fi
done
[ -n "$CHECKER" ] || { echo "FATAL: Isaac compatibility checker missing" >&2; exit 2; }
timeout 10m "$CHECKER" --/app/quitAfter=10 --no-window \
    >"$OUTPUT/compatibility_check.log" 2>&1
timeout 15m /usr/local/bin/roborender-trainer-sm90 \
    python /opt/trainer_import_gate.py | tee "$OUTPUT/sealed_imports.json"
timeout 20m /usr/local/bin/roborender-trainer-sm90 \
    python /opt/isaac_camera_smoke.py --output-dir "$OUTPUT" \
    | tee "$OUTPUT/camera_stdout.log"
test -s "$OUTPUT/isaac_rgb.png"
test -s "$OUTPUT/isaac_depth.npy"
grep -q '"status": "PASS"' "$OUTPUT/isaac_camera_smoke.json"
echo "OCI_NRT_H100_TRAINER_SEALED_PASS"
