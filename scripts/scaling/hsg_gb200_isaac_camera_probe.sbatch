#!/bin/bash
# Dynamic GB200 gate for native Isaac Kit/Vulkan/RTX camera execution.
#SBATCH --account=llmservice_nemotron_ultra
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --mem=230G
#SBATCH --time=00:45:00
#SBATCH --job-name=rr_gb200_isaac_camera
set -Eeuo pipefail

STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ldu/roborender_gb200_sm100}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
export ROBORENDER_STAGE_ROOT=$STAGE_ROOT ROOT
source "$ROOT/scripts/setup/hsg_gb200_env.sh"
OUTPUT=${OUTPUT:-$STAGE_ROOT/validation/isaac_base_camera_smoke}
mkdir -p "$OUTPUT" "$HOME_OVERRIDE"

if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    exec srun --nodes=1 --ntasks=1 --gpus-per-task=4 --kill-on-bad-exit=1 \
        --container-image="$HSG_ISAAC_CONTAINER_IMAGE" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT" \
        --container-workdir="$ROOT" \
        env ROBORENDER_CONTAINERIZED=1 ROBORENDER_STAGE_ROOT="$STAGE_ROOT" \
            ROOT="$ROOT" OUTPUT="$OUTPUT" HOME="$HOME_OVERRIDE" \
            OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 \
            bash "$ROOT/scripts/scaling/hsg_gb200_isaac_camera_probe.sbatch"
fi

export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
CHECKER=
for candidate in /isaac-sim/isaac-sim.compatibility_check.sh /isaac-sim/compatibility_check.sh; do
    if [ -x "$candidate" ]; then CHECKER=$candidate; break; fi
done
[ -n "$CHECKER" ] || { echo "FATAL: Isaac compatibility checker missing" >&2; exit 2; }
set +e
timeout 10m "$CHECKER" --/app/quitAfter=10 --no-window \
    >"$OUTPUT/compatibility_check.log" 2>&1
CHECK_RC=$?
timeout 15m /isaac-sim/python.sh \
    "$ROOT/containers/gb200/isaac_sim_camera_smoke.py" --output-dir "$OUTPUT" \
    >"$OUTPUT/camera_stdout.log" 2>"$OUTPUT/camera_stderr.log"
CAMERA_RC=$?
set -e
printf 'compatibility_check_rc=%s\ncamera_rc=%s\n' "$CHECK_RC" "$CAMERA_RC" \
    | tee "$OUTPUT/status.txt"
[ "$CHECK_RC" -eq 0 ] || { tail -120 "$OUTPUT/compatibility_check.log" >&2; exit "$CHECK_RC"; }
[ "$CAMERA_RC" -eq 0 ] || { tail -120 "$OUTPUT/camera_stderr.log" >&2; exit "$CAMERA_RC"; }
test -s "$OUTPUT/base_rgb.png"
test -s "$OUTPUT/base_depth.npy"
grep -q '"status": "PASS"' "$OUTPUT/isaac_base_camera_smoke.json"
echo "HSG_GB200_ISAAC_DYNAMIC_PASS"
