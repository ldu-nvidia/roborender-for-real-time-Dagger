#!/bin/bash
# Apply entrypoint/smoke fixes to the already-built sealed ARM64 trainer image.
#SBATCH --account=llmservice_nemotron_ultra
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --mem=230G
#SBATCH --time=00:45:00
#SBATCH --job-name=rr_gb200_trainer_reseal
set -Eeuo pipefail

STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ldu/roborender_gb200_sm100}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
BASE=${BASE:-$STAGE_ROOT/containers/roborender-isaac-gr00t-arm64-sm100-isaac5.1-r4.sqsh}
OUTPUT=${OUTPUT:-$STAGE_ROOT/containers/roborender-isaac-gr00t-arm64-sm100-isaac5.1-r5.sqsh}
PARTIAL="$OUTPUT.partial.${SLURM_JOB_ID:-$$}"
export STAGE_ROOT ROOT BASE OUTPUT
[ -s "$BASE" ] || { echo "FATAL: base trainer image missing: $BASE" >&2; exit 2; }
[ ! -e "$OUTPUT" ] || { echo "FATAL: refusing to overwrite $OUTPUT" >&2; exit 2; }
[ ! -e "$PARTIAL" ] || { echo "FATAL: stale partial output exists: $PARTIAL" >&2; exit 2; }

if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    set +e
    srun --nodes=1 --ntasks=1 --gpus-per-task=4 --kill-on-bad-exit=1 \
        --container-image="$BASE" --container-writable --container-remap-root \
        --container-save="$PARTIAL" --container-mounts="$STAGE_ROOT:$STAGE_ROOT" \
        --container-workdir=/opt/GR00T-VisualSim2Real \
        env ROBORENDER_CONTAINERIZED=1 STAGE_ROOT="$STAGE_ROOT" ROOT="$ROOT" \
            BASE="$BASE" OUTPUT="$OUTPUT" OMNI_KIT_ACCEPT_EULA=YES \
            OMNI_KIT_ALLOW_ROOT=1 bash "$ROOT/scripts/scaling/hsg_gb200_trainer_reseal.sbatch"
    rc=$?
    set -e
    [ "$rc" -eq 0 ] || exit "$rc"
    [ -s "$PARTIAL" ] || { echo "FATAL: Pyxis did not save $PARTIAL" >&2; exit 2; }
    mv "$PARTIAL" "$OUTPUT"
    sha256sum "$OUTPUT" | tee "$OUTPUT.sha256"
    echo "HSG_GB200_TRAINER_RESEALED=$OUTPUT"
    exit 0
fi

install -m 0555 "$ROOT/containers/gb200/trainer_exec.sh" /usr/local/bin/roborender-trainer-sm100
install -m 0444 "$ROOT/containers/gb200/trainer_import_gate.py" /opt/trainer_import_gate.py
install -m 0444 "$ROOT/containers/gb200/isaac_camera_smoke.py" /opt/isaac_camera_smoke.py
/opt/viral-sm100/bin/python -m pip install "toml==0.10.2"
/opt/viral-sm100/bin/python -m pip install --no-build-isolation \
    -e /opt/IsaacLab/source/isaaclab_contrib
/usr/local/bin/roborender-trainer-sm100 python /opt/trainer_import_gate.py
echo "HSG_GB200_TRAINER_RESEAL_CONTENT_PASS"
