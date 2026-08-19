#!/bin/bash
# Online image-model DAgger. The student acts, the teacher supervises, and some or all
# environments have their RGB observations replaced by Image3F renders every step.
#
#   sbatch --export=ALL,MIX=0.5,CKPT=<warm start> dagger_train.sbatch
#
# MIX = fraction of environments left on RAY-TRACED observations (videogen.raytraced_mix_prob):
#
#   MIX=0.0   fully image-generated  — every env relabelled; maximum renderer throughput
#                                      needed; trains purely on the generated distribution
#   MIX=0.5   mixed (default)        — half ray-traced, half generated; what round 4 used
#                                      (69.3%); hedges against renderer artefacts
#   MIX=1.0   fully ray-traced       — no relabelling at all; equivalent to plain DAgger,
#                                      useful as a control
#
# Renderer sizing: roughly one warm image3f server per 5 relabelled envs. With NUM_ENVS=16,
# MIX=0.5 needs ~2 servers and MIX=0.0 needs ~3. Too few servers does not fail, it just
# serialises the rollout — watch "Iteration time" (expect ~145 s at 16 envs / 3 servers).
#
# Variables:
#   MIX         ray-traced fraction, default 0.5
#   CKPT        warm-start checkpoint (default: round-4 final)
#   NUM_ENVS    default 16
#   EXP / PROJ  experiment / project name (default derived from MIX)
#   ANCHORS     anchor pool dir, default anchor_pool_2048
#   ANCHOR_IDX  which stored anchor frame, default 0
#
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=dagger_train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/dagger_train_3view_%j.log
set -euo pipefail

ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}
source "$ROOT/scripts/setup/oci_env.sh"
if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    exec bash "$ROOT/scripts/setup/reexec_in_oci_container.sh" "$ROOT/scripts/train/dagger_train_3view.sbatch"
fi
MIX=${MIX:-0.5}
NUM_ENVS=${NUM_ENVS:-16}
ANCHORS=${ANCHORS:-$ROBORENDER_STAGE_ROOT/assets/anchor_pool_2048}
ANCHOR_IDX=${ANCHOR_IDX:-0}
CKPT=${CKPT:-none}
MIXTAG=$(python3 -c "print({'0.0':'fullgen','1.0':'fullrt'}.get('$MIX','mix$MIX'))")
PROJ=${PROJ:-image_dagger_$MIXTAG}
EXP=${EXP:-dagger_$MIXTAG}

python3 -c "
mix = float('$MIX')
assert 0.0 <= mix <= 1.0, 'MIX must be in [0,1], got %r' % mix" || exit 1

export HOME=$HOME_OVERRIDE OMNI_KIT_ACCEPT_EULA=YES

# CKPT=none trains without a warm start. Required for the 3-view policy: its vision_obs
# is 3x taller, so single-view checkpoints (offline BC, image DAgger rounds 1-4) carry
# incompatible vision weights and cannot be loaded.
if [ "$CKPT" = none ]; then
    CKPT_ARG=()
    echo "warm start: none (vision trained from scratch)"
else
    [ -f "$CKPT" ] || { echo "FATAL: no such warm-start checkpoint: $CKPT"; exit 1; }
    CKPT_ARG=(checkpoint="$CKPT")
fi

# Collect every endpoint file whose host currently runs an i3f_warm_server. A stale endpoint
# left over from a finished job points the trainer at a dead host and kills the run hours in.
LIVE_NODES=$(squeue -u "$USER" -h -n i3f_warm_server -o '%N' | tr ',' '\n' | sort -u)
ENDPOINTS=()
for EP in "$ROOT"/image3f_server_endpoint.txt "$ROOT"/image3f_server_endpoint_[0-9].txt; do
    [ -f "$EP" ] || continue
    H=$(awk '{print $1}' "$EP" | cut -d. -f1)
    grep -qx "$H" <<<"$LIVE_NODES" && ENDPOINTS+=("$EP") && echo "  live renderer: $(cat "$EP")"
done

if [ "$MIX" = "1.0" ]; then
    echo "MIX=1.0 (fully ray-traced): no renderers required"
elif [ ${#ENDPOINTS[@]} -eq 0 ]; then
    echo "FATAL: MIX=$MIX needs relabelling but no live image3f server was found."
    echo "       Start them first:  sbatch --time=3-00:00:00 scripts/server/image3f_server.sbatch"
    exit 1
else
    NEED=$(python3 -c "import math; print(max(1, math.ceil($NUM_ENVS * (1 - $MIX) / 5)))")
    [ ${#ENDPOINTS[@]} -lt "$NEED" ] && echo \
      "WARNING: ${#ENDPOINTS[@]} renderer(s) for $NUM_ENVS envs at MIX=$MIX; ~$NEED recommended — expect slow iterations"
fi

# hydra list syntax: [a,b,c]
EP_LIST=$(IFS=,; echo "[${ENDPOINTS[*]}]")

echo "=================================================================="
echo " MIX=$MIX ($MIXTAG)   envs=$NUM_ENVS   renderers=${#ENDPOINTS[@]}"
echo " warm start : $CKPT"
echo " anchors    : $ANCHORS (index $ANCHOR_IDX)"
echo " run        : $PROJ / $EXP"
echo "=================================================================="

source "$VIRAL_VENV_ACTIVATE"
cd "$ROOT/GR00T-VisualSim2Real"
HYDRA_FULL_ERROR=1 accelerate launch --num_processes 1 \
    gr00t/rl/train_agent_trl.py \
    +exp=loco_manip/wsdpt_image_dagger_3view \
    num_envs="$NUM_ENVS" \
    headless=True \
    "${CKPT_ARG[@]}" \
    algo.config.videogen.raytraced_mix_prob="$MIX" \
    algo.config.videogen.anchor_pool_dir="$ANCHORS" \
    algo.config.videogen.anchor_index="$ANCHOR_IDX" \
    algo.config.videogen.image3f_endpoint_files="$EP_LIST" \
    experiment_name="$EXP" \
    project_name="$PROJ"
echo RUN_DONE
