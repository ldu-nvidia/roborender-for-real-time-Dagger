#!/bin/bash
# Evaluate a policy checkpoint under a KNOWN-GOOD config.
#
#   sbatch --export=ALL,CKPT=<path>[,EPISODES=300][,OBS=raytraced|generated] eval_policy.sbatch
#
# Why this script exists: eval_agent_trl.py loads config.yaml from the CHECKPOINT'S OWN
# directory. Evaluating a DAgger checkpoint straight out of its run directory therefore
# scores it in that run's environment, not the baseline's — which silently invalidated
# three rounds of results. This script always stages the checkpoint next to an explicit
# reference config so every number is comparable.
#
# Variables:
#   CKPT       (required) checkpoint .pt to evaluate
#   EPISODES   episode count. Default 300 (SE ~2.7 pts); 150 gives SE ~4.0
#   OBS        raytraced (default) | generated
#   CONFIG     override the reference config (defaults per OBS, see below)
#   ENDPOINT   OBS=generated only: renderer endpoint file
#   NUM_ENVS   parallel envs, default 4
#
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=eval_policy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/eval_policy_%j.log
set -euo pipefail

ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}
source "$ROOT/scripts/setup/oci_env.sh"
if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    exec bash "$ROOT/scripts/setup/reexec_in_oci_container.sh" "$ROOT/scripts/eval/eval_policy.sbatch"
fi
REPO=$GR00T_ROOT
# Reference configs. RAYTRACED = the offline-BC baseline's own config, i.e. the environment
# every headline number in README.md was measured in. GENERATED additionally needs the
# videogen/image_dagger block, so it comes from a DAgger run.
BASELINE_CFG=${BASELINE_CFG:-}
RELABEL_CFG=${RELABEL_CFG:-}

: "${CKPT:?set CKPT=<path to .pt>}"
EPISODES=${EPISODES:-300}
OBS=${OBS:-raytraced}
NUM_ENVS=${NUM_ENVS:-4}
ENDPOINT=${ENDPOINT:-$ROOT/image3f_server_endpoint.txt}

export HOME=$HOME_OVERRIDE OMNI_KIT_ACCEPT_EULA=YES

[ -f "$CKPT" ] || { echo "FATAL: no such checkpoint: $CKPT"; exit 1; }

case "$OBS" in
  raytraced)
    : "${CONFIG:=${BASELINE_CFG:?set CONFIG or BASELINE_CFG to a staged config}}"
    ;;
  generated)
    : "${CONFIG:=${RELABEL_CFG:?set CONFIG or RELABEL_CFG to a staged config}}"
    export IMAGE_DAGGER_EVAL_RELABEL=1
    # A stale endpoint file outlives its server and points the client at a dead host,
    # so require one whose node still has a running i3f_warm_server job.
    [ -f "$ENDPOINT" ] || { echo "FATAL: OBS=generated needs a renderer; $ENDPOINT missing"; exit 1; }
    EP_HOST=$(awk '{print $1}' "$ENDPOINT" | cut -d. -f1)
    squeue -u "$USER" -h -n i3f_warm_server -o '%N' | tr ',' '\n' | grep -qx "$EP_HOST" \
      || { echo "FATAL: $ENDPOINT points at $EP_HOST but no i3f_warm_server runs there (stale)"; exit 1; }
    echo "renderer: $(cat "$ENDPOINT")"
    ;;
  *) echo "FATAL: OBS must be raytraced or generated (got '$OBS')"; exit 1 ;;
esac
[ -f "$CONFIG" ] || { echo "FATAL: no such config: $CONFIG"; exit 1; }

STAGE=$ROOT/.eval_stage/${SLURM_JOB_ID:-manual}
mkdir -p "$STAGE"
cp "$CKPT" "$STAGE/last.pt"
cp "$CONFIG" "$STAGE/config.yaml"
# Score every env through the renderer, not half of them.
[ "$OBS" = generated ] && sed -i 's/raytraced_mix_prob: .*/raytraced_mix_prob: 0.0/' "$STAGE/config.yaml"

echo "=================================================================="
echo " checkpoint : $CKPT"
echo " config     : $CONFIG"
echo " obs        : $OBS      episodes: $EPISODES      envs: $NUM_ENVS"
echo "=================================================================="

source "$VIRAL_VENV_ACTIVATE"
cd "$REPO"
python gr00t/rl/eval_agent_trl.py \
    +checkpoint="$STAGE/last.pt" \
    +num_envs="$NUM_ENVS" +headless=True \
    algo.config.eval.num_eval_episodes="$EPISODES" \
    +simulator.config.render_results=true

LOG=$ROBORENDER_STAGE_ROOT/logs/eval_policy_${SLURM_JOB_ID:-manual}.log
echo "RESULT $(basename "$CKPT") obs=$OBS eps=$EPISODES :: $(grep -hoE 'success rate.*' "$LOG" | tail -1)"
rm -rf "$STAGE"
echo EVAL_DONE
