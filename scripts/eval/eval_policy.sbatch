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
#SBATCH --partition=viscam-hi
#SBATCH --account=viscam
#SBATCH --job-name=eval_policy
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --exclude=viscam10,viscam12
#SBATCH --output=/svl/u/ravenh/fast_roborender/logs_archive/eval_policy_%j.log
set -euo pipefail

ROOT=/svl/u/ravenh/fast_roborender
REPO=$ROOT/GR00T-VisualSim2Real
# Reference configs. RAYTRACED = the offline-BC baseline's own config, i.e. the environment
# every headline number in README.md was measured in. GENERATED additionally needs the
# videogen/image_dagger block, so it comes from a DAgger run.
BASELINE_CFG=$REPO/logs_rl/videogen_offline_bc/videogen_offline_bc-20260721_204447/config.yaml
RELABEL_CFG=$REPO/logs_rl/image_dagger_r4/image_dagger_run4-20260814_131202/config.yaml

: "${CKPT:?set CKPT=<path to .pt>}"
EPISODES=${EPISODES:-300}
OBS=${OBS:-raytraced}
NUM_ENVS=${NUM_ENVS:-4}
ENDPOINT=${ENDPOINT:-$ROOT/image3f_server_endpoint.txt}

export HOME=/sailhome/ravenh
export XDG_CACHE_HOME=/svl/u/ravenh/.cache_xdg XDG_DATA_HOME=/svl/u/ravenh/.share_xdg
export TMPDIR=/svl/u/ravenh/tmp OMNI_KIT_ACCEPT_EULA=YES

[ -f "$CKPT" ] || { echo "FATAL: no such checkpoint: $CKPT"; exit 1; }

case "$OBS" in
  raytraced) CONFIG=${CONFIG:-$BASELINE_CFG} ;;
  generated)
    CONFIG=${CONFIG:-$RELABEL_CFG}
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

source /svl/u/ravenh/miniconda3/etc/profile.d/conda.sh
conda activate viral
cd "$REPO"
python gr00t/rl/eval_agent_trl.py \
    +checkpoint="$STAGE/last.pt" \
    +num_envs="$NUM_ENVS" +headless=True \
    algo.config.eval.num_eval_episodes="$EPISODES" \
    +simulator.config.render_results=true

LOG=$ROOT/logs_archive/eval_policy_${SLURM_JOB_ID:-manual}.log
echo "RESULT $(basename "$CKPT") obs=$OBS eps=$EPISODES :: $(grep -hoE 'success rate.*' "$LOG" | tail -1)"
rm -rf "$STAGE"
echo EVAL_DONE
