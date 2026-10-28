#!/bin/bash
#SBATCH --partition=viscam
#SBATCH --account=viscam
#SBATCH --job-name=raven_server
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/raven_server_standalone_%j.log
set -e
export HOME=/sailhome/ravenh
cd /svl/u/ravenh/fast_roborender
PORT=$((18000 + SLURM_JOB_ID % 1000))
echo "$(hostname) $PORT" > ${ENDPOINT_FILE:-/svl/u/ravenh/fast_roborender/raven_server_endpoint.txt}
echo "endpoint: $(hostname):$PORT"
exec /svl/u/ravenh/fast_roborender/run_raven_server.sh --host 0.0.0.0 --port $PORT ${RAVEN_SERVER_ARGS:-}
