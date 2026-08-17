#!/bin/bash
#SBATCH --job-name=i3f_warm_server
#SBATCH --partition=viscam-hi
#SBATCH --account=viscam
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/i3f_warm_server_%j.log
export HOME=/sailhome/ravenh
export PYTHONDONTWRITEBYTECODE=1
ROOT=/vision/u/ravenh/roborender_image_model_accelerated
export IMAGE3F_ROOT="$ROOT"
export ROBORENDER_MODEL_BASE=/vision/u/ravenh/roborender_image3f_portable_buildonly_v1/assets/models
export ENDPOINT_FILE=${ENDPOINT_FILE:-/svl/u/ravenh/fast_roborender/image3f_server_endpoint.txt}
export IMAGE3F_QUEUE=${IMAGE3F_QUEUE:-/svl/u/ravenh/fast_roborender/image3f_queue}
rm -f $IMAGE3F_QUEUE/READY
exec "$ROOT/.venv/bin/python" /svl/u/ravenh/fast_roborender/image3f_server.py
