#!/bin/bash
#SBATCH --partition=viscam
#SBATCH --account=viscam
#SBATCH --job-name=teacher_anchor
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/teacher_anchor_%j.log
set -x
export HOME=/sailhome/ravenh
cd /svl/u/ravenh/fast_roborender
FR=/svl/u/ravenh/fast_roborender
PORT=$((18000 + SLURM_JOB_ID % 1000))
/svl/u/ravenh/fast_roborender/run_raven_server.sh --host 127.0.0.1 --port $PORT --teacher_lora /svl/u/ravenh/renderscale/Sim2RealGen_Video_Model/lora_ckpts/Wan2.1-Fun-V1.1-1.3B-Mixed-DROID-AGIBOT-3View-DualControl-PrevFrame-droidnew_continue_lora/step-5294.safetensors &
SERVER_PID=$!
CHUNKS="${CHUNKS_OVERRIDE:- /svl/u/ravenh/fast_roborender/videogen_dataset/chunk_16222830_0000123.npz /svl/u/ravenh/fast_roborender/videogen_dataset/chunk_16222830_0000404.npz /svl/u/ravenh/fast_roborender/videogen_dataset/chunk_16222830_0000777.npz /svl/u/ravenh/fast_roborender/videogen_dataset/chunk_16222830_0001111.npz /svl/u/ravenh/fast_roborender/videogen_dataset/chunk_16222830_0001666.npz /svl/u/ravenh/fast_roborender/videogen_dataset/chunk_16222830_0001999.npz}"
/svl/u/ravenh/miniconda3/envs/gr00t/bin/python render_teacher_fullres.py \
  --chunks $CHUNKS --out "$FR/teacher_fullres" --port $PORT
RC=$?
kill $SERVER_PID 2>/dev/null
[ $RC -ne 0 ] && { echo RENDER_FAIL; exit 1; }
echo ANCHOR_JOB_DONE
