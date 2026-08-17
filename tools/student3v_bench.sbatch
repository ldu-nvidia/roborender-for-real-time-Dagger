#!/bin/bash
#SBATCH --partition=viscam-hi
#SBATCH --account=viscam
#SBATCH --job-name=student3v_bench
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/student3v_bench_%j.log
set -x
export HOME=/sailhome/ravenh
cd /svl/u/ravenh/fast_roborender
PORT=$((18000 + SLURM_JOB_ID % 1000))
/svl/u/ravenh/fast_roborender/run_raven_server.sh --host 127.0.0.1 --port $PORT \
  --student_pth /svl/u/ravenh/fast_roborender/roborender-acceleration/assets/student/raven_native3view_ladd_5step_iter5250.pth &
SERVER_PID=$!
FR=/svl/u/ravenh/fast_roborender
CHUNKS="$FR/videogen_dataset/chunk_16222830_0001309.npz $FR/videogen_dataset/chunk_16222830_0000051.npz $FR/videogen_dataset/chunk_16222830_0000501.npz"
/svl/u/ravenh/miniconda3/envs/gr00t/bin/python render_teacher_fullres.py \
  --chunks $CHUNKS --out "$FR/student3v_fullres" --port $PORT
RC=$?
kill $SERVER_PID 2>/dev/null
[ $RC -ne 0 ] && { echo BENCH_FAIL; exit 1; }
echo BENCH_DONE
