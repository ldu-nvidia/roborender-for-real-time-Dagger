#!/bin/bash
#SBATCH --partition=sc-loprio
#SBATCH --account=viscam
#SBATCH --job-name=raven_srv_test
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/test_raven_server.log
set -e
cd /svl/u/ravenh/fast_roborender
SERVER_LOG=/svl/u/ravenh/fast_roborender/raven_server_test.log
/svl/u/ravenh/fast_roborender/run_raven_server.sh --port 18913 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
until grep -q RAVEN_SERVER_READY "$SERVER_LOG" 2>/dev/null; do
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER_DIED"; tail -40 "$SERVER_LOG"; exit 1; fi
  sleep 5
done
echo "server ready, running client test"
roborender-acceleration/.venv/bin/python /svl/u/ravenh/fast_roborender/archive/test_raven_server.py
kill $SERVER_PID
echo SERVER_TEST_DONE
