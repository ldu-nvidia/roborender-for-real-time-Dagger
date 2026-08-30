#!/usr/bin/env bash
# Immutable Lepton Job entrypoint for one camera-free 8xH200 learner attempt.
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

fatal() { echo "FATAL: $*" >&2; exit 2; }

required=(
  ROUND_DAGGER_LEPTON_BACKEND_CONTRACT
  ROUND_DAGGER_LEPTON_BACKEND_CONTRACT_SHA256
  ROUND_DAGGER_LEPTON_RUN_REQUEST
  ROUND_DAGGER_LEPTON_RUN_REQUEST_SHA256
  ROUND_DAGGER_LEPTON_SEMANTIC_ATTEMPT
)
for name in "${required[@]}"; do
  [[ -n ${!name:-} ]] || fatal "missing required Lepton Job environment: $name"
done
[[ ${WANDB_API_KEY:+injected} == injected ]] || \
  fatal "W&B credential was not injected by Lepton Secret"

case "$ROUND_DAGGER_LEPTON_BACKEND_CONTRACT" in
  /workspace/projects/roborender/*) ;;
  *) fatal "backend contract is outside the persistent RoboRender project" ;;
esac
case "$ROUND_DAGGER_LEPTON_RUN_REQUEST" in
  /workspace/projects/roborender/*) ;;
  *) fatal "run request is outside the persistent RoboRender project" ;;
esac
[[ $ROUND_DAGGER_LEPTON_BACKEND_CONTRACT_SHA256 =~ ^[0-9a-f]{64}$ ]] || \
  fatal "backend contract SHA-256 is malformed"
[[ $ROUND_DAGGER_LEPTON_RUN_REQUEST_SHA256 =~ ^[0-9a-f]{64}$ ]] || \
  fatal "run request SHA-256 is malformed"
[[ $ROUND_DAGGER_LEPTON_SEMANTIC_ATTEMPT =~ ^[1-3]$ ]] || \
  fatal "semantic attempt is outside the bridge-owned retry budget"
[[ -f $ROUND_DAGGER_LEPTON_BACKEND_CONTRACT && \
   ! -L $ROUND_DAGGER_LEPTON_BACKEND_CONTRACT ]] || \
  fatal "backend contract is missing or unsafe"
[[ $(sha256sum "$ROUND_DAGGER_LEPTON_BACKEND_CONTRACT" | awk '{print $1}') == \
   "$ROUND_DAGGER_LEPTON_BACKEND_CONTRACT_SHA256" ]] || \
  fatal "backend contract SHA-256 mismatch"
[[ -f $ROUND_DAGGER_LEPTON_RUN_REQUEST && ! -L $ROUND_DAGGER_LEPTON_RUN_REQUEST ]] || \
  fatal "run request is missing or unsafe"
[[ $(sha256sum "$ROUND_DAGGER_LEPTON_RUN_REQUEST" | awk '{print $1}') == \
   "$ROUND_DAGGER_LEPTON_RUN_REQUEST_SHA256" ]] || \
  fatal "run request SHA-256 mismatch"

python_path=$(
  PYTHONDONTWRITEBYTECODE=1 python3 - "$ROUND_DAGGER_LEPTON_BACKEND_CONTRACT" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
python = value.get("container", {}).get("python")
if not isinstance(python, str) or not python.startswith("/workspace/projects/roborender/"):
    raise SystemExit(2)
print(python)
PY
) || fatal "cannot resolve pinned learner Python"
[[ -x $python_path ]] || fatal "pinned learner Python is not executable"

source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
export PYTHONPATH="$source_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_path" -m round_dagger.lepton_job_execute \
  --contract "$ROUND_DAGGER_LEPTON_BACKEND_CONTRACT" \
  --contract-sha256 "$ROUND_DAGGER_LEPTON_BACKEND_CONTRACT_SHA256" \
  --request "$ROUND_DAGGER_LEPTON_RUN_REQUEST" \
  --request-sha256 "$ROUND_DAGGER_LEPTON_RUN_REQUEST_SHA256" \
  --semantic-attempt "$ROUND_DAGGER_LEPTON_SEMANTIC_ATTEMPT"
