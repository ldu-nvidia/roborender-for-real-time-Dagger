#!/usr/bin/env bash
# Source-only regression gate. No simulator, GPU allocation or live service.
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$ROOT"
PYTHON=${ROBORENDER_TEST_PYTHON:-python3}
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export PYTHONPATH="$ROOT:$ROOT/pipeline:$ROOT/pipeline/dagger/rtx/code/src:$ROOT/pipeline/deployment/rtx_actor:$ROOT/pipeline/deployment/throughput${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m pytest -c "$ROOT/pytest.ini" "$@"
while IFS= read -r -d '' file; do
    [[ -f $file && $file == *.sh ]] || continue
    bash -n "$file"
done < <(git ls-files -z --cached --others --exclude-standard)
git diff --check
