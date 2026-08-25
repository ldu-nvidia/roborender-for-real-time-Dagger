#!/usr/bin/env bash
# Bounded rank-zero W&B gate around run_actor_validation.sh.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
VALIDATION="$ROOT/scripts/round_dagger/run_actor_validation.sh"
RUN_DIR=${TRAIN_RUN_DIR:?missing TRAIN_RUN_DIR}
RANKS=${ROUND_DAGGER_VALIDATION_RANKS:-1}
ACTOR_RANK_RAW=${ROUND_DAGGER_VALIDATION_ACTOR_RANK:-}
ACTOR_RANK=${ACTOR_RANK_RAW:-0}
STEPS=${ROUND_DAGGER_VALIDATION_STEPS:-2}

case "$RUN_DIR" in /*) ;; *) echo "TRAIN_RUN_DIR must be absolute" >&2; exit 2 ;; esac
[[ $(realpath -m -- "$RUN_DIR") == "$RUN_DIR" ]] || {
    echo "TRAIN_RUN_DIR must be a canonical path without symlink traversal" >&2
    exit 2
}
[[ $RANKS =~ ^(1|2)$ ]] || { echo "validation ranks must be exactly 1 or 2" >&2; exit 2; }
if ((RANKS == 2)) && [[ -z $ACTOR_RANK_RAW ]]; then
    echo "two-rank bounded validation requires an explicit isolated actor rank" >&2
    exit 2
fi
[[ $ACTOR_RANK =~ ^(0|1)$ ]] || {
    echo "validation actor rank must be exactly 0 or 1" >&2; exit 2;
}
((ACTOR_RANK < RANKS)) || {
    echo "validation actor rank must be smaller than validation ranks" >&2; exit 2;
}
[[ $STEPS =~ ^[1-9][0-9]*$ ]] || {
    echo "validation steps must be a positive integer" >&2; exit 2;
}

# Logical nonzero actors never initialize, resume, or upload a W&B sidecar.
if ((ACTOR_RANK != 0)); then
    exec /usr/bin/env -u WANDB_API_KEY bash "$VALIDATION"
fi

[[ ${DRY_RUN:-0} == 0 ]] || {
    echo "bounded real W&B validation rejects DRY_RUN" >&2; exit 2;
}
((STEPS >= 2)) || {
    echo "bounded W&B validation needs at least two steps for real MP4 evidence" >&2
    exit 2
}

WANDB_PYTHON=${ROUND_DAGGER_WANDB_PYTHON:?missing ROUND_DAGGER_WANDB_PYTHON}
WANDB_PROJECT=${ROUND_DAGGER_WANDB_PROJECT:?missing ROUND_DAGGER_WANDB_PROJECT}
WANDB_ENTITY=${ROUND_DAGGER_WANDB_ENTITY:?missing ROUND_DAGGER_WANDB_ENTITY}
WANDB_EMAIL=${ROUND_DAGGER_WANDB_EXPECTED_EMAIL:?missing ROUND_DAGGER_WANDB_EXPECTED_EMAIL}
READY_TIMEOUT=${ROUND_DAGGER_WANDB_READY_TIMEOUT_SECONDS:-120}
READBACK_TIMEOUT=${ROUND_DAGGER_WANDB_READBACK_TIMEOUT_SECONDS:-180}
POLL_SECONDS=${ROUND_DAGGER_WANDB_POLL_SECONDS:-5}
ERROR_SCANS=${ROUND_DAGGER_WANDB_MAX_ERROR_SCANS:-4}

case "$WANDB_PYTHON" in
    /*) ;;
    *) echo "W&B sidecar Python must be an absolute path" >&2; exit 2 ;;
esac
# Virtualenv and Conda entry points are normally symbolic links.  Preserve the
# entry-point spelling (Python uses it to discover the environment), but bind
# it to a canonical regular executable before launching anything online.
WANDB_PYTHON_TARGET=$(realpath -e -- "$WANDB_PYTHON") || {
    echo "W&B sidecar Python is missing or unsafe" >&2; exit 2;
}
[[ -f $WANDB_PYTHON_TARGET && ! -L $WANDB_PYTHON_TARGET && \
   -x $WANDB_PYTHON_TARGET ]] || {
    echo "W&B sidecar Python is missing, unsafe, or not executable" >&2; exit 2;
}
for value in "$READY_TIMEOUT" "$READBACK_TIMEOUT" "$POLL_SECONDS" "$ERROR_SCANS"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || {
        echo "W&B validation timing values must be positive integers" >&2; exit 2;
    }
done

# The actor image entrypoint sources Isaac Sim's Python 3.11 environment.  The
# W&B sidecar deliberately uses an independent Python 3.10 virtualenv, so it
# must not inherit Isaac's standard-library/site-package search path (doing so
# fails before W&B starts with NamespaceLoader/SRE ABI mismatches).  Keep the
# actor environment unchanged and sanitize only the two external-Python calls.
# An exact repository-only PYTHONPATH retains imports of round_dagger/scripts;
# the selected virtualenv still discovers its own site-packages normally.
WANDB_RUNTIME_ENV=(
    /usr/bin/env
    -u PYTHONHOME
    -u PYTHONUSERBASE
    -u VIRTUAL_ENV
    -u CONDA_PREFIX
    -u CONDA_DEFAULT_ENV
    PYTHONNOUSERSITE=1
    PYTHONPATH="$ROOT"
)

umask 077
mkdir -p "$RUN_DIR"
for directory in "$RUN_DIR/telemetry" "$RUN_DIR/logs" "$RUN_DIR/validation"; do
    [[ ! -L $directory ]] || {
        echo "unsafe bounded W&B run subdirectory" >&2; exit 2;
    }
    mkdir -p "$directory"
    [[ $(realpath -m -- "$directory") == "$directory" ]] || {
        echo "bounded W&B run subdirectory escaped the run directory" >&2; exit 2;
    }
done
READY="$RUN_DIR/telemetry/WANDB_WATCHER_READY.json"
DONE="$RUN_DIR/WANDB_VALIDATION_DONE"
LOG="$RUN_DIR/logs/wandb_validation_sidecar.log"
READBACK="$RUN_DIR/validation/wandb_actor_validation_readback.json"
BASELINE="$RUN_DIR/validation/wandb_actor_validation_local_baseline.json"
for marker in "$READY" "$DONE"; do
    if [[ -e $marker || -L $marker ]]; then
        [[ -f $marker && ! -L $marker ]] || {
            echo "unsafe bounded W&B lifecycle marker" >&2; exit 2;
        }
        rm -f -- "$marker"
    fi
done
if [[ -e $READBACK || -L $READBACK ]]; then
    [[ -f $READBACK && ! -L $READBACK ]] || {
        echo "unsafe bounded W&B readback path" >&2; exit 2;
    }
    rm -f -- "$READBACK"
fi

python3 - "$RUN_DIR" "$BASELINE" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2])
inventory = {}
for kind, suffixes in {
    "scalar": (".json",),
    "image": (".png", ".jpg", ".jpeg", ".webp"),
    "video": (".mp4", ".webm", ".mov"),
}.items():
    directory = root / "telemetry" / f"{kind}s"
    records = {}
    if directory.is_dir() and not directory.is_symlink():
        for path in directory.rglob("*"):
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in suffixes:
                stat = path.stat()
                records[str(path.relative_to(root))] = [
                    stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
                ]
    inventory[kind] = records
destination.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(inventory, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY

WATCHER=(
    "$WANDB_PYTHON" "$ROOT/scripts/scaling/wandb_run_watcher.py"
    --run-dir "$RUN_DIR"
    --project "$WANDB_PROJECT"
    --entity "$WANDB_ENTITY"
    --expected-email "$WANDB_EMAIL"
    --authorization-scope team
    --telemetry-scope bounded-validation
    --poll-interval-s "$POLL_SECONDS"
    --max-consecutive-error-scans "$ERROR_SCANS"
    --ready-marker "$READY"
    --finish-marker WANDB_VALIDATION_DONE
)
[[ -z ${ROUND_DAGGER_WANDB_NAME:-} ]] || WATCHER+=(--name "$ROUND_DAGGER_WANDB_NAME")
[[ -z ${ROUND_DAGGER_WANDB_GROUP:-} ]] || WATCHER+=(--group "$ROUND_DAGGER_WANDB_GROUP")

# Ensure a shared scheduler rank cannot make this logical rank-zero sidecar a
# no-op.  Actor identity remains explicit and independent of Torch collectives.
env -u RANK -u SLURM_PROCID -u LOCAL_RANK \
    "${WANDB_RUNTIME_ENV[@]}" \
    "${WATCHER[@]}" >"$LOG" 2>&1 &
WATCHER_PID=$!

finish_watcher() {
    local status=0
    if [[ -n ${WATCHER_PID:-} ]]; then
        if kill -0 "$WATCHER_PID" 2>/dev/null; then
            : >"$DONE"
        fi
        wait "$WATCHER_PID" || status=$?
    fi
    WATCHER_PID=
    return "$status"
}
trap 'finish_watcher || true' EXIT
trap 'finish_watcher || true; exit 130' INT
trap 'finish_watcher || true; exit 143' TERM

ready=0
for ((second=0; second<READY_TIMEOUT; second++)); do
    if [[ -s $READY && ! -L $READY ]]; then
        ready=1
        break
    fi
    if ! kill -0 "$WATCHER_PID" 2>/dev/null; then
        wait "$WATCHER_PID" || true
        WATCHER_PID=
        echo "bounded W&B sidecar exited before readiness" >&2
        exit 2
    fi
    sleep 1
done
[[ $ready == 1 ]] || {
    finish_watcher || true
    echo "bounded W&B sidecar readiness timed out" >&2
    exit 2
}

# The gate owns exact rank-zero telemetry cadence.  A successful bounded run
# must create current scalar, PNG, and H.264 evidence; learner telemetry remains
# a separate scalar-only stream.
export IMAGE_DAGGER_TELEMETRY_REQUIRED=1
export IMAGE_DAGGER_TELEMETRY_REQUIRE_VIDEO=1
export IMAGE_DAGGER_TELEMETRY_SCALAR_EVERY=1
# Step zero always emits one current PNG; avoid hundreds of redundant contact
# sheets when the bounded rung intentionally crosses the 256-frame lag.
export IMAGE_DAGGER_TELEMETRY_IMAGE_EVERY=$STEPS
export IMAGE_DAGGER_TELEMETRY_CLIP_EVERY=$STEPS
if ((STEPS < 32)); then
    export IMAGE_DAGGER_TELEMETRY_CLIP_FRAMES=$STEPS
else
    export IMAGE_DAGGER_TELEMETRY_CLIP_FRAMES=32
fi

set +e
/usr/bin/env -u WANDB_API_KEY bash "$VALIDATION"
ACTOR_STATUS=$?
set -e
if [[ $ACTOR_STATUS -ne 0 ]]; then
    finish_watcher || true
    echo "bounded actor validation failed" >&2
    exit "$ACTOR_STATUS"
fi

# Reject stale files from an earlier invocation before the final upload drain.
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$RUN_DIR" "$BASELINE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
baseline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
for kind, suffixes in {
    "scalar": (".json",),
    "image": (".png", ".jpg", ".jpeg", ".webp"),
    "video": (".mp4", ".webm", ".mov"),
}.items():
    directory = root / "telemetry" / f"{kind}s"
    current = []
    if directory.is_dir() and not directory.is_symlink():
        for path in directory.rglob("*"):
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in suffixes:
                stat = path.stat()
                fingerprint = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
                relative = str(path.relative_to(root))
                if baseline.get(kind, {}).get(relative) != fingerprint:
                    current.append(path)
    if not current:
        raise SystemExit(f"current bounded validation produced no {kind} telemetry")
PY

set +e
finish_watcher
WATCHER_STATUS=$?
set -e
if [[ $WATCHER_STATUS -ne 0 ]]; then
    echo "bounded W&B scalar/image/video sidecar failed" >&2
    exit "$WATCHER_STATUS"
fi

"${WANDB_RUNTIME_ENV[@]}" \
    "$WANDB_PYTHON" "$ROOT/scripts/round_dagger/readback_actor_validation_wandb.py" \
    --run-dir "$RUN_DIR" \
    --project "$WANDB_PROJECT" \
    --entity "$WANDB_ENTITY" \
    --expected-email "$WANDB_EMAIL" \
    --authorization-scope team \
    --timeout-s "$READBACK_TIMEOUT" \
    --poll-interval-s "$POLL_SECONDS" \
    --output "$READBACK"

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$READBACK" <<'PY'
import sys
from pathlib import Path
from round_dagger.artifacts import load_strict_json

report = load_strict_json(Path(sys.argv[1]))
expected = {
    "schema": "fast_roborender.round_dagger.wandb_actor_validation_readback.v1",
    "status": "PASS",
    "telemetry_scope": "bounded-validation",
    "trainer_run_required": False,
}
for key, value in expected.items():
    if report.get(key) != value:
        raise SystemExit(f"bounded W&B terminal report mismatch for {key}")
PY

trap - EXIT INT TERM
printf 'ROUND_ACTOR_VALIDATION_WANDB_OK readback=%q\n' "$READBACK"
