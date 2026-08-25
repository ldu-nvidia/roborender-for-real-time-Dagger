#!/usr/bin/env bash
# Export a fresh ImageNet-initialized GR00T VisionActorObjPred checkpoint.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
GR00T_PYTHON=${GR00T_PYTHON:?set GR00T_PYTHON to the validated Isaac/GR00T Python}
ROUND_DAGGER_BOOTSTRAP_OUTPUT=${ROUND_DAGGER_BOOTSTRAP_OUTPUT:?set an absolute bootstrap .pt output}
ROUND_DAGGER_TEACHER_CHECKPOINT=${ROUND_DAGGER_TEACHER_CHECKPOINT:?set the independent expert checkpoint}
ROUND_DAGGER_TEACHER_SHA256=${ROUND_DAGGER_TEACHER_SHA256:?set the expert checkpoint SHA-256}
RUN_DIR=${ROUND_DAGGER_BOOTSTRAP_RUN_DIR:?set an absolute run directory}
DRY_RUN=${DRY_RUN:-0}

case "$ROUND_DAGGER_BOOTSTRAP_OUTPUT:$ROUND_DAGGER_TEACHER_CHECKPOINT:$RUN_DIR" in
    /*:/*:/*) ;;
    *) echo "bootstrap, teacher, and run paths must be absolute" >&2; exit 2 ;;
esac
[[ -x $GR00T_PYTHON ]] || { echo "GR00T_PYTHON is not executable" >&2; exit 2; }
[[ -f $ROUND_DAGGER_TEACHER_CHECKPOINT && ! -L $ROUND_DAGGER_TEACHER_CHECKPOINT ]] || {
    echo "teacher checkpoint is missing or unsafe" >&2; exit 2;
}
[[ $(sha256sum "$ROUND_DAGGER_TEACHER_CHECKPOINT" | awk '{print $1}') == "$ROUND_DAGGER_TEACHER_SHA256" ]] || {
    echo "teacher checkpoint SHA-256 mismatch" >&2; exit 2;
}
for suffix in '' '.policy_spec.json' '.manifest.json'; do
    [[ ! -e ${ROUND_DAGGER_BOOTSTRAP_OUTPUT}${suffix} && ! -L ${ROUND_DAGGER_BOOTSTRAP_OUTPUT}${suffix} ]] || {
        echo "refusing to replace bootstrap artifact: ${ROUND_DAGGER_BOOTSTRAP_OUTPUT}${suffix}" >&2
        exit 2
    }
done

COMMAND=(
    "$GR00T_PYTHON" "$GR00T_ROOT/gr00t/rl/train_agent_trl.py"
    +exp=loco_manip/wsdpt_round_actor_3view
    num_envs=1
    algo.config.num_steps_per_env=1
    algo.config.num_learning_epochs=0
    algo.config.rollout_with_teacher_num_steps=0
    algo.config.enforce_teacher_rollout=false
    algo.config.ratio_teacher_rollout=0.0
    algo.trl.num_total_batches=1
    algo.trl.total_episodes=null
    checkpoint=null
    resume=false
    teacher_actor_path="$ROUND_DAGGER_TEACHER_CHECKPOINT"
    simulator.config.render_results=false
    headless=true
    use_wandb=false
    experiment_name=fresh-round-dagger-bootstrap
    experiment_dir="$RUN_DIR"
    hydra.run.dir="$RUN_DIR/hydra"
    output_dir="$RUN_DIR/output"
)

if [[ $DRY_RUN == 1 ]]; then
    printf 'DRY_RUN'
    printf ' %q' env \
        IMAGE_DAGGER_SIMULATION_APP_SKIP_CLEANUP=1 \
        ROUND_DAGGER_INITIALIZE_ONLY=1 \
        ROUND_DAGGER_BOOTSTRAP_OUTPUT="$ROUND_DAGGER_BOOTSTRAP_OUTPUT" \
        PYTHONPATH="$ROOT:$GR00T_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$RUN_DIR"
cd "$GR00T_ROOT"
env \
    IMAGE_DAGGER_SIMULATION_APP_SKIP_CLEANUP=1 \
    ROUND_DAGGER_INITIALIZE_ONLY=1 \
    ROUND_DAGGER_BOOTSTRAP_OUTPUT="$ROUND_DAGGER_BOOTSTRAP_OUTPUT" \
    PYTHONPATH="$ROOT:$GR00T_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "${COMMAND[@]}"

env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$GR00T_PYTHON" - \
    "$ROUND_DAGGER_BOOTSTRAP_OUTPUT" <<'PY'
import sys
from pathlib import Path
from round_dagger.artifacts import load_strict_json, sha256_file
from round_dagger.gr00t_policy import load_policy_spec

checkpoint = Path(sys.argv[1])
if checkpoint.is_symlink() or not checkpoint.is_file():
    raise SystemExit("bootstrap checkpoint was not published")
spec_path = checkpoint.with_suffix(checkpoint.suffix + ".policy_spec.json")
manifest_path = checkpoint.with_suffix(checkpoint.suffix + ".manifest.json")
spec = load_policy_spec(spec_path)
manifest = load_strict_json(manifest_path)
if manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
    raise SystemExit("bootstrap manifest/checkpoint mismatch")
if manifest.get("policy_spec_sha256") != sha256_file(spec_path):
    raise SystemExit("bootstrap manifest/policy-spec mismatch")
if spec.get("checkpoint_sha256") != manifest["checkpoint_sha256"]:
    raise SystemExit("policy spec/checkpoint mismatch")
print(f"FRESH_BOOTSTRAP_OK checkpoint={checkpoint} sha256={manifest['checkpoint_sha256']}")
PY
