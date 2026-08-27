#!/bin/bash
# Pre-flight check for the videogen pipeline. Read-only: verifies that everything a run
# needs is present and reports PASS/WARN/FAIL per item. Run it on a login node before
# submitting, or inside a job to debug a failure.
#
#   bash scripts/setup/check_env.sh
#
# WARN means "fine unless you need that stage" (e.g. no dataset if you only run DAgger).
set -uo pipefail

# Paths default to the OCI/Lustre layout; override any of them for another machine, e.g.
#   ROOT=$(git rev-parse --show-toplevel) \
#   IMAGE3F=/opt/roborender_image_model_accelerated \
#   VIRAL_PY=$CONDA_PREFIX/bin/python  bash scripts/setup/check_env.sh
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
ACCEL=${ACCEL:-$STAGE_ROOT/assets/roborender-acceleration}
IMAGE3F=${IMAGE3F:-$STAGE_ROOT/assets/roborender_image_model_accelerated}
IMAGE3F_PY=${IMAGE3F_PY:-$STAGE_ROOT/runtime/envs/image3f-cu128/bin/python}
MODEL_BASE=${MODEL_BASE:-$STAGE_ROOT/assets/models}
ANCHORS=${ANCHORS:-$STAGE_ROOT/assets/anchor_pool_2048}
VIRAL_PY=${VIRAL_PY:-$STAGE_ROOT/runtime/envs/viral/bin/python}
WANDB_PY=${WANDB_PY:-$STAGE_ROOT/runtime/envs/wandb-host/bin/python}
TEACHER_CKPT=${TEACHER_CKPT:-$STAGE_ROOT/assets/policies/expert_labeler/model_step_044500.pt}
EXPECTED_TEACHER_SHA256=${EXPECTED_TEACHER_SHA256:-943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38}
FAIL=0

pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
sec()  { printf '\n== %s ==\n' "$1"; }

sec "host telemetry environment"
if [ -x "$WANDB_PY" ] && "$WANDB_PY" -c 'import wandb' 2>/dev/null; then
    pass "host W&B Python at $WANDB_PY"
else
    fail "no host W&B env — run scripts/scaling/oci_host_wandb_setup.sbatch"
fi

sec "container trainer environment"
TRAINER_MANIFEST=$STAGE_ROOT/runtime/envs/trainer_environment_manifest.txt
if [ -s "$TRAINER_MANIFEST" ]; then
    CURRENT_GR00T_REV=$(git -C "$GR00T_ROOT" rev-parse HEAD 2>/dev/null || true)
    MANIFEST_GR00T_REV=$(awk -F= '$1 == "gr00t_revision" {print $2}' "$TRAINER_MANIFEST")
    if [ -n "$CURRENT_GR00T_REV" ] && [ "$MANIFEST_GR00T_REV" = "$CURRENT_GR00T_REV" ]; then
        pass "pinned Isaac/GR00T environment manifest matches $CURRENT_GR00T_REV"
    else
        fail "trainer manifest GR00T revision ${MANIFEST_GR00T_REV:-missing} does not match source ${CURRENT_GR00T_REV:-unknown}"
    fi
    HYDRA_CONTRACT_REPORT=$(awk -F= '$1 == "production_hydra_contract" {print substr($0, index($0, "=") + 1)}' "$TRAINER_MANIFEST")
    HYDRA_CONTRACT_SHA256=$(awk -F= '$1 == "production_hydra_contract_sha256" {print $2}' "$TRAINER_MANIFEST")
    if [ -n "$CURRENT_GR00T_REV" ] && [ -s "$HYDRA_CONTRACT_REPORT" ] && \
            [ "$(sha256sum "$HYDRA_CONTRACT_REPORT" | awk '{print $1}')" = "$HYDRA_CONTRACT_SHA256" ] && \
            python3 - "$HYDRA_CONTRACT_REPORT" "$CURRENT_GR00T_REV" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["schema"] == "fast_roborender.production_hydra_contract.v1"
assert payload["status"] == "validated"
assert payload["gr00t_revision"] == sys.argv[2]
assert payload["policy_views"] == ["head_color", "hand_left", "hand_right"]
assert payload["renderer_views"] == ["head_color", "hand_left", "hand_right"]
assert payload["policy_image_shape"] == [324, 192, 3]
assert payload["checks"] and all(payload["checks"].values())
PY
    then
        pass "exact production Hydra composition report"
    else
        fail "production Hydra composition report is missing, stale, or invalid"
    fi
else
    fail "no trainer manifest — run scripts/scaling/oci_h100_trainer_setup.sbatch"
fi
if command -v python3 >/dev/null 2>&1 && [ -d "$GR00T_ROOT" ]; then
    CURRENT_GR00T_REV=${CURRENT_GR00T_REV:-$(git -C "$GR00T_ROOT" rev-parse HEAD 2>/dev/null || true)}
    if [ -n "$CURRENT_GR00T_REV" ] && python3 "$ROOT/scripts/scaling/install_gr00t_integration.py" \
            --gr00t-root "$GR00T_ROOT" --expected-revision "$CURRENT_GR00T_REV" \
            --json >/dev/null 2>&1; then
        pass "exact three-view trainer/config/runtime-gate integration"
    else
        fail "three-view GR00T integration check failed"
    fi
else
    fail "cannot validate the three-view GR00T integration"
fi

sec "render bundles"
if [ -x "$ACCEL/.venv/bin/python" ]; then
    pass "LADD/teacher bundle venv"
else
    warn "no venv at $ACCEL/.venv — needed for offline rendering and the teacher anchor server"
fi
if [ -s "$STAGE_ROOT/runtime/envs/image3f-cu128/.oci_environment_ready" ]; then
    pass "Image3F OCI environment marker at $STAGE_ROOT/runtime/envs/image3f-cu128"
else
    fail "no Image3F OCI environment — run scripts/scaling/oci_h100_image3f_setup.sbatch"
fi
ENGINE_MANIFEST=$IMAGE3F/assets/generated/engines/current/engine_build_manifest.json
I3F_CURRENT_ENGINES=("$IMAGE3F"/assets/generated/engines/current/*.engine)
if [ -s "$ENGINE_MANIFEST" ] && [ -e "${I3F_CURRENT_ENGINES[0]}" ]; then
    pass "target-local Image3F TensorRT engine and build manifest"
else
    fail "no validated target-local Image3F TensorRT engine — run scripts/scaling/oci_h100_image3f_setup.sbatch"
fi
if [ -f "$IMAGE3F/adapter/rgb030-depth030-step-5000.safetensors" ]; then
    pass "Image3F adapter (rgb030-depth030-step-5000)"
else
    fail "Image3F adapter missing under $IMAGE3F/adapter/"
fi
if [ -d "$MODEL_BASE/PAI/Wan2.1-Fun-V1.1-1.3B-Control" ]; then
    pass "foundation weights (ROBORENDER_MODEL_BASE=$MODEL_BASE)"
else
    fail "foundation weights missing at $MODEL_BASE"
fi

sec "TensorRT engines (per GPU arch, per TRT version — built locally, never shipped)"
shopt -s nullglob
I3F_ENG=("$IMAGE3F"/assets/generated/engines/*/*.engine)
if [ ${#I3F_ENG[@]} -gt 0 ]; then
    for e in "${I3F_ENG[@]}"; do pass "image3f: $(basename "$(dirname "$e")")/$(basename "$e")"; done
else
    warn "no Image3F engine — it will run the slower PyTorch DiT path (~+25 ms/frame)"
fi
LADD_ENG=("$ACCEL"/assets/engines/generated/*/)
if [ ${#LADD_ENG[@]} -gt 0 ]; then
    pass "LADD student engines: $(for d in "${LADD_ENG[@]}"; do basename "$d"; done | tr '\n' ' ')"
else
    warn "no LADD engines (only needed for the accelerated 2-view student)"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    [ -n "$GPU" ] && pass "GPU visible: $GPU — engines must match this arch"
else
    warn "no GPU on this node (expected on a login node)"
fi

sec "data and anchors"
if [ -d "$ANCHORS" ]; then
    N=$(ls "$ANCHORS"/anchor_*.npz 2>/dev/null | wc -l)
    [ "$N" -gt 0 ] && pass "anchor pool: $N anchors at $ANCHORS" \
                   || fail "anchor pool dir exists but is empty — run scripts/anchors/anchor_pool.sbatch"
else
    fail "no anchor pool at $ANCHORS — DAgger needs it (scripts/anchors/anchor_pool.sbatch)"
fi
if [ -d "$ROOT/videogen_dataset" ]; then
    N=$(ls "$ROOT"/videogen_dataset/chunk_*.npz 2>/dev/null | wc -l)
    pass "videogen dataset: $N chunks"
else
    warn "no videogen_dataset/ — only needed for offline BC (scripts/data/collect_data.sbatch)"
fi

sec "baseline checkpoints"
if [ -s "$TEACHER_CKPT" ] && [ "$(sha256sum "$TEACHER_CKPT" | awk '{print $1}')" = "$EXPECTED_TEACHER_SHA256" ]; then
    pass "expert labeler checkpoint and SHA-256"
else
    fail "expert labeler checkpoint missing or hash mismatch: $TEACHER_CKPT"
fi
BC=${BASELINE_RUN:-$ROOT/GR00T-VisualSim2Real/logs_rl/videogen_offline_bc/videogen_offline_bc-20260721_204447}
[ -f "$BC/config.yaml" ] && pass "baseline eval config (eval_policy.sbatch stages against it)" \
                         || warn "historical baseline config missing: $BC/config.yaml"
[ -f "$BC/model_step_011000.pt" ] && pass "offline-BC baseline checkpoint (57.3%)" \
                                  || warn "offline-BC baseline checkpoint missing"

sec "renderers currently running"
LIVE=$(command -v squeue >/dev/null 2>&1 && squeue -u "$USER" -h -n i3f_warm_server -o '%N %M' 2>/dev/null || true)
if [ -n "$LIVE" ]; then
    while read -r n t; do pass "i3f_warm_server on $n (up $t)"; done <<<"$LIVE"
    for EP in "$ROOT"/image3f_server_endpoint*.txt; do
        [ -f "$EP" ] || continue
        H=$(awk '{print $1}' "$EP" | cut -d. -f1)
        grep -q "$H" <<<"$LIVE" && pass "endpoint $(basename "$EP") -> $(cat "$EP")" \
                                || warn "STALE endpoint $(basename "$EP") -> $(cat "$EP") (no server there)"
    done
else
    warn "no i3f_warm_server running (start before DAgger or OBS=generated evals)"
fi

sec "slurm"
if command -v sacctmgr >/dev/null 2>&1; then
    sacctmgr -nP show assoc where user="$USER" format=Account 2>/dev/null | sed 's/|$//' | sort -u \
      | grep -q '^general_cs_infra$' && pass "general_cs_infra account available" \
      || warn "general_cs_infra account not found"
else
    warn "no Slurm on this machine - launch training directly (see README, Running on another machine)"
fi
df -h "$STAGE_ROOT" 2>/dev/null | tail -1 \
  | awk -v root="$STAGE_ROOT" '{print "  INFO  " root " " $4 " free (" $5 " used)"}'

printf '\n'
[ "$FAIL" -eq 0 ] && printf '\033[32mREADY\033[0m — %d hard failures\n' "$FAIL" \
                  || printf '\033[31mNOT READY\033[0m — %d hard failure(s) above\n' "$FAIL"
exit $((FAIL > 0))
