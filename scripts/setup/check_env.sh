#!/bin/bash
# Pre-flight check for the videogen pipeline. Read-only: verifies that everything a run
# needs is present and reports PASS/WARN/FAIL per item. Run it on a login node before
# submitting, or inside a job to debug a failure.
#
#   bash scripts/setup/check_env.sh
#
# WARN means "fine unless you need that stage" (e.g. no dataset if you only run DAgger).
set -uo pipefail

ROOT=/svl/u/ravenh/fast_roborender
ACCEL=$ROOT/roborender-acceleration
IMAGE3F=/vision/u/ravenh/roborender_image_model_accelerated
MODEL_BASE=/vision/u/ravenh/roborender_image3f_portable_buildonly_v1/assets/models
ANCHORS=/vision/u/ravenh/anchor_pool_2048
VIRAL_PY=/svl/u/ravenh/miniconda3/envs/viral/bin/python
FAIL=0

pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
sec()  { printf '\n== %s ==\n' "$1"; }

sec "trainer environment (conda: viral)"
if [ -x "$VIRAL_PY" ]; then
    pass "python $($VIRAL_PY -V 2>&1 | cut -d' ' -f2) at $VIRAL_PY"
    for mod in torch isaaclab gr00t; do
        if v=$("$VIRAL_PY" -c "import $mod,sys; print(getattr($mod,'__version__',''))" 2>/dev/null); then
            pass "import $mod ${v:+($v)}"
        else
            fail "import $mod — rerun scripts/setup/setup_viral_env.sh"
        fi
    done
else
    fail "no trainer env at $VIRAL_PY — run scripts/setup/setup_viral_env.sh"
fi

sec "render bundles"
if [ -x "$ACCEL/.venv/bin/python" ]; then
    pass "LADD/teacher bundle venv"
else
    warn "no venv at $ACCEL/.venv — needed for offline rendering and the teacher anchor server"
fi
if [ -x "$IMAGE3F/.venv/bin/python" ]; then
    pass "Image3F bundle venv"
else
    fail "no venv at $IMAGE3F/.venv — needed for DAgger; see scripts/server/image3f_ft_setup.sbatch"
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
BC=$ROOT/GR00T-VisualSim2Real/logs_rl/videogen_offline_bc/videogen_offline_bc-20260721_204447
[ -f "$BC/config.yaml" ] && pass "baseline eval config (eval_policy.sbatch stages against it)" \
                         || fail "baseline config missing: $BC/config.yaml"
[ -f "$BC/model_step_011000.pt" ] && pass "offline-BC baseline checkpoint (57.3%)" \
                                  || warn "offline-BC baseline checkpoint missing"

sec "renderers currently running"
LIVE=$(squeue -u "$USER" -h -n i3f_warm_server -o '%N %M' 2>/dev/null)
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
sacctmgr -n show assoc user="$USER" format=account 2>/dev/null | tr -d ' ' | sort -u \
  | grep -q viscam && pass "viscam account available" || warn "viscam account not found"
df -h /svl/u/ravenh 2>/dev/null | tail -1 | awk '{print "  INFO  /svl/u/ravenh " $4 " free (" $5 " used)"}'

printf '\n'
[ "$FAIL" -eq 0 ] && printf '\033[32mREADY\033[0m — %d hard failures\n' "$FAIL" \
                  || printf '\033[31mNOT READY\033[0m — %d hard failure(s) above\n' "$FAIL"
exit $((FAIL > 0))
