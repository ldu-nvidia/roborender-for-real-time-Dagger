#!/usr/bin/env bash
# Container entrypoint for the dedicated fixed-256/step-100 validation.
set -Eeuo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
PYTHON=${ROUND_DAGGER_H100_PYTHON:-/opt/viral-h100/bin/python}
MODE=${1:?usage: $0 preflight|learner-sequence}
CAMPAIGN=${ROUND_DAGGER_HELDOUT100_CAMPAIGN_ROOT:?missing campaign mount}
OUTPUT=${ROUND_DAGGER_HELDOUT100_OUTPUT_ROOT:?missing learner-output mount}
JOB=${ROUND_DAGGER_HELDOUT100_JOB_ROOT:?missing job scratch mount}
SOURCE_SHA=${ROUND_DAGGER_HELDOUT100_SOURCE_INVENTORY_SHA256:?missing source identity}
SNAPSHOT=${ROUND_DAGGER_HELDOUT100_REPLAY_SNAPSHOT:?missing training replay snapshot}
SNAPSHOT_FILE_SHA=${ROUND_DAGGER_HELDOUT100_REPLAY_SNAPSHOT_FILE_SHA256:?missing training replay file identity}
SNAPSHOT_SHA=${ROUND_DAGGER_HELDOUT100_REPLAY_SNAPSHOT_SHA256:?missing training replay semantic identity}
HELDOUT=${ROUND_DAGGER_HELDOUT100_HELDOUT_SNAPSHOT:?missing fixed-256 held-out snapshot}
CHECKPOINT=${ROUND_DAGGER_HELDOUT100_POLICY_CHECKPOINT:?missing parent checkpoint}
CHECKPOINT_SHA=${ROUND_DAGGER_HELDOUT100_POLICY_SHA256:?missing parent checkpoint identity}
SPEC=${ROUND_DAGGER_HELDOUT100_POLICY_SPEC:?missing policy spec}
SPEC_SHA=${ROUND_DAGGER_HELDOUT100_POLICY_SPEC_SHA256:?missing policy spec identity}
MANIFEST=${ROUND_DAGGER_HELDOUT100_POLICY_MANIFEST:?missing policy manifest}
MANIFEST_SHA=${ROUND_DAGGER_HELDOUT100_POLICY_MANIFEST_SHA256:?missing policy manifest identity}
RUNTIME_SHA=${ROUND_DAGGER_HELDOUT100_RUNTIME_POLICY_SHA256:?missing runtime policy identity}
CONFIG_SHA=${ROUND_DAGGER_HELDOUT100_LEARNER_CONFIG_SHA256:?missing learner bundle identity}
CONTRACT=${ROUND_DAGGER_HELDOUT100_CONTRACT:?missing heldout100 contract}
CONTRACT_SHA=${ROUND_DAGGER_HELDOUT100_CONTRACT_SHA256:?missing heldout100 contract identity}
BINDING_SHA=${ROUND_DAGGER_HELDOUT100_BINDING_SHA256:?missing fresh-campaign binding identity}
PRODUCTION_CONTRACT_SHA=${ROUND_DAGGER_HELDOUT100_PRODUCTION_CONTRACT_SHA256:?missing campaign-owned production contract identity}
HELDOUT_FILE_SHA=${ROUND_DAGGER_HELDOUT100_HELDOUT_SNAPSHOT_FILE_SHA256:?missing held-out file identity}
HELDOUT_SHA=${ROUND_DAGGER_HELDOUT100_HELDOUT_SNAPSHOT_SHA256:?missing held-out semantic identity}
LEARNER_IDENTITY_SHA=${ROUND_DAGGER_HELDOUT100_LEARNER_IDENTITY_SHA256:?missing learner identity}
CAMPAIGN_ID=${ROUND_DAGGER_HELDOUT100_CAMPAIGN_ID:?missing campaign ID}
MACRO_ROUND=${ROUND_DAGGER_HELDOUT100_MACRO_ROUND:?missing macro round}
CONTAINER_PATH=${ROUND_DAGGER_HELDOUT100_CONTAINER_HOST_PATH:?missing mounted container path}
CONTAINER_SHA=${ROUND_DAGGER_HELDOUT100_CONTAINER_SHA256:?missing container identity}
RUN_ID=${ROUND_DAGGER_HELDOUT100_WANDB_RUN_ID:?missing W&B run ID}
PROJECT=${ROUND_DAGGER_WANDB_PROJECT:?missing W&B project}
ENTITY=${ROUND_DAGGER_WANDB_ENTITY:?missing W&B entity}
EXPECTED_EMAIL=${ROUND_DAGGER_WANDB_EXPECTED_EMAIL:?missing W&B account}
AUTH_SCOPE=${ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE:?missing W&B authorization scope}

[[ -x $PYTHON ]] || fatal "H100 Python is unavailable"
[[ $CAMPAIGN == /* && -d $CAMPAIGN && ! -L $CAMPAIGN ]] || fatal "campaign mount is unsafe"
[[ $OUTPUT == "$CAMPAIGN"/* && -d $OUTPUT && ! -L $OUTPUT ]] || fatal "output mount is not nested"
[[ $JOB == /round_dagger_heldout100_job && -d $JOB && ! -L $JOB ]] || fatal "job scratch mount is non-canonical"
[[ $HELDOUT == "$CAMPAIGN/heldout_snapshot.json" ]] || \
    fatal "fixed-256 held-out snapshot is not campaign-owned"
[[ $CONTRACT == "$ROOT/configs/round_dagger/validation_learner_1x8h100_heldout100.json" ]] || fatal "heldout100 contract path changed"
[[ -n $CAMPAIGN_ID && $MACRO_ROUND == 0 ]] || fatal "campaign/round identity is invalid"
[[ $PROJECT == roborender-round-dagger-validation && $ENTITY == nvidia && \
   $EXPECTED_EMAIL == ldu@nvidia.com && $AUTH_SCOPE == team ]] || fatal "W&B contract changed"
for name in SOURCE_SHA SNAPSHOT_FILE_SHA SNAPSHOT_SHA CHECKPOINT_SHA SPEC_SHA \
            MANIFEST_SHA RUNTIME_SHA CONFIG_SHA CONTRACT_SHA LEARNER_IDENTITY_SHA \
            CONTAINER_SHA BINDING_SHA PRODUCTION_CONTRACT_SHA HELDOUT_FILE_SHA \
            HELDOUT_SHA; do
    [[ ${!name} =~ ^[0-9a-f]{64}$ ]] || fatal "$name is not a lowercase SHA-256"
done
for path in "$SNAPSHOT" "$HELDOUT" "$CHECKPOINT" "$SPEC" "$MANIFEST" "$CONTRACT" "$CONTAINER_PATH"; do
    [[ -f $path && ! -L $path ]] || fatal "required input is unsafe: $path"
done
[[ $SPEC == "${CHECKPOINT}.policy_spec.json" && $MANIFEST == "${CHECKPOINT}.manifest.json" ]] || fatal "policy sidecars are not canonical"
[[ $(sha256sum "$CONTRACT" | awk '{print $1}') == "$CONTRACT_SHA" ]] || fatal "contract digest mismatch"
[[ $(sha256sum "$CONTAINER_PATH" | awk '{print $1}') == "$CONTAINER_SHA" ]] || fatal "container digest mismatch"
[[ $(sha256sum "$SNAPSHOT" | awk '{print $1}') == "$SNAPSHOT_FILE_SHA" ]] || fatal "training snapshot digest mismatch"
[[ $(sha256sum "$CHECKPOINT" | awk '{print $1}') == "$CHECKPOINT_SHA" ]] || fatal "checkpoint digest mismatch"
[[ $(sha256sum "$SPEC" | awk '{print $1}') == "$SPEC_SHA" ]] || fatal "policy spec digest mismatch"
[[ $(sha256sum "$MANIFEST" | awk '{print $1}') == "$MANIFEST_SHA" ]] || fatal "policy manifest digest mismatch"
[[ $(sha256sum "$HELDOUT" | awk '{print $1}') == "$HELDOUT_FILE_SHA" ]] || fatal "held-out snapshot digest mismatch"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PYTHON" "$ROOT/scripts/round_dagger/h100_heldout100_validation.py" contract \
    --path "$CONTRACT" --sha256 "$CONTRACT_SHA" --source-root "$ROOT" >/dev/null
"$PYTHON" "$ROOT/scripts/round_dagger/h100_heldout100_validation.py" snapshot \
    --path "$HELDOUT" --campaign-root "$CAMPAIGN" \
    --campaign-id "$CAMPAIGN_ID" --file-sha256 "$HELDOUT_FILE_SHA" \
    --semantic-sha256 "$HELDOUT_SHA" >/dev/null
BOUND=$(
    "$PYTHON" "$ROOT/scripts/round_dagger/h100_heldout100_validation.py" binding \
        --contract "$CONTRACT" --contract-sha256 "$CONTRACT_SHA" \
        --source-root "$ROOT" --campaign-root "$CAMPAIGN" \
        --campaign-id "$CAMPAIGN_ID" --heldout-snapshot "$HELDOUT" \
        --heldout-snapshot-file-sha256 "$HELDOUT_FILE_SHA" \
        --heldout-snapshot-sha256 "$HELDOUT_SHA" \
        --production-contract-sha256 "$PRODUCTION_CONTRACT_SHA" | \
        "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["binding_sha256"])'
)
[[ $BOUND == "$BINDING_SHA" ]] || fatal "fresh-campaign binding digest mismatch"
[[ $RUN_ID == "rrd-m000000-id${LEARNER_IDENTITY_SHA}-b${BINDING_SHA:0:32}" ]] || \
    fatal "W&B run ID is not binding-qualified"
"$PYTHON" -m round_dagger.policy_bundle \
    --checkpoint "$CHECKPOINT" --policy-spec "$SPEC" --manifest "$MANIFEST" \
    --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
    --expected-runtime-policy-sha256 "$RUNTIME_SHA" >/dev/null

mount_gate() {
    "$PYTHON" - "$CAMPAIGN" "$OUTPUT" "$(dirname "$HELDOUT")" <<'PY'
import os
import sys

campaign, output, heldout = map(os.path.realpath, sys.argv[1:])
mounts = {}
with open("/proc/self/mountinfo", encoding="utf-8") as handle:
    for line in handle:
        left, _right = line.rstrip("\n").split(" - ", 1)
        fields = left.split()
        mountpoint = fields[4].replace("\\040", " ").replace("\\011", "\t")
        mounts[os.path.realpath(mountpoint)] = set(fields[5].split(","))
if campaign not in mounts or "ro" not in mounts[campaign]:
    raise SystemExit("campaign mount is not exactly read-only")
if output not in mounts or "rw" not in mounts[output]:
    raise SystemExit("learner output mount is not exactly writable")
if heldout not in mounts or "ro" not in mounts[heldout]:
    raise SystemExit("fixed-256 held-out mount is not exactly read-only")
if os.path.commonpath((campaign, output)) != campaign or campaign == output:
    raise SystemExit("learner output is not nested below campaign")
PY
}

configure_wandb() {
    WANDB_RUNTIME="$JOB/wandb"
    mkdir -p "$WANDB_RUNTIME/runtime" "$WANDB_RUNTIME/cache" "$WANDB_RUNTIME/config" "$JOB/evidence"
    export HOME=/home/ldu WANDB_MODE=online WANDB_DIR="$WANDB_RUNTIME/runtime"
    export WANDB_CACHE_DIR="$WANDB_RUNTIME/cache" WANDB_CONFIG_DIR="$WANDB_RUNTIME/config" WANDB_CONSOLE=off
    [[ -f $HOME/.netrc && ! -L $HOME/.netrc ]] || fatal "mounted W&B netrc is unsafe"
    case $(stat -c %a "$HOME/.netrc") in 400|600) ;; *) fatal "mounted W&B netrc mode is unsafe" ;; esac
}

case "$MODE" in
preflight)
    mount_gate
    configure_wandb
    [[ ${CUDA_VISIBLE_DEVICES:-} != "" ]] || fatal "preflight lacks a GPU"
    "$PYTHON" "$ROOT/scripts/scaling/wandb_preflight.py" \
        --expected-email "$EXPECTED_EMAIL" --entity "$ENTITY" \
        --authorization-scope team --format json >"$JOB/evidence/wandb_preflight.json"
    "$PYTHON" - <<'PY' >"$JOB/evidence/cuda_import_gate.json"
import json
import torch
import gr00t
from round_dagger.gr00t_policy import create_vision_actor_obj_pred

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("one-GPU preflight must see exactly one CUDA device")
if list(torch.cuda.get_device_capability(0)) != [9, 0]:
    raise SystemExit("one-GPU preflight did not receive H100 SM90")
if not torch.distributed.is_nccl_available():
    raise SystemExit("NCCL is unavailable")
x = torch.arange(4096, device="cuda", dtype=torch.float32)
if x.square().sum().item() <= 0:
    raise SystemExit("CUDA kernel probe failed")
torch.cuda.synchronize()
print(json.dumps({
    "status": "PASS", "visible_cuda_devices": 1,
    "compute_capability": [9, 0], "nccl_available": True,
    "gr00t_import": getattr(gr00t, "__version__", "import-ok"),
    "factory": f"{create_vision_actor_obj_pred.__module__}:{create_vision_actor_obj_pred.__name__}",
}, sort_keys=True))
PY
    chmod 0444 "$JOB/evidence/wandb_preflight.json" "$JOB/evidence/cuda_import_gate.json"
    echo "OCI_NRT_H100_HELDOUT100_PREFLIGHT_PASS"
    ;;
learner-sequence)
    [[ ${ROUND_DAGGER_LEARNER_NPROC_PER_NODE:-} == 8 && \
       ${ROUND_DAGGER_LEARNER_BATCH_SIZE:-} == 1 && \
       ${ROUND_DAGGER_LEARNER_MAX_STEPS:-} == 100 && \
       ${ROUND_DAGGER_LEARNER_SAVE_EVERY:-} == 5 ]] || fatal "learner topology/config changed"
    [[ -f $JOB/evidence/wandb_preflight.json && -f $JOB/evidence/cuda_import_gate.json ]] || fatal "preflight evidence is missing"
    mount_gate
    configure_wandb
    [[ -z $(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit) ]] || fatal "learner output must start empty"

    "$PYTHON" - "$SNAPSHOT" "$SNAPSHOT_FILE_SHA" "$SNAPSHOT_SHA" "$CAMPAIGN_ID" \
        "$MACRO_ROUND" "$CHECKPOINT_SHA" "$RUNTIME_SHA" <<'PY'
import sys
from round_dagger.artifacts import load_strict_json
from round_dagger.learner import partition_shards
from round_dagger.replay import load_snapshot

path, file_sha, semantic_sha, campaign, macro_round, parent, runtime = sys.argv[1:]
snapshot = load_snapshot(path, expected_file_sha256=file_sha)
if snapshot["snapshot_sha256"] != semantic_sha or snapshot["campaign_id"] != campaign:
    raise SystemExit("training replay identity mismatch")
if snapshot["through_macro_round"] != int(macro_round) or snapshot["expected_train_cohorts"] != 4:
    raise SystemExit("training replay frontier mismatch")
shards = snapshot["resolved_shards"]
if len(shards) < 8:
    raise SystemExit("training replay leaves fewer shards than ranks")
for rank in range(8):
    if not partition_shards(shards, rank=rank, world_size=8, seed=23):
        raise SystemExit(f"training replay leaves rank {rank} empty")
current = [row for row in snapshot["cohorts"] if row["macro_round"] == int(macro_round)]
if len(current) != 4:
    raise SystemExit("training replay lacks exactly four current-round cohorts")
for row in current:
    manifest = load_strict_json(row["cohort_manifest_path"])
    expected = {"campaign_id": campaign, "macro_round": int(macro_round), "purpose": "train",
                "policy_sha256": parent, "runtime_policy_sha256": runtime, "sealed": True}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise SystemExit("current training cohort lineage mismatch")
PY

    BUNDLE="$OUTPUT/validation_learner_bundle.json"
    "$PYTHON" "$ROOT/scripts/round_dagger/prepare_gr00t_learner_bundle.py" \
        --base-policy-spec "$SPEC" --parent-checkpoint "$CHECKPOINT" \
        --expected-parent-checkpoint-sha256 "$CHECKPOINT_SHA" \
        --runtime-policy-sha256 "$RUNTIME_SHA" --campaign-id "$CAMPAIGN_ID" \
        --macro-round "$MACRO_ROUND" --code-sha256 "$SOURCE_SHA" \
        --batch-size 1 --max-steps 100 --save-every 5 --learning-rate 2.5e-5 \
        --weight-decay 0.01 --seed 17 --partition-seed 23 \
        --heldout-snapshot "$HELDOUT" --heldout-eval-interval 100 \
        --output "$BUNDLE" --expected-bundle-sha256 "$CONFIG_SHA" >/dev/null

    COMMON=(
        "$ROOT/scripts/round_dagger/train_learner.py" --bundle "$BUNDLE"
        --snapshot "$SNAPSHOT" --heldout-snapshot "$HELDOUT"
        --model-factory round_dagger.gr00t_policy:create_vision_actor_obj_pred
        --adapter-factory round_dagger.gr00t_policy:vision_actor_obj_pred_adapter_factory
        --initial-checkpoint "$CHECKPOINT" --expected-snapshot-file-sha256 "$SNAPSHOT_FILE_SHA"
        --learner-identity-sha256 "$LEARNER_IDENTITY_SHA"
        --runtime-container-sha256 "$CONTAINER_SHA" --learner-contract-sha256 "$CONTRACT_SHA"
        --max-steps 100 --save-every 5 --heldout-eval-interval 100 --device cuda
    )
    FULL="$OUTPUT/uninterrupted"
    RESUMED="$OUTPUT/resumed"
    mkdir "$FULL" "$RESUMED"
    "$PYTHON" -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --node-rank 0 \
        --master-addr 127.0.0.1 --master-port 29841 "${COMMON[@]}" \
        --output-dir "$FULL" --telemetry-jsonl "$FULL/telemetry.jsonl" \
        --heldout-telemetry-jsonl "$FULL/heldout.jsonl"
    "$PYTHON" -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --node-rank 0 \
        --master-addr 127.0.0.1 --master-port 29842 "${COMMON[@]}" \
        --output-dir "$RESUMED" --telemetry-jsonl "$RESUMED/telemetry.jsonl" \
        --heldout-telemetry-jsonl "$RESUMED/heldout.jsonl" \
        --wandb-project "$PROJECT" --wandb-entity "$ENTITY" --wandb-run-id "$RUN_ID" \
        --wandb-run-binding-sha256 "$BINDING_SHA" \
        --wandb-resume-never --no-publish-result --validation-stop-at-step 50
    cp --reflink=auto --preserve=mode "$RESUMED/restart.pt" "$OUTPUT/intentional_stop_step_50.pt"
    chmod 0444 "$OUTPUT/intentional_stop_step_50.pt"
    "$PYTHON" -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --node-rank 0 \
        --master-addr 127.0.0.1 --master-port 29843 "${COMMON[@]}" \
        --output-dir "$RESUMED" --telemetry-jsonl "$RESUMED/telemetry.jsonl" \
        --heldout-telemetry-jsonl "$RESUMED/heldout.jsonl" \
        --wandb-project "$PROJECT" --wandb-entity "$ENTITY" --wandb-run-id "$RUN_ID" \
        --wandb-run-binding-sha256 "$BINDING_SHA" \
        --resume-from "$RESUMED/restart.pt"

    RESUME_EVIDENCE="$JOB/evidence/strict_resume_invocation.json"
    "$PYTHON" - "$OUTPUT/intentional_stop_step_50.pt" "$RESUMED/restart.pt" \
        "$RUN_ID" "$LEARNER_IDENTITY_SHA" "$CONFIG_SHA" "$SOURCE_SHA" \
        "$SNAPSHOT_FILE_SHA" "$SNAPSHOT_SHA" "$CONTAINER_SHA" "$CONTRACT_SHA" \
        "$HELDOUT_FILE_SHA" "$HELDOUT_SHA" "$BINDING_SHA" \
        "$PRODUCTION_CONTRACT_SHA" "$RESUME_EVIDENCE" <<'PY'
import os
import stat
import sys
from round_dagger.artifacts import publish_immutable_json, sha256_file
from round_dagger.checkpoint_compat import load_restricted_restart_document
from round_dagger.learner import CHECKPOINT_SCHEMA

(stopped_path, resumed_path, run_id, identity, config, code, replay_file, replay,
 container, contract, heldout_file, heldout, binding, production, output) = sys.argv[1:]
stopped = load_restricted_restart_document(stopped_path, map_location="cpu")
resumed = load_restricted_restart_document(resumed_path, map_location="cpu")
for doc, step in ((stopped, 50), (resumed, 100)):
    if doc.get("schema") != CHECKPOINT_SCHEMA or doc.get("completed_steps") != step:
        raise SystemExit("strict-resume checkpoint frontier mismatch")
    expected = {"learner_identity_sha256": identity, "learner_config_sha256": config,
                "code_sha256": code, "replay_snapshot_file_sha256": replay_file,
                "replay_snapshot_sha256": replay, "runtime_container_sha256": container,
                "learner_contract_sha256": contract}
    if any(doc.get(k) != v for k, v in expected.items()):
        raise SystemExit("strict-resume checkpoint identity mismatch")
report = {
    "schema": "fast_roborender.round_dagger.h100_heldout100_strict_resume.v1",
    "status": "PASS", "run_id": run_id, "stop_step": 50, "target_step": 100,
    "first_invocation_resume_mode": "never", "second_invocation_resume_mode": "must",
    "learner_identity_sha256": identity, "learner_config_sha256": config,
    "learner_code_sha256": code, "replay_snapshot_file_sha256": replay_file,
    "replay_snapshot_sha256": replay,
    "heldout_snapshot_file_sha256": heldout_file,
    "heldout_snapshot_sha256": heldout,
    "validation_binding_sha256": binding,
    "production_contract_sha256": production,
    "runtime_container_sha256": container, "learner_contract_sha256": contract,
    "stopped_checkpoint_sha256": sha256_file(stopped_path),
    "resumed_checkpoint_sha256": sha256_file(resumed_path),
    "canonical_wandb_lineage": True,
}
publish_immutable_json(output, report)
os.chmod(output, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
PY

    printf -v ROUND_PADDED '%06d' "$((10#$MACRO_ROUND))"
    FULL_RESULT="$FULL/learner_result_round_${ROUND_PADDED}.json"
    RESUMED_RESULT="$RESUMED/learner_result_round_${ROUND_PADDED}.json"
    RESTART_REPORT="$OUTPUT/restart_equivalence.json"
    "$PYTHON" "$ROOT/scripts/round_dagger/verify_learner_restart_equivalence.py" \
        --uninterrupted-checkpoint "$FULL/restart.pt" --resumed-checkpoint "$RESUMED/restart.pt" \
        --expected-completed-steps 100 --uninterrupted-result "$FULL_RESULT" \
        --resumed-result "$RESUMED_RESULT" --output "$RESTART_REPORT" >/dev/null

    READBACK="$OUTPUT/heldout100_wandb_readback.json"
    "$PYTHON" "$ROOT/scripts/round_dagger/h100_heldout100_validation.py" readback \
        --run-id "$RUN_ID" --uninterrupted-telemetry "$FULL/telemetry.jsonl" \
        --resumed-telemetry "$RESUMED/telemetry.jsonl" \
        --uninterrupted-heldout "$FULL/heldout.jsonl" --resumed-heldout "$RESUMED/heldout.jsonl" \
        --uninterrupted-result "$FULL_RESULT" --resumed-result "$RESUMED_RESULT" \
        --uninterrupted-checkpoint "$FULL/restart.pt" \
        --stopped-checkpoint "$OUTPUT/intentional_stop_step_50.pt" \
        --resumed-checkpoint "$RESUMED/restart.pt" --resume-evidence "$RESUME_EVIDENCE" \
        --campaign-id "$CAMPAIGN_ID" --macro-round "$MACRO_ROUND" \
        --learner-code-sha256 "$SOURCE_SHA" --learner-config-sha256 "$CONFIG_SHA" \
        --model-lineage-sha256 "$CHECKPOINT_SHA" \
        --replay-snapshot-file-sha256 "$SNAPSHOT_FILE_SHA" \
        --replay-snapshot-sha256 "$SNAPSHOT_SHA" \
        --parent-runtime-policy-sha256 "$RUNTIME_SHA" \
        --learner-identity-sha256 "$LEARNER_IDENTITY_SHA" \
        --runtime-container-sha256 "$CONTAINER_SHA" \
        --validation-contract-sha256 "$CONTRACT_SHA" --project "$PROJECT" \
        --validation-binding-sha256 "$BINDING_SHA" \
        --production-contract-sha256 "$PRODUCTION_CONTRACT_SHA" \
        --heldout-snapshot-file-sha256 "$HELDOUT_FILE_SHA" \
        --heldout-snapshot-sha256 "$HELDOUT_SHA" \
        --expected-email "$EXPECTED_EMAIL" --entity "$ENTITY" \
        --authorization-scope "$AUTH_SCOPE" --output "$READBACK" >/dev/null

    "$PYTHON" "$ROOT/scripts/round_dagger/h100_heldout100_validation.py" attest \
        --contract "$CONTRACT" --contract-sha256 "$CONTRACT_SHA" --source-root "$ROOT" \
        --campaign-root "$CAMPAIGN" \
        --heldout-snapshot "$HELDOUT" --campaign-id "$CAMPAIGN_ID" \
        --heldout-snapshot-file-sha256 "$HELDOUT_FILE_SHA" \
        --heldout-snapshot-sha256 "$HELDOUT_SHA" \
        --production-contract-sha256 "$PRODUCTION_CONTRACT_SHA" \
        --validation-binding-sha256 "$BINDING_SHA" \
        --readback "$READBACK" --restart-report "$RESTART_REPORT" \
        --output "$OUTPUT/success_attestation.json" >/dev/null
    echo "OCI_NRT_H100_HELDOUT100_LEARNER_SEQUENCE_PASS"
    ;;
*) fatal "unknown heldout100 mode: $MODE" ;;
esac
