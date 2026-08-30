#!/usr/bin/env bash
# Validation-only container entrypoint for the staged OCI-NRT H100 live gates.
set -Eeuo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
PYTHON=${ROUND_DAGGER_H100_PYTHON:-/opt/viral-h100/bin/python}
MODE=${1:?usage: $0 preflight|learner-sequence}
CAMPAIGN=${ROUND_DAGGER_VALIDATION_CAMPAIGN_ROOT:?missing campaign mount}
OUTPUT=${ROUND_DAGGER_VALIDATION_OUTPUT_ROOT:?missing learner-output mount}
JOB=${ROUND_DAGGER_VALIDATION_JOB_ROOT:?missing job scratch mount}
SOURCE_SHA=${ROUND_DAGGER_VALIDATION_SOURCE_INVENTORY_SHA256:?missing source identity}
SNAPSHOT=${ROUND_DAGGER_VALIDATION_REPLAY_SNAPSHOT:?missing replay snapshot}
SNAPSHOT_FILE_SHA=${ROUND_DAGGER_VALIDATION_REPLAY_SNAPSHOT_FILE_SHA256:?missing replay snapshot file identity}
SNAPSHOT_SHA=${ROUND_DAGGER_VALIDATION_REPLAY_SNAPSHOT_SHA256:?missing replay snapshot semantic identity}
CHECKPOINT=${ROUND_DAGGER_VALIDATION_POLICY_CHECKPOINT:?missing policy checkpoint}
CHECKPOINT_SHA=${ROUND_DAGGER_VALIDATION_POLICY_SHA256:?missing policy checkpoint identity}
SPEC=${ROUND_DAGGER_VALIDATION_POLICY_SPEC:?missing policy spec}
SPEC_SHA=${ROUND_DAGGER_VALIDATION_POLICY_SPEC_SHA256:?missing policy spec identity}
MANIFEST=${ROUND_DAGGER_VALIDATION_POLICY_MANIFEST:?missing policy manifest}
MANIFEST_SHA=${ROUND_DAGGER_VALIDATION_POLICY_MANIFEST_SHA256:?missing policy manifest identity}
RUNTIME_SHA=${ROUND_DAGGER_VALIDATION_RUNTIME_POLICY_SHA256:?missing runtime policy identity}
CONFIG_SHA=${ROUND_DAGGER_VALIDATION_LEARNER_CONFIG_SHA256:?missing learner config identity}
CONTRACT=${ROUND_DAGGER_VALIDATION_CONTRACT:?missing validation execution contract}
CONTRACT_SHA=${ROUND_DAGGER_VALIDATION_CONTRACT_SHA256:?missing validation contract identity}
LEARNER_IDENTITY_SHA=${ROUND_DAGGER_VALIDATION_LEARNER_IDENTITY_SHA256:?missing learner semantic identity}
CAMPAIGN_ID=${ROUND_DAGGER_VALIDATION_CAMPAIGN_ID:?missing validation campaign ID}
MACRO_ROUND=${ROUND_DAGGER_VALIDATION_MACRO_ROUND:?missing validation macro round}
PROJECT=${ROUND_DAGGER_WANDB_PROJECT:?missing W&B project}
ENTITY=${ROUND_DAGGER_WANDB_ENTITY:?missing W&B entity}
EXPECTED_EMAIL=${ROUND_DAGGER_WANDB_EXPECTED_EMAIL:?missing W&B expected email}
AUTH_SCOPE=${ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE:?missing W&B authorization scope}
WANDB_RUN_ID=${ROUND_DAGGER_VALIDATION_WANDB_RUN_ID:?missing canonical W&B run ID}
CONTAINER_SHA=${ROUND_DAGGER_VALIDATION_CONTAINER_SHA256:?missing runtime container identity}

[[ -x $PYTHON ]] || fatal "H100 Python is unavailable"
[[ $CAMPAIGN == /* && -d $CAMPAIGN && ! -L $CAMPAIGN ]] || \
    fatal "campaign mount is unsafe"
[[ $OUTPUT == "$CAMPAIGN"/* && -d $OUTPUT && ! -L $OUTPUT ]] || \
    fatal "learner-output mount must be nested below the campaign mount"
[[ $JOB == /round_dagger_validation_job && -d $JOB && ! -L $JOB ]] || \
    fatal "job scratch mount is non-canonical"
[[ $OUTPUT != "$JOB"* && $JOB != "$OUTPUT"* ]] || \
    fatal "job scratch must be outside returned learner output"
for name in SOURCE_SHA SNAPSHOT_FILE_SHA SNAPSHOT_SHA CHECKPOINT_SHA SPEC_SHA \
            MANIFEST_SHA RUNTIME_SHA CONFIG_SHA CONTRACT_SHA \
            LEARNER_IDENTITY_SHA CONTAINER_SHA; do
    [[ ${!name} =~ ^[0-9a-f]{64}$ ]] || fatal "$name is not a lowercase SHA-256"
done
[[ $MACRO_ROUND =~ ^[0-9]+$ ]] || fatal "macro round is invalid"
[[ $CAMPAIGN_ID =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || \
    fatal "validation campaign ID is malformed"
[[ $PROJECT =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ && \
   $WANDB_RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || \
    fatal "validation W&B identifiers are malformed"
[[ $ENTITY == nvidia && $EXPECTED_EMAIL == ldu@nvidia.com && \
   $AUTH_SCOPE == team ]] || fatal "validation W&B authorization changed"
for path in "$SNAPSHOT" "$CHECKPOINT" "$SPEC" "$MANIFEST"; do
    [[ $path == "$CAMPAIGN"/* && -f $path && ! -L $path ]] || \
        fatal "validation input is outside the read-only campaign: $path"
done
[[ $CONTRACT == "$ROOT/configs/round_dagger/validation_learner_1x8h100_2step.json" && \
   -f $CONTRACT && ! -L $CONTRACT ]] || fatal "validation contract is non-canonical"
[[ $(sha256sum "$CONTRACT" | awk '{print $1}') == "$CONTRACT_SHA" ]] || \
    fatal "validation contract digest mismatch"
[[ $SPEC == "${CHECKPOINT}.policy_spec.json" && \
   $MANIFEST == "${CHECKPOINT}.manifest.json" ]] || \
    fatal "policy sidecars are not canonically co-located"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PYTHON" - <<'PY'
import sys

import numpy as np

if np.__version__ != "1.26.0":
    raise SystemExit(f"expected NumPy 1.26.0, found {np.__version__}")
if not any(
    "/isaac-sim/extscache/omni.kit.pip_archive-" in entry
    for entry in sys.path
):
    raise SystemExit("Isaac Sim Python environment is absent from sys.path")
PY
[[ $(sha256sum "$SNAPSHOT" | awk '{print $1}') == "$SNAPSHOT_FILE_SHA" ]] || \
    fatal "replay snapshot file digest mismatch"
[[ $(sha256sum "$CHECKPOINT" | awk '{print $1}') == "$CHECKPOINT_SHA" ]] || \
    fatal "policy checkpoint digest mismatch"
[[ $(sha256sum "$SPEC" | awk '{print $1}') == "$SPEC_SHA" ]] || \
    fatal "policy spec digest mismatch"
[[ $(sha256sum "$MANIFEST" | awk '{print $1}') == "$MANIFEST_SHA" ]] || \
    fatal "policy manifest digest mismatch"
"$PYTHON" -m round_dagger.policy_bundle \
    --checkpoint "$CHECKPOINT" --policy-spec "$SPEC" --manifest "$MANIFEST" \
    --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
    --expected-runtime-policy-sha256 "$RUNTIME_SHA" >/dev/null
"$PYTHON" - "$CONTRACT" "$CAMPAIGN_ID" "$MACRO_ROUND" "$SOURCE_SHA" \
    "$CONFIG_SHA" "$SNAPSHOT_FILE_SHA" "$SNAPSHOT_SHA" "$CHECKPOINT_SHA" \
    "$RUNTIME_SHA" "$CONTAINER_SHA" "$CONTRACT_SHA" "$LEARNER_IDENTITY_SHA" \
    "$WANDB_RUN_ID" "$PROJECT" <<'PY'
import json
import sys
from round_dagger.learner_identity import learner_identity_sha256, learner_wandb_run_id

(path, campaign, macro_round, code, config, replay_file, replay, parent,
 parent_runtime, container, contract, expected_identity, expected_run_id,
 project) = sys.argv[1:]
value = json.load(open(path, encoding="utf-8"))
if value.get("schema") != "fast_roborender.round_dagger.validation_learner_contract.v1":
    raise SystemExit("validation contract schema mismatch")
if value.get("allocation") != {
    "nodes": 1, "gpus_per_node": 8, "gpu_architecture": "H100-SM90",
    "backend": "nccl", "batch_size_per_rank": 1, "global_batch_size": 8,
} or value.get("optimization") != {
    "optimizer_steps": 2, "save_every_steps": 1, "intentional_stop_step": 1,
    "learning_rate": 2.5e-5, "weight_decay": 0.01, "seed": 17,
    "partition_seed": 23,
}:
    raise SystemExit("validation execution contract mismatch")
if value.get("restart_validation") != {
    "uninterrupted_comparator": "local_only",
    "wandb_lineage": "single_canonical_run",
    "first_invocation_resume": "allow", "second_invocation_resume": "must",
}:
    raise SystemExit("validation restart lineage contract mismatch")
if value.get("wandb", {}).get("project") != project:
    raise SystemExit("validation W&B project contract mismatch")
if value.get("container", {}).get("sha256") != container:
    raise SystemExit("validation runtime container contract mismatch")
identity = learner_identity_sha256(
    campaign_id=campaign, macro_round=int(macro_round), learner_code_sha256=code,
    learner_config_sha256=config, replay_snapshot_file_sha256=replay_file,
    replay_snapshot_sha256=replay, parent_checkpoint_sha256=parent,
    parent_runtime_policy_sha256=parent_runtime,
    runtime_container_sha256=container, learner_contract_sha256=contract,
)
run_id = learner_wandb_run_id(
    campaign_id=campaign, macro_round=int(macro_round),
    learner_identity_sha256=identity,
)
if identity != expected_identity or run_id != expected_run_id:
    raise SystemExit("validation semantic learner identity mismatch")
PY
mount_gate() {
    "$PYTHON" - "$CAMPAIGN" "$OUTPUT" <<'PY'
import json
import os
import sys

campaign, output = map(os.path.realpath, sys.argv[1:])
mounts = {}
with open("/proc/self/mountinfo", encoding="utf-8") as handle:
    for line in handle:
        left, right = line.rstrip("\n").split(" - ", 1)
        fields = left.split()
        mountpoint = fields[4].replace("\\040", " ").replace("\\011", "\t")
        mounts[os.path.realpath(mountpoint)] = set(fields[5].split(","))
if campaign not in mounts or "ro" not in mounts[campaign]:
    raise SystemExit("campaign does not have an exact read-only mount")
if output not in mounts or "rw" not in mounts[output]:
    raise SystemExit("learner output does not have an exact writable nested mount")
if os.path.commonpath((campaign, output)) != campaign or campaign == output:
    raise SystemExit("learner output is not nested below campaign")
print(json.dumps({"campaign_mount": "ro", "learner_output_mount": "rw"}))
PY
}

configure_wandb_runtime() {
    WANDB_RUNTIME=/round_dagger_validation_job/wandb
    [[ $WANDB_RUNTIME != "$OUTPUT"* ]] || fatal "W&B runtime leaked into returned output"
    mkdir -p "$WANDB_RUNTIME/runtime" "$WANDB_RUNTIME/cache" \
        "$WANDB_RUNTIME/config" "$JOB/evidence"
    export HOME=/home/ldu
    export WANDB_MODE=online
    export WANDB_DIR="$WANDB_RUNTIME/runtime"
    export WANDB_CACHE_DIR="$WANDB_RUNTIME/cache"
    export WANDB_CONFIG_DIR="$WANDB_RUNTIME/config"
    export WANDB_CONSOLE=off
    [[ -f $HOME/.netrc && ! -L $HOME/.netrc ]] || fatal "mounted W&B netrc is unsafe"
    case $(stat -c %a "$HOME/.netrc") in 400|600) ;; \
        *) fatal "mounted W&B netrc mode must be 0400 or 0600" ;; esac
}

case "$MODE" in
preflight)
    [[ ${CUDA_VISIBLE_DEVICES:-} != "" ]] || fatal "preflight lacks a visible GPU"
    mount_gate >/dev/null
    configure_wandb_runtime

    WANDB_REPORT="$JOB/evidence/wandb_preflight.json"
    IMPORT_REPORT="$JOB/evidence/cuda_import_gate.json"
    [[ ! -e $WANDB_REPORT && ! -L $WANDB_REPORT ]] || fatal "W&B report already exists"
    [[ ! -e $IMPORT_REPORT && ! -L $IMPORT_REPORT ]] || fatal "import report already exists"
    "$PYTHON" "$ROOT/scripts/scaling/wandb_preflight.py" \
        --expected-email ldu@nvidia.com --entity nvidia \
        --authorization-scope team --format json >"$WANDB_REPORT"
    "$PYTHON" - "$IMPORT_REPORT" "$CAMPAIGN" "$OUTPUT" <<'PY'
import json
import os
import platform
import sys
from pathlib import Path

import gr00t
import numpy as np
import torch
from round_dagger.gr00t_policy import create_vision_actor_obj_pred

output, campaign, learner_output = map(Path, sys.argv[1:])
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("one-GPU preflight must see exactly one CUDA device")
capability = list(torch.cuda.get_device_capability(0))
if capability != [9, 0]:
    raise SystemExit(f"expected H100 SM90, found {capability}")
if not torch.distributed.is_nccl_available():
    raise SystemExit("PyTorch NCCL backend is unavailable")
sample = torch.arange(4096, device="cuda", dtype=torch.float32)
if sample.square().sum().item() <= 0:
    raise SystemExit("CUDA kernel validation failed")
torch.cuda.synchronize()
report = {
    "schema": "fast_roborender.round_dagger.h100_import_gate.v1",
    "status": "PASS",
    "architecture": platform.machine(),
    "visible_cuda_devices": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0),
    "compute_capability": capability,
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "numpy_version": np.__version__,
    "isaac_pip_archive_on_sys_path": any(
        "/isaac-sim/extscache/omni.kit.pip_archive-" in entry
        for entry in sys.path
    ),
    "nccl_available": True,
    "gr00t_import": getattr(gr00t, "__version__", "import-ok"),
    "gr00t_factory": f"{create_vision_actor_obj_pred.__module__}:{create_vision_actor_obj_pred.__name__}",
    "campaign_mount": "ro",
    "learner_output_mount": "rw",
}
temporary = output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
    chmod 0444 "$WANDB_REPORT" "$IMPORT_REPORT"
    # Exercise the nested writable output mount without leaving a non-artifact
    # in the returned tree.
    PROBE="$OUTPUT/.writable_mount_probe"
    [[ ! -e $PROBE && ! -L $PROBE ]] || fatal "writable-mount probe path already exists"
    printf '%s\n' validation >"$PROBE"
    [[ $(<"$PROBE") == validation ]] || fatal "writable learner-output probe failed"
    unlink "$PROBE"
    echo "OCI_NRT_H100_PREFLIGHT_PASS"
    ;;
learner-sequence)
    [[ ${ROUND_DAGGER_LEARNER_NPROC_PER_NODE:-} == 8 ]] || \
        fatal "validation learner must use exactly eight ranks"
    [[ ${ROUND_DAGGER_LEARNER_BATCH_SIZE:-} == 1 && \
       ${ROUND_DAGGER_LEARNER_MAX_STEPS:-} == 2 && \
       ${ROUND_DAGGER_LEARNER_SAVE_EVERY:-} == 1 ]] || \
        fatal "validation learner must use batch1, two updates, save1"
    [[ -f $JOB/evidence/cuda_import_gate.json && \
       -f $JOB/evidence/wandb_preflight.json ]] || \
        fatal "one-GPU preflight evidence is missing"
    mount_gate >/dev/null
    configure_wandb_runtime
    [[ -z $(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit) ]] || \
        fatal "validation learner output must start empty"

    REPLAY_SHARDS=$("$PYTHON" - "$SNAPSHOT" "$SNAPSHOT_FILE_SHA" \
        "$SNAPSHOT_SHA" "$CAMPAIGN_ID" "$MACRO_ROUND" "$CHECKPOINT_SHA" \
        "$RUNTIME_SHA" <<'PY'
import sys
from round_dagger.artifacts import load_strict_json
from round_dagger.learner import partition_shards
from round_dagger.replay import load_snapshot

(
    snapshot_path, file_sha, semantic_sha, campaign_id, macro_round,
    parent_sha, runtime_sha,
) = sys.argv[1:]
snapshot = load_snapshot(snapshot_path, expected_file_sha256=file_sha)
if snapshot["snapshot_sha256"] != semantic_sha:
    raise SystemExit("snapshot semantic digest mismatch")
if snapshot["campaign_id"] != campaign_id:
    raise SystemExit("validation campaign differs from replay campaign")
if snapshot["through_macro_round"] != int(macro_round):
    raise SystemExit("validation macro round differs from replay frontier")
if snapshot["expected_train_cohorts"] != 4:
    raise SystemExit("validation replay must preserve four train cohorts per round")
if snapshot["view_contract"] != {
    "dtype": "uint8", "layout": "vertical", "shape": [324, 192, 3],
    "views": ["head_color", "hand_left", "hand_right"],
}:
    raise SystemExit("validation replay three-view contract mismatch")
if snapshot["action_contract"] != {"dtype": "float32", "shape": [26]}:
    raise SystemExit("validation replay action contract mismatch")
shards = snapshot["resolved_shards"]
if len(shards) < 8:
    raise SystemExit("validation replay requires at least eight shards")
assignments = [
    partition_shards(shards, rank=rank, world_size=8, seed=23)
    for rank in range(8)
]
empty = [rank for rank, assigned in enumerate(assignments) if not assigned]
if empty:
    raise SystemExit(f"validation replay leaves NCCL ranks without data: {empty}")
transition_counts = [
    sum(shard["transition_count"] for shard in assigned)
    for assigned in assignments
]
if sum(transition_counts) != snapshot["transition_count"]:
    raise SystemExit("validation replay partition total differs from snapshot")
print(
    "H100_VALIDATION_REPLAY_PARTITION "
    f"transitions_by_rank={transition_counts} "
    f"global={sum(transition_counts)}",
    file=sys.stderr,
)
current = [row for row in snapshot["cohorts"] if row["macro_round"] == int(macro_round)]
if len(current) != 4:
    raise SystemExit("validation replay lacks exactly four current-round cohorts")
for row in current:
    manifest = load_strict_json(row["cohort_manifest_path"])
    expected = {
        "campaign_id": campaign_id,
        "macro_round": int(macro_round),
        "purpose": "train",
        "policy_sha256": parent_sha,
        "runtime_policy_sha256": runtime_sha,
        "sealed": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise SystemExit("current replay cohort does not match parent/runtime policy")
print(len(shards))
PY
)

    BUNDLE="$OUTPUT/validation_learner_bundle.json"
    "$PYTHON" "$ROOT/scripts/round_dagger/prepare_gr00t_learner_bundle.py" \
        --base-policy-spec "$SPEC" \
        --parent-checkpoint "$CHECKPOINT" \
        --expected-parent-checkpoint-sha256 "$CHECKPOINT_SHA" \
        --runtime-policy-sha256 "$RUNTIME_SHA" \
        --campaign-id "$CAMPAIGN_ID" --macro-round "$MACRO_ROUND" \
        --code-sha256 "$SOURCE_SHA" --batch-size 1 --max-steps 2 \
        --save-every 1 --learning-rate 2.5e-5 --weight-decay 0.01 \
        --seed 17 --partition-seed 23 --output "$BUNDLE" \
        --expected-bundle-sha256 "$CONFIG_SHA" >/dev/null

    TRAINER_COMMON=(
        "$ROOT/scripts/round_dagger/train_learner.py"
        --bundle "$BUNDLE" --snapshot "$SNAPSHOT"
        --model-factory round_dagger.gr00t_policy:create_vision_actor_obj_pred
        --adapter-factory round_dagger.gr00t_policy:vision_actor_obj_pred_adapter_factory
        --initial-checkpoint "$CHECKPOINT"
        --expected-snapshot-file-sha256 "$SNAPSHOT_FILE_SHA"
        --learner-identity-sha256 "$LEARNER_IDENTITY_SHA"
        --runtime-container-sha256 "$CONTAINER_SHA"
        --learner-contract-sha256 "$CONTRACT_SHA"
        --max-steps 2 --save-every 1 --device cuda
    )

    FULL="$OUTPUT/uninterrupted"
    RESUMED="$OUTPUT/resumed"
    mkdir "$FULL" "$RESUMED"
    "$PYTHON" -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
        --node-rank 0 --master-addr 127.0.0.1 --master-port 29741 \
        "${TRAINER_COMMON[@]}" --output-dir "$FULL" \
        --telemetry-jsonl "$FULL/telemetry.jsonl"
    "$PYTHON" -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
        --node-rank 0 --master-addr 127.0.0.1 --master-port 29742 \
        "${TRAINER_COMMON[@]}" --output-dir "$RESUMED" \
        --telemetry-jsonl "$RESUMED/telemetry.jsonl" \
        --wandb-project "$PROJECT" --wandb-entity "$ENTITY" \
        --wandb-run-id "$WANDB_RUN_ID" \
        --no-publish-result --validation-stop-at-step 1
    cp --reflink=auto --preserve=mode "$RESUMED/restart.pt" \
        "$OUTPUT/intentional_stop_step_1.pt"
    chmod 0444 "$OUTPUT/intentional_stop_step_1.pt"
    "$PYTHON" -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
        --node-rank 0 --master-addr 127.0.0.1 --master-port 29743 \
        "${TRAINER_COMMON[@]}" --output-dir "$RESUMED" \
        --telemetry-jsonl "$RESUMED/telemetry.jsonl" \
        --wandb-project "$PROJECT" --wandb-entity "$ENTITY" \
        --wandb-run-id "$WANDB_RUN_ID" \
        --resume-from "$RESUMED/restart.pt"

    RESUME_EVIDENCE="$JOB/evidence/strict_resume_invocation.json"
    "$PYTHON" - "$OUTPUT/intentional_stop_step_1.pt" "$RESUMED/restart.pt" \
        "$WANDB_RUN_ID" "$CONFIG_SHA" "$LEARNER_IDENTITY_SHA" \
        "$CONTAINER_SHA" "$CONTRACT_SHA" "$SOURCE_SHA" "$SNAPSHOT_FILE_SHA" \
        "$SNAPSHOT_SHA" "$CHECKPOINT_SHA" "$RUNTIME_SHA" \
        "$RESUME_EVIDENCE" <<'PY'
import os
import stat
import sys

from round_dagger.artifacts import publish_immutable_json, sha256_file
from round_dagger.checkpoint_compat import load_restricted_restart_document
from round_dagger.learner import CHECKPOINT_SCHEMA

(stopped_path, resumed_path, run_id, config_sha, identity_sha, container_sha,
 contract_sha, code_sha, snapshot_file_sha, snapshot_sha, parent_sha,
 parent_runtime_sha, output) = sys.argv[1:]
stopped = load_restricted_restart_document(stopped_path, map_location="cpu")
resumed = load_restricted_restart_document(resumed_path, map_location="cpu")
for document, step, label in ((stopped, 1, "stopped"), (resumed, 2, "resumed")):
    if document.get("schema") != CHECKPOINT_SCHEMA or document.get("completed_steps") != step:
        raise SystemExit(f"{label} strict-resume checkpoint mismatch")
    if document.get("learner_config_sha256") != config_sha:
        raise SystemExit(f"{label} strict-resume config mismatch")
    expected_identity = {
        "learner_identity_sha256": identity_sha,
        "runtime_container_sha256": container_sha,
        "learner_contract_sha256": contract_sha,
        "code_sha256": code_sha,
        "replay_snapshot_file_sha256": snapshot_file_sha,
        "replay_snapshot_sha256": snapshot_sha,
        "model_lineage_sha256": parent_sha,
    }
    if any(document.get(key) != value for key, value in expected_identity.items()):
        raise SystemExit(f"{label} strict-resume semantic identity mismatch")
report = {
    "schema": "fast_roborender.round_dagger.h100_validation_strict_resume.v2",
    "status": "PASS",
    "run_id": run_id,
    "first_invocation_resume_mode": "allow",
    "second_invocation_resume_mode": "must",
    "stop_step": 1,
    "target_step": 2,
    "learner_config_sha256": config_sha,
    "learner_identity_sha256": identity_sha,
    "learner_code_sha256": code_sha,
    "replay_snapshot_file_sha256": snapshot_file_sha,
    "replay_snapshot_sha256": snapshot_sha,
    "parent_checkpoint_sha256": parent_sha,
    "parent_runtime_policy_sha256": parent_runtime_sha,
    "runtime_container_sha256": container_sha,
    "learner_contract_sha256": contract_sha,
    "uninterrupted_comparator_wandb": False,
    "canonical_wandb_lineage": True,
    "stopped_checkpoint_sha256": sha256_file(stopped_path),
    "resumed_checkpoint_sha256": sha256_file(resumed_path),
}
publish_immutable_json(output, report)
os.chmod(output, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
PY

    printf -v ROUND_PADDED '%06d' "$((10#$MACRO_ROUND))"
    FULL_RESULT="$FULL/learner_result_round_${ROUND_PADDED}.json"
    RESUMED_RESULT="$RESUMED/learner_result_round_${ROUND_PADDED}.json"
    RESTART_REPORT="$OUTPUT/restart_equivalence.json"
    "$PYTHON" "$ROOT/scripts/round_dagger/verify_learner_restart_equivalence.py" \
        --uninterrupted-checkpoint "$FULL/restart.pt" \
        --resumed-checkpoint "$RESUMED/restart.pt" \
        --expected-completed-steps 2 \
        --uninterrupted-result "$FULL_RESULT" \
        --resumed-result "$RESUMED_RESULT" --output "$RESTART_REPORT" >/dev/null

    # This is immutable authenticated evidence returned with the learner
    # artifacts, not a W&B SDK runtime directory.  Keep its name distinct so
    # static audits cannot confuse it with the job-scratch-only W&B runtime.
    WANDB_READBACK="$OUTPUT/learner_telemetry_readback.json"
    "$PYTHON" "$ROOT/scripts/round_dagger/h100_validation_wandb_readback.py" \
        --run-id "$WANDB_RUN_ID" \
        --uninterrupted-telemetry "$FULL/telemetry.jsonl" \
        --resumed-telemetry "$RESUMED/telemetry.jsonl" \
        --uninterrupted-result "$FULL_RESULT" --resumed-result "$RESUMED_RESULT" \
        --resume-evidence "$RESUME_EVIDENCE" --campaign-id "$CAMPAIGN_ID" \
        --macro-round "$MACRO_ROUND" --learner-config-sha256 "$CONFIG_SHA" \
        --learner-code-sha256 "$SOURCE_SHA" \
        --model-lineage-sha256 "$CHECKPOINT_SHA" \
        --replay-snapshot-file-sha256 "$SNAPSHOT_FILE_SHA" \
        --replay-snapshot-sha256 "$SNAPSHOT_SHA" \
        --parent-runtime-policy-sha256 "$RUNTIME_SHA" \
        --runtime-container-sha256 "$CONTAINER_SHA" \
        --validation-contract-sha256 "$CONTRACT_SHA" \
        --learner-identity-sha256 "$LEARNER_IDENTITY_SHA" \
        --uninterrupted-checkpoint "$FULL/restart.pt" \
        --stopped-checkpoint "$OUTPUT/intentional_stop_step_1.pt" \
        --resumed-checkpoint "$RESUMED/restart.pt" \
        --project "$PROJECT" \
        --expected-email "$EXPECTED_EMAIL" --entity "$ENTITY" \
        --authorization-scope "$AUTH_SCOPE" --output "$WANDB_READBACK" >/dev/null

    "$PYTHON" "$ROOT/scripts/round_dagger/h100_validation_attestation.py" \
        --source-root "$ROOT" --source-inventory-sha256 "$SOURCE_SHA" \
        --container "$ROUND_DAGGER_VALIDATION_CONTAINER_HOST_PATH" \
        --container-sha256 "$ROUND_DAGGER_VALIDATION_CONTAINER_SHA256" \
        --validation-contract "$CONTRACT" \
        --validation-contract-sha256 "$CONTRACT_SHA" \
        --learner-identity-sha256 "$LEARNER_IDENTITY_SHA" \
        --parent-runtime-policy-sha256 "$RUNTIME_SHA" \
        --campaign-root "$CAMPAIGN" --snapshot "$SNAPSHOT" \
        --snapshot-file-sha256 "$SNAPSHOT_FILE_SHA" \
        --snapshot-sha256 "$SNAPSHOT_SHA" --replay-shards "$REPLAY_SHARDS" \
        --policy-checkpoint "$CHECKPOINT" \
        --policy-checkpoint-sha256 "$CHECKPOINT_SHA" \
        --policy-spec "$SPEC" --policy-spec-sha256 "$SPEC_SHA" \
        --policy-manifest "$MANIFEST" --policy-manifest-sha256 "$MANIFEST_SHA" \
        --bundle "$BUNDLE" --config-sha256 "$CONFIG_SHA" \
        --import-report "$JOB/evidence/cuda_import_gate.json" \
        --wandb-report "$JOB/evidence/wandb_preflight.json" \
        --wandb-readback "$WANDB_READBACK" \
        --uninterrupted-telemetry "$FULL/telemetry.jsonl" \
        --resumed-telemetry "$RESUMED/telemetry.jsonl" \
        --resume-evidence "$RESUME_EVIDENCE" \
        --uninterrupted-checkpoint "$FULL/restart.pt" \
        --uninterrupted-result "$FULL_RESULT" \
        --interrupted-checkpoint "$OUTPUT/intentional_stop_step_1.pt" \
        --resumed-checkpoint "$RESUMED/restart.pt" \
        --resumed-result "$RESUMED_RESULT" --restart-report "$RESTART_REPORT" \
        --campaign-id "$CAMPAIGN_ID" --macro-round "$MACRO_ROUND" \
        --output "$OUTPUT/success_attestation.json"
    echo "OCI_NRT_H100_LEARNER_SEQUENCE_PASS"
    ;;
*) fatal "unknown validation mode: $MODE" ;;
esac
