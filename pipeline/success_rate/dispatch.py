"""Consolidated success evaluation dispatch; versioned protocols remain distinct."""
from __future__ import annotations


# ---- scaled success dispatch ----

from copy import deepcopy as scaled_deepcopy
import fcntl as scaled_fcntl
import hashlib as scaled_hashlib
import json as scaled_json
import os as scaled_os
from pathlib import Path as scaled_Path
import re as scaled_re
from typing import Protocol as scaled_Protocol

from pipeline.success_rate import scaled_success_contract as scaled_scaled
from pipeline.deployment.rtx_actor.rendered_endpoint_pool import expected_physical_gpu as scaled_expected_physical_gpu

scaled_PLAN_SCHEMA = "roborender.scaled-success-dispatch-plan.v1"
scaled_COMMIT_SCHEMA = "roborender.scaled-success-cohort-commit.v1"
scaled_LIMIT = 64 << 20
scaled_MAX_FINAL_RESULT_LIMIT = 512 << 20
scaled_OWNER_KEYS = ("worker_instance_id", "source_sha256", "native_pid", "gpu_uuid")


def scaled_canonical(value):
    return scaled_json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def scaled_digest(value):
    return scaled_hashlib.sha256(scaled_canonical(value)).hexdigest()


def scaled__sha(value):
    return isinstance(value, str) and scaled_re.fullmatch(r"[0-9a-f]{64}", value) is not None


def scaled_read(path, *, limit=scaled_LIMIT):
    if type(limit) is not int or not 1 <= limit <= scaled_MAX_FINAL_RESULT_LIMIT:
        raise ValueError('bounded dispatcher read limit required')
    path = scaled_Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("bounded regular dispatcher evidence required")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("dispatcher evidence exceeded bound")
    return scaled_json.loads(raw)


def scaled_immutable(path, value, *, limit=scaled_LIMIT):
    """No-clobber publication, fsynced before any associated native mutation."""
    path = scaled_Path(path)
    if type(limit) is not int or not 1 <= limit <= scaled_MAX_FINAL_RESULT_LIMIT:
        raise ValueError('bounded dispatcher publication limit required')
    raw = scaled_canonical(value) + b"\n"
    if len(raw) > limit or path.is_symlink():
        raise ValueError("invalid immutable dispatcher evidence")
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("immutable dispatcher evidence changed: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = scaled_os.open(path, scaled_os.O_WRONLY | scaled_os.O_CREAT | scaled_os.O_EXCL | scaled_os.O_NOFOLLOW, 0o444)
    with scaled_os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        scaled_os.fsync(stream.fileno())
    directory = scaled_os.open(path.parent, scaled_os.O_RDONLY | scaled_os.O_DIRECTORY)
    try:
        scaled_os.fsync(directory)
    finally:
        scaled_os.close(directory)


def scaled__owners(owners):
    if not isinstance(owners, list) or not 1 <= len(owners) <= 2:
        raise ValueError("one or two explicitly qualified RTX owners required")
    for row in owners:
        if (not isinstance(row, dict) or not all(key in row for key in scaled_OWNER_KEYS)
                or not scaled_re.fullmatch(r"[0-9a-f]{32}", row.get("worker_instance_id", ""))
                or not scaled__sha(row.get("source_sha256"))
                or type(row.get("native_pid")) is not int or row["native_pid"] <= 0
                or type(row.get("physical_gpu")) is not int or row["physical_gpu"] not in (0, 1)
                or not isinstance(row.get("gpu_uuid"), str) or not row["gpu_uuid"].startswith("GPU-")):
            raise ValueError("exact qualified source/native/GPU owner identity required")
    for key in ("physical_gpu", "gpu_uuid", "worker_instance_id"):
        if len({row[key] for row in owners}) != len(owners):
            raise ValueError("one distinct owner per GPU required")
    if len({row["source_sha256"] for row in owners}) != 1:
        raise ValueError("paired owners must share the same frozen source")
    return scaled_deepcopy(sorted(owners, key=lambda row: row['physical_gpu']))


def scaled_build_plan(panel, config, owners, *, command_prefix, arm="bc", qualification_evidence=None,
               qualification_evidence_sha256=None):
    """Return deterministic immutable intents and exact canonical staged bytes.

    The transport receives each artifact as {path, raw_sha256, value}; it must
    stage canonical(value), WITHOUT a trailing newline, and reverify that hash.
    Checkpoint staging/qualification is a separate required driver admission.
    """
    panel = scaled_scaled.require_panel(panel)
    owners = scaled__owners(owners)
    physical_assignments = [scaled_expected_physical_gpu(config.get('renderer', {}), i)
                            for i in range(len(panel['cohorts']))]
    if any(gpu is not None for gpu in physical_assignments):
        if {row['physical_gpu'] for row in owners} != set(physical_assignments):
            raise ValueError('sharded evaluation needs exactly its prescribed physical GPU owners')
    if (not isinstance(command_prefix, str)
            or not scaled_re.fullmatch(r"[a-z0-9][a-z0-9-]{0,69}", command_prefix)):
        raise ValueError("new lowercase bounded campaign command prefix required")
    if arm not in ("bc", "dagger"):
        raise ValueError("only frozen student policy arms may be evaluated")
    checkpoint = config.get("checkpoint_sha256")
    if not scaled__sha(checkpoint):
        raise ValueError("configuration must bind the selected checkpoint")
    qualification = panel["scaled_protocol"]["purpose"] == "qualification"
    if qualification:
        if qualification_evidence is not None or qualification_evidence_sha256 is not None:
            raise ValueError("qualification probes cannot claim prior production readiness")
    else:
        from pipeline.success_rate.scaled_success_qualification import validate_qualification
        validate_qualification(qualification_evidence, source_sha256=owners[0]["source_sha256"],
            panel=panel, config=config)
        actual_sha = scaled_digest(qualification_evidence)
        if qualification_evidence_sha256 is not None and qualification_evidence_sha256 != actual_sha:
            raise ValueError("actual qualification evidence raw digest differs")
        qualification_evidence_sha256 = actual_sha

    def artifact(name, value):
        return dict(path=command_prefix + "/" + name, raw_sha256=scaled_digest(value), value=scaled_deepcopy(value))

    panel_file, config_file = artifact("panel.json", panel), artifact("config.json", config)
    proof_file = None if qualification else artifact("qualification.json", qualification_evidence)
    commands = []
    for index in range(len(panel["cohorts"])):
        expected_gpu = physical_assignments[index]
        owner_index = (index % len(owners) if expected_gpu is None else
                       next(i for i, row in enumerate(owners) if row['physical_gpu'] == expected_gpu))
        owner, cid = owners[owner_index], f"{command_prefix}-c{index:03d}"
        partition = scaled_scaled.single_partition(panel, [index])
        admission = scaled_scaled.build_admission(panel, config, source_sha256=owner["source_sha256"],
            worker_id=owner["worker_instance_id"], command_id=cid, output_name=cid,
            cohort_indices=[index], qualification_evidence_sha256=qualification_evidence_sha256, arm=arm)
        partition_file = artifact(f"partition-{index:03d}.json", partition)
        admission_file = artifact(f"admission-{index:03d}.json", admission)
        body = dict(schema="roborender.isaac-resident.v1", worker_instance_id=owner["worker_instance_id"],
            command_id=cid, operation=scaled_scaled.OPERATION, payload=dict(
                panel_file=panel_file["path"], panel_sha256=panel_file["raw_sha256"],
                checkpoint_sha256=checkpoint, arm=arm, output_name=cid, **scaled_scaled.evaluation_options(panel),
                rendered_observation_config_file=config_file["path"],
                rendered_observation_config_sha256=config_file["raw_sha256"],
                partition_file=partition_file["path"], partition_sha256=partition_file["raw_sha256"],
                scaled_admission_file=admission_file["path"], scaled_admission_sha256=admission_file["raw_sha256"]))
        if proof_file is not None:
            body["payload"].update(qualification_file=proof_file["path"], qualification_sha256=proof_file["raw_sha256"])
        commands.append(dict(cohort_index=index, owner_index=owner_index, body=body,
            payload_sha256=scaled_digest(body), artifacts=[partition_file, admission_file]))
    return dict(schema=scaled_PLAN_SCHEMA, panel=panel, config=scaled_deepcopy(config), owners=owners, arm=arm,
        command_prefix=command_prefix, checkpoint_sha256=checkpoint,
        qualification_evidence_sha256=qualification_evidence_sha256, commands=commands,
        qualification_evidence=scaled_deepcopy(qualification_evidence),
        shared_artifacts=[panel_file, config_file] + ([] if proof_file is None else [proof_file]),
        historical_outcomes_reused=False, allocation_release_authorized=False,
        stop_boundary="one_complete_cohort_per_native_command")


def scaled_validate_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("scaled dispatcher plan required")
    expected = scaled_build_plan(plan["panel"], plan["config"], plan["owners"],
        command_prefix=plan["command_prefix"], arm=plan["arm"],
        qualification_evidence=plan["qualification_evidence"],
        qualification_evidence_sha256=plan["qualification_evidence_sha256"])
    if plan != expected:
        raise ValueError("immutable cohort/source/owner assignment changed")
    return scaled_deepcopy(expected)


class scaled_Transport(scaled_Protocol):
    """Driver implements bounded, authenticated and exact-owner-attested I/O."""
    def get_ready(self, owner): ...
    def get_command(self, owner, command_id): ...
    def stage_artifacts(self, owner, artifacts): ...
    def submit(self, owner, body): ...


def scaled_validate_ready(value, owner):
    if (not isinstance(value, dict) or any(value.get(key) != owner[key] for key in scaled_OWNER_KEYS)
            or value.get("native_initializations") != 1 or value.get("role") != "isaac_resident"
            or value.get("schema") != "roborender.isaac-resident.v1"):
        raise ValueError("actual native lifetime/source/GPU differs; replay forbidden")
    phase = value.get("phase")
    if phase not in ("IDLE", "QUEUED", "RUNNING", "CRASH_UNCERTAIN", "STOPPED"):
        raise ValueError("unknown native owner phase")
    if value.get("ok") is not (phase not in ("CRASH_UNCERTAIN", "STOPPED")):
        raise ValueError("native readiness contradicts phase")
    if phase == "IDLE" and value.get("active_command_id") is not None:
        raise ValueError("idle owner has an active command")
    return value


def scaled_validate_complete(plan, entry, row, resource_validator):
    """Independently check science and require real hash-bound resource audit.

    resource_validator(owner, native_receipt) is REQUIRED; the driver must read
    and verify the actual guard in the attested native owner's filesystem, e.g.
    production_resident_pool.validate_guard, not trust a boolean in a response.
    """
    owner = plan["owners"][entry["owner_index"]]
    receipt = row.get("receipt")
    if (row.get("status") != "COMPLETE" or row.get("payload_sha256") != entry["payload_sha256"]
            or not isinstance(receipt, dict) or receipt.get("complete") is not True
            or receipt.get("source_sha256") != owner["source_sha256"]
            or receipt.get("native_pid") != owner["native_pid"]
            or receipt.get("native_initializations") != 1 or receipt.get("optimizer_steps") != 0
            or receipt.get("purpose") != "test" or receipt.get("profile") != "explicit_evaluation"):
        raise ValueError("completed native receipt identity or science envelope differs")
    result = scaled_scaled.validate_result(plan["panel"], receipt.get("result"), plan["checkpoint_sha256"],
        rendered_config=plan["config"])
    if (result.get("schema") != scaled_scaled.PARTITION_RESULT_SCHEMA
            or result["partition_result"]["partition"]["cohort_indices"] != [entry["cohort_index"]]
            or result["partition_result"]["arm"] != plan["arm"]):
        raise ValueError("native receipt does not belong to this exact cohort/arm")
    if not callable(resource_validator):
        raise ValueError("actual owned resource evidence verifier required")
    resource_validator(owner, receipt)
    return result


class scaled_Coordinator:
    """Bounded tick API; construction and transport injection do not start GPUs."""
    def _validate_plan(self, plan):
        return scaled_validate_plan(plan)

    def _validate_complete(self, entry, row):
        return scaled_validate_complete(self.plan, entry, row, self.resource_validator)

    def _aggregate(self, values):
        return scaled_scaled.aggregate_results(self.plan["panel"], values, self.plan["checkpoint_sha256"],
            self.plan["config"], arm=self.plan["arm"])

    def __init__(self, plan, root, transport, *, resource_validator, max_submit_attempts=3, result_limit=scaled_LIMIT,
                 stop_peer_on_uncertainty=False):
        self.plan = self._validate_plan(plan)
        self.root, self.transport = scaled_Path(root), transport
        self.resource_validator = resource_validator
        if not callable(resource_validator):
            raise ValueError("a real native resource-evidence validator is mandatory")
        if type(max_submit_attempts) is not int or not 1 <= max_submit_attempts <= 3:
            raise ValueError("submission retry bound must be 1..3")
        self.max_submit_attempts = max_submit_attempts
        if type(stop_peer_on_uncertainty) is not bool:
            raise ValueError('explicit paired uncertainty policy must be boolean')
        self.stop_peer_on_uncertainty=stop_peer_on_uncertainty
        if type(result_limit) is not int or not scaled_LIMIT <= result_limit <= scaled_MAX_FINAL_RESULT_LIMIT:
            raise ValueError('bounded explicit final aggregate limit required')
        self.result_limit = result_limit
        if self.root.is_symlink():
            raise ValueError("dispatcher root cannot be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self.plan_sha = scaled_digest(self.plan)
        self._verified_commits = {}
        scaled_immutable(self.root / "plan.json", self.plan)
        policy = dict(max_submit_attempts=max_submit_attempts,
            native_replay_allowed=False, compute_release_authorized=False)
        if result_limit != scaled_LIMIT:
            policy['final_aggregate_limit_bytes'] = result_limit
        if stop_peer_on_uncertainty:
            policy['stop_peer_on_uncertainty']=True
        scaled_immutable(self.root / "controller-policy.json", policy)

    def _path(self, entry):
        return self.root / f"cohort-{entry['cohort_index']:03d}"

    def _acceptance(self, entry):
        return dict(schema="roborender.isaac-resident.v1", command_id=entry["body"]["command_id"],
            worker_instance_id=entry["body"]["worker_instance_id"],
            payload_sha256=entry["payload_sha256"], status="accepted")

    def _commit(self, entry, row):
        result = self._validate_complete(entry, row)
        value = dict(schema=scaled_COMMIT_SCHEMA, plan_sha256=self.plan_sha, cohort_index=entry["cohort_index"],
            owner=self.plan["owners"][entry["owner_index"]], command_id=entry["body"]["command_id"],
            payload_sha256=entry["payload_sha256"], native_command_receipt=scaled_deepcopy(row), result=result)
        scaled_immutable(self._path(entry) / "commit.json", value)
        self._verified_commits[entry["cohort_index"]] = (scaled_digest(value), scaled_deepcopy(value))
        return value

    def _read_commit(self, entry):
        path = self._path(entry) / "commit.json"
        if not path.exists():
            return None
        value = scaled_read(path)
        if path.read_bytes() != scaled_canonical(value) + b"\n":
            raise ValueError("immutable commit raw bytes changed")
        cached = self._verified_commits.get(entry["cohort_index"])
        if cached is not None and scaled_digest(value) == cached[0]:
            return scaled_deepcopy(cached[1])
        # Existing commits are revalidated against actual science/resource proof,
        # never adopted merely because a filename or completed counter exists.
        return self._commit(entry, value["native_command_receipt"])

    def _step(self, entry):
        owner, path = self.plan["owners"][entry["owner_index"]], self._path(entry)
        cid = entry["body"]["command_id"]
        scaled_immutable(path / "intent.json", dict(plan_sha256=self.plan_sha, entry=entry, owner=owner))
        ready = scaled_validate_ready(self.transport.get_ready(scaled_deepcopy(owner)), owner)
        row = self.transport.get_command(scaled_deepcopy(owner), cid)
        if row is not None:
            if (not isinstance(row, dict) or row.get("payload_sha256") != entry["payload_sha256"]
                    or row.get("status") not in ("QUEUED", "RUNNING", "COMPLETE", "CRASH_UNCERTAIN")):
                raise ValueError("native command identity/status differs")
            scaled_immutable(path / "acceptance.json", self._acceptance(entry))
            if row["status"] == "COMPLETE":
                self._commit(entry, row)
                return "COMMITTED"
            if row["status"] == "CRASH_UNCERTAIN":
                scaled_immutable(path / "terminal-fault.json", row)
                return "NATIVE_TERMINAL_FAULT"
            if ready["phase"] in ("CRASH_UNCERTAIN", "STOPPED"):
                return "NATIVE_TERMINAL_FAULT"
            if ready["phase"] in ("QUEUED", "RUNNING") and ready.get("active_command_id") != cid:
                raise ValueError("owner is running a different command")
            return "IN_FLIGHT"
        if (path / "acceptance.json").exists():
            raise ValueError("previously accepted native command disappeared; replay forbidden")
        if ready["phase"] in ("CRASH_UNCERTAIN", "STOPPED"):
            return "NATIVE_TERMINAL_FAULT"
        if ready["phase"] != "IDLE":
            if ready.get("active_command_id") != cid:
                raise ValueError("owner busy with unrelated work; no submission")
            return "RECONCILING_ACCEPTANCE"
        attempts = sorted(path.glob("submit-attempt-*.json"))
        if [p.name for p in attempts] != [f"submit-attempt-{i:02d}.json" for i in range(len(attempts))]:
            raise ValueError("submission attempt journal is not a prefix")
        for prior in attempts:
            if scaled_read(prior) != dict(plan_sha256=self.plan_sha, body=entry["body"], owner=owner):
                raise ValueError("submission attempt identity changed")
        if (self.root / "STOP").exists():
            return "STOPPED_UNCERTAIN_ACCEPTANCE" if attempts else "NOT_STARTED_STOPPED"
        if len(attempts) >= self.max_submit_attempts:
            return "SUBMISSION_LIMIT_REQUIRES_RECONCILIATION"
        self.transport.stage_artifacts(scaled_deepcopy(owner), scaled_deepcopy(self.plan["shared_artifacts"] + entry["artifacts"]))
        # Re-attest and GET again after staging: staging is not authority to POST.
        ready = scaled_validate_ready(self.transport.get_ready(scaled_deepcopy(owner)), owner)
        if self.transport.get_command(scaled_deepcopy(owner), cid) is not None:
            return "RECONCILING_ACCEPTANCE"
        if ready["phase"] != "IDLE" or (self.root / "STOP").exists():
            return "RECONCILING_ACCEPTANCE" if ready["phase"] != "IDLE" else "NOT_STARTED_STOPPED"
        scaled_immutable(path / f"submit-attempt-{len(attempts):02d}.json",
            dict(plan_sha256=self.plan_sha, body=entry["body"], owner=owner))
        ack = self.transport.submit(scaled_deepcopy(owner), scaled_deepcopy(entry["body"]))
        if ack != self._acceptance(entry):
            raise ValueError("native acceptance differs from exact durable intent")
        scaled_immutable(path / "acceptance.json", ack)
        return "SUBMITTED"

    def tick(self):
        """One bounded pass. Caller chooses polling interval/deadline, never replay."""
        lock_path = self.root / "controller.lock"
        if lock_path.is_symlink():
            raise ValueError("controller lock cannot be a symlink")
        with lock_path.open("a") as lock:
            scaled_fcntl.flock(lock, scaled_fcntl.LOCK_EX | scaled_fcntl.LOCK_NB)
            if scaled_read(self.root / "plan.json") != self.plan:
                raise ValueError("durable dispatcher plan changed")
            commits = {entry["cohort_index"]: self._read_commit(entry) for entry in self.plan["commands"]}
            states = []
            pending_owners=[]
            for owner_index in range(len(self.plan["owners"])):
                pending = [entry for entry in self.plan["commands"]
                    if entry["owner_index"] == owner_index and commits[entry["cohort_index"]] is None]
                if pending:pending_owners.append((owner_index,pending[0]))
            if self.stop_peer_on_uncertainty:
                # Reconcile every previously attempted command before admitting
                # untouched peer work, regardless of physical GPU ordering.
                def previously_attempted(row):
                    path=self._path(row[1])
                    return (path/'acceptance.json').exists() or any(path.glob('submit-attempt-*.json'))
                pending_owners.sort(key=lambda row:not previously_attempted(row))
            for owner_index,entry in pending_owners:
                try:
                    status = self._step(entry)
                except (TimeoutError, ConnectionError):
                    status = "TRANSPORT_UNKNOWN_RECONCILE_SAME_OWNER"
                states.append(dict(owner_index=owner_index, cohort_index=entry["cohort_index"], status=status))
                if status == "COMMITTED":
                    commits[entry["cohort_index"]] = self._read_commit(entry)
                # A known fault prevents admitting more peer work in this tick.
                if (status == "NATIVE_TERMINAL_FAULT" or self.stop_peer_on_uncertainty and status in {
                        'TRANSPORT_UNKNOWN_RECONCILE_SAME_OWNER','SUBMISSION_LIMIT_REQUIRES_RECONCILIATION',
                        'STOPPED_UNCERTAIN_ACCEPTANCE','NOT_STARTED_STOPPED','RECONCILING_ACCEPTANCE'}):
                    break
            values = [commits[i]["result"] for i in sorted(commits) if commits[i] is not None]
            result = None
            if len(values) == len(self.plan["commands"]):
                result = self._aggregate(values)
                scaled_immutable(self.root / "result.json", result, limit=self.result_limit)
            return dict(schema="roborender.scaled-success-dispatch-progress.v1", plan_sha256=self.plan_sha,
                complete=result is not None, committed_cohorts=len(values), expected_cohorts=len(self.plan["commands"]),
                stop_requested=(self.root / "STOP").exists(), workers=states,
                compute_release_authorized=False, historical_outcomes_reused=False,
                qualification_only=self.plan["panel"]["scaled_protocol"]["purpose"] == "qualification")


# ---- episode dispatch ----
from copy import deepcopy as pooled_deepcopy

from pipeline.success_rate import scaled_success_dispatch as pooled_scaled
from pipeline.success_rate import episode_protocol as pooled_benchmark

pooled_SCHEMA = 'roborender.raven-pooled-dispatch-plan.v1'


def pooled_build_plan(panel, config, owners, *, command_prefix, qualification_only, arm='dagger'):
    value = pooled_scaled.build_plan(panel, config, owners, command_prefix=command_prefix, arm=arm)
    if len(value['owners']) != 2 or len(value['commands']) != 2:
        raise ValueError('pooled benchmark requires both independently scoped B16 owners')
    for entry in value['commands']:
        body = entry['body']; payload = body['payload']; owner = value['owners'][entry['owner_index']]
        spec = pooled_benchmark.build_benchmark(panel, config, owner=owner, command_id=body['command_id'],
            output_name=payload['output_name'], cohort_index=entry['cohort_index'], qualification_only=qualification_only, arm=arm)
        artifact = dict(path=command_prefix+f"/benchmark-{entry['cohort_index']:03d}.json",
            raw_sha256=pooled_benchmark.digest(spec), value=spec)
        entry['artifacts'] = [entry['artifacts'][0], artifact]
        payload.pop('scaled_admission_file'); payload.pop('scaled_admission_sha256')
        payload.update(benchmark_file=artifact['path'], benchmark_sha256=artifact['raw_sha256'])
        body['operation'] = pooled_benchmark.OPERATION
        entry['payload_sha256'] = pooled_scaled.digest(body)
    value.update(schema=pooled_SCHEMA, qualification_only=qualification_only,
        minimum_completed_episodes_per_actor=32 if qualification_only else 150,
        population='two_independent_B16_replicates_not_one_global_B32_pool',
        stop_boundary='each_actor_next_loop_check_after_all_simultaneous_completions',
        fixed295_outcomes_imported=False)
    return value


def pooled_validate_plan(value):
    expected = pooled_build_plan(value['panel'], value['config'], value['owners'], command_prefix=value['command_prefix'],
        qualification_only=value['qualification_only'], arm=value['arm'])
    if value != expected:
        raise ValueError('pooled dispatch plan changed')
    return pooled_deepcopy(expected)


def pooled_validate_complete(plan, entry, row, resource_validator):
    pooled_validate_plan(plan)
    if entry not in plan['commands']:
        raise ValueError('pooled completion is not an admitted actor command')
    owner = plan['owners'][entry['owner_index']]; receipt = row.get('receipt', {})
    if (row.get('status') != 'COMPLETE' or row.get('payload_sha256') != entry['payload_sha256']
            or receipt.get('complete') is not True or receipt.get('source_sha256') != owner['source_sha256']
            or receipt.get('command_id') != entry['body']['command_id']
            or receipt.get('worker_instance_id') != owner['worker_instance_id']
            or receipt.get('native_pid') != owner['native_pid'] or receipt.get('native_initializations') != 1
            or type(receipt.get('kit_identity')) is not int or receipt['kit_identity'] <= 0
            or receipt.get('same_kit') is not True or receipt.get('physics_pause_requested') is not True
            or any(type(receipt.get(key)) is not int or receipt[key] < 1
                for key in ('scene_initializations','phase_resets'))
            or receipt.get('optimizer_steps') != 0 or receipt.get('purpose') != 'test'
            or receipt.get('profile') != 'explicit_evaluation'):
        raise ValueError('actual complete native pooled benchmark receipt required')
    spec = entry['artifacts'][1]['value']
    result = pooled_benchmark.validate_result(receipt.get('result'), spec)
    from pipeline.success_rate.rendered_student_success_contract import validate_cohort_observations
    validate_cohort_observations(result, count=16, config=plan['config'], panel=plan['panel'],
        cohort_index=entry['cohort_index'], _pooled_benchmark=spec)
    if not callable(resource_validator):
        raise ValueError('actual owned resource verifier required')
    resource_validator(owner, receipt)
    return result


def pooled_aggregate(plan, results):
    pooled_validate_plan(plan)
    if not isinstance(results, list) or len(results) != 2:
        raise ValueError('both independent pooled actor results required')
    rows = [pooled_benchmark.validate_result(result, entry['artifacts'][1]['value'])
        for result, entry in zip(results, plan['commands'])]
    successes = sum(row['exported_successes'] for row in rows)
    completed = sum(row['completed'] for row in rows)
    return dict(schema='roborender.raven-pooled-aggregate.v1', checkpoint_sha256=plan['checkpoint_sha256'],
        qualification_only=plan['qualification_only'], benchmark_complete=not plan['qualification_only'],
        population=plan['population'], actor_results=[dict(physical_gpu=i,
            successes=row['exported_successes'], completed=row['completed'], overshoot=row['overshoot'],
            result_sha256=pooled_benchmark.digest(row)) for i, row in enumerate(rows)],
        successes=successes, completed=completed, weighted_success_rate=successes/completed,
        fixed295_equivalent=False, full295_motion_coverage=False, published_numeric_equivalence=False,
        compute_release_authorized=False)


# ---- dispatch ----
from copy import deepcopy as deepcopy
import re as re

from . import scaled_success_dispatch as student
from . import scaled_teacher_benchmark as teacher
from . import episode_dispatch as pooled

FIXED_PROTOCOL = "scaled-fixed295-b16-v1"
POOLED_PROTOCOL = "raven-pooled-b16x2-v1"
TEACHER_SCHEMA = "roborender.teacher-success-dispatch-plan.v1"


def build_plan(panel, owners, *, policy, protocol, command_prefix, config=None,
               arm=None, qualification_only=False, qualification_evidence=None,
               shadow_checkpoint_sha256=None):
    if type(qualification_only) is not bool:
        raise ValueError("qualification_only must be an explicit Boolean")
    if policy not in {"student", "teacher"}:
        raise ValueError("explicit student or teacher policy required")
    if policy == "student":
        if arm not in {"bc", "dagger"} or config is None or shadow_checkpoint_sha256 is not None:
            raise ValueError("student requires its bc/dagger arm and rendered config, not a teacher shadow")
        if protocol == POOLED_PROTOCOL:
            if qualification_evidence is not None:
                raise ValueError("pooled benchmark cannot inherit fixed295 qualification")
            return pooled.build_plan(panel, config, owners, command_prefix=command_prefix,
                qualification_only=qualification_only, arm=arm)
        if protocol != FIXED_PROTOCOL or qualification_only:
            raise ValueError("matched fixed295 requires the complete test, not a qualification subset")
        teacher.require_panel(panel)  # Identical physical population for both arms.
        return student.build_plan(panel, config, owners, command_prefix=command_prefix,
            arm=arm, qualification_evidence=qualification_evidence)
    if protocol != FIXED_PROTOCOL:
        raise ValueError("teacher pooled/B32 evaluation is not admitted; no silent benchmark substitution")
    if config is not None or arm is not None or qualification_only or qualification_evidence is not None:
        raise ValueError("teacher uses privileged state; student config/arm/qualification must be absent")
    panel = teacher.require_panel(panel)
    owners = student._owners(owners)
    if len(owners) != 2 or not isinstance(command_prefix, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,69}", command_prefix):
        raise ValueError("teacher requires both exact native owners and a fresh bounded command prefix")

    def artifact(name, value):
        return dict(path=command_prefix + "/" + name, raw_sha256=student.digest(value), value=deepcopy(value))

    panel_file = artifact("panel.json", panel)
    commands = []
    for i in range(len(panel["cohorts"])):
        owner_index = i % 2
        cid = f"{command_prefix}-c{i:03d}"
        spec = teacher.build_benchmark(panel, owner=owners[owner_index], command_id=cid,
            output_name=cid, cohort_index=i, shadow_checkpoint_sha256=shadow_checkpoint_sha256)
        spec_file = artifact(f"benchmark-{i:03d}.json", spec)
        payload = dict(panel_file=panel_file["path"], panel_sha256=panel_file["raw_sha256"],
            benchmark_file=spec_file["path"], benchmark_sha256=spec_file["raw_sha256"],
            checkpoint_sha256=shadow_checkpoint_sha256, output_name=cid, **teacher.evaluation_options())
        body = dict(schema="roborender.isaac-resident.v1", worker_instance_id=owners[owner_index]["worker_instance_id"],
            command_id=cid, operation=teacher.OPERATION, payload=payload)
        commands.append(dict(cohort_index=i, owner_index=owner_index, body=body,
            payload_sha256=student.digest(body), artifacts=[spec_file]))
    return dict(schema=TEACHER_SCHEMA, policy="teacher", protocol=protocol, panel=panel,
        owners=owners, command_prefix=command_prefix, commands=commands, shared_artifacts=[panel_file],
        shadow_not_behavior_checkpoint_sha256=shadow_checkpoint_sha256,
        observation_mode="privileged_state", compute_release_authorized=False,
        stop_boundary="one_complete_cohort_per_native_command", historical_outcomes_reused=False)


def validate_plan(value):
    if not isinstance(value, dict):
        raise ValueError("success-rate policy plan must be an object")
    if value.get("schema") == pooled.SCHEMA:
        return pooled.validate_plan(value)
    if value.get("schema") == student.PLAN_SCHEMA:
        return student.validate_plan(value)
    if value.get("schema") != TEACHER_SCHEMA:
        raise ValueError("unknown success-rate policy plan")
    expected = build_plan(value["panel"], value["owners"], policy="teacher", protocol=FIXED_PROTOCOL,
        command_prefix=value["command_prefix"], shadow_checkpoint_sha256=value["shadow_not_behavior_checkpoint_sha256"])
    if value != expected:
        raise ValueError("teacher policy plan identity changed")
    return expected


def aggregate(plan, results):
    plan = validate_plan(plan)
    if plan["schema"] == pooled.SCHEMA:
        return pooled.aggregate(plan, results)
    if not isinstance(results, list) or len(results) != len(plan["commands"]):
        raise ValueError("all planned cohort results are required; partial is not complete")
    if plan["schema"] == TEACHER_SCHEMA:
        for result, entry in zip(results, plan["commands"]):
            if result.get("benchmark") != entry["artifacts"][0]["value"]:
                raise ValueError("teacher result belongs to a different owner/command/plan")
        value = teacher.aggregate_results(plan["panel"], results)
        value["sanity_check"] = dict(expected_success_rate_gt=0.80,
            expectation_met=value["success_rate"] > 0.80,
            advisory_only=True, low_rate_requires_investigation=value["success_rate"] <= 0.80,
            certifies_student_renderer=False)
        return value
    return student.scaled.aggregate_results(plan["panel"], results, plan["checkpoint_sha256"],
        plan["config"], arm=plan["arm"])


class TeacherCoordinator(student.Coordinator):
    """Reuse the same durable no-replay dispatch, locks and warm native owners."""
    def _validate_plan(self, plan):
        value = validate_plan(plan)
        if value["schema"] != TEACHER_SCHEMA:
            raise ValueError("teacher coordinator requires the teacher plan")
        return value

    def _aggregate(self, values):
        return aggregate(self.plan, values)

    def _validate_complete(self, entry, row):
        owner = self.plan["owners"][entry["owner_index"]]
        receipt = row.get("receipt", {})
        spec = entry["artifacts"][0]["value"]
        if (row.get("status") != "COMPLETE" or row.get("payload_sha256") != entry["payload_sha256"]
                or any(receipt.get(k) != v for k, v in dict(complete=True, purpose="test", optimizer_steps=0,
                    source_sha256=owner["source_sha256"], native_pid=owner["native_pid"],
                    native_initializations=1, profile="explicit_evaluation", same_kit=True,
                    physics_pause_requested=True, command_id=spec["command_id"],
                    worker_instance_id=owner["worker_instance_id"]).items())
                or type(receipt.get("kit_identity")) is not int or receipt["kit_identity"] <= 0):
            raise ValueError("complete exact-owner warm teacher receipt required")
        result = teacher.validate_result(self.plan["panel"], receipt.get("result"))
        if result["benchmark"] != spec:
            raise ValueError("teacher receipt is not this exact admitted command")
        self.resource_validator(owner, receipt)
        return result


class PooledCoordinator(student.Coordinator):
    def _validate_plan(self, plan):
        return pooled.validate_plan(plan)

    def _validate_complete(self, entry, row):
        return pooled.validate_complete(self.plan, entry, row, self.resource_validator)

    def _aggregate(self, values):
        return pooled.aggregate(self.plan, values)


def coordinator(plan, root, transport, *, resource_validator):
    """Bind, but do not tick, the selected existing-owner dispatch path.

Use a new run root for a policy switch. Immutable plans reject switching the
behavior policy underneath an already accepted command. Admission and native
resource verification remain mandatory; this is not a capacity bypass.
    """
    value = validate_plan(plan)
    cls = (TeacherCoordinator if value["schema"] == TEACHER_SCHEMA else
           PooledCoordinator if value["schema"] == pooled.SCHEMA else student.Coordinator)
    return cls(value, root, transport, resource_validator=resource_validator,
        max_submit_attempts=1, stop_peer_on_uncertainty=True)


class TeacherTransport:
    """Teacher has no rendering dependency; retain source/owner/I/O guards."""
    def __init__(self, *args, **kwargs):
        # Keep planning/help stdlib-only; load native I/O only on construction.
        from .paired_scaled_runtime import Transport
        self._transport = Transport(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_transport"), name)

    def stage_artifacts(self, owner, artifacts):
        if not isinstance(artifacts, list) or len(artifacts) != 2:
            raise ValueError("teacher staging requires only its panel and benchmark")
        panel_file, benchmark_file = artifacts
        panel = teacher.require_panel(panel_file["value"])
        benchmark = teacher.validate_benchmark(benchmark_file["value"], panel)
        if benchmark["owner"] != {k: owner[k] for k in teacher.OWNER_FIELDS}:
            raise ValueError("teacher artifact owner changed")
        for artifact in artifacts:
            if student.digest(artifact["value"]) != artifact["raw_sha256"]:
                raise ValueError("teacher staged artifact changed")
        for artifact in artifacts:
            raw = student.canonical(artifact["value"])
            answer = self.invoke(owner, "stage",
                artifact={k: artifact[k] for k in ("path", "raw_sha256")}, input_bytes=raw)
            if answer != dict(staged=True, raw_sha256=artifact["raw_sha256"]):
                raise ValueError("actual staged teacher bytes differ")


def transport_for(policy, *args, **kwargs):
    if policy not in {"student", "teacher"}:
        raise ValueError("explicit student or teacher policy required")
    if policy == "teacher":
        return TeacherTransport(*args, **kwargs)
    from .paired_scaled_runtime import Transport
    return Transport(*args, **kwargs)
