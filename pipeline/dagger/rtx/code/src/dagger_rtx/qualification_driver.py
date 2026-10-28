"""Three-window, two-actor runtime check; never changes production admission.

The existing owner/orchestrator still owns capture, exactly-once stepping,
episode resets and sealing. This wrapper owns only the persistent lifecycle
and the checkpoint barrier. A restart is forbidden without native restoration.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time

from .contracts import require_sha256
from .durable import atomic_write_json
from .http_transport import TransientTransportError

MODE = "multi-window-check"
ACK = "I_ACKNOWLEDGE_THREE_WINDOWS_AND_REAL_LEARNER_UPDATES"
GPU_UUIDS = {1: "GPU-0c1949ae-379a-2f2e-0f23-621de2eee409",
             2: "GPU-0540f144-9826-ba8f-140d-af4f95a475ce"}


def validate_config(value: dict) -> dict:
    expected = {"schema", "run_id", "windows", "num_envs_per_gpu", "physical_rtx_gpus",
                "initial_checkpoint_sha256", "seed", "gateway_pids", "source_sha256",
                "teacher_beta", "wandb_run_name", "wandb_run_id", "wandb_mode", "ack_loss_probe"}
    if set(value) != expected or value["schema"] not in {"roborender.beta-runtime-check.v1", "roborender.beta-runtime-check.v2"}:
        raise ValueError("runtime-check config schema/keys differ")
    if not re.fullmatch(r"rtx-beta-migration-runtime-[a-z0-9-]+", value["run_id"]):
        raise ValueError("runtime check requires an isolated identity")
    single = value["schema"] == "roborender.beta-runtime-check.v2"
    if (type(value["windows"]) is not int or value["windows"] != 3
            or type(value["num_envs_per_gpu"]) is not int
            or value["num_envs_per_gpu"] not in ((8,) if single else (4, 6))
            or value["physical_rtx_gpus"] != ([1] if single else [1, 2]) or value["teacher_beta"] != 0.75):
        raise ValueError("runtime check requires the frozen v1 pair or v2 single-GPU B8 topology, three windows, beta=.75")
    if type(value["seed"]) is not int or not 0 <= value["seed"] < 2**31 - 1:
        raise ValueError("invalid simulator seed")
    pids = value["gateway_pids"]
    count = 1 if single else 4
    if len(pids) != count or len(set(pids)) != count or any(type(x) is not int or x <= 0 for x in pids):
        raise ValueError("live gateway count differs from the frozen qualification topology")
    for key in ("source_sha256", "initial_checkpoint_sha256"):
        require_sha256(value[key], key)
    if (not value["wandb_run_name"].startswith("smoke-dagger-")
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value["wandb_run_id"])):
        raise ValueError("isolated smoke W&B identity required")
    if value["wandb_mode"] not in {"online", "offline"}:
        raise ValueError("runtime-check W&B must be explicitly online or offline")
    if value["ack_loss_probe"] != {"gpu": 1, "window_index": 1, "step_index": 5}:
        raise ValueError("only the single registered post-commit response-loss probe is permitted")
    return value


def read_config(path: Path, expected_sha: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("runtime-check configuration digest changed")
    return validate_config(json.loads(raw))


def actor_ids(config: dict) -> tuple[str, ...]:
    return tuple(f"{config['run_id']}-g{gpu}-lane-{lane:03d}"
                 for gpu in config["physical_rtx_gpus"] for lane in range(config["num_envs_per_gpu"]))


class CommitAckLossProbe:
    """Drop one real committed response and close sockets, then allow retry.

    This is an explicit test double around transport delivery, NOT fake data
    or a fake backend. No request, response, action or receipt is modified.
    """
    def __init__(self, client, evidence: Path):
        self.client, self.evidence = client, evidence
        self.injected = False

    def __getattr__(self, key):
        return getattr(self.client, key)

    def commit_batch(self, selections, local_commits, idempotency_keys):
        response = self.client.commit_batch(selections, local_commits, idempotency_keys)
        key = selections[0]["step_key"]
        if (not self.injected and key["rollout_window_id"] == "window-000001"
                and key["rollout_step_index"] == 5):
            self.injected = True
            atomic_write_json(self.evidence, {
                "injected": True, "kind": "discard_committed_response_then_close_sockets",
                "step_key": key, "idempotency_keys": list(idempotency_keys),
                "remote_receipts": response, "native_step_replay_allowed": False,
            })
            # Transport.close resets only its HTTP connection; the sharded
            # executor remains open and can reconnect on the identical retry.
            for gateway in getattr(self.client, 'gateways', (self.client,)):
                gateway.transport.close()
            raise TransientTransportError("registered runtime-check lost commit acknowledgment")
        return response


class QualificationDriver:
    state_filename = "qualification-state.json"

    def normalize_config(self, config, config_sha):
        return validate_config(config)

    def __init__(self, runner, request, root: Path, config: dict, config_sha: str,
                 gpu: int, *, timeout_s=600, sleep=time.sleep, monotonic=time.monotonic):
        self.runner, self.request, self.root = runner, request, Path(root)
        config = self.normalize_config(config, config_sha)
        self.config, self.sha, self.gpu = config, config_sha, gpu
        self.round_id = config.get("round_id", 1)
        self.windows = config["windows"]
        if (gpu not in config["physical_rtx_gpus"] or runner.num_envs != config["num_envs_per_gpu"]
                or runner.actor_prefix != f"{config['run_id']}-g{gpu}"):
            raise ValueError("runtime-check actor identity mismatch")
        self.path = self.root / self.state_filename
        if self.path.exists():
            raise RuntimeError("native runtime-check restart forbidden; simulator state is not restored")
        self.timeout, self.sleep, self.monotonic = timeout_s, sleep, monotonic
        self.state = {"run_id": config["run_id"], "round_id": self.round_id, "phase": "CREATED", "windows": [],
                      "lifecycle_initializations": 0, "physical_gpu": gpu,
                      "config_sha256": config_sha, "uncertain_reason": None}

    def save(self, phase, **updates):
        self.state.update(phase=phase, **updates)
        atomic_write_json(self.path, self.state)
        print(json.dumps({"runtime_check": phase, "gpu": self.gpu, **updates}), flush=True)

    def status(self):
        value = self.request("GET", f"/v1/status/{self.config['run_id']}")
        run = value.get("run", {})
        if (not value.get("ok") or run.get("config_sha256") != self.config.get("coordinator_config_sha256", self.sha)
                or tuple(run.get("actor_ids", ())) != actor_ids(self.config)
                or run.get("current_round") != self.round_id or run.get("beta") != self.config["teacher_beta"]
                or run.get("rollout_steps") != 81
                or run.get("validation_interval_iterations") != self.windows):
            raise RuntimeError("coordinator runtime-check identity mismatch")
        if run.get("status") == "learner_failed":
            raise RuntimeError("runtime-check learner failed")
        return value

    def wait_candidate(self, index, parent):
        deadline = self.monotonic() + self.timeout
        while self.monotonic() < deadline:
            status = self.status()
            row = next((r for r in status["iterations"] if r["round_id"] == self.round_id
                        and r["iteration_id"] == index + 1), None)
            if row and row.get("candidate_checkpoint_sha256"):
                candidate = require_sha256(row["candidate_checkpoint_sha256"], "candidate")
                barriers = [r for r in status["barriers"] if r["round_id"] == self.round_id
                            and r["iteration_id"] == index + 1]
                if (row["input_checkpoint_sha256"] != parent or candidate == parent
                        or len(barriers) != len(actor_ids(self.config))
                        or {b["actor_id"] for b in barriers} != set(actor_ids(self.config))
                        or any(b["checkpoint_sha256"] != parent or b["chunk_ids"] != [
                            f"d{self.round_id}-{b['actor_id']}-window-{index:06d}"] for b in barriers)):
                    raise RuntimeError("learner candidate/barrier lineage mismatch")
                if index == self.windows - 1:
                    if row["status"] not in {"awaiting_validation", "validating"} or status["run"]["status"] != row["status"]:
                        raise RuntimeError("final qualification budget did not stop")
                else:
                    next_row = next((r for r in status["iterations"] if r["round_id"] == self.round_id
                                     and r["iteration_id"] == index + 2), {})
                    if (row["status"] != "advanced" or next_row.get("status") != "collecting"
                            or next_row.get("input_checkpoint_sha256") != candidate
                            or status["run"]["active_checkpoint_sha256"] != candidate):
                        raise RuntimeError("new checkpoint not active before next native window")
                return candidate
            self.sleep(1)
        raise TimeoutError("runtime-check checkpoint barrier timed out")

    def should_finish(self, index, candidate):
        return index == self.windows - 1

    def completion_metadata(self):
        return {}

    def run(self):
        parent = self.config["initial_checkpoint_sha256"]
        try:
            initial = self.status()
            if (len([r for r in initial["iterations"] if r["round_id"] == self.round_id]) != 1
                    or any(r["round_id"] == self.round_id for r in initial["barriers"])
                    or initial["run"]["status"] != "collecting"
                    or initial["run"]["active_checkpoint_sha256"] != parent):
                raise RuntimeError("runtime check is not fresh")
            self.runner.lifecycle.begin()
            self.save("READY", lifecycle_initializations=1)
            for index in range(self.windows):
                if self.runner.phase_telemetry is not None:
                    self.runner.phase_telemetry.set_vector_step_offset(index * 81)
                self.save("COLLECTING", window_index=index, input_checkpoint_sha256=parent)
                result = self.runner.run_window(index, manage_lifecycle=False)
                from .process_memory import compact_host_heap
                compact_host_heap("qualification_window_sealed")
                if (result["transition_batches"] != 81
                        or result["physical_transition_rows"] != 81 * self.config["num_envs_per_gpu"]):
                    raise RuntimeError("incomplete runtime-check window")
                for lane in range(self.config["num_envs_per_gpu"]):
                    actor = f"{self.runner.actor_prefix}-lane-{lane:03d}"
                    ack = self.request("POST", "/v1/barriers", {
                        "run_id": self.config["run_id"], "round_id": self.round_id, "actor_id": actor,
                        "checkpoint_sha256": parent, "chunk_ids": [f"d{self.round_id}-{actor}-window-{index:06d}"]})
                    if (not ack.get("ok") or ack.get("iteration_id") != index + 1
                            or ack.get("expected") != len(actor_ids(self.config))):
                        raise RuntimeError("actor barrier rejected")
                self.save("WAITING_FOR_CHECKPOINT", window_index=index)
                candidate = self.wait_candidate(index, parent)
                self.state["windows"].append({"index": index, "input_checkpoint_sha256": parent,
                                               "output_checkpoint_sha256": candidate})
                parent = candidate
                if self.should_finish(index, candidate):
                    break
            completed = len(self.state["windows"])
            self.save("COMPLETE", transition_batches=81 * completed,
                      physical_transition_rows=81 * completed * self.config["num_envs_per_gpu"],
                      final_candidate_checkpoint_sha256=parent, **self.completion_metadata())
            return dict(self.state)
        except BaseException as exc:
            self.save("CRASH_UNCERTAIN", uncertain_reason=str(exc)[:512])
            raise
        finally:
            self.runner.lifecycle.finish()


class BetaRoundDriver(QualificationDriver):
    """Same exactly-once lifecycle; explicit production contract, no fault injection."""
    state_filename = "beta-round-state.json"

    def normalize_config(self, config, config_sha):
        from .beta_round import driver_config
        return driver_config(config, config_sha)
