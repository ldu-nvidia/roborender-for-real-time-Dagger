"""One genuine 81-step held-out window per actor, with no learner handoff."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .beta_round import canonical
from .contracts import require_sha256
from .durable import atomic_write_json

SCHEMA = "roborender.dagger-heldout-actor.v1"


def read_actor(path, raw_sha):
    require_sha256(raw_sha, "held-out actor sidecar")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != raw_sha:
        raise ValueError("held-out actor sidecar changed")
    value = json.loads(raw)
    fields = {"schema", "registration", "registration_sha256", "actor_prefix", "num_envs", "physical_gpu_index",
              "gpu_uuid", "simulator_seed", "motion_assignment_path", "motion_assignment_sha256", "gateway_pids", "source_sha256"}
    if set(value) != fields or value["schema"] != SCHEMA:
        raise ValueError("invalid held-out actor fields/schema")
    registration = value["registration"]
    required = {"schema", "purpose", "run_id", "round_id", "actor_ids", "initial_checkpoint_sha256",
                "teacher_checkpoint_sha256", "policy_spec_sha256", "motion_inventory_sha256", "teacher_beta", "collection_plan_sha256"}
    if (set(registration) != required or registration["schema"] != "roborender.dagger-validation-collection.v1"
            or registration["purpose"] != "validation" or not registration["run_id"].startswith("heldout-")
            or type(registration["round_id"]) is not int or registration["round_id"] not in (1, 2, 3, 4)
            or registration["teacher_beta"] != {1: .75, 2: .5, 3: .25, 4: 0.}[registration["round_id"]]
            or hashlib.sha256(canonical(registration)).hexdigest() != value["registration_sha256"]):
        raise ValueError("held-out collection registration identity differs")
    for key in ("initial_checkpoint_sha256", "teacher_checkpoint_sha256", "policy_spec_sha256", "motion_inventory_sha256", "collection_plan_sha256"):
        require_sha256(registration[key], key)
    # Metadata bound only; physical launch admission and native phase memory
    # guards remain mandatory. Scaled plans cap validation at B16 per actor.
    if type(value["num_envs"]) is not int or not 1 <= value["num_envs"] <= 16:
        raise ValueError("held-out B must stay within the qualified B6 ceiling")
    from .beta_round import gpu_uuids, LEO_SCHEMA
    if gpu_uuids({"schema": LEO_SCHEMA}).get(value["physical_gpu_index"]) != value["gpu_uuid"]:
        raise ValueError("held-out phase must bind one exact owned Leo GPU")
    if type(value["simulator_seed"]) is not int or not 0 <= value["simulator_seed"] < 2**31:
        raise ValueError("held-out phase requires an explicit simulator seed")
    if not value["actor_prefix"].startswith(registration["run_id"] + "-"):
        raise ValueError("held-out actor namespace differs")
    actors = [f"{value['actor_prefix']}-lane-{lane:03d}" for lane in range(value["num_envs"])]
    if not set(actors) <= set(registration["actor_ids"]):
        raise ValueError("held-out actor lanes are not admitted")
    for key in ("motion_assignment_sha256", "source_sha256"):
        require_sha256(value[key], key)
    if not Path(value["motion_assignment_path"]).is_absolute():
        raise ValueError("held-out native motion assignment path must be absolute")
    pids = value["gateway_pids"]
    if len(pids) not in (1, 2, 4) or len(pids) != len(set(pids)) or any(type(pid) is not int or pid < 1 for pid in pids):
        raise ValueError("bounded held-out phase needs genuine live gateway PIDs")
    return value


def verify_registration(spec, request):
    response = request("GET", "/v1/validation-collections/" + spec["registration"]["run_id"])
    collection = response.get("collection", {})
    if (not response.get("ok") or collection.get("registration") != spec["registration"]
            or collection.get("registration_sha256") != spec["registration_sha256"]
            or collection.get("status") != "collecting" or collection.get("optimizer_authorized") is not False):
        raise RuntimeError("held-out collection owner/parent/purpose differs")
    actors = {f"{spec['actor_prefix']}-lane-{lane:03d}" for lane in range(spec["num_envs"])}
    if actors & {row["actor_id"] for row in collection["actors"]}:
        raise RuntimeError("completed native held-out actors must not be replayed")
    return collection


class HeldoutCollectionDriver:
    def __init__(self, runner, request, root, spec, sidecar_sha):
        self.runner, self.request, self.spec = runner, request, spec
        self.path = Path(root) / "heldout-collection-state.json"
        if self.path.exists():
            raise RuntimeError("native held-out lifecycle restart forbidden")
        if runner.num_envs != spec["num_envs"] or runner.actor_prefix != spec["actor_prefix"]:
            raise ValueError("held-out runner topology differs")
        self.state = {"schema": "roborender.heldout-actor-completion.v1", "run_id": spec["registration"]["run_id"],
            "round_id": spec["registration"]["round_id"], "actor_prefix": spec["actor_prefix"],
            "registration_sha256": spec["registration_sha256"], "actor_sidecar_sha256": sidecar_sha,
            "purpose": "validation", "optimizer_steps": 0, "phase": "CREATED"}

    def run(self):
        try:
            verify_registration(self.spec, self.request)
            self.state["phase"] = "COLLECTING"
            atomic_write_json(self.path, self.state)
            result = self.runner.run_window(0)
            if result["transition_batches"] != 81 or result["physical_transition_rows"] != 81 * self.spec["num_envs"]:
                raise RuntimeError("held-out collection did not finish exactly one complete window")
            chunks = []
            for lane in range(self.spec["num_envs"]):
                actor = f"{self.spec['actor_prefix']}-lane-{lane:03d}"
                chunk = f"d{self.state['round_id']}-{actor}-window-000000"
                ack = self.request("POST", "/v1/validation-collections/finish-actor", {
                    "run_id": self.state["run_id"], "actor_id": actor, "chunk_id": chunk,
                    "registration_sha256": self.spec["registration_sha256"]})
                if not ack.get("ok") or ack.get("actor_id") != actor or ack.get("chunk_id") != chunk or ack.get("optimizer_authorized") is not False:
                    raise RuntimeError("held-out completion acknowledgement differs")
                chunks.append(chunk)
            self.state.update(phase="COMPLETE", chunk_ids=chunks, transition_batches=81,
                              physical_transition_rows=81 * self.spec["num_envs"], native_restart_allowed=False)
            atomic_write_json(self.path, self.state)
            return dict(self.state)
        except BaseException as exc:
            self.state.update(phase="CRASH_UNCERTAIN", uncertain_reason=type(exc).__name__, native_restart_allowed=False)
            atomic_write_json(self.path, self.state)
            raise
