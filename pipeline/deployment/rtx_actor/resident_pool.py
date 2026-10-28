"""One qualified standing owner per Leo GPU; sequential phase admission."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

from leo_binding import GPU_UUIDS
from qualify_graphics import LABEL
from resident_isaac_client import ResidentClient, canonical
from run_throughput_pair import container_memory_sample, memory_kib
from resident_memory_admission import GIB, admit_future_caps

SCHEMA = "roborender.leo-resident-pool.v1"


def validate_pool(pool):
    if set(pool) != {"schema", "source_sha256", "jobs_root", "assets_root", "workers", "qualification"} or pool["schema"] != SCHEMA:
        raise ValueError("explicit qualified resident pool required")
    if len(pool["workers"]) != 2:
        raise ValueError("one standing worker on each of two Leo RTX devices required")
    for gpu, row in enumerate(pool["workers"]):
        if set(row) != {"physical_gpu", "gpu_uuid", "port", "token_file", "runtime_root", "container_name",
                        "worker_instance_id", "memory_gib", "evaluation_memory_gib"}:
            raise ValueError("unknown resident binding fields")
        if (row["physical_gpu"] != gpu or row["gpu_uuid"] != GPU_UUIDS[gpu] or row["port"] != 18640+gpu
                or row["memory_gib"] != 21 or row["evaluation_memory_gib"] != 28):
            raise ValueError("resident host/GPU/memory topology differs")
        for key in ("token_file", "runtime_root"):
            if not Path(row[key]).is_absolute():
                raise ValueError("resident paths must be explicit")
    qualified = pool["qualification"]
    if set(qualified) != {"path", "sha256"}:
        raise ValueError("native qualification receipt required")
    return pool


def clients(pool):
    validate_pool(pool)
    path = Path(pool["qualification"]["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != pool["qualification"]["sha256"]:
        raise RuntimeError("resident native qualification identity differs")
    receipt = json.loads(path.read_bytes())
    if (receipt.get("complete") is not True or receipt.get("source_sha256") != pool["source_sha256"]
            or receipt.get("profiles") != ["training_b6", "evaluation_b5", "training_b6"]
            or receipt.get("native_initializations_per_worker") != 1
            or receipt.get("paired_initial_states") is not True):
        raise RuntimeError("resident pool is not natively qualified")
    values = []
    for row in pool["workers"]:
        client = ResidentClient(row["port"], row["token_file"], source_sha256=pool["source_sha256"], gpu_uuid=row["gpu_uuid"])
        client.instance = row["worker_instance_id"]
        client.ready()
        values.append(client)
    return values


def quiescent(pool):
    values = clients(pool)
    return all(client.ready()["phase"] == "IDLE" for client in values)


def evaluation_memory(pool, enable):
    """Raise only one owned cap, with an aggregate host-headroom check."""
    if not quiescent(pool):
        raise RuntimeError("memory change requires both native owners idle")
    current = [container_memory_sample(row["container_name"]) for row in pool["workers"]]
    # MemAvailable already includes reclaimable file cache. Do not credit
    # memory.current again: only attributed anon/shmem can offset future caps.
    limits = [28 if enable else 21, 21]
    admission = admit_future_caps(memory_kib()["MemAvailable"] * 1024,
                                  current, [limit * GIB for limit in limits])
    if not admission["admitted"]:
        raise RuntimeError("aggregate resident evaluation RAM admission failed")
    row = pool["workers"][0]
    info = json.loads(subprocess.check_output(["docker", "inspect", row["container_name"]]))[0]
    if info["Config"].get("Labels", {}).get(LABEL) != row["container_name"]:
        raise RuntimeError("resident container owner mismatch")
    limit = 28 if enable else 21
    subprocess.run(["docker", "update", "--memory", f"{limit}g", "--memory-swap", f"{limit}g", row["container_name"]],
                   check=True, stdout=subprocess.DEVNULL, timeout=15)
    return admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--action", choices=["status", "submit", "wait", "evaluation-memory", "training-memory", "shutdown", "abort"], required=True)
    parser.add_argument("--commands", type=Path)
    args = parser.parse_args()
    pool = validate_pool(json.loads(args.pool.read_bytes()))
    if args.action == "abort":
        commands = json.loads(args.commands.read_bytes())
        expected = {row["gpu"]: row["command_id"] for row in commands}
        if set(expected) != {0, 1}:
            raise ValueError("abort must bind both exact owners")
        from qualify_graphics import cleanup_owned
        for gpu, worker in enumerate(pool["workers"]):
            state = json.loads((Path(worker["runtime_root"])/"resident-state.json").read_bytes())
            if state["worker_instance_id"] != worker["worker_instance_id"]:
                raise RuntimeError("abort owner generation changed")
            if state["phase"] not in {"IDLE", "STOPPED"}:
                if not state.get("active_command_id", "").startswith(expected[gpu]+"-r"):
                    raise RuntimeError("cannot stop another campaign's resident command")
                cleanup_owned(worker["container_name"])
        print(json.dumps({"complete": True, "native_restart_allowed": False}))
        return
    values = clients(pool)
    if args.action == "status":
        output = {"workers": [client.ready() for client in values], "quiescent": quiescent(pool)}
    elif args.action in {"evaluation-memory", "training-memory"}:
        admission = evaluation_memory(pool, args.action == "evaluation-memory")
        output = {"complete": True, "evaluation_memory": args.action == "evaluation-memory",
                  "memory_admission": admission}
    else:
        commands = json.loads(args.commands.read_bytes()) if args.commands else []
        if args.action == "shutdown":
            commands = [{"gpu": gpu, "command_id": "explicit-shutdown", "operation": "shutdown", "payload": {}}
                        for gpu in range(2)]
        if not commands or len({row["gpu"] for row in commands}) != len(commands) or any(type(row["gpu"]) is not int or row["gpu"] not in (0, 1) for row in commands):
            raise ValueError("one command per selected GPU is required")
        def action(row):
            client = values[row["gpu"]]
            if args.action == "wait":
                return client.wait(row["command_id"])
            return client.submit(row["command_id"], row["operation"], row["payload"])
        with ThreadPoolExecutor(len(commands)) as executor:
            output = {"results": list(executor.map(action, commands))}
    print(json.dumps(output, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
