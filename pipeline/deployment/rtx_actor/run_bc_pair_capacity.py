"""Isolated Leo B2+B2 full-window capacity qualification, never production.

The adapter and child launcher come from an independently frozen source. This
host-only wrapper adds aggregate admission and samples both owned cgroups. It
does not change native collection, reset, writer, or optimization semantics.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roborender.infrastructure.source_locations import resolve_source_release


SCHEMA = "roborender.bc-pair-capacity.v1"
MEMORY_GIB = 23
ADMISSION_GIB = 2 * MEMORY_GIB + 6
RUNTIME_FLOOR_GIB = 8


def validate_spec(value):
    if (not isinstance(value, dict) or set(value) != {"schema", "pair_id", "source_root",
            "source_manifest", "source_sha256", "runtime_root", "actors", "timeout_seconds"}
            or value["schema"] != SCHEMA
            or not re.fullmatch(r"bc-pair-leo-[a-z0-9-]{1,64}", str(value["pair_id"]))
            or not re.fullmatch(r"[0-9a-f]{64}", str(value["source_sha256"]))
            or type(value["timeout_seconds"]) is not int or not 300 <= value["timeout_seconds"] <= 1800
            or not isinstance(value["actors"], list) or len(value["actors"]) != 2):
        raise ValueError("invalid bounded BC pair qualification")
    for gpu, row in enumerate(value["actors"]):
        if (not isinstance(row, dict) or set(row) != {"physical_gpu", "session_file", "session_sha256", "requests_dir"}
                or type(row["physical_gpu"]) is not int or row["physical_gpu"] != gpu
                or not re.fullmatch(r"[0-9a-f]{64}", str(row["session_sha256"]))):
            raise ValueError("pair must bind exactly Leo GPU0 and GPU1")
    if len({row["session_sha256"] for row in value["actors"]}) != 2:
        raise ValueError("independent actor session identities are required")
    return value


def admission(memory):
    if memory["MemAvailable"] < ADMISSION_GIB * 1024**2:
        raise RuntimeError("BC pair requires aggregate 46 GiB caps plus 6 GiB admission cushion")


def safety(initial, current, samples, memory_safety):
    if current["MemAvailable"] < RUNTIME_FLOOR_GIB * 1024**2:
        raise RuntimeError("BC pair crossed its 8 GiB runtime RAM floor")
    for sample in samples:
        if sample.get("events", {}).get("oom", 0):
            raise RuntimeError("owned BC pair cgroup encountered OOM pressure")
    return memory_safety(initial, current, samples)


def verify_requests(session, requests):
    if (len(requests) != 1 or len(session["requests"]) != 1
            or len(requests[0]["motions"]) != 2 or requests[0]["chunks_per_env"] != [1, 1]
            or requests[0]["max_windows"] != 4):
        raise ValueError("capacity scope must remain one B2 cohort, quota1/lane, max4 windows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    raw = args.spec.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.spec_sha256:
        raise ValueError("pair specification identity changed")
    spec = validate_spec(json.loads(raw))
    source = resolve_source_release(spec["source_root"], error="pair must use frozen native source")
    sys.path.insert(0, str(source / "pipeline/deployment/rtx_actor"))
    from qualify_leo import source_gate
    from run_bc_session import read_requests
    from run_success_diagnostic import gpu_gate
    from run_throughput_pair import memory_kib, memory_safety, owned_memory_sample
    from qualify_graphics import cleanup_owned
    from isaac6_bc_production import publish_json
    if source_gate(source, Path(spec["source_manifest"])) != spec["source_sha256"]:
        raise ValueError("pair frozen source identity differs")
    runtime = Path(spec["runtime_root"]).resolve()
    if runtime.exists() or not runtime.is_relative_to("/home/ldu/roborender/runtime/bc-sessions"):
        raise ValueError("fresh scoped pair runtime required")
    actors, seeds, ids = [], set(), set()
    for row in spec["actors"]:
        raw_session = Path(row["session_file"]).read_bytes()
        if hashlib.sha256(raw_session).hexdigest() != row["session_sha256"]:
            raise ValueError("actor session identity changed")
        session, requests = read_requests(json.loads(raw_session), row["requests_dir"])
        verify_requests(session, requests)
        seeds.add(requests[0]["process_seed"])
        ids.add(session["session_id"])
        child = runtime / f"gpu{row['physical_gpu']}"
        name = "bc-cohort-leo-" + session["session_id"].removeprefix("bc-session-leo-")
        command = [sys.executable, str(source / "pipeline/deployment/rtx_actor/run_bc_session.py"),
            "--session", row["session_file"], "--session-sha256", row["session_sha256"],
            "--requests-dir", row["requests_dir"], "--source-root", str(source),
            "--source-manifest", spec["source_manifest"], "--runtime", str(child),
            "--physical-gpu", str(row["physical_gpu"]), "--memory-gib", str(MEMORY_GIB),
            "--timeout", str(spec["timeout_seconds"]), "--execute"]
        actors.append({"gpu": row["physical_gpu"], "container": name, "runtime": child, "command": command})
    if len(ids) != 2 or len(seeds) != 2:
        raise ValueError("pair requires distinct fresh sessions and simulator seeds")
    plan = {"schema": SCHEMA, "pair_id": spec["pair_id"], "source_sha256": spec["source_sha256"],
        "spec_sha256": args.spec_sha256, "per_actor_cap_gib": MEMORY_GIB,
        "aggregate_admission_gib": ADMISSION_GIB, "runtime_floor_gib": RUNTIME_FLOOR_GIB,
        "num_envs_per_gpu": 2, "max_windows_per_gpu": 4, "quota_per_lane": 1,
        "production_admitted": False, "H200_used": False, "optimizer_steps": 0,
        "actors": [{**a, "runtime": str(a["runtime"])} for a in actors]}
    if not args.execute:
        print(json.dumps(plan), flush=True)
        return
    if socket.gethostname() != "leo-rtx-6k":
        raise RuntimeError("BC pair is Leo-only")
    for actor in actors:
        gpu_gate(actor["gpu"], deployment_host="leo-rtx-6k")
    initial = memory_kib()
    admission(initial)
    runtime.mkdir(parents=True)
    publish_json(runtime / "pair-launch.json", plan)
    processes, failure, samples = [], None, []
    minimum = initial["MemAvailable"]
    started = time.monotonic()

    def interrupted(signum, frame):
        raise InterruptedError("owned BC pair stop requested")

    signal.signal(signal.SIGTERM, interrupted)
    with ExitStack() as stack:
        try:
            for actor in actors:
                log = stack.enter_context((runtime / f"gpu{actor['gpu']}-host.log").open("xb"))
                processes.append(subprocess.Popen(actor["command"], stdout=log, stderr=subprocess.STDOUT))
            trace = stack.enter_context((runtime / "memory.jsonl").open("x"))
            while True:
                current = memory_kib()
                minimum = min(minimum, current["MemAvailable"])
                samples = [owned_memory_sample(a["container"], p, a["runtime"] / "host-result.json",
                    completion_field="passed", completion_value=True) for a, p in zip(actors, processes)]
                attribution = safety(initial, current, samples, memory_safety)
                trace.write(json.dumps({"elapsed_seconds": time.monotonic() - started,
                    "host": current, "owned": samples, "attribution": attribution}, sort_keys=True) + "\n")
                trace.flush()
                if all(p.poll() is not None for p in processes):
                    break
                if time.monotonic() - started > spec["timeout_seconds"] + 30:
                    raise TimeoutError("bounded BC pair qualification timeout")
                time.sleep(3)
        except BaseException as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for actor in actors:
                cleanup_owned(actor["container"])
            for process in processes:
                if process.poll() is None:
                    process.wait(timeout=35)
    results = []
    for actor in actors:
        path = actor["runtime"] / "host-result.json"
        native = actor["runtime"] / "bc-session-result.json"
        results.append({"physical_gpu": actor["gpu"],
            "host": json.loads(path.read_bytes()) if path.is_file() else {},
            "native": json.loads(native.read_bytes()) if native.is_file() else {}})
    passed = failure is None and len(processes) == 2 and all(p.returncode == 0 for p in processes)
    report = {"schema": SCHEMA, "pair_id": spec["pair_id"], "passed": passed, "failure": failure,
        "elapsed_seconds": time.monotonic() - started, "minimum_host_available_kib": minimum,
        "per_actor_cap_gib": MEMORY_GIB, "runtime_floor_gib": RUNTIME_FLOOR_GIB,
        "actors": results, "production_admitted": False}
    publish_json(runtime / "pair-result.json", report)
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
