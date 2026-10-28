"""Bounded, identity-pinned RTX trial; never promotes checkpoints.

Caps each actor's host RAM, forbids swap, and abandons both scoped containers
if host headroom falls below 6 GiB (12 GiB in conservative Leo probes).
Leo uses physical GPU0+GPU1; the legacy Beta layout is GPU1+GPU2.
All incomplete rollout data is diagnostic. No Cosmos allocation is released.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roborender.infrastructure.source_locations import resolve_native_code

from materialize_isaac6 import materialize
from qualify_native_isaac6 import (ANCHORS, BETA_BENCHMARK_GPUS, HOSTNAME, IMAGE,
                                  LABEL, RAVEN, REPO, command, cleanup_owned)
from native_thp_policy import INHERITED, DISABLED, validate_policy


def production_capacity_options(*,policy,mode,host,gpus,steps,memory_gib,compact,conservative):
    """Opt-in only; preserve historical probes and require production guards."""
    validate_policy(policy)
    if policy == DISABLED and (mode != 'teacher-capacity' or host != 'leo-rtx-6k'
            or gpus != [0,1] or steps != 10 or memory_gib != 21 or not compact or not conservative):
        raise ValueError('disable-v1 requires paired Leo ten-step teacher probe with 21GiB and conservative guards')


def attest_capacity_scope(sample):
    """Same read-only systemd creation-policy check as production owners."""
    from run_production_resident_isaac import attest_systemd_scope
    if type(sample.get('pid')) is not int or sample['pid'] <= 0:
        raise RuntimeError('actual capacity native PID required for scope attestation')
    entry = next(line.split(':',2)[2] for line in Path(f'/proc/{sample["pid"]}/cgroup').read_text().splitlines()
                 if line.startswith('0::'))
    return attest_systemd_scope(Path('/sys/fs/cgroup') / entry.lstrip('/'))


def validate_capacity_native_memory(result):
    thp=result.get('native_thp_policy',{});memory=result.get('cgroup_memory',{})
    if (thp.get('schema')!='roborender.native-thp-policy.v1' or thp.get('policy')!=DISABLED
            or thp.get('disabled_after')!=1 or type(thp.get('native_pid'))is not int or thp['native_pid']<=0
            or thp.get('global_host_policy_modified')is not False
            or memory.get('memory.max')!=str(21*(1<<30))
            or memory.get('memory.swap.current')!='0' or memory.get('memory.swap.max')!='0'):
        raise RuntimeError('native capacity THP/zero-swap policy receipt mismatch')


def memory_kib():
    return {k: int(v.split()[0]) for k, v in
            (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}


def memory_safety(initial, current, owned_samples):
    """Attribute swap to owned cgroups; host-wide swap is diagnostic only."""
    if current["MemAvailable"] < 6 * 1024**2:
        raise RuntimeError("host RAM safety floor reached; abandoning workload, no replay")
    for sample in owned_samples:
        if sample.get("oom_killed") or sample.get("events", {}).get("oom_kill", 0):
            raise RuntimeError("owned native container OOM; no replay")
        if sample.get("memory_swap.current", 0):
            raise RuntimeError("owned native container used forbidden swap")
    growth = max(0, initial["SwapFree"] - current["SwapFree"])
    if growth > 128 * 1024:
        # Missing counters cannot attest that an apparent swap change is
        # outside the owned workload. Fresh samples are required for this case.
        if (not owned_samples or any(
                (sample.get("sampling_error") or "memory_swap.current" not in sample)
                and not (sample.get("lifecycle") == "completed_attested"
                         and sample.get("exit_code") == 0
                         and sample.get("native_pass_receipt") is True
                         and isinstance(sample.get("completion_receipt_sha256"), str)
                         and len(sample["completion_receipt_sha256"]) == 64)
                for sample in owned_samples)):
            raise RuntimeError("host swap changed and owned memory attribution is unavailable")
    return {"host_swap_growth_kib": growth, "owned_swap_bytes": 0,
            "host_swap_outside_owned_cgroups": growth > 128 * 1024}


def validate_layout(gpus, memory_gib, fanout, deployment_host=HOSTNAME):
    if deployment_host == "leo-rtx-6k":
        allowed = ([0], [0, 1])
    elif deployment_host == HOSTNAME:
        allowed = ([1], [1, 2])
    else:
        raise ValueError("unknown RTX host")
    if gpus not in allowed:
        raise ValueError("GPU layout differs from the explicitly bound host")
    if ((len(gpus) == 1 and (memory_gib not in (23, 32, 40) or fanout != 1))
            or (len(gpus) == 2 and (memory_gib not in (20, 21, 22, 23, 24) or fanout not in (2, 4)))):
        raise ValueError("unsafe RAM budget or unsupported gateway topology")
    return len(gpus) * memory_gib + 6


def conservative_capacity_safety(initial, current, samples, gpu_rows, memory_gib):
    """Stop a diagnostic well before its caps; never search by provoking OOM."""
    memory_safety(initial, current, samples)
    if current["MemAvailable"] < 12 * 1024**2:
        raise RuntimeError("capacity probe reached its 12 GiB host reserve")
    for sample in samples:
        if sample.get("lifecycle") == "completed_attested":
            continue
        stat = sample.get("stat", {})
        if sample.get("sampling_error") or not {"anon", "shmem", "kernel"} <= set(stat):
            raise RuntimeError("capacity probe lacks fresh owned memory attribution")
        if sample.get("events", {}).get("oom", 0):
            raise RuntimeError("capacity probe observed owned OOM pressure")
        committed = sum(stat[key] for key in ("anon", "shmem", "kernel"))
        if committed >= (memory_gib - 4) * 1024**3:
            raise RuntimeError("capacity probe reached its 4 GiB private-memory margin")
    if not gpu_rows:
        raise RuntimeError("capacity probe lacks GPU memory samples")
    for row in gpu_rows:
        _index, used, total, _util, _power = [float(x.strip()) for x in row.split(",")]
        if total <= 0 or used >= .75 * total:
            raise RuntimeError("capacity probe reached 75 percent GPU memory ceiling")


def container_memory_sample(name):
    """Small cgroup counters distinguish anonymous, file-cache and kernel RAM."""
    try:
        info = json.loads(subprocess.check_output(["docker", "inspect", name],
                                                 stderr=subprocess.PIPE, timeout=3))[0]
        if info["Config"].get("Labels", {}).get(LABEL) != name:
            raise RuntimeError("resource sampler owner mismatch")
        pid = info["State"]["Pid"]
        result = {"container": name, "pid": pid, "oom_killed": info["State"].get("OOMKilled", False)}
        if not pid:
            return result
        entry = next(line.split(":", 2)[2] for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
                     if line.startswith("0::"))
        cgroup = Path("/sys/fs/cgroup") / entry.lstrip("/")
        for field in ("current", "peak", "swap.current"):
            result["memory_" + field] = int((cgroup / ("memory." + field)).read_text())
        for field in ("stat", "events"):
            values = dict(line.split() for line in (cgroup / ("memory." + field)).read_text().splitlines())
            result[field] = {k: int(v) for k, v in values.items()
                             if field == "events" or k in {"anon", "file", "shmem", "kernel", "file_dirty", "file_writeback"}}
        return result
    except (OSError, ValueError, KeyError, StopIteration, subprocess.SubprocessError) as exc:
        return {"container": name, "sampling_error": str(exc)[:180]}


def native_completion(process, receipt, *, completion_field="result", completion_value="PASS",
                      receipt_max_bytes=1024 * 1024):
    """An exited native owner needs both code zero and its durable PASS file."""
    if type(receipt_max_bytes) is not int or not 0 < receipt_max_bytes <= 32 * 1024**2:
        raise ValueError("native receipt bound must be explicit and at most32MiB")
    code = process.poll()
    if code is None:
        return None
    if code != 0:
        raise RuntimeError("one native actor exited unsuccessfully")
    receipt = Path(receipt)
    if receipt.is_symlink() or not receipt.is_file() or receipt.stat().st_size > receipt_max_bytes:
        raise RuntimeError("native actor exited without a bounded PASS receipt")
    raw = receipt.read_bytes()
    value = json.loads(raw).get(completion_field)
    if type(value) is not type(completion_value) or value != completion_value:
        raise RuntimeError("native actor exited without a complete PASS receipt")
    return {"lifecycle": "completed_attested", "exit_code": 0,
            "native_pass_receipt": True,
            "completion_receipt_sha256": hashlib.sha256(raw).hexdigest()}


def owned_memory_sample(name, process, receipt, *, completion_field="result", completion_value="PASS",
                        receipt_max_bytes=1024 * 1024):
    """Reconcile cgroup disappearance with this exact owner's native exit.

    Docker --rm can remove a cgroup between sampling and child.poll(). Never
    reuse stale counters or excuse a running/failed owner: only a durable PASS
    plus observed exit zero replaces the no-longer-existing cgroup.
    """
    options = {"completion_field": completion_field, "completion_value": completion_value,
               "receipt_max_bytes": receipt_max_bytes}
    complete = native_completion(process, receipt, **options)
    if complete is not None:
        return {"container": name, **complete}
    sample = container_memory_sample(name)
    if sample.get("sampling_error") or "memory_swap.current" not in sample:
        # The child can finish while docker inspect or /proc is being read.
        complete = native_completion(process, receipt, **options)
        if complete is None and Path(receipt).is_file():
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            complete = native_completion(process, receipt, **options)
        if complete is not None:
            return {"container": name, **complete}
    return sample


def main():
    def interrupted(signum, frame):
        raise InterruptedError("owned pair runner stop requested")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deployment-host", choices=[HOSTNAME, "leo-rtx-6k"], default=HOSTNAME)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, choices=[2, 4, 6, 8, 12, 16, 28, 32, 40, 64], required=True)
    parser.add_argument("--rtx-gpus", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--mode", choices=["capture-only", "teacher-capacity", "teacher-camera-soak", "recorded-action-camera-soak", "capacity-probe", "single-window", "multi-window-check", "beta-production"], default="capacity-probe")
    parser.add_argument("--capacity-steps", type=int, choices=[3, 10, 324], default=10)
    parser.add_argument("--warm-camera-plan", type=Path,
                        help="Isolated B16 soak: exact fresh/warm scene comparison, no production admission")
    parser.add_argument("--recorded-action-reference", type=Path,
                        help="Exclusive fresh diagnostic, never replay into production")
    parser.add_argument("--qualification-config", type=Path)
    parser.add_argument("--source-manifest", type=Path, help="Required for beta production; verified once before launch")
    parser.add_argument("--supervisor-config", type=Path)
    parser.add_argument("--supervisor-status-dir", type=Path)
    parser.add_argument("--gateway-pids", default="")
    parser.add_argument("--fanout", type=int, choices=[1, 2, 4], default=4)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--seed", type=int, default=610091)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--memory-gib", type=int, choices=[20, 21, 22, 23, 24, 32, 40], default=23,
                        help="Per-actor RAM cap; B6 exceeded 23 GiB in the sustained October 10 check")
    parser.add_argument("--compact-host-heap", action="store_true",
                        help="Opt-in two-arena CPU allocator and unused-heap release at lifecycle barriers")
    parser.add_argument("--conservative-capacity-guard", action="store_true",
                        help="Diagnostic-only: 12GiB host reserve, 4GiB actor margin, <=75%% VRAM")
    parser.add_argument('--native-thp-policy', choices=(INHERITED,DISABLED), default=INHERITED,
                        help='Opt-in paired teacher probe: match production disable-v1 and systemd zero-swap policy')
    args = parser.parse_args()
    production_capacity_options(policy=args.native_thp_policy,mode=args.mode,host=args.deployment_host,
        gpus=args.rtx_gpus,steps=args.capacity_steps,memory_gib=args.memory_gib,
        compact=args.compact_host_heap,conservative=args.conservative_capacity_guard)
    recorded = args.mode == "recorded-action-camera-soak"
    camera_soak = args.mode in {"teacher-camera-soak", "recorded-action-camera-soak"}
    teacher_probe = args.mode in {"teacher-capacity", "teacher-camera-soak", "recorded-action-camera-soak"}
    if recorded != (args.recorded_action_reference is not None) or (recorded and args.warm_camera_plan is None):
        raise ValueError("recorded reference requires exclusive isolated warm-scene diagnostic")
    if args.warm_camera_plan is not None:
        if not camera_soak or args.rtx_gpus != [0, 1]:
            raise ValueError("warm comparison requires paired camera soak")
        from warm_camera_diagnostic import read_plan
        warm_camera_sha = hashlib.sha256(args.warm_camera_plan.read_bytes()).hexdigest()
        warm_plan = read_plan(args.warm_camera_plan, warm_camera_sha)
        if recorded:
            from recorded_action_camera import read_reference
            from warm_camera_diagnostic import canonical
            recorded_sha = hashlib.sha256(args.recorded_action_reference.read_bytes()).hexdigest()
            read_reference(args.recorded_action_reference, recorded_sha,
                hashlib.sha256(canonical(warm_plan['arms']['0'][-1]['assignment'])).hexdigest())
    if camera_soak:
        if (args.num_envs != 16 or args.capacity_steps != 324
                or not args.conservative_capacity_guard):
            raise ValueError("camera soak requires conservative B16/324-step diagnostic")
    elif args.capacity_steps == 324:
        raise ValueError("324 steps are exclusive to the camera-soak diagnostic")
    if args.num_envs in (32, 64) and not (
            args.deployment_host == 'leo-rtx-6k' and args.mode == 'teacher-capacity'
            and args.conservative_capacity_guard):
        raise ValueError('32/64 probes require Leo teacher-only mode and conservative memory guards')
    required_available_gib = validate_layout(args.rtx_gpus, args.memory_gib, args.fanout, args.deployment_host)
    if args.conservative_capacity_guard:
        if args.deployment_host != "leo-rtx-6k" or args.mode not in {"teacher-capacity", "teacher-camera-soak", "recorded-action-camera-soak", "capture-only"}:
            raise ValueError("conservative capacity guard is Leo diagnostic-only")
        required_available_gib = len(args.rtx_gpus) * args.memory_gib + 12
    leo = args.deployment_host == "leo-rtx-6k"
    role = "leo" if leo else "beta"
    if leo:
        from leo_binding import GPU_UUIDS as bound_gpu_uuids
    else:
        bound_gpu_uuids = BETA_BENCHMARK_GPUS
    import re
    beta = args.mode == "beta-production"
    pattern = "dagger-" + role + r"-[a-z0-9-]{1,64}" if beta else "rtx-" + role + r"-migration-[a-z0-9-]+"
    if socket.gethostname() != args.deployment_host or not re.fullmatch(pattern, args.run_id):
        raise RuntimeError("wrong host/run scope")
    if not 300 <= args.timeout <= (21600 if beta else 3600 if camera_soak else 1800):
        raise RuntimeError("unbounded timeout or mutable source root")
    try:
        args.code_root = resolve_native_code(args.code_root)
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError("unbounded timeout or mutable source root") from error
    connected = args.mode not in {"capture-only", "teacher-capacity", "teacher-camera-soak", "recorded-action-camera-soak"}
    if teacher_probe and not leo:
        raise RuntimeError("teacher-capacity qualification is Leo-only")
    if teacher_probe:
        if args.source_manifest is None:
            raise RuntimeError("teacher-capacity qualification requires a frozen source manifest")
        from qualify_leo import source_gate
        source_gate(args.code_root.resolve().parents[3], args.source_manifest)
    pids = [int(x) for x in args.gateway_pids.split(",") if x]
    expected_gateways = 1 if len(args.rtx_gpus) == 1 else 4
    if connected and (len(pids) != expected_gateways or len(set(pids)) != expected_gateways or args.token_file is None
                      or not args.token_file.is_file() or args.token_file.stat().st_mode & 0o077):
        raise RuntimeError("the attested gateway topology and an owner-only token are required")
    if beta:
        import sys
        sys.path.insert(0, str(args.code_root.resolve() / "src"))
        from dagger_rtx.beta_round import validate_config, deployment_host as config_host, gpu_uuids
        if args.qualification_config is None or args.source_manifest is None:
            raise RuntimeError("beta production requires immutable config and source manifest")
        config = validate_config(json.loads(args.qualification_config.read_bytes()), runtime=True)
        if (args.rtx_gpus != list(gpu_uuids(config)) or config_host(config) != args.deployment_host
                or args.num_envs != 6 or args.fanout != 4
                or args.memory_gib != 21 or not args.compact_host_heap
                or config["run_id"] != args.run_id or config["seed"] != args.seed
                or config["h200"]["gateway_pids"] != pids):
            raise RuntimeError("beta production launcher/config mismatch")
        # Source-only cold-path check; never scan weights or rollout payloads.
        source_root = args.code_root.resolve().parents[3]
        sys.path.insert(0, str(source_root))
        from pipeline.deployment.source_snapshot import canonical, file_record
        manifest = json.loads(args.source_manifest.read_bytes())
        rows = [file_record(source_root, row["path"]) for row in manifest["files"]]
        if (rows != manifest["files"] or hashlib.sha256(canonical(rows)).hexdigest() != config["source_sha256"]
                or manifest["source_sha256"] != config["source_sha256"]):
            raise RuntimeError("beta frozen source/config digest mismatch")
    if args.mode == "multi-window-check":
        import sys
        sys.path.insert(0, str(args.code_root.resolve() / "src"))
        from dagger_rtx.qualification_driver import validate_config
        if args.qualification_config is None:
            raise RuntimeError("missing frozen qualification configuration")
        if args.supervisor_config is None or args.supervisor_status_dir is None:
            raise RuntimeError("runtime check requires run-owned supervised communication")
        config = validate_config(json.loads(args.qualification_config.read_bytes()))
        if (args.num_envs != config["num_envs_per_gpu"] or args.fanout != expected_gateways or config["run_id"] != args.run_id
                or args.rtx_gpus != config["physical_rtx_gpus"]
                or config["seed"] != args.seed or config["gateway_pids"] != pids):
            raise RuntimeError("runtime-check launcher/config mismatch")
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", IMAGE]))[0]
    if image["Id"] != IMAGE or image["Config"]["Labels"].get("org.roborender.runtime") != "isaac6-lab3-migration-candidate":
        raise RuntimeError("candidate image mismatch")
    for _ in range(3):
        rows = subprocess.check_output(["nvidia-smi", "-i", ",".join(map(str, args.rtx_gpus)), "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                        "--format=csv,noheader,nounits"], text=True).splitlines()
        if len(rows) != len(args.rtx_gpus):
            raise RuntimeError("GPU inventory is incomplete")
        for row in rows:
            index, gpu, memory, util = [v.strip() for v in row.split(",")]
            if int(index) not in args.rtx_gpus or bound_gpu_uuids.get(int(index)) != gpu or int(memory) > 2048 or int(util) > 10:
                raise RuntimeError("one selected GPU is not idle and identity-verified")
        time.sleep(1)
    initial_memory = memory_kib()
    if initial_memory["MemAvailable"] < required_available_gib * 1024 * 1024:
        raise RuntimeError("insufficient host RAM for both bounded actors plus headroom")
    output = (REPO / f"runtime/{role}-production" / args.run_id / f"round-{config['round_id']}"
              if beta else REPO / "runtime/throughput-sweeps" / args.run_id)
    output.mkdir(parents=True, exist_ok=False)
    barrier = output / "barrier"
    barrier.mkdir()
    processes, containers, logs = [], [], []
    receipts = []
    plans = []
    scope_admissions = {}
    started = time.monotonic()
    failure = None
    try:
        for lane_rank, gpu in enumerate(args.rtx_gpus):
            actor_output = output / f"gpu{gpu}"
            actor_output.mkdir()
            (actor_output / "home").mkdir()
            cache = REPO / ("runtime/isaac6-cache" if gpu == 1 else f"runtime/isaac6-cache-gpu{gpu}")
            if leo:
                cache = REPO / f"runtime/isaac6-cache-leo-gpu{gpu}"
            cache.mkdir(exist_ok=True, parents=True)
            overlay = actor_output / "overlay"
            materialize(RAVEN, overlay, args.code_root.resolve() / "../config/assets.manifest.json")
            name = args.run_id + (f"-r{config['round_id']}" if beta else "") + f"-g{gpu}"
            shard_pids = pids if args.fanout in (1, 4) else pids[lane_rank*2:(lane_rank+1)*2]
            base = 18620 if args.fanout in (1, 4) else 18620 + lane_rank*2
            cmd = command(name, actor_output, overlay, args.code_root, args.num_envs,
                          mode=args.mode, run_id=args.run_id, token_file=args.token_file,
                          gateway_pids=shard_pids, gateway_base=base, physical_gpu=gpu,
                          seed=args.seed + lane_rank, memory_gib=args.memory_gib, cpu_count=12,
                          benchmark_barrier=barrier if connected or len(args.rtx_gpus) > 1 else None,
                          qualification_config=args.qualification_config,
                          supervisor_config=args.supervisor_config,
                          supervisor_status_dir=args.supervisor_status_dir,
                          compact_host_heap=args.compact_host_heap, deployment_host=args.deployment_host,
                          capacity_steps=args.capacity_steps, warm_camera_plan=args.warm_camera_plan,
                          recorded_action_reference=args.recorded_action_reference,
                          warm_camera_source_sha256=(json.loads(args.source_manifest.read_bytes())["source_sha256"]
                                                     if args.warm_camera_plan is not None else None),
                          native_thp_policy=args.native_thp_policy)
            plans.append({"gpu": gpu, "command": cmd})
            log = (actor_output / "console.log").open("wb")
            logs.append(log)
            processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
            containers.append(name)
            receipts.append(actor_output / ("native-service-result.json" if connected else "capture-smoke.json"))
        (output / "plan.json").write_text(json.dumps({"args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                                                    "actors": plans, "initial_memory_kib": initial_memory}, indent=2))
        print(json.dumps({"phase": "initializing_real_actors", "actors": len(args.rtx_gpus), "output": str(output), "environments": len(args.rtx_gpus)*args.num_envs}), flush=True)
        with (output / "resources.jsonl").open("a") as resources:
            next_gpu_sample = 0.0
            while any(p.poll() is None for p in processes):
                # Check terminal owners before evaluating live resource guards.
                # This also handles the all-exited race after the while test.
                complete = [native_completion(p, receipt) for p, receipt in zip(processes, receipts)]
                if all(value is not None for value in complete):
                    break
                mem = memory_kib()
                ready = len(list(barrier.glob("*.ready.json")))
                sample = {"time_ns": time.time_ns(), "elapsed_s": time.monotonic()-started,
                          "mem_available_kib": mem["MemAvailable"], "swap_free_kib": mem["SwapFree"],
                          "actors_ready": ready}
                if time.monotonic() >= next_gpu_sample:
                    try:
                        sample["gpu_samples"] = subprocess.check_output(
                            ["nvidia-smi", "-i", ",".join(map(str, args.rtx_gpus)),
                             "--query-gpu=" + ("index,memory.used,memory.total,utilization.gpu,power.draw"
                                if args.conservative_capacity_guard else "index,memory.used,utilization.gpu,power.draw"),
                             "--format=csv,noheader,nounits"], text=True, timeout=5).splitlines()
                    except (subprocess.SubprocessError, OSError) as exc:
                        sample["gpu_sample_error"] = repr(exc)
                    sample["container_memory"] = [owned_memory_sample(name, p, receipt)
                        for name, p, receipt in zip(containers, processes, receipts)]
                    if args.native_thp_policy == DISABLED:
                        for name, observation in zip(containers,sample['container_memory']):
                            if name not in scope_admissions and observation.get('pid',0)>0:
                                scope_admissions[name]=attest_capacity_scope(observation)
                    next_gpu_sample = time.monotonic() + (2 if args.conservative_capacity_guard else 10)
                # Attribute any host swap change before deciding to abandon.
                # Cgroup swap is disabled by the Docker memory/swap caps.
                owned = sample.get("container_memory")
                if owned is None and initial_memory["SwapFree"] - mem["SwapFree"] > 128 * 1024:
                    owned = [owned_memory_sample(name, p, receipt)
                        for name, p, receipt in zip(containers, processes, receipts)]
                    sample["container_memory"] = owned
                sample["memory_safety"] = memory_safety(initial_memory, mem, owned or [])
                if args.conservative_capacity_guard and "container_memory" in sample:
                    # A short launcher grace permits Docker to create the
                    # cgroup; after that missing counters fail closed.
                    live = sample["container_memory"]
                    if time.monotonic()-started > 15 or all("stat" in s or s.get("lifecycle") == "completed_attested" for s in live):
                        conservative_capacity_safety(initial_memory, mem, live,
                            sample.get("gpu_samples", []), args.memory_gib)
                resources.write(json.dumps(sample) + "\n")
                resources.flush()
                for process, receipt in zip(processes, receipts):
                    native_completion(process, receipt)
                if ready == len(args.rtx_gpus) and not (barrier / "GO").exists():
                    if args.native_thp_policy == DISABLED and len(scope_admissions)!=2:
                        raise RuntimeError('paired systemd zero-swap admission required before any teacher steps')
                    (barrier / "GO").write_text(str(time.time_ns()))
                    print(json.dumps({"phase": "all_actors_ready_starting_collection", "actors": ready, "elapsed_s": time.monotonic()-started}), flush=True)
                if time.monotonic()-started > args.timeout:
                    raise RuntimeError("bounded pair timeout")
                time.sleep(2)
    except BaseException as exc:
        failure = repr(exc)
        (barrier / "ABORT").touch()
    finally:
        for name in containers:
            cleanup_owned(name)
        for p in processes:
            p.wait(timeout=30)
        for log in logs:
            log.close()
    # Only exact generated evidence files are made host-readable.
    evidence_files = []
    for gpu in args.rtx_gpus:
        for filename in ("capture-smoke.json", "native-service-result.json", "run-telemetry.jsonl", "capacity-telemetry.jsonl", "phase-boundaries.jsonl", "capacity-report.json", "qualification-state.json", "beta-round-state.json", "ack-loss-probe.json"):
            if (output / f"gpu{gpu}" / filename).is_file():
                evidence_files.append(f"/evidence/gpu{gpu}/{filename}")
    if evidence_files:
        subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/chown",
                        "--mount", f"type=bind,src={output},dst=/evidence", IMAGE,
                        f"{os.getuid()}:{os.getgid()}", *evidence_files], check=True, timeout=30)
    results = []
    for gpu in args.rtx_gpus:
        receipt = output / f"gpu{gpu}" / ("capture-smoke.json" if not connected else "native-service-result.json")
        results.append(json.loads(receipt.read_text()) if receipt.exists() else {})
    passed = failure is None and all(p.returncode == 0 for p in processes) and all(r.get("result") == "PASS" for r in results)
    if teacher_probe:
        passed = passed and all(r.get("mode") == args.mode
            and r.get("num_envs") == args.num_envs and r.get("environment_step_calls") == args.capacity_steps
            and r.get("physical_transition_rows") == args.capacity_steps * args.num_envs
            and r.get("optimizer_steps") == 0 and r.get("chunks_published") == 0
            and r.get("H200_renderer_tested") is False and r.get("production_qualified") is False for r in results)
    if args.native_thp_policy == DISABLED:
        try:
            if len(scope_admissions)!=2:
                raise RuntimeError('both actual capacity systemd scopes must be attested')
            for result in results:validate_capacity_native_memory(result)
        except RuntimeError as exc:
            passed=False
            if failure is None:failure=str(exc)
    if args.warm_camera_plan is not None:
        for gpu, result in zip(args.rtx_gpus, results):
            warm = result.get("warm_scene_diagnostic", {})
            total = 324 if gpu == 0 else 357
            passed = passed and (warm.get("plan_sha256") == warm_camera_sha
                and warm.get("arm") == ("fresh" if gpu == 0 else "warm")
                and warm.get("scene_count") == (1 if gpu == 0 else 12)
                and warm.get("native_initializations") == 1
                and warm.get("total_environment_step_calls") == total
                and warm.get("total_physical_transition_rows") == total * 16)
            if recorded:
                trace = result.get('recorded_action_diagnostic', {})
                passed = passed and (trace.get('reference_sha256') == recorded_sha
                    and trace.get('recorded_vector_steps') == 129
                    and trace.get('recorded_transition_rows') == 2064
                    and trace.get('production_replay_allowed') is False
                    and trace.get('simulator_restore_claimed') is False)
    if not passed and failure is None:
        failure = 'native actor exited without a complete PASS receipt; zero container exit code is insufficient'
    report = {"passed": passed, "failure": failure, "run_id": args.run_id, "output": str(output),
              "mode": args.mode, "num_envs_per_gpu": args.num_envs, "physical_gpus": args.rtx_gpus,
              "exit_codes": [p.returncode for p in processes], "elapsed_seconds": time.monotonic()-started,
              "production_promoted": False, "automatic_learning": args.mode in {"multi-window-check", "beta-production"},
              **({"round_id": config["round_id"], "config_sha256": __import__('hashlib').sha256(args.qualification_config.read_bytes()).hexdigest()}
                 if beta else {})}
    if teacher_probe:
        report.update(source_sha256=json.loads(args.source_manifest.read_bytes())['source_sha256'],
            deployment_host=args.deployment_host,capacity_steps=args.capacity_steps,
            memory_gib_per_actor=args.memory_gib,conservative_capacity_guard=args.conservative_capacity_guard)
    if args.native_thp_policy == DISABLED:
        report.update(native_thp_policy=DISABLED,systemd_scope_admissions=scope_admissions,
                      production_memory_policy_matched=passed)
    if args.warm_camera_plan is not None:
        report.update(warm_camera_plan_sha256=warm_camera_sha,
                      total_vector_steps_per_actor=[324, 357],
                      diagnostic_only=True, mixed_policy_tested=False)
        if recorded:
            report.update(recorded_action_reference_sha256=recorded_sha,
                          live_student_inference=False, production_replay_allowed=False)
    (output / "pair-result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
