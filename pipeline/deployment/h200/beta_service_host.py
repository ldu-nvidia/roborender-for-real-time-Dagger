"""Own a beta service bundle and its exclusive persistent ledger.

This supervisor runs on either a reserved four-H200 workload or the CPU access
pod, never both. It exits if a child fails and stops only children it started.
An explicit deadline also protects against an abandoned management connection.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

SOURCE_ROOT = Path(__file__).resolve().parents[3]
if str(SOURCE_ROOT) not in sys.path:sys.path.insert(0, str(SOURCE_ROOT))


def atomic(path, value):
    part = path.with_suffix(".part")
    part.write_text(json.dumps(value, sort_keys=True) + "\n")
    part.replace(path)


def launch_plan(source, state, token, output, mode, *, retained_fleet=None):
    source, state, token, output = map(Path, (source, state, token, output))
    if mode not in {"gpu", "cpu"} or not all(p.is_absolute() for p in (source, state, token, output)):
        raise ValueError("explicit source/state/token/output paths and service mode required")
    env = {"DAGGER_SERVICE_STATE_ROOT": str(state), "DAGGER_SERVICE_TOKEN_FILE": str(token),
           "ROBORENDER_ASSET_ROOT": "/workspace/project/roborender",
           "IMAGE3F_PYTHON": "/workspace/project/roborender/renderer/envs/image3f-py313-cu130-trt10133/venv/bin/python",
           "DAGGER_LEARNER_WORKER": str(source / "pipeline/dagger/run_raven_learner_worker_h200.sh"),
           "DAGGER_RENDER_POOL_ROOT": str(output / "render_pool"),
           "DAGGER_RENDER_GPUS": "0,1,2,3", "IMAGE3F_PORTS": "18500,18501,18502,18503",
           "DAGGER_GATEWAY_PORTS": "18620,18621,18622,18623",
           "DAGGER_STUDENT_DEVICES": "cuda:0,cuda:1,cuda:2,cuda:3",
           "DAGGER_EPHEMERAL_WINDOW_CACHE": "false", "WANDB_MODE": "online", "WANDB_ENTITY": "nvidia"}
    roles = {"coordinator": 18610, "registry": 18612, "evaluation": 18614}
    if retained_fleet is not None:
        from pipeline.deployment.h200.retained_renderer_attachment import validate_fleet
        validate_fleet(retained_fleet)
        if mode != 'gpu':raise ValueError('retained renderer attachment requires GPU services')
        for key in ('DAGGER_RENDER_GPUS','IMAGE3F_PORTS','DAGGER_STUDENT_DEVICES'):
            env.pop(key)
        env['DAGGER_GATEWAY_PORTS'] = ','.join(str(port) for port in range(18620,18628))
    if mode == "gpu":
        roles = {"coordinator": 18610, "trajectory": 18611, "registry": 18612,
                 "learner": 18613, "evaluation": 18614}
    else:
        env.update(DAGGER_SERVICE_PYTHON=sys.executable, CUDA_VISIBLE_DEVICES="")
    commands = [(name, ["bash", str(source / "pipeline/dagger/run_h200_service.sh"), name], [port])
                for name, port in roles.items()]
    if mode == 'gpu' and retained_fleet is not None:
        commands += [(f'gateway-{slot}', ['bash', str(source / 'pipeline/dagger/run_h200_service.sh'), 'render-student'],
                      [18620 + slot]) for slot in range(8)]
    elif mode == "gpu":
        commands.append(("render_pool", ["bash", str(source / "pipeline/dagger/run_h200_render_pool.sh")],
                         list(range(18620, 18624))))
    return env, commands


def gateway_environment(fleet, slot):
    """One new student gateway per existing physical renderer GPU, no pooling."""
    row = fleet['workers'][slot]
    return dict(CUDA_VISIBLE_DEVICES=row['gpu_uuid'], DAGGER_SERVICE_PORT=str(18620 + slot),
                DAGGER_STUDENT_DEVICE='cuda:0', DAGGER_STUDENT_DEVICES='cuda:0',
                IMAGE3F_PORT=str(row['port']), IMAGE3F_PORTS=str(row['port']))


def learner_environment(fleet):
    """Pin serial full-population updates to one retained H200 before Torch loads.

    Gateways retain their own eight UUID bindings. The learner shares slot0;
    its existing actual-memory admission reserves renderer/headroom capacity.
    """
    from pipeline.deployment.h200.retained_renderer_attachment import validate_fleet
    validate_fleet(fleet)
    return dict(CUDA_VISIBLE_DEVICES=fleet['workers'][0]['gpu_uuid'])


def durable_intent(path, value):
    """Never overwrite or retry a possibly accepted service launch."""
    raw = (json.dumps(value, sort_keys=True) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def probe(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=2) as response:
        value = json.loads(response.read(16385))
    if value.get("ok") is not True or type(value.get("pid")) is not int:
        raise RuntimeError("invalid service readiness")
    return value


def require_allocation_identity(run_id, retained_owner=None):
    # sshd SetEnv supplies this from the allocation entrypoint, not the
    # client request/process spec. SSH otherwise discards the parent env.
    expected = retained_owner if retained_owner is not None else run_id
    if not expected or os.environ.get("ROBORENDER_BETA_RUN_ID") != expected:
        raise RuntimeError("reserved pod run identity differs")


def validate_lifetime(ttl, retained_owner, continuous):
    """Continuous services stop only their children, never their allocation.

    This is an explicit new deployment mode, not a larger legacy TTL. It is
    restricted to a retained allocation whose entrypoint owner is verified.
    """
    if continuous:
        if not retained_owner or ttl != 24000:
            raise ValueError("continuous services require a retained owner and no TTL override")
    elif not 60 <= ttl <= 43200:
        raise ValueError("service lifetime must be bounded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "state", "token", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--mode", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--state-identity", required=True)
    parser.add_argument("--ttl", type=int, default=24000)
    parser.add_argument("--netrc", type=Path)
    parser.add_argument("--retained-owner")
    parser.add_argument("--continuous-services", action="store_true")
    parser.add_argument('--attach-retained-renderer-fleet', type=Path)
    parser.add_argument('--renderer-fleet-sha256')
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--source-sha256')
    args = parser.parse_args()
    validate_lifetime(args.ttl, args.retained_owner, args.continuous_services)
    if args.retained_owner is not None and args.mode != "gpu":
        raise ValueError("retained allocation owner is GPU-only")
    expected = hashlib.sha256((args.run_id + ":beta-service-ledger-v1").encode()).hexdigest()
    if args.state_identity != expected:
        raise ValueError("ledger identity differs")
    if (args.token.stat().st_mode & 0o077 or len(args.token.read_text().strip()) < 32
            or args.token.is_symlink()):
        raise ValueError("unsafe token file")
    fleet = attachment = None
    fields = (args.attach_retained_renderer_fleet, args.renderer_fleet_sha256, args.source_manifest, args.source_sha256)
    if any(fields):
        if not all(fields) or args.mode != 'gpu' or not args.continuous_services:
            raise ValueError('explicit complete retained8 source/fleet attachment required')
        from pipeline.deployment.h200.retained_renderer_attachment import (
            read_bound, validate_fleet, observe, verify_release, gateway_identity, OWNER)
        if args.retained_owner != OWNER:
            raise ValueError('attachment cannot change the retained allocation owner')
        verify_release(args.source, args.source_manifest, args.source_sha256)
        fleet = validate_fleet(read_bound(args.attach_retained_renderer_fleet, args.renderer_fleet_sha256))
        attachment = observe(fleet)
    if args.mode == "gpu" and fleet is None:
        require_allocation_identity(args.run_id, args.retained_owner)
        devices = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
        if len(devices) != 4 or any("H200" not in row for row in devices):
            raise RuntimeError("this workload requires exactly four H200s")
        clients = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if clients.strip():
            raise RuntimeError("reserved workload already has GPU clients; refusing takeover")
    overrides, commands = launch_plan(args.source, args.state, args.token, args.output, args.mode, retained_fleet=fleet)
    if fleet is not None:overrides['ROBORENDER_SERVICE_SOURCE_SHA256'] = args.source_sha256
    if args.netrc:
        if args.netrc.stat().st_mode & 0o077 or args.netrc.is_symlink():
            raise ValueError("W&B credential reference must be owner-only")
        overrides["NETRC"] = str(args.netrc)
    if fleet is not None and args.state.exists():
        raise RuntimeError('retained8 attachment requires a fresh ledger namespace; no replay/adoption')
    args.state.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=False)
    if fleet is not None:
        durable_intent(args.output / 'launch-intent.json', dict(source_sha256=args.source_sha256,
            fleet_sha256=args.renderer_fleet_sha256, run_id=args.run_id, state_identity=args.state_identity,
            commands=commands, renderer_start_authorized=False, renderer_stop_authorized=False,
            automatic_restart=False, compute_release_authorized=False))
        durable_intent(args.output / 'renderer-attachment.json', attachment)
    children, logs, stop = [], [], False
    def terminate(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    with (args.state / "bundle-owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = args.state / "bundle-identity.json"
        identity = {"run_id": args.run_id, "state_identity": args.state_identity}
        if identity_path.exists():
            if json.loads(identity_path.read_bytes()) != identity:
                raise RuntimeError("shared ledger belongs to a different campaign")
        else:
            atomic(identity_path, identity)
        env = dict(os.environ, **overrides)
        started = time.monotonic()
        ready = {}; gateways = []
        error = None
        try:
            # Bind checks prevent a pre-existing server's /ready from being
            # mistaken for our own newly launched process.
            for _, _, ports in commands:
                for port in ports:
                    with socket.socket() as check:
                        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        check.bind(("127.0.0.1", port))
            for name, argv, ports in commands:
                log = (args.output / (name + ".log")).open("xb")
                logs.append(log)
                child_env = dict(env)
                if fleet is not None and name.startswith('gateway-'):
                    child_env.update(gateway_environment(fleet, int(name.split('-')[1])))
                elif fleet is not None and name == 'learner':
                    child_env.update(learner_environment(fleet))
                if fleet is not None:
                    durable_intent(args.output / (name + '-launch-intent.json'), dict(argv=argv, ports=ports,
                        source_sha256=args.source_sha256, automatic_restart=False, compute_release_authorized=False))
                child = subprocess.Popen(argv, env=child_env, cwd=args.source, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                deadline = time.monotonic() + (1200 if name == "render_pool" else 90)
                for port in ports:
                    while True:
                        if stop or any(p.poll() is not None for p in children):
                            raise RuntimeError("service owner stopped before readiness")
                        try:
                            ready[str(port)] = probe(port)
                            if fleet is not None and name.startswith('gateway-'):
                                gateways.append(gateway_identity(args.source, args.source_sha256, fleet,
                                    int(name.split('-')[1]), ready[str(port)], expected_pid=child.pid))
                            break
                        except OSError:
                            if time.monotonic() > deadline:
                                raise TimeoutError("service readiness deadline")
                            time.sleep(1)
            extra = {} if fleet is None else dict(source_sha256=args.source_sha256,
                renderer_fleet_sha256=args.renderer_fleet_sha256, gateway_process_receipts=gateways,
                gateway_identities=[{k:v for k,v in row.items() if k != 'process_identity'} for row in gateways],
                retained_renderer_attachment=attachment, renderer_start_count=0, renderer_stop_authorized=False)
            atomic(args.output / "ready.json", {"ok": True, **identity, "mode": args.mode,
                                               "owner_pid": os.getpid(), "services": ready, **extra})
            while not stop and (args.continuous_services or time.monotonic() - started < args.ttl):
                if any(p.poll() is not None for p in children):
                    raise RuntimeError("a service exited; stopping this entire owned bundle")
                time.sleep(1)
        except BaseException as exc:
            error = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            for child in reversed(children):
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
            for child in reversed(children):
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
            for log in logs:
                log.close()
            atomic(args.output / "stopped.json", {**identity, "owned_children_stopped": True,
                "error": error, "expired": not args.continuous_services and not stop and error is None,
                "allocation_release_requested": False})


if __name__ == "__main__":
    main()
