"""CPU-only live Beta tunnel checks, with optional scoped reconnect injection.

Never allocates a GPU, imports a model, touches credentials, or replays a
simulator action. Only the explicitly owned management SSH child may be stopped.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import secrets
import shlex
import subprocess
import time

SSH = ["ssh", "-F", "/workspace/.ssh/roborender_beta_access_config",
       "-o", "ConnectTimeout=8", "roborender-beta-direct"]
STATUS = "/home/ldu/.local/share/roborender-beta-access/status.json"
UNIT = "roborender-beta-access.service"
HOSTNAME = "465d6b0-lcedt"
ROLE = "beta"
PORT = 22032


def select_host(host):
    global SSH, STATUS, UNIT, HOSTNAME, ROLE, PORT
    if host not in {"465d6b0-lcedt", "leo-rtx-6k"}:
        raise ValueError("unknown explicitly pinned workstation")
    ROLE = "leo" if host == "leo-rtx-6k" else "beta"
    HOSTNAME, PORT = host, 22033 if ROLE == "leo" else 22032
    SSH = ["ssh", "-F", f"/workspace/.ssh/roborender_{ROLE}_access_config",
           "-o", "ConnectTimeout=8", f"roborender-{ROLE}-direct"]
    STATUS = f"/home/ldu/.local/share/roborender-{ROLE}-access/status.json"
    UNIT = f"roborender-{ROLE}-access.service"


def remote(command, *, data=None, timeout=40):
    result = subprocess.run(SSH + [shlex.join(command)], input=data,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Beta probe failed (exit {result.returncode})")
    return result.stdout


def snapshot():
    code = f"""import json,pathlib,subprocess,socket
s=json.loads(pathlib.Path({STATUS!r}).read_text())
props=subprocess.check_output(['systemctl','--user','show',{UNIT!r},'-p','ActiveState','-p','SubState','-p','MainPID'],text=True)
service=dict(x.split('=',1) for x in props.splitlines())
linger=subprocess.check_output(['loginctl','show-user','ldu','-p','Linger','--value'],text=True).strip()
enabled=subprocess.check_output(['systemctl','--user','is-enabled',{UNIT!r}],text=True).strip()
print(json.dumps(dict(status=s,service=service,linger=linger,enabled=enabled,hostname=socket.gethostname())))
"""
    value = json.loads(remote(["python3", "-c", code]))
    if (value["hostname"] != HOSTNAME or value["status"]["state"] != "ready"
            or value["service"]["ActiveState"] != "active" or value["enabled"] != "enabled"
            or value["linger"] != "yes" or value["status"]["health"]["ok"] is not True):
        raise RuntimeError("Beta service is not persistently ready")
    stamp = datetime.datetime.fromisoformat(value["status"]["updated_at_utc"])
    age = (datetime.datetime.now(datetime.timezone.utc) - stamp).total_seconds()
    if not 0 <= age < 75:
        raise RuntimeError("Beta health receipt is stale")
    return value


def stop_script(ssh_pid, parent_pid):
    # Stop only an exactly attested child. This script is sent on the very same
    # tunnel; EOF after SIGTERM is expected, not proof of completed recovery.
    return f"""import os,pathlib,signal
pid={int(ssh_pid)}; parent={int(parent_pid)}
argv=pathlib.Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(bytes([0]))[:-1]
prefix=[b'ssh',b'-F',b'/home/ldu/.local/share/roborender-{ROLE}-access/ssh.conf']
legacy=prefix+[b'-NT',b'-R',b'127.0.0.1:{PORT}:127.0.0.1:22',b'roborender-cosmos-access']
current=prefix+[b'-o',b'ControlMaster=yes',b'-o',b'ControlPath=/home/ldu/.local/share/roborender-{ROLE}-access/master.sock',b'-o',b'ControlPersist=no',b'-o',b'ConnectTimeout=65']+legacy[3:]
assert argv in (legacy,current), 'not the owned management SSH child'
proc=pathlib.Path('/proc/'+str(pid)+'/status').read_text().splitlines()
assert int(next(x for x in proc if x.startswith('PPid:')).split()[1])==parent
supervisor=pathlib.Path('/proc/'+str(parent)+'/cmdline').read_bytes().split(bytes([0]))
assert any(x.endswith(b'/{ROLE}_access.py') for x in supervisor) and b'run' in supervisor
os.kill(pid,signal.SIGTERM)
"""


def verify(output, *, exercise_reconnect=False):
    if output.exists():
        raise ValueError("new evidence path required")
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    before = snapshot()
    samples = []
    for size in (4096, 1 << 20, 1 << 20):
        data = secrets.token_bytes(size)
        began = time.monotonic()
        received = remote(["python3", "-c", "import sys;sys.stdout.buffer.write(sys.stdin.buffer.read())"], data=data)
        if received != data:
            raise RuntimeError("bidirectional binary payload mismatch")
        samples.append({"bytes_each_direction": size, "sha256": hashlib.sha256(data).hexdigest(),
                        "round_trip_seconds_including_ssh": time.monotonic() - began})
    reconnect = {"exercised": False}
    if exercise_reconnect:
        old_pid = before["status"]["ssh_pid"]
        parent_pid = int(before["service"]["MainPID"])
        began = time.monotonic()
        try:
            remote(["python3", "-c", stop_script(old_pid, parent_pid)], timeout=15)
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        print(json.dumps({"phase": "owned_tunnel_interrupted_waiting_for_reconnect"}), flush=True)
        deadline = time.monotonic() + 180
        after = None
        while time.monotonic() < deadline:
            try:
                observed = snapshot()
                if observed["status"]["ssh_pid"] != old_pid:
                    after = observed
                    break
            except (RuntimeError, subprocess.TimeoutExpired):
                pass
            time.sleep(3)
        if after is None:
            raise RuntimeError("owned tunnel did not prove fresh authenticated recovery")
        if after["service"]["MainPID"] != before["service"]["MainPID"]:
            raise RuntimeError("recovery changed supervisor unexpectedly")
        reconnect = {"exercised": True, "old_ssh_pid": old_pid, "new_ssh_pid": after["status"]["ssh_pid"],
                     "recovery_seconds": time.monotonic() - began, "supervisor_preserved": True}
    after = snapshot()
    result = {"schema": f"roborender.{ROLE}-connectivity-validation.v1", "passed": True,
              "started_at_utc": start, "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "before": before, "after": after, "binary_roundtrips": samples, "reconnect": reconnect,
              "alpha_used": False, "beta_used": ROLE == "beta", "mac_runtime_used": False, "gpu_allocations": 0,
              "native_training_or_evaluation_tested": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exercise-reconnect", action="store_true")
    parser.add_argument("--deployment-host", choices=["465d6b0-lcedt", "leo-rtx-6k"], default="465d6b0-lcedt")
    args = parser.parse_args()
    select_host(args.deployment_host)
    print(json.dumps(verify(args.output, exercise_reconnect=args.exercise_reconnect), sort_keys=True))
