"""Scoped CPU-side enrollment/health for the Beta-owned reverse SSH tunnel."""
from __future__ import annotations
import argparse
import base64
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import socket
import subprocess

ROOT = Path("/workspace/project/roborender")
SSH_ROOT = Path("/workspace/.ssh")
CONFIG = SSH_ROOT / "roborender_beta_access_config"
KNOWN = SSH_ROOT / "roborender_beta_access_known_hosts"
IDENTITY = SSH_ROOT / "roborender_alpha_beta_migration_20261008"
ALIAS = "roborender-beta-direct"
PORT = 22032
BETA_HOST = "465d6b0-lcedt"
BETA_PIN = "SHA256:FGgy9h5kjzygHcUtkXTNKSYdUnX7Sp5LI+3wHJOHUzc"
MARKER = "roborender-beta-access-20261012"


def public_key(value):
    if not isinstance(value, str) or "\n" in value:
        raise ValueError("single Ed25519 public key required")
    parts = value.split()
    if len(parts) not in (2, 3) or parts[0] != "ssh-ed25519":
        raise ValueError("Ed25519 public key required")
    raw = base64.b64decode(parts[1], validate=True)
    if len(raw) != 51 or raw[:19] != b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20":
        raise ValueError("invalid Ed25519 public key")
    return " ".join(parts[:2])


def fingerprint(key):
    raw = base64.b64decode(public_key(key).split()[1])
    return "SHA256:" + base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")


def publish_exact(path, content, mode=0o600):
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError("refusing symlink target")
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError("existing managed file differs; explicit reconciliation required")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)


def authorized_line(key):
    command = f"/usr/bin/python3 {ROOT}/pipeline/deployment/connectivity/cpu_endpoint.py health"
    return (f'restrict,port-forwarding,permitlisten="127.0.0.1:{PORT}",'
            f'permitopen="127.0.0.1:1",command="{command}" {public_key(key)} {MARKER}\n')


def append_authorization(path, line):
    """Add only our dedicated restricted key; never remove/replace other keys."""
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError("authorized_keys is a symlink; reconcile explicitly")
    lock = path.with_name("roborender-beta-access.lock")
    with lock.open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        old = path.read_text() if path.exists() else ""
        if line.rstrip() in old.splitlines():
            return False
        key_blob = line.split("ssh-ed25519 ", 1)[1].split()[0]
        if MARKER in old or key_blob in old:
            raise RuntimeError("dedicated key already has different authorization")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(("\n" if old and not old.endswith("\n") else "") + line)
            stream.flush()
            os.fsync(stream.fileno())
        return True


def enroll(payload):
    if os.geteuid() != 0 or not socket.gethostname().startswith("leo-cosmos-h200-access-"):
        raise RuntimeError("enrollment requires the existing CPU access pod")
    if set(payload) != {"tunnel_public_key", "beta_host_public_key"}:
        raise ValueError("unexpected enrollment fields")
    beta_key = public_key(payload["beta_host_public_key"])
    if fingerprint(beta_key) != BETA_PIN:
        raise RuntimeError("Beta host key differs from authenticated migration pin")
    cpu_key = public_key(Path("/etc/ssh/ssh_host_ed25519_key.pub").read_text().strip())
    conf = (f"Host {ALIAS}\n HostName 127.0.0.1\n Port {PORT}\n User ldu\n HostKeyAlias {ALIAS}\n"
            f" IdentityFile {IDENTITY}\n IdentitiesOnly yes\n BatchMode yes\n"
            f" StrictHostKeyChecking yes\n UserKnownHostsFile {KNOWN}\n GlobalKnownHostsFile /dev/null\n"
            " ForwardAgent no\n ControlMaster no\n ControlPath none\n ConnectTimeout 8\n"
            " ServerAliveInterval 10\n ServerAliveCountMax 2\n")
    SSH_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not IDENTITY.is_file() or IDENTITY.stat().st_mode & 0o077:
        raise RuntimeError("approved CPU management identity unavailable or not private")
    publish_exact(KNOWN, (ALIAS + " " + beta_key + "\n").encode())
    publish_exact(CONFIG, conf.encode())
    ssh_dir = Path(pwd.getpwuid(0).pw_dir) / ".ssh"
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    append_authorization(ssh_dir / "authorized_keys", authorized_line(payload["tunnel_public_key"]))
    return {"ok": True, "cpu_hostname": socket.gethostname(), "reverse_port": PORT,
            "cpu_host_public_key": cpu_key,
            "cpu_management_public_key": public_key(IDENTITY.with_suffix(".pub").read_text().strip())}


def health():
    if os.environ.get("SSH_ORIGINAL_COMMAND") not in (None, "roborender-health"):
        raise ValueError("dedicated tunnel key permits health checks only")
    result = subprocess.run(["ssh", "-F", str(CONFIG), ALIAS, "/bin/hostname"],
                            capture_output=True, timeout=12)
    ok = result.returncode == 0 and result.stdout.strip().decode() == BETA_HOST
    return {"ok": ok, "cpu_hostname": socket.gethostname(),
            "beta_hostname": BETA_HOST if ok else None, "reverse_port": PORT,
            "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["enroll", "health"])
    parser.add_argument("--payload")
    args = parser.parse_args()
    try:
        value = enroll(json.loads(base64.b64decode(args.payload, validate=True))) if args.action == "enroll" else health()
        print(json.dumps(value, sort_keys=True))
        return 0 if value["ok"] else 1
    except Exception as error:
        print(json.dumps({"ok": False, "error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
