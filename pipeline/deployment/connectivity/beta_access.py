"""Install/supervise the Beta-owned, Alpha/Mac-independent management tunnel.

This program provisions connectivity only. It never imports CUDA, starts a
simulator, allocates compute, restarts a workload, or copies a cloud credential.
"""
from __future__ import annotations
import argparse
import base64
import datetime
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import threading
import time

from cloud_proxy import CPU_POD, run_code
from cpu_endpoint import public_key, fingerprint, publish_exact, BETA_PIN, BETA_HOST

BASE = Path("/home/ldu/.local/share/roborender-beta-access")
SDK_PYTHON = Path("/home/ldu/.local/share/roborender-cloud/venv-compatible/bin/python")
TUNNEL_KEY = Path("/home/ldu/.ssh/roborender_beta_access_20261012")
UNIT_NAME = "roborender-beta-access.service"
CPU_ALIAS = "roborender-cosmos-access"
CPU_PIN = "SHA256:sVIM/mBcMijiqJnzkfPhBK1VDJAQ6LQeTFoNctQEyS0"
PORT = 22032
MARKER = "# Managed by RoboRender beta access v1"
CPU_ENDPOINT = "/workspace/project/roborender/pipeline/deployment/connectivity/cpu_endpoint.py"
CONTROL = BASE / "master.sock"


def checked_config(config, *, installed=False):
    keys = {"cpu_endpoint_path", "cpu_endpoint_sha256"}
    if installed:
        keys |= {"schema", "ssh_config"}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError("unexpected access configuration")
    if config["cpu_endpoint_path"] != CPU_ENDPOINT or not re.fullmatch(r"[a-f0-9]{64}", config["cpu_endpoint_sha256"]):
        raise ValueError("CPU endpoint identity is not pinned")
    if installed and (config["schema"] != "roborender.beta-access.v1" or config["ssh_config"] != str(BASE / "ssh.conf")):
        raise ValueError("installed route identity differs")
    return config


def phase(name):
    print(json.dumps({"phase": name, "alpha_required": False, "mac_required_after_install": False}), flush=True)


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    if path.is_symlink() or temp.is_symlink():
        raise RuntimeError("refusing a symlink state file")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(data, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def require_beta():
    if socket.gethostname() != BETA_HOST or pwd.getpwuid(os.getuid()).pw_name != "ldu":
        raise RuntimeError("installer/supervisor must run as ldu on RTX Beta")
    if not (3, 9) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("Lepton needs compatible Python 3.9 through 3.13")


def run(argv, timeout=30, **kwargs):
    result = subprocess.run(list(map(str, argv)), timeout=timeout, capture_output=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"{Path(str(argv[0])).name} failed with exit {result.returncode}")
    return result.stdout


def enroll(config):
    from cloud_proxy import run_code
    cpu_path = config["cpu_endpoint_path"]
    digest = config["cpu_endpoint_sha256"]
    beta_host = public_key(Path("/etc/ssh/ssh_host_ed25519_key.pub").read_text().strip())
    if fingerprint(beta_host) != BETA_PIN:
        raise RuntimeError("Beta host public key changed")
    payload = {"tunnel_public_key": public_key(TUNNEL_KEY.with_suffix(".pub").read_text().strip()),
               "beta_host_public_key": beta_host}
    encoded = base64.b64encode(json.dumps(payload, sort_keys=True).encode()).decode()
    code = ("import hashlib,pathlib,runpy,sys; "
            f"p=pathlib.Path({cpu_path!r}); "
            f"assert hashlib.sha256(p.read_bytes()).hexdigest()=={digest!r}; "
            f"sys.argv=[str(p),'enroll','--payload',{encoded!r}]; "
            "runpy.run_path(str(p),run_name='__main__')")
    value = json.loads(run_code(code))
    if (value.get("ok") is not True or value.get("reverse_port") != PORT
            or not value.get("cpu_hostname", "").startswith(CPU_POD + "-")
            or fingerprint(value["cpu_host_public_key"]) != CPU_PIN):
        raise RuntimeError("CPU enrollment/identity failed")
    return value


def make_ssh_config(python, proxy, known):
    cmd = shlex.join([str(python), str(proxy), CPU_POD])
    return (f"Host {CPU_ALIAS}\n HostName {CPU_POD}\n HostKeyAlias {CPU_ALIAS}\n User root\n"
            f" IdentityFile {TUNNEL_KEY}\n IdentitiesOnly yes\n BatchMode yes\n"
            f" UserKnownHostsFile {known}\n GlobalKnownHostsFile /dev/null\n StrictHostKeyChecking yes\n"
            " ControlMaster no\n ControlPath none\n ForwardAgent no\n ConnectTimeout 25\n"
            " ServerAliveInterval 15\n ServerAliveCountMax 3\n ExitOnForwardFailure yes\n"
            f" ProxyCommand {cmd}\n")


def service_unit(python, script, config):
    values = [str(python), str(script), str(config)]
    if any(not Path(v).is_absolute() or any(c in v for c in '\n\r"%') for v in values):
        raise ValueError("invalid service paths")
    return (MARKER + "\n[Unit]\nDescription=RoboRender Beta-owned Cosmos access\n"
            "StartLimitIntervalSec=0\n\n[Service]\nType=simple\n"
            f'ExecStart="{python}" "{script}" run --config "{config}"\n'
            "Restart=always\nRestartSec=30\nTimeoutStopSec=20\nKillMode=control-group\n"
            "WatchdogSec=180\nNotifyAccess=main\n"
            "UMask=0077\nNoNewPrivileges=yes\nEnvironment=PYTHONUNBUFFERED=1\n"
            "Environment=LOGURU_LEVEL=ERROR\n\n[Install]\nWantedBy=default.target\n")


def management_key(key):
    """Keep the approved CPU key local; existing restricted entries are retained."""
    key = public_key(key)
    path = Path("/home/ldu/.ssh/authorized_keys")
    if path.is_symlink():
        raise RuntimeError("authorized_keys symlink requires explicit reconciliation")
    with (BASE / "enrollment.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        old = path.read_text() if path.exists() else ""
        if key.split()[1] in old:
            return
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(("\n" if old and not old.endswith("\n") else "")
                         + "restrict " + key + " roborender-cpu-management\n")
            stream.flush()
            os.fsync(stream.fileno())


def install(public_config):
    require_beta()
    checked_config(public_config)
    os.umask(0o077)
    phase("beta_identity_checked")
    BASE.mkdir(parents=True, exist_ok=True, mode=0o700)
    if BASE.is_symlink():
        raise RuntimeError("refusing symlink install root")
    Path("/home/ldu/.ssh").mkdir(mode=0o700, exist_ok=True)
    if TUNNEL_KEY.is_symlink() or TUNNEL_KEY.with_suffix(".pub").is_symlink():
        raise RuntimeError("refusing symlink key path")
    if not TUNNEL_KEY.exists():
        run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "roborender-beta-access", "-f", TUNNEL_KEY])
    if TUNNEL_KEY.stat().st_mode & 0o077:
        raise RuntimeError("dedicated tunnel key must be private")
    phase("cloud_enrollment_start")
    identity = enroll(public_config)
    phase("cloud_enrollment_complete")
    management_key(identity["cpu_management_public_key"])
    source = Path(__file__).resolve().parent
    known, ssh_config = BASE / "known_hosts", BASE / "ssh.conf"
    publish_exact(known, (CPU_ALIAS + " " + identity["cpu_host_public_key"] + "\n").encode())
    publish_exact(ssh_config, make_ssh_config(SDK_PYTHON, source / "cloud_proxy.py", known).encode())
    config = dict(public_config, schema="roborender.beta-access.v1", ssh_config=str(ssh_config))
    config_path = BASE / "config.json"
    publish_exact(config_path, (json.dumps(config, sort_keys=True) + "\n").encode())
    unit_dir = Path("/home/ldu/.config/systemd/user")
    unit_dir.mkdir(parents=True, exist_ok=True)
    publish_exact(unit_dir / UNIT_NAME, service_unit(SDK_PYTHON, source / "beta_access.py", config_path).encode())
    # user services require a running user manager. Never edit root/global units.
    os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus")
    run(["systemctl", "--user", "daemon-reload"])
    run(["systemctl", "--user", "enable", "--now", UNIT_NAME])
    phase("user_service_started")
    linger = run(["loginctl", "show-user", "ldu", "-p", "Linger", "--value"]).strip() == b"yes"
    if not linger:
        # A normal user may enable their own lingering on some systems. No sudo,
        # prompts, or global policy edits are attempted by the installer.
        try:
            subprocess.run(["loginctl", "--no-ask-password", "enable-linger", "ldu"], capture_output=True, timeout=15)
        except subprocess.TimeoutExpired:
            pass
        linger = run(["loginctl", "show-user", "ldu", "-p", "Linger", "--value"]).strip() == b"yes"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            value = json.loads((BASE / "status.json").read_text())
            stamp = datetime.datetime.fromisoformat(value["updated_at_utc"])
            fresh = 0 <= (datetime.datetime.now(datetime.timezone.utc) - stamp).total_seconds() < 45
            if value.get("state") == "ready" and fresh:
                print(json.dumps({"ready": True, "persistent_after_logout": linger,
                                  "alpha_required": False, "mac_required": False, "reverse_port": PORT}))
                if not linger:
                    print("User-manager lingering is not enabled; ask the operator to enable it for ldu.")
                return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(2)
    raise RuntimeError("service installed but not ready; inspect systemctl --user status roborender-beta-access")


def ssh_args(config):
    return ["ssh", "-F", config["ssh_config"]]


def tunnel_args(config):
    # Private master belongs only to this service. Cold cloud-shell startup is
    # inside SSH's ConnectTimeout, so allow its bounded stages to finish.
    return ssh_args(config) + ["-o", "ControlMaster=yes", "-o", f"ControlPath={CONTROL}",
                              "-o", "ControlPersist=no", "-o", "ConnectTimeout=65",
                              "-NT", "-R", f"127.0.0.1:{PORT}:127.0.0.1:22", CPU_ALIAS]


def health(config):
    # Reuse the authenticated tunnel instead of opening another cloud shell
    # every 30 s. A missing/dead master MUST NOT silently fall back to a new
    # connection: ProxyCommand=false closes that check/use race.
    run(ssh_args(config) + ["-S", str(CONTROL), "-O", "check", CPU_ALIAS], timeout=3)
    value = json.loads(run(ssh_args(config) + ["-S", str(CONTROL), "-o", "ProxyCommand=/usr/bin/false",
                                            CPU_ALIAS, "roborender-health"], timeout=20))
    if (value.get("ok") is not True or value.get("beta_hostname") != BETA_HOST
            or not value.get("cpu_hostname", "").startswith(CPU_POD + "-")
            or value.get("reverse_port") != PORT):
        raise RuntimeError("authenticated bidirectional health failed")
    return value


def stop_owned(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    except ProcessLookupError:
        pass


def remove_stale_control_socket():
    """Remove only our dead Unix socket, never an active master or other file.

    SIGKILL cannot unlink OpenSSH's socket. A subsequent SSH can otherwise
    disable multiplexing while the health checker keeps using the dead path.
    The supervisor lock serializes our callers; inode checks reject replacement.
    """
    if CONTROL.parent != BASE or BASE.is_symlink():
        raise RuntimeError('control socket is outside the private service directory')
    parent = BASE.stat()
    if parent.st_uid != os.getuid() or parent.st_mode & 0o022:
        raise RuntimeError('control directory is not privately owned')
    try:
        before = CONTROL.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISSOCK(before.st_mode) or before.st_uid != os.getuid():
        raise RuntimeError('refusing non-owned or non-socket control path')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        try:
            probe.connect(str(CONTROL))
        except OSError as error:
            if error.errno != errno.ECONNREFUSED:
                raise RuntimeError('control socket state is ambiguous') from error
        else:
            raise RuntimeError('an active private SSH master already exists')
    after = CONTROL.lstat()
    if (before.st_dev, before.st_ino, before.st_uid, before.st_mode) != (
            after.st_dev, after.st_ino, after.st_uid, after.st_mode):
        raise RuntimeError('control socket changed during stale check')
    CONTROL.unlink()
    return True


def watchdog_ping():
    """Main-loop heartbeat only: a stuck SDK/SSH call must not look healthy."""
    address=os.environ.get('NOTIFY_SOCKET')
    expected=os.environ.get('WATCHDOG_PID')
    if not address or (expected is not None and expected!=str(os.getpid())):
        return False
    if address.startswith('@'):address='\0'+address[1:]
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as notifier:
            notifier.settimeout(1)
            notifier.connect(address)
            notifier.sendall(b'WATCHDOG=1')
        return True
    except OSError:
        return False


def supervise(config):
    require_beta()
    checked_config(config, installed=True)
    os.umask(0o077)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())
    last_state = None

    def status(state, **extra):
        nonlocal last_state
        value = {"schema": "roborender.beta-access.health.v1", "state": state,
                 "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}
        atomic_json(BASE / "status.json", value)
        if state != last_state:
            print(json.dumps(value), flush=True)
            last_state = state

    with (BASE / "supervisor.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failures = 0
        while not stop.is_set():
            watchdog_ping()
            process = None
            stage = "ssh_connect"
            try:
                # Enrollment already happened at install. Retry it after two
                # failed connection cycles to restore a restarted CPU pod's
                # authorization, not on every ordinary transport reconnect.
                if failures and failures % 2 == 0:
                    stage = "restore_enrollment"
                    status("connecting", stage=stage)
                    enroll(config)
                stage = "ssh_connect"
                status("connecting", stage=stage)
                remove_stale_control_socket()
                process = subprocess.Popen(tunnel_args(config),
                                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                           start_new_session=True)
                failed = 0
                # Detect an exited SSH process promptly, independently of the
                # more expensive 30-second authenticated health cadence.
                next_health = time.monotonic() + 3
                startup_deadline = time.monotonic() + 75
                while not stop.wait(1):
                    watchdog_ping()
                    if process.poll() is not None:
                        raise RuntimeError("owned SSH tunnel exited")
                    if time.monotonic() < next_health:
                        continue
                    stage = "authenticated_return_hop"
                    try:
                        observation = health(config)
                    except Exception:
                        if last_state != "ready" and time.monotonic() < startup_deadline:
                            next_health = time.monotonic() + 3
                            continue
                        failed += 1
                        if failed >= 2:
                            raise RuntimeError("two authenticated health checks failed")
                        next_health = time.monotonic() + 5
                    else:
                        failed = failures = 0
                        status("ready", ssh_pid=process.pid, health=observation)
                        next_health = time.monotonic() + 30
            except Exception as error:
                failures += 1
                status("reconnecting", stage=stage, error_type=type(error).__name__, retry=failures)
            finally:
                if process is not None:
                    stop_owned(process)
            stop.wait(min(30, 5 * 2 ** min(failures, 3)))
        status("stopped")


def upgrade(public_config):
    """Upgrade only this service's attested unit/proxy path; retain rollback."""
    require_beta()
    checked_config(public_config)
    config_path = BASE / "config.json"
    installed = checked_config(json.loads(config_path.read_bytes()), installed=True)
    if any(installed[k] != v for k, v in public_config.items()):
        raise RuntimeError("endpoint identity changed; not a routine service upgrade")
    unit_path = Path("/home/ldu/.config/systemd/user") / UNIT_NAME
    old = unit_path.read_text()
    match = re.search(r'ExecStart="([^"]+)" "([^"]+)" run --config "([^"]+)"', old)
    if not match:
        raise RuntimeError("unrecognized service unit; refusing replacement")
    python, script, config = map(Path, match.groups())
    if python != SDK_PYTHON or config != config_path or script.name != "beta_access.py" or script.parent.parent != BASE / "versions":
        raise RuntimeError("service is outside owned release scope")
    expected_unit=service_unit(python,script,config)
    if old not in (expected_unit,expected_unit.replace('WatchdogSec=180\nNotifyAccess=main\n','')):
        raise RuntimeError("service unit has unreviewed customizations")
    ssh_path = Path(installed["ssh_config"])
    old_ssh = ssh_path.read_bytes()
    if old_ssh != make_ssh_config(python, script.parent / "cloud_proxy.py", BASE / "known_hosts").encode():
        raise RuntimeError("SSH configuration has unreviewed customizations")
    source = Path(__file__).resolve().parent
    if source.parent != BASE / "versions" or not re.fullmatch(r"[a-f0-9]{64}", source.name):
        raise RuntimeError("upgrade source must be an immutable installed release")
    backup = BASE / "rollback" / hashlib.sha256(old.encode()).hexdigest()
    backup.mkdir(parents=True, exist_ok=True, mode=0o700)
    publish_exact(backup / "service.unit", old.encode())
    publish_exact(backup / "ssh.conf", old_ssh)
    for path, content in ((ssh_path, make_ssh_config(python, source / "cloud_proxy.py", BASE / "known_hosts").encode()),
                          (unit_path, service_unit(python, source / "beta_access.py", config).encode())):
        if path.is_symlink():
            raise RuntimeError("refusing symlink managed configuration")
        temporary = path.with_name(path.name + ".upgrade")
        publish_exact(temporary, content)
        os.replace(temporary, path)
    run(["systemctl", "--user", "daemon-reload"])
    run(["systemctl", "--user", "restart", "--no-block", UNIT_NAME])
    phase("owned_service_upgrade_restart_requested")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["install", "upgrade", "run", "check"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text())
        if args.action == "install":
            install(config)
        elif args.action == "upgrade":
            upgrade(config)
        elif args.action == "run":
            supervise(config)
        else:
            checked_config(config, installed=True)
            print(json.dumps(health(config), sort_keys=True))
    except Exception as error:
        print("RoboRender Beta access failed: " + type(error).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
