"""Real loopback SSH regression: allocation identity must survive sshd."""
from test.deployment.source_paths import REPO_ROOT
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import pytest

from pipeline.deployment.h200.beta_service_host import require_allocation_identity


def test_service_identity_guard_still_fails_closed(monkeypatch):
    monkeypatch.delenv("ROBORENDER_BETA_RUN_ID", raising=False)
    with pytest.raises(RuntimeError, match="run identity differs"):
        require_allocation_identity("dagger-leo-test")
    monkeypatch.setenv("ROBORENDER_BETA_RUN_ID", "dagger-leo-other")
    with pytest.raises(RuntimeError, match="run identity differs"):
        require_allocation_identity("dagger-leo-test")
    monkeypatch.setenv("ROBORENDER_BETA_RUN_ID", "dagger-leo-test")
    require_allocation_identity("dagger-leo-test")


@pytest.fixture
def ssh_identity_directory():
    # StrictModes correctly rejects authorized_keys below world-writable /tmp.
    # A private temporary directory beneath the login home keeps that check on.
    with tempfile.TemporaryDirectory(prefix="roborender-sshd-test-", dir=pwd.getpwuid(os.geteuid()).pw_dir) as path:
        yield Path(path)


@pytest.mark.parametrize("server_setenv", [False, True])
def test_real_ssh_preserves_only_server_bound_allocation_identity(ssh_identity_directory, server_setenv):
    tmp_path = ssh_identity_directory
    sshd, ssh, keygen = (shutil.which(name) for name in ("sshd", "ssh", "ssh-keygen"))
    if not all((sshd, ssh, keygen)) or os.geteuid() != 0:
        pytest.skip("isolated sshd integration requires root and local OpenSSH")
    root = REPO_ROOT
    run_id = "dagger-leo-ssh-identity-test"
    user = pwd.getpwuid(os.geteuid()).pw_name
    for name in ("host", "client"):
        subprocess.run([keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(tmp_path/name)], check=True)
    (tmp_path/"authorized_keys").write_bytes((tmp_path/"client.pub").read_bytes())
    (tmp_path/"authorized_keys").chmod(0o600)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    known = tmp_path/"known_hosts"
    known.write_text(f"[127.0.0.1]:{port} " + (tmp_path/"host.pub").read_text())
    config = tmp_path/"sshd_config"
    config.write_text("\n".join([
        f"Port {port}", "ListenAddress 127.0.0.1", f"HostKey {tmp_path/'host'}",
        f"PidFile {tmp_path/'sshd.pid'}", f"AuthorizedKeysFile {tmp_path/'authorized_keys'}",
        "StrictModes yes", "PubkeyAuthentication yes", "AuthenticationMethods publickey",
        "PasswordAuthentication no", "KbdInteractiveAuthentication no", "PermitRootLogin prohibit-password",
        "UsePAM yes", "AllowTcpForwarding no", "AllowAgentForwarding no", "X11Forwarding no",
        "PermitUserEnvironment no", "PermitTTY no", f"AllowUsers {user}", "LogLevel VERBOSE",
        # Deliberately admit the attempted client spoof: server SetEnv must win.
        "AcceptEnv ROBORENDER_BETA_RUN_ID ROBORENDER_BETA_ROUND_ID", "",
    ]))
    argv = [sshd, "-D", "-e", "-f", str(config)]
    if server_setenv:
        entry = (root/"pipeline/deployment/h200/beta_pod_entrypoint.sh").read_text()
        option = re.findall(r'-o "(SetEnv=[^"\n]+)"', entry)
        assert len(option) == 1
        option = option[0].replace("$ROBORENDER_BETA_RUN_ID", run_id).replace("$ROBORENDER_BETA_ROUND_ID", "1")
        assert "$" not in option
        argv += ["-o", option]
    # Inheritance alone reproduces the original bug in a real SSH session.
    env = dict(os.environ, ROBORENDER_BETA_RUN_ID=run_id, ROBORENDER_BETA_ROUND_ID="1")
    with (tmp_path/"sshd.log").open("wb") as log:
        server = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+10
            while True:
                if server.poll() is not None:
                    raise AssertionError((tmp_path/"sshd.log").read_text())
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.2):
                        break
                except OSError:
                    if time.monotonic() > deadline: raise
                    time.sleep(.05)
            base = [ssh, "-F", "/dev/null", "-p", str(port), "-i", str(tmp_path/"client"),
                    "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                    "-o", f"UserKnownHostsFile={known}", "-o", "ConnectTimeout=5"]
            probe = shlex.join([sys.executable, "-c", "import json,os;print(json.dumps({k:os.getenv(k) for k in "
                "('ROBORENDER_BETA_RUN_ID','ROBORENDER_BETA_ROUND_ID')}))"])
            def query(extra=()):
                result = subprocess.run([*base, *extra, f"{user}@127.0.0.1", probe],
                                        capture_output=True, text=True, timeout=10, check=True)
                return json.loads(result.stdout)
            actual = query()
            assert actual == {"ROBORENDER_BETA_RUN_ID": run_id if server_setenv else None,
                              "ROBORENDER_BETA_ROUND_ID": "1" if server_setenv else None}
            if server_setenv:
                assert query(["-o", "SetEnv=ROBORENDER_BETA_RUN_ID=wrong-run ROBORENDER_BETA_ROUND_ID=4"]) == actual
                check = shlex.join([sys.executable, "-c", f"import sys;sys.path.insert(0,{str(root)!r});"
                    "from pipeline.deployment.h200.beta_service_host import require_allocation_identity;"
                    f"require_allocation_identity({run_id!r});print('identity-pass')"])
                result = subprocess.run([*base, f"{user}@127.0.0.1", check], capture_output=True,
                                        text=True, timeout=10, check=True)
                assert result.stdout.strip() == "identity-pass"
        finally:
            server.terminate()
            server.wait(timeout=10)
