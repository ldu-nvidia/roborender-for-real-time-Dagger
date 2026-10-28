"""CPU-only protocol/security regressions; NOT a live cloud qualification."""
from test.deployment.source_paths import CONNECTIVITY_ROOT
import base64
import hashlib
import json
import os
import socket
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import zipfile

import pytest

HERE = CONNECTIVITY_ROOT
sys.path.insert(0, str(HERE))
import beta_access as beta
import cloud_proxy as cloud
import cpu_endpoint as cpu
from build_bundle import build
from bundle_main import unpack

CPU_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIC9QSbt8+DhCPb76OtKNV6vtFc2luqtHX0br/A4ox9ZF"
BETA_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILR7lJiAL56+hSM5+O1QCfz7zbRVYPkGepUVtjlxFIFA"
URL = "https://gateway.dgxc-lepton.nvidia.com/api/v2/workspaces/b5k2m9x7"


def test_stale_socket_cleanup_preserves_active_master(tmp_path,monkeypatch):
    tmp_path.chmod(0o700)
    path=tmp_path/'master.sock'
    monkeypatch.setattr(beta,'BASE',tmp_path)
    monkeypatch.setattr(beta,'CONTROL',path)
    assert beta.remove_stale_control_socket() is False
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as master:
        master.bind(str(path));master.listen(1)
        with pytest.raises(RuntimeError,match='active private'):
            beta.remove_stale_control_socket()
        assert path.exists()
    assert beta.remove_stale_control_socket() is True
    assert not path.exists()


@pytest.mark.parametrize('kind',['regular','symlink','shared_directory'])
def test_stale_socket_cleanup_rejects_unsafe_paths(tmp_path,monkeypatch,kind):
    tmp_path.chmod(0o700)
    path=tmp_path/'master.sock'
    monkeypatch.setattr(beta,'BASE',tmp_path)
    monkeypatch.setattr(beta,'CONTROL',path)
    if kind=='regular':path.write_text('preserve')
    elif kind=='symlink':path.symlink_to(tmp_path/'other')
    else:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as master:master.bind(str(path))
        tmp_path.chmod(0o777)
    with pytest.raises(RuntimeError):beta.remove_stale_control_socket()
    assert path.exists() or path.is_symlink()


def test_owned_tunnel_watchdog_notifies_only_main_loop_process(tmp_path,monkeypatch):
    path=str(tmp_path/'notify.sock')
    with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as receiver:
        receiver.bind(path);receiver.settimeout(1)
        monkeypatch.setenv('NOTIFY_SOCKET',path)
        monkeypatch.setenv('WATCHDOG_PID',str(os.getpid()))
        assert beta.watchdog_ping() is True
        assert receiver.recv(100)==b'WATCHDOG=1'
        monkeypatch.setenv('WATCHDOG_PID',str(os.getpid()+1))
        assert beta.watchdog_ping() is False
    monkeypatch.delenv('NOTIFY_SOCKET')
    assert beta.watchdog_ping() is False
    unit=beta.service_unit('/python','/source/beta_access.py','/config.json')
    assert 'WatchdogSec=180\nNotifyAccess=main\n' in unit
    assert 'KillMode=control-group' in unit


def test_authenticated_pins_and_key_validation():
    assert cpu.fingerprint(CPU_KEY) == beta.CPU_PIN
    assert cpu.fingerprint(BETA_KEY) == cpu.BETA_PIN
    assert cpu.public_key(CPU_KEY + " comment") == CPU_KEY
    for value in (CPU_KEY + "\n", "ssh-rsa AAAA", "ssh-ed25519 AAAA", CPU_KEY + " a b"):
        with pytest.raises(ValueError): cpu.public_key(value)


@pytest.mark.parametrize("pod", [cloud.CPU_POD, "leo-rr-beta-0123456789ab-r1", "leo-rr-beta-0123456789ab-r4"])
def test_cloud_scope_accepts_only_cpu_or_exact_owned_campaign_pod(pod):
    value = cloud.websocket_url(URL, pod, "replica-0")
    assert value == URL.replace("https:", "wss:") + f"/deployments/{pod}/replicas/replica-0/shell"


@pytest.mark.parametrize("url,pod,replica", [
    (URL.replace("https:", "http:"), cloud.CPU_POD, "r"),
    (URL.replace("gateway.dgxc-lepton.nvidia.com", "example.com"), cloud.CPU_POD, "r"),
    (URL + "?other=1", cloud.CPU_POD, "r"),
    (URL, "unrelated-workload", "r"),
    (URL, "leo-rr-beta-0123456789ab-r5", "r"),
    (URL, cloud.CPU_POD, "../../bad"),
])
def test_cloud_scope_rejects_unapproved_destinations(url, pod, replica):
    with pytest.raises(ValueError): cloud.websocket_url(url, pod, replica)


def test_replica_get_and_websocket_verify_destinations_before_auth(monkeypatch):
    response = SimpleNamespace(status_code=200, json=lambda: [{"metadata": {"id": "replica-0"}}])
    client = SimpleNamespace(url=URL, _header={"Authorization": "TEST_ONLY"}, _get=Mock(return_value=response))
    record = SimpleNamespace(client=Mock(return_value=client))
    connect = Mock(return_value=object())
    monkeypatch.setitem(sys.modules, "leptonai.api.v2.workspace_record", SimpleNamespace(WorkspaceRecord=record))
    monkeypatch.setitem(sys.modules, "websocket", SimpleNamespace(create_connection=connect))
    cloud.open_shell(cloud.CPU_POD)
    client._get.assert_called_once_with(f"/deployments/{cloud.CPU_POD}/replicas", allow_redirects=False)
    _, kwargs = connect.call_args
    assert kwargs["redirect_limit"] == 0 and "sslopt" not in kwargs
    assert kwargs["origin"] == cloud.ORIGIN and kwargs["subprotocols"] == ["v4.channel.k8s.io"]
    client.url = "https://example.com"
    client._get.reset_mock()
    with pytest.raises(ValueError): cloud.open_shell(cloud.CPU_POD)
    client._get.assert_not_called()


def test_preamble_marker_cannot_match_terminal_echo_and_preserves_ssh_banner():
    marker = b"\x1eSSH_TEST\x1f"
    command = cloud.marker_command(marker)
    assert marker not in command.encode()
    assert subprocess.check_output(["bash", "-c", command]) == marker
    ws = SimpleNamespace(settimeout=Mock(), recv=Mock(side_effect=[
        b"\x01prompt " + command.encode(), b"\x02ignored-stderr", b"\x01\x1eSSH_", b"\x01TEST\x1fSSH-2.0-test\r\n"]))
    before, after = cloud.read_until(ws, marker)
    assert before.startswith(b"prompt") and after == b"SSH-2.0-test\r\n"


def test_bounded_shell_preamble_and_exit_status():
    ws = SimpleNamespace(settimeout=Mock(), recv=Mock(return_value=b"\x01" + b"x" * 100))
    with pytest.raises(RuntimeError): cloud.read_until(ws, b"absent", limit=20)
    with pytest.raises(EOFError): cloud.channel_payload(b"")
    with pytest.raises(RuntimeError): cloud.channel_payload(b"\x03remote error")


def test_provisioning_combined_markers_and_json(monkeypatch):
    monkeypatch.setattr(cloud.secrets, "token_hex", lambda n: "a" * 24)
    ws = SimpleNamespace(settimeout=Mock(), send_binary=Mock(), close=Mock(), recv=Mock(return_value=(
        b"\x01\x1eRR_" + b"a" * 24 + b'\x1f{"ok":true}\r\n\x1eEND_' + b"a" * 24 + b"\x1f")))
    monkeypatch.setattr(cloud, "open_shell", lambda pod: ws)
    assert json.loads(cloud.run_code("print('test')")) == {"ok": True}
    assert ws.send_binary.call_args.args[0].startswith(b"\x00stty -echo;")
    ws.close.assert_called_once()


def test_binary_output_partial_writes(monkeypatch):
    values = []
    def partial(fd, data):
        values.append(bytes(data[:3]))
        return min(3, len(data))
    monkeypatch.setattr(cloud.os, "write", partial)
    cloud.write_all(1, bytes(range(32)))
    assert b"".join(values) == bytes(range(32))


def test_key_authorization_preserves_others_and_limits_forwarding(tmp_path):
    path = tmp_path / "authorized_keys"
    path.write_text("existing-user-key\n")
    line = cpu.authorized_line(CPU_KEY)
    assert 'restrict,port-forwarding,permitlisten="127.0.0.1:22032"' in line
    assert 'permitopen="127.0.0.1:1"' in line
    assert "cpu_endpoint.py health" in line
    assert cpu.append_authorization(path, line)
    assert not cpu.append_authorization(path, line)
    assert path.read_text() == "existing-user-key\n" + line
    with pytest.raises(RuntimeError): cpu.append_authorization(path, line.replace("22032", "22033"))


def test_managed_files_never_overwrite_or_follow_symlinks(tmp_path):
    path = tmp_path / "managed"
    cpu.publish_exact(path, b"original")
    cpu.publish_exact(path, b"original")
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(RuntimeError): cpu.publish_exact(path, b"changed")
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(RuntimeError): cpu.publish_exact(link, b"original")


def test_ssh_and_user_service_are_loopback_owned_and_alpha_free(tmp_path):
    conf = beta.make_ssh_config(beta.SDK_PYTHON, "/home/ldu/cloud_proxy.py", "/home/ldu/pins")
    assert "StrictHostKeyChecking yes" in conf and "ForwardAgent no" in conf
    assert "rtx-6k-alpha" not in conf and "ProxyJump" not in conf
    path = tmp_path / "ssh.conf"
    path.write_text(conf)
    result = subprocess.check_output(["ssh", "-G", "-F", str(path), beta.CPU_ALIAS], text=True)
    assert "stricthostkeychecking true" in result
    assert "controlmaster false" in result
    unit = beta.service_unit(beta.SDK_PYTHON, "/home/ldu/beta_access.py", "/home/ldu/config.json")
    assert "Restart=always" in unit and "KillMode=control-group" in unit
    assert "sudo" not in unit and "CUDA" not in unit and "Mac" not in unit


def test_health_requires_real_return_hop_and_expected_host(monkeypatch):
    config = {"ssh_config": "/home/ldu/.local/share/roborender-beta-access/ssh.conf"}
    good = {"ok": True, "beta_hostname": cpu.BETA_HOST, "cpu_hostname": cloud.CPU_POD + "-test", "reverse_port": 22032}
    monkeypatch.setattr(beta, "run", lambda *a, **k: json.dumps(good).encode())
    assert beta.health(config) == good
    good["beta_hostname"] = "wrong-host"
    with pytest.raises(RuntimeError): beta.health(config)
    monkeypatch.setenv("SSH_ORIGINAL_COMMAND", "some-other-command")
    with pytest.raises(ValueError): cpu.health()


def test_health_reuses_private_master_without_cloud_fallback(monkeypatch):
    config = {"ssh_config": str(beta.BASE / "ssh.conf")}
    good = {"ok": True, "beta_hostname": cpu.BETA_HOST, "cpu_hostname": cloud.CPU_POD + "-test", "reverse_port": 22032}
    runner = Mock(side_effect=[b"", json.dumps(good).encode()])
    monkeypatch.setattr(beta, "run", runner)
    assert beta.health(config) == good
    check, request = [c.args[0] for c in runner.call_args_list]
    assert "-O" in check and "check" in check and str(beta.CONTROL) in check
    assert "ProxyCommand=/usr/bin/false" in request and str(beta.CONTROL) in request
    command = beta.tunnel_args(config)
    assert "ControlMaster=yes" in command and "ConnectTimeout=65" in command
    assert command[-1] == beta.CPU_ALIAS and command[-2] == "127.0.0.1:22032:127.0.0.1:22"


def test_live_reconnect_probe_checks_exact_child_and_supervisor():
    from verify_live import stop_script
    source = stop_script(100, 99)
    compile(source, "reconnect-probe", "exec")
    assert "argv in (legacy,current)" in source and "PPid:" in source
    assert "beta_access.py" in source and "os.kill(pid,signal.SIGTERM)" in source
    assert "killpg" not in source


def test_cpu_health_does_not_claim_success_for_open_port_only(monkeypatch):
    monkeypatch.delenv("SSH_ORIGINAL_COMMAND", raising=False)
    called = Mock(return_value=SimpleNamespace(returncode=0, stdout=b"wrong-host\n"))
    monkeypatch.setattr(cpu.subprocess, "run", called)
    assert cpu.health()["ok"] is False
    assert called.call_args.args[0][-1] == "/bin/hostname"
    called.return_value.stdout = (cpu.BETA_HOST + "\n").encode()
    assert cpu.health()["ok"] is True


def test_configuration_is_pinned_and_rejects_extra_fields():
    value = {"cpu_endpoint_path": beta.CPU_ENDPOINT, "cpu_endpoint_sha256": "a" * 64}
    beta.checked_config(value)
    installed = dict(value, schema="roborender.beta-access.v1", ssh_config=str(beta.BASE / "ssh.conf"))
    beta.checked_config(installed, installed=True)
    with pytest.raises(ValueError): beta.checked_config(dict(value, token="TEST_ONLY"))
    with pytest.raises(ValueError): beta.checked_config(dict(value, cpu_endpoint_path="/different.py"))
    with pytest.raises(ValueError): beta.checked_config(dict(installed, ssh_config="/different"), installed=True)


def test_source_only_bundle_is_reproducible_and_checked(tmp_path):
    a, b = tmp_path / "a.pyz", tmp_path / "b.pyz"
    first, second = build(a), build(b)
    assert first["sha256"] == second["sha256"]
    assert first["members"] == sorted(["cloud_proxy.py", "cpu_endpoint.py", "beta_access.py", "public-config.json", "MANIFEST.json", "__main__.py"])
    target = unpack(a, tmp_path / "install")
    assert unpack(a, tmp_path / "install") == target
    (target / "cloud_proxy.py").write_text("changed")
    with pytest.raises(RuntimeError): unpack(a, tmp_path / "install")
    config = json.loads((target / "public-config.json").read_text())
    assert config["cpu_endpoint_sha256"] == hashlib.sha256((HERE / "cpu_endpoint.py").read_bytes()).hexdigest()


def test_bundle_rejects_traversal_and_changed_payload(tmp_path):
    archive = tmp_path / "bad.pyz"
    build(archive)
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr("../outside", "no")
    with pytest.raises(ValueError): unpack(archive, tmp_path / "install")
