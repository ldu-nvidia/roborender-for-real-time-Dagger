from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import socket

import pytest

from tunnel_supervisor import IdentityMismatch, Supervisor, read_config, ssh_command


def configuration(tmp_path):
    for name in ("ssh.config", "known_hosts"):
        (tmp_path / name).touch(mode=0o600)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return {"schema": "roborender.ssh-supervisor.v1", "run_id": "test-supervised",
        "ssh_config": str(tmp_path / "ssh.config"), "known_hosts": str(tmp_path / "known_hosts"),
        "ssh_host": "owned-host", "lifetime_seconds": 1800,
        "forwards": [{"local_port": port, "remote_port": 18620,
                      "expected_ready": {"ok": True, "role": "render_student", "pid": 123}}]}


def test_config_and_ssh_keep_host_verification_and_loopback_binding(tmp_path):
    config = configuration(tmp_path)
    path = tmp_path / "supervisor.json"
    path.write_text(json.dumps(config))
    loaded, digest = read_config(path)
    assert loaded == config and len(digest) == 64
    command = ssh_command(loaded, loaded["forwards"][0])
    assert "StrictHostKeyChecking=yes" in command and "ForwardAgent=no" in command
    assert command[command.index("-L") + 1].startswith("127.0.0.1:")
    assert "ControlPath=none" in command
    (tmp_path / "ssh.config").chmod(0o666)
    with pytest.raises(ValueError, match="unsafe"):
        read_config(path)


def test_dead_owned_forward_reconnects_without_taking_over_other_processes(tmp_path, monkeypatch):
    killed, spawned = [], []
    monkeypatch.setattr(os, "killpg", lambda pid, signal: killed.append(pid))

    class Child:
        def __init__(self):
            self.pid, self.returncode = 40000 + len(spawned), None
        def poll(self):
            return self.returncode
        def wait(self, timeout):
            self.returncode = -15

    def popen(*args, **kwargs):
        child = Child()
        spawned.append(child)
        return child

    config, clock = configuration(tmp_path), [0.]
    root = tmp_path / "state"
    root.mkdir()
    supervisor = Supervisor(config, "a" * 64, root, popen=popen,
        health=lambda row: (True, None), monotonic=lambda: clock[0])
    supervisor.events = io.StringIO()
    with ThreadPoolExecutor(max_workers=1) as pool:
        supervisor.cycle(pool)
        assert supervisor.entries[0]["healthy"]
        spawned[0].returncode = 255
        clock[0] = 2.
        supervisor.cycle(pool)
        assert not supervisor.entries[0]["healthy"]
        clock[0] = 3.
        supervisor.cycle(pool)
        assert supervisor.entries[0]["healthy"] and len(spawned) == 2
        supervisor.terminate(supervisor.entries[0])
    assert killed == [spawned[0].pid, spawned[1].pid]
    assert json.loads((root / "status.json").read_text())["forwards"][0]["starts"] == 2


def test_changed_service_identity_fails_closed(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    root = tmp_path / "state"
    root.mkdir()

    class Child:
        pid = 40000
        def poll(self):
            return None
        def wait(self, timeout):
            return -15

    def mismatch(_row):
        raise IdentityMismatch("gateway changed")

    monkeypatch.setattr(os, "killpg", lambda *args: None)
    supervisor = Supervisor(config, "a" * 64, root, popen=lambda *a, **kw: Child(), health=mismatch)
    supervisor.events = io.StringIO()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(IdentityMismatch):
            supervisor.cycle(pool)
    supervisor.terminate(supervisor.entries[0])


def test_short_health_jitter_does_not_restart_live_owned_ssh(tmp_path, monkeypatch):
    clock, killed, spawned = [0.], [], []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.append(pid))
    class Child:
        pid = 43001
        def poll(self): return None
        def wait(self, timeout): return -15
    def popen(*args, **kwargs):
        spawned.append(Child())
        return spawned[-1]
    root = tmp_path / "jitter"
    root.mkdir()
    supervisor = Supervisor(configuration(tmp_path), "a" * 64, root, popen=popen,
        monotonic=lambda: clock[0], health=lambda row: (clock[0] == 0 or clock[0] >= 12, "TimeoutError"))
    supervisor.events = io.StringIO()
    with ThreadPoolExecutor(max_workers=1) as pool:
        supervisor.cycle(pool)
        for timestamp in (1., 3., 7.):
            clock[0] = timestamp
            supervisor.cycle(pool)
        assert len(spawned) == 1 and not killed
        assert not supervisor.entries[0]["healthy"]
        clock[0] = 12.
        supervisor.cycle(pool)
        assert supervisor.entries[0]["healthy"] and len(spawned) == 1
    supervisor.terminate(supervisor.entries[0])


def test_continuous_unresponsive_link_restarts_after_grace_and_still_has_deadline(tmp_path, monkeypatch):
    from tunnel_supervisor import RECOVERY_SECONDS, UNRESPONSIVE_GRACE_SECONDS
    clock, killed, spawned = [0.], [], []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.append(pid))
    class Child:
        def __init__(self): self.pid = 44000 + len(spawned)
        def poll(self): return None
        def wait(self, timeout): return -15
    def popen(*args, **kwargs):
        child = Child()
        spawned.append(child)
        return child
    root = tmp_path / "permanent"
    root.mkdir()
    supervisor = Supervisor(configuration(tmp_path), "a" * 64, root, popen=popen,
        monotonic=lambda: clock[0], health=lambda row: (clock[0] == 0, "TimeoutError"))
    supervisor.events = io.StringIO()
    with ThreadPoolExecutor(max_workers=1) as pool:
        supervisor.cycle(pool)
        for timestamp in (1., 6., 11.):
            clock[0] = timestamp
            supervisor.cycle(pool)
        assert not killed
        clock[0] = 1 + UNRESPONSIVE_GRACE_SECONDS
        supervisor.cycle(pool)
        assert killed == [spawned[0].pid]
        clock[0] += 1
        supervisor.cycle(pool)
        assert len(spawned) == 2
        clock[0] = 1 + RECOVERY_SECONDS + .1
        with pytest.raises(RuntimeError, match="exceeded recovery budget"):
            supervisor.cycle(pool)
    supervisor.terminate(supervisor.entries[0])


def test_retained_transport_recovers_after_long_outage_without_native_restart(tmp_path, monkeypatch):
    from tunnel_supervisor import PRODUCTION_SCHEMA, RECOVERY_SECONDS
    clock=[0.];config=configuration(tmp_path);config['schema']=PRODUCTION_SCHEMA
    root=tmp_path/'retained';root.mkdir()
    class Child:
        pid=45101
        def poll(self):return None
        def wait(self,timeout):return -15
    monkeypatch.setattr(os,'killpg',lambda *a:None)
    supervisor=Supervisor(config,'a'*64,root,popen=lambda *a,**kw:Child(),
        monotonic=lambda:clock[0],health=lambda row:(clock[0]>RECOVERY_SECONDS,'TimeoutError'))
    supervisor.events=io.StringIO()
    with ThreadPoolExecutor(max_workers=1) as pool:
        supervisor.cycle(pool)
        clock[0]=RECOVERY_SECONDS+10
        supervisor.cycle(pool)
        assert supervisor.entries[0]['healthy']
        supervisor.health=lambda row:(_ for _ in ()).throw(IdentityMismatch('changed'))
        with pytest.raises(IdentityMismatch):supervisor.cycle(pool)
    supervisor.terminate(supervisor.entries[0])


@pytest.mark.parametrize("live_listener", [False, True])
def test_launch_tolerates_time_wait_but_refuses_live_listener(tmp_path, monkeypatch, live_listener):
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    address = listener.getsockname()
    listener.listen()
    client = socket.socket()
    client.connect(address)
    accepted, _ = listener.accept()
    accepted.shutdown(socket.SHUT_WR)
    assert client.recv(1) == b""
    accepted.close()
    client.close()
    if not live_listener:
        listener.close()

    config = configuration(tmp_path)
    config["forwards"][0]["local_port"] = address[1]
    root = tmp_path / "time-wait-state"
    root.mkdir()
    spawned = []
    class Child:
        pid = 41000
        def wait(self, timeout):
            return -15
    def popen(*args, **kwargs):
        spawned.append(args)
        return Child()
    monkeypatch.setattr(os, "killpg", lambda *args: None)
    supervisor = Supervisor(config, "a" * 64, root, popen=popen)
    supervisor.events = io.StringIO()
    try:
        if live_listener:
            with pytest.raises(OSError):
                supervisor.launch(supervisor.entries[0])
            assert not spawned
        else:
            supervisor.launch(supervisor.entries[0])
            assert len(spawned) == 1
            supervisor.terminate(supervisor.entries[0])
    finally:
        listener.close()
