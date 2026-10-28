from test.deployment.source_paths import MACOS_ROOT
import hashlib
import json
import plistlib
from pathlib import Path
import subprocess
import sys

import pytest
from pipeline.deployment.macos import install_bridge as bridge

from pipeline.deployment.macos.install_bridge import (
    BOOTSTRAP_SHA, LABEL, HealthGate, launch_spec, probe_bridge, publish,
    verify_bootstrap, verify_owned_plist,
)


def test_pinned_bootstrap_is_existing_reviewed_source():
    source = (MACOS_ROOT / "bootstrap-mac-bridge.sh")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == BOOTSTRAP_SHA
    verify_bootstrap(source.read_bytes())
    with pytest.raises(RuntimeError):
        verify_bootstrap(b"unreviewed bridge")


def test_launch_agent_is_user_scoped_persistent_and_private(tmp_path):
    spec = launch_spec(tmp_path, tmp_path / "logs", "/usr/bin/python3", False)
    assert spec["Label"] == LABEL
    assert spec["KeepAlive"] and spec["RunAtLoad"]
    assert spec["ThrottleInterval"] == 30 and spec["Umask"] == 0o077
    assert spec["ProgramArguments"] == ["/usr/bin/python3", str(tmp_path / "install_bridge.py"), "run"]
    assert "SSH_AUTH_SOCK" not in spec["EnvironmentVariables"]
    assert not any("sudo" in str(v) for v in spec.values())
    assert plistlib.loads(plistlib.dumps(spec)) == spec


def test_awake_setting_is_explicit_and_ac_only(tmp_path):
    spec = launch_spec(tmp_path, tmp_path, "/usr/bin/python3", True)
    assert spec["ProgramArguments"][:2] == ["/usr/bin/caffeinate", "-s"]


def test_foreign_or_symlink_launchagent_is_not_replaced(tmp_path):
    path = tmp_path / "test.plist"
    publish(path, plistlib.dumps({"Label": "somebody.else"}))
    with pytest.raises(RuntimeError):
        verify_owned_plist(path, tmp_path)
    owned = tmp_path / "owned.plist"
    publish(owned, plistlib.dumps(launch_spec(tmp_path, tmp_path, "/usr/bin/python3", False)))
    verify_owned_plist(owned, tmp_path)
    link = tmp_path / "symlink.plist"
    link.symlink_to(owned)
    with pytest.raises(RuntimeError):
        verify_owned_plist(link, tmp_path)
    with pytest.raises(RuntimeError):
        publish(link, b"do not overwrite")


def test_atomic_install_has_owner_only_permissions(tmp_path):
    path = tmp_path / "owned.py"
    publish(path, b"first", 0o700)
    publish(path, b"second", 0o700)
    assert path.read_bytes() == b"second"
    assert path.stat().st_mode & 0o777 == 0o700


def health_payload():
    return {"control_hostname": bridge.EXPECTED_CONTROL, "targets": {
        name: {"returncode": 0, "hostname": host}
        for name, host in bridge.EXPECTED_TARGETS.items()}}


def test_health_requires_real_verified_forwarded_hostnames():
    def runner(args, timeout):
        assert "ControlPath=/tmp/owned/%C" in args and "ControlMaster=no" in args
        assert "StrictHostKeyChecking=yes" in args and timeout == 25
        assert "/workspace/.ssh/roborender_migration_config" in args[-1]
        assert "roborender-migration-" in args[-1]
        return subprocess.CompletedProcess(args, 0, json.dumps(health_payload()))
    assert probe_bridge("/tmp/owned/%C", runner=runner)["ok"]


@pytest.mark.parametrize("failure", ["wrong_cpu", "wrong_beta", "port_closed", "local_master_only"])
def test_master_alive_is_never_proof_that_bridge_is_healthy(failure):
    payload = health_payload()
    if failure == "wrong_cpu":
        payload["control_hostname"] = "unrelated-pod"
    elif failure == "wrong_beta":
        payload["targets"]["beta"]["hostname"] = "unrelated-machine"
    elif failure == "port_closed":
        payload["targets"]["beta"] = {"returncode": 255, "hostname": ""}
    output = "Master running (pid=37805)" if failure == "local_master_only" else json.dumps(payload)
    assert not probe_bridge("/tmp/owned/%C", runner=lambda args, timeout:
                            subprocess.CompletedProcess(args, 0, output))["ok"]


def test_post_authentication_hang_is_bounded_and_unhealthy():
    def runner(args, timeout):
        raise TimeoutError("fixture")
    assert probe_bridge("/tmp/owned/%C", runner=runner) == {
        "ok": False, "reason": "control_session_timeout"}
    with pytest.raises(TimeoutError, match="deadline"):
        bridge.bounded_command([sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.05)


def test_only_repeated_failures_restart_and_success_resets_count():
    gate = HealthGate()
    assert not gate.ready
    assert not gate.observe({"ok": False})
    assert not gate.observe({"ok": True}) and gate.ready
    assert not gate.observe({"ok": False}) and not gate.ready
    assert gate.observe({"ok": False})


def test_shutdown_can_only_exit_this_attempts_private_masters(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(bridge, "run_private_worker", lambda app, control: 7)
    def runner(args, timeout):
        seen.append(args)
        assert args[2].startswith("ControlPath=/tmp/rr-bridge-")
        assert args[2].endswith("/%C")
    monkeypatch.setattr(bridge, "bounded_command", runner)
    assert bridge.run_worker(tmp_path) == 7
    assert [args[-1] for args in seen] == ["cosmos-h200", "rtx-6k-alpha", "rtx-6k-beta"]
    assert all(args[-3:-1] == ["-O", "exit"] for args in seen)


def test_worker_reconnects_even_while_bootstrap_process_is_alive(tmp_path, monkeypatch):
    script = tmp_path / "bootstrap-mac-bridge.sh"
    script.write_text("#!/bin/bash\necho 'Bridge forwards installed on the private SSH connection'\nsleep 60\n")
    monkeypatch.setattr(bridge, "verify_bootstrap", lambda _: None)
    monkeypatch.setattr(bridge, "HEALTH_INTERVAL", 0)
    monkeypatch.setattr(bridge, "probe_bridge", lambda _: {"ok": False, "reason": "round_trip_failed"})
    old_term, old_int = bridge.signal.getsignal(bridge.signal.SIGTERM), bridge.signal.getsignal(bridge.signal.SIGINT)
    try:
        assert bridge.run_private_worker(tmp_path, "/tmp/fixture/%C") == 1
    finally:
        bridge.signal.signal(bridge.signal.SIGTERM, old_term)
        bridge.signal.signal(bridge.signal.SIGINT, old_int)
    state = json.loads((tmp_path / "bridge-status.json").read_text())
    assert state["state"] == "reconnecting" and state["consecutive_failures"] == 2


def agent_fixture(tmp_path):
    app = tmp_path / "app"
    logs = tmp_path / "logs"
    app.mkdir()
    logs.mkdir()
    plist = tmp_path / (LABEL + ".plist")
    publish(plist, plistlib.dumps(launch_spec(app, logs, "/usr/bin/python3", True)))
    return app, logs, plist


def test_upgrade_restarts_owned_registration_without_bootstrap_race(tmp_path):
    app, logs, plist = agent_fixture(tmp_path)
    calls = []
    def runner(args, timeout):
        calls.append(args)
        out = f"path = {plist}\nstate = running\nenvironment = SECRET_NOT_FOR_REPORT" if args[1] == "print" else ""
        return subprocess.CompletedProcess(args, 0, out, "")
    bridge.activate_agent(plist, app, logs, reuse_registration=True, runner=runner)
    assert [args[1] for args in calls] == ["print", "enable", "kickstart"]
    assert calls[-1][2] == "-kp"
    report = (logs / "activation.json").read_text()
    assert "SECRET_NOT_FOR_REPORT" not in report
    assert json.loads(report)["activated"]


def test_reload_waits_for_removal_and_retries_transient_bootstrap5(tmp_path):
    app, logs, plist = agent_fixture(tmp_path)
    calls, waits = [], []
    reads, starts = 0, 0
    def runner(args, timeout):
        nonlocal reads, starts
        calls.append(args)
        op = args[1]
        if op == "print":
            reads += 1
            # Initial and first unload poll are still registered. Two absent
            # results follow, then the second bootstrap has registered it.
            exists = reads <= 2 or starts == 2
            return subprocess.CompletedProcess(args, 0 if exists else 113,
                                               f"path = {plist}\n" if exists else "", "")
        if op == "bootstrap":
            starts += 1
            return subprocess.CompletedProcess(args, 5 if starts == 1 else 0, "", "fixture error5")
        return subprocess.CompletedProcess(args, 0, "", "")
    bridge.activate_agent(plist, app, logs, reuse_registration=False, runner=runner, sleep=waits.append)
    assert starts == 2 and waits == [1, 1]
    ops = [a[1] for a in calls]
    assert ops.index("bootstrap") > ops.index("bootout") + 2
    assert json.loads((logs / "activation.json").read_text())["activated"]


def test_failed_unload_never_bootstraps_over_existing_job(tmp_path):
    app, logs, plist = agent_fixture(tmp_path)
    calls = []
    def runner(args, timeout):
        calls.append(args)
        return subprocess.CompletedProcess(args, 5 if args[1] == "bootout" else 0,
                                           f"path = {plist}\n" if args[1] == "print" else "", "")
    with pytest.raises(RuntimeError, match="did not unload"):
        bridge.activate_agent(plist, app, logs, reuse_registration=False, runner=runner, sleep=lambda _: None)
    assert not any("bootstrap" in a for a in calls)
    assert not json.loads((logs / "activation.json").read_text())["activated"]


def test_loaded_foreign_path_and_unknown_print_error_fail_closed(tmp_path):
    app, logs, plist = agent_fixture(tmp_path)
    for status, output in [(0, "path = /tmp/someone-else.plist\n"), (5, "")]:
        calls = []
        def runner(args, timeout):
            calls.append(args)
            return subprocess.CompletedProcess(args, status if args[1] == "print" else 0,
                                               output if args[1] == "print" else "", "")
        with pytest.raises(RuntimeError):
            bridge.activate_agent(plist, app, logs, reuse_registration=True, runner=runner, sleep=lambda _: None)
        assert [a[1] for a in calls] == ["print"]


def test_sanitized_activation_report_upload_uses_fresh_verified_cpu_route(tmp_path):
    report = {"service": "gui/502/com.roborender.migration-bridge", "events": [], "activated": False}
    publish(tmp_path / "activation.json", json.dumps(report).encode())
    def runner(args, timeout, input_data):
        assert "ControlPath=none" in args and "StrictHostKeyChecking=yes" in args
        assert args[-2] == "cosmos-h200" and timeout == 20
        assert json.loads(input_data) == report
        assert "mac-bridge-activation-" in args[-1]
        return subprocess.CompletedProcess(args, 0, "", "")
    assert bridge.upload_activation_report(tmp_path, runner=runner)
