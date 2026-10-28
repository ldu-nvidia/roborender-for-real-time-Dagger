import json

import pytest

from dagger_rtx.connection_health import ConnectionHealthGate, qualification_connection_options
from dagger_rtx.contracts import ContractError
from dagger_rtx.http_transport import TransientTransportError


def snapshot():
    return {"schema": "roborender.ssh-supervisor.v1", "run_id": "test-run", "config_sha256": "a" * 64,
            "state": "ready", "updated_at_ns": 100_000_000_000,
            "forwards": [{"healthy": True, "expected_ready": {"role": "render_student", "pid": p}}
                         for p in (11, 12, 13, 14)]}


def gate(tmp_path, value=None):
    path = tmp_path / "status.json"
    path.write_text(json.dumps(value or snapshot()))
    return ConnectionHealthGate(path, run_id="test-run", config_sha256="a" * 64,
                                gateway_pids=(11, 12, 13, 14), time_ns=lambda: 101_000_000_000)


def test_ready_gate_reads_atomic_replacement_not_old_inode(tmp_path):
    health = gate(tmp_path)
    health()
    value = snapshot()
    value["state"] = "recovering"
    new = tmp_path / "status.part"
    new.write_text(json.dumps(value))
    new.replace(health.path)
    with pytest.raises(TransientTransportError):
        health()


@pytest.mark.parametrize("key,value", [("run_id", "another-run"), ("config_sha256", "b" * 64),
                                       ("state", "failed"), ("state", "expired"),
                                       ("updated_at_ns", 200_000_000_000)])
def test_identity_and_terminal_errors_never_retry(tmp_path, key, value):
    state = snapshot()
    state[key] = value
    with pytest.raises(ContractError):
        gate(tmp_path, state)()


def test_missing_stale_or_recovering_status_is_transient(tmp_path):
    health = gate(tmp_path)
    health.path.unlink()
    with pytest.raises(TransientTransportError):
        health()
    value = snapshot()
    value["updated_at_ns"] = 1
    health.path.write_text(json.dumps(value))
    with pytest.raises(TransientTransportError):
        health()


def test_qualification_retry_budget_waits_for_supervisor_without_changing_legacy_default(tmp_path):
    from dagger_rtx.window_runner import TransportRetryPolicy
    options = qualification_connection_options({"DAGGER_CONNECTION_STATUS": str(tmp_path / "status.json"),
        "DAGGER_CONNECTION_CONFIG_SHA256": "a" * 64}, run_id="test-run", gateway_pids=(11, 12, 13, 14))
    assert options["retry_policy"].max_recovery_s == 90
    assert options["retry_policy"].max_attempts == 60
    assert isinstance(options["request_guard"], ConnectionHealthGate)
    assert TransportRetryPolicy().max_attempts == 3
    with pytest.raises(ContractError):
        qualification_connection_options({}, run_id="test-run", gateway_pids=(11,))


def test_production_connection_budget_is_bounded_and_explicit(tmp_path):
    env = {"DAGGER_CONNECTION_STATUS": str(tmp_path / "status.json"),
           "DAGGER_CONNECTION_CONFIG_SHA256": "a" * 64}
    options = qualification_connection_options(env, run_id="test-run",
        gateway_pids=(11, 12, 13, 14), production=True)
    assert options["retry_policy"].max_recovery_s == 300
    assert options["retry_policy"].max_attempts == 180
    for invalid in (1, "true", None):
        with pytest.raises(ContractError):
            qualification_connection_options(env, run_id="test-run", gateway_pids=(11,),
                                             production=invalid)
