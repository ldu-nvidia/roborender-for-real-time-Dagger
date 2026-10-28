from test.deployment.source_paths import DEPLOYMENT_ROOT
import copy
import json
import subprocess
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.deployment.h200.beta_pod import name_for, operate, workspace_client, read_deployment
from pipeline.deployment.h200.beta_service_host import launch_plan
from pipeline.deployment.owned_process import start, status
from pipeline.dagger.services.client import IdempotentControlClient, ServiceClient, ServiceError, RetryableTransportError
from pipeline.deployment.beta_lease_watchdog import heartbeat_fresh
from pipeline.deployment.beta_site import Site, validate_site


def test_remote_metadata_survives_realistic_reconnect_without_payload_change(monkeypatch):
    from pipeline.deployment import beta_site as module
    clock = [0.]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    calls = []
    def invoke(cmd, **kw):
        calls.append((cmd, kw["input"]))
        return SimpleNamespace(returncode=255 if clock[0] < 25 else 0, stdout=b"saved-result", stderr=b"Connection refused")
    monkeypatch.setattr(module.subprocess, "run", invoke)
    site = Site.__new__(Site)
    site.ssh = lambda host: ["ssh", "pinned-beta"]
    assert site.command(["read-result"], "beta", data=b"same") == b"saved-result"
    assert len(calls) > 3 and all(call == calls[0] for call in calls)
    assert 25 <= clock[0] < 40


@pytest.mark.parametrize("stderr", [b"Permission denied (publickey)", b"Host key verification failed",
                                     b"REMOTE HOST IDENTIFICATION HAS CHANGED"])
def test_security_errors_are_not_retried_or_exposed(monkeypatch, stderr):
    from pipeline.deployment import beta_site as module
    run = Mock(return_value=SimpleNamespace(returncode=255, stdout=b"", stderr=stderr+b" secret"))
    monkeypatch.setattr(module.subprocess, "run", run)
    site = Site.__new__(Site)
    site.ssh = lambda host: ["ssh", "pinned-beta"]
    with pytest.raises(RuntimeError, match="exit=255") as error:
        site.command(["read-result"], "beta")
    assert "secret" not in str(error.value) and run.call_count == 1


def test_remote_recovery_bounded_and_non_transport_failures_not_retried(monkeypatch):
    from pipeline.deployment import beta_site as module
    clock = [0.]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    monkeypatch.setattr(module, "TRANSPORT_RECOVERY_SECONDS", 20)
    run = Mock(return_value=SimpleNamespace(returncode=255, stdout=b"", stderr=b"Connection refused"))
    monkeypatch.setattr(module.subprocess, "run", run)
    site = Site.__new__(Site)
    site.ssh = lambda host: ["ssh", "pinned-beta"]
    with pytest.raises(RuntimeError, match="recovery deadline"):
        site.command(["read-result"], "beta", timeout=10)
    assert clock[0] == 20
    run.reset_mock()
    run.return_value.returncode = 1
    with pytest.raises(RuntimeError, match="exit=1"):
        site.command(["read-result"], "beta")
    assert run.call_count == 1
    run.reset_mock()
    run.side_effect = subprocess.TimeoutExpired("local", 1)
    with pytest.raises(subprocess.TimeoutExpired):
        site.command(["local-work"], timeout=1)
    assert run.call_count == 1


def test_allocator_uses_explicit_v2_workspace_and_bounded_calls():
    client = SimpleNamespace(url="https://gateway.dgxc-lepton.nvidia.com/api/v2/workspaces/b5k2m9x7", _header={})
    record = SimpleNamespace(client=Mock(return_value=client))
    assert workspace_client(record) is client
    record.client.assert_called_once_with("b5k2m9x7")
    assert client._timeout == 12
    assert client._header == {"Origin": "https://dashboard.dgxc-lepton.nvidia.com"}
    for bad in ("https://api.lepton.ai/api/v1/workspaces/b5k2m9x7", "http://gateway.dgxc-lepton.nvidia.com/api/v2/workspaces/b5k2m9x7",
                "https://gateway.dgxc-lepton.nvidia.com/api/v2/workspaces/other"):
        client.url = bad
        with pytest.raises(RuntimeError, match="workspace_api_origin_mismatch"):
            workspace_client(record)


def test_beta_owned_site_has_no_alpha_or_self_ssh_hop():
    config = json.loads((DEPLOYMENT_ROOT / "profiles/beta_site.direct.example.json").read_text())
    site = Site.__new__(Site)
    site.site = validate_site(config)
    site.pod = name_for("dagger-beta-test", 1)
    beta_proxy = site.gpu_proxy("beta")
    assert beta_proxy == [config["broker_python"], config["broker_proxy"], site.pod]
    local_proxy = site.gpu_proxy("local")
    assert local_proxy[0] == "ssh" and local_proxy[-4] == "roborender-beta-direct"
    assert "roborender-migration-alpha" not in local_proxy and "rtx_proxy_ssh_config" not in config
    with pytest.raises(ValueError): validate_site(dict(config, broker_ssh_host="roborender-migration-alpha"))
    with pytest.raises(ValueError): validate_site(dict(config, rtx_proxy_ssh_host="old-route"))
    with pytest.raises(ValueError): validate_site(dict(config, broker_proxy="/some/unpinned/proxy.py"))


def test_legacy_site_retains_explicit_old_route_without_implicit_fallback():
    config = json.loads((DEPLOYMENT_ROOT / "profiles/beta_site.example.json").read_text())
    site = Site.__new__(Site)
    site.site = validate_site(config)
    site.pod = name_for("dagger-beta-test", 1)
    assert site.gpu_proxy("beta")[:3] == ["ssh", "-F", config["rtx_proxy_ssh_config"]]
    assert config["rtx_proxy_ssh_host"] in site.gpu_proxy("beta")


def test_orphan_guard_requires_fresh_exact_scheduler_identity():
    now = 1_000_000_000_000
    beat = {"run_id": "dagger-beta-test", "updated_at_ns": now - 20_000_000_000}
    assert heartbeat_fresh(beat, "dagger-beta-test", now)
    assert not heartbeat_fresh(beat, "dagger-beta-other", now)
    assert not heartbeat_fresh(beat, "dagger-beta-test", now + 301_000_000_000)
    assert not heartbeat_fresh({**beat, "updated_at_ns": now + 20_000_000_000}, "dagger-beta-test", now)


def test_scoped_stop_waits_for_supervisor_identity(monkeypatch):
    site = Site.__new__(Site)
    site.process = Mock(side_effect=[{"state": state} for state in
                                    ("STARTING", "RUNNING", "RUNNING", "STOPPED")])
    monkeypatch.setattr("pipeline.deployment.beta_site.time.sleep", lambda _: None)
    site.stop_process("actors", "beta")
    assert [call.args for call in site.process.call_args_list] == [
        ("actors", "status"), ("actors", "status"), ("actors", "stop"), ("actors", "status")]


@pytest.mark.parametrize("path", ["/v1/jobs", "/v1/rounds/start", "/v1/iterations/advance", "/v2/evaluations/x/outcomes"])
def test_lost_control_ack_retries_identical_bytes(monkeypatch, path):
    sent = []
    def request(self, method, route, body=None, content_type=None):
        sent.append((method, route, body))
        if len(sent) == 1: raise ConnectionResetError("lost reply")
        return {"ok": True}
    monkeypatch.setattr(ServiceClient, "_request", request)
    client = IdempotentControlClient("http://localhost", "test", sleep=lambda _: None)
    assert client.post(path, {"job_id": "same-id", "x": 1}) == {"ok": True}
    assert len(sent) == 2 and sent[0] == sent[1]


def test_no_semantic_retry_or_native_action_retry(monkeypatch):
    request = Mock(side_effect=ServiceError("409 conflict"))
    monkeypatch.setattr(ServiceClient, "_request", request)
    client = IdempotentControlClient("http://localhost", "test", sleep=lambda _: None)
    with pytest.raises(ServiceError): client.post("/v1/jobs", {"job_id": "same"})
    assert request.call_count == 1
    with pytest.raises(ValueError): client.post("/v1/env/step", {})
    assert request.call_count == 1
    request.side_effect = TimeoutError()
    with pytest.raises(RetryableTransportError): client.get("/ready")
    assert request.call_count == 6


def fake_client():
    class Spec(SimpleNamespace):
        def model_copy(self, deep): return copy.deepcopy(self)
    source = SimpleNamespace(metadata=SimpleNamespace(id_="leo-cosmos-h200-access"),
        spec=Spec(resource_requirement=SimpleNamespace(), reservation_config=SimpleNamespace(),
                  queue_config=SimpleNamespace(priority_class="mid-4000", can_preempt=None,
                                               can_be_preempted=None),
                  container=SimpleNamespace(), envs=[SimpleNamespace(name="unused", value="unused")]))
    def get(name):
        if name == "leo-cosmos-h200-access": return source
        raise cloud_error(404)
    return SimpleNamespace(deployment=SimpleNamespace(list_all=Mock(return_value=[]),
        get=Mock(side_effect=get), create=Mock(), delete=Mock()))


def cloud_error(code):
    error = RuntimeError("response text must not be exposed")
    error.response = SimpleNamespace(status_code=code)
    return error


def test_scoped_cloud_read_handles_404_without_list_get_race():
    client = fake_client()
    assert read_deployment(client, "deleted") is None
    client.deployment.list_all.assert_not_called()
    assert operate(client, "status", "dagger-beta-unit", 1)["present"] is False


def test_cloud_read_retries_only_transient_reads_and_is_bounded():
    client = fake_client()
    expected = SimpleNamespace()
    client.deployment.get.side_effect = [cloud_error(503), TimeoutError(), expected]
    sleep = Mock()
    assert read_deployment(client, "owned", sleep=sleep) is expected
    assert client.deployment.get.call_count == 3
    client.deployment.create.assert_not_called()
    client.deployment.delete.assert_not_called()
    for code in (401, 403, 409):
        client.deployment.get.reset_mock(side_effect=True)
        client.deployment.get.side_effect = cloud_error(code)
        with pytest.raises(RuntimeError): read_deployment(client, "owned", sleep=Mock())
        assert client.deployment.get.call_count == 1
    client.deployment.get.reset_mock(side_effect=True)
    client.deployment.get.side_effect = cloud_error(503)
    with pytest.raises(RuntimeError): read_deployment(client, "owned", sleep=Mock())
    assert client.deployment.get.call_count == 3


def test_actual_sdk_decode_error_is_retried_but_never_exposed():
    from pipeline.deployment.h200.beta_pod import AllocationReadError
    client=fake_client()
    error=RuntimeError("You encountered a programming error. Please report this, and include the following debug info: "
                       "response returned status 200, but the content cannot be decoded as json. secret-body")
    client.deployment.get.side_effect=[error,SimpleNamespace()]
    assert read_deployment(client,"owned",sleep=Mock()) is not None
    assert client.deployment.get.call_count==2
    client.deployment.get.side_effect=error
    with pytest.raises(AllocationReadError) as caught:read_deployment(client,"owned",sleep=Mock())
    assert caught.value.stage=="deployment_read" and caught.value.retryable
    assert "secret-body" not in str(caught.value)
    client.deployment.get.reset_mock(side_effect=True)
    client.deployment.get.side_effect=RuntimeError("unknown programming error")
    with pytest.raises(RuntimeError):read_deployment(client,"owned",sleep=Mock())
    assert client.deployment.get.call_count==1


def test_broker_read_error_recovers_without_recreating_allocation(monkeypatch):
    from pipeline.deployment import beta_site as module
    clock=[0.];monkeypatch.setattr(module.time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(module.time,"sleep",lambda s:clock.__setitem__(0,clock[0]+s))
    error={"schema":"roborender.allocation-error.v1","error_type":"AllocationReadError","status_code":None,
           "stage":"replica_read","action":"inspect","retryable":True}
    responses=[SimpleNamespace(returncode=1,stdout=json.dumps(error).encode(),stderr=b""),
               SimpleNamespace(returncode=0,stdout=b'{"present":true}',stderr=b"")]
    run=Mock(side_effect=responses);monkeypatch.setattr(module.subprocess,"run",run)
    site=Site.__new__(Site);site.ssh=lambda host:["ssh","pinned-leo"]
    assert site.command(["read-only-inspect"],"broker")==b'{"present":true}'
    assert run.call_count==2 and run.call_args_list[0].args==run.call_args_list[1].args


def test_replica_invariant_has_bounded_reason_and_is_not_retried():
    from pipeline.deployment.h200.beta_pod import AllocationInvariantError, inspect_replicas
    response = SimpleNamespace(raise_for_status=Mock(), json=Mock(return_value=[{}, {}]))
    client = SimpleNamespace(_get=Mock(return_value=response))
    with pytest.raises(AllocationInvariantError) as caught:
        inspect_replicas(client, "owned", sleep=Mock())
    assert caught.value.reason_code == "multiple_replica_response"
    assert caught.value.stage == "replica_read" and caught.value.retryable is False
    assert client._get.call_count == 1


def test_broker_invariant_reason_is_sanitized_and_never_retried(monkeypatch):
    from pipeline.deployment import beta_site as module
    for reason, visible in (("multiple_replica_response", True), ("credential=secret-body", False)):
        error = {"schema": "roborender.allocation-error.v1", "error_type": "AllocationInvariantError",
                 "status_code": None, "stage": "replica_read", "action": "inspect", "retryable": False,
                 "reason_code": reason}
        run = Mock(return_value=SimpleNamespace(returncode=1, stdout=json.dumps(error).encode(), stderr=b""))
        monkeypatch.setattr(module.subprocess, "run", run)
        site = Site.__new__(Site)
        site.ssh = lambda host: ["ssh", "pinned-leo"]
        with pytest.raises(RuntimeError) as caught:
            site.command(["read-only-inspect"], "broker")
        assert (reason in str(caught.value)) is visible
        assert "secret-body" not in str(caught.value) and run.call_count == 1


def test_plan_never_allocates_and_create_has_approved_shape_only():
    client = fake_client()
    args = ("dagger-beta-unit", 1, ["ssh-ed25519 AAAA first", "ssh-ed25519 AAAA second"])
    plan = operate(client, "plan", *args)
    client.deployment.create.assert_not_called()
    assert plan["allow_burst"] is False and plan["shape"] == "custom.gpu.4xh200"
    operate(client, "create", *args)
    request = client.deployment.create.call_args.args[0]
    assert request.spec.reservation_config.reservation_id == "cosmos-solution-architect-xcwtffiq"
    assert request.spec.reservation_config.allow_burst_to_other_reservations is False
    assert request.spec.container.command[-1].endswith("beta_pod_entrypoint.sh")
    assert len(name_for(args[0], 1)) < 36


def test_one_gpu_production_bc_stays_scoped_and_reserved():
    client = fake_client()
    args = ("dagger-leo-bc-prod-20261015", 1,
            ["ssh-ed25519 AAAA first", "ssh-ed25519 AAAA second"])
    plan = operate(client, "plan", *args, gpus=1)
    assert plan["shape"] == "custom.gpu.1xh200"
    assert plan["reservation"] == "cosmos-solution-architect-xcwtffiq"
    assert plan["allow_burst"] is False
    client.deployment.create.assert_not_called()
    operate(client, "create", *args, gpus=1)
    request = client.deployment.create.call_args.args[0]
    assert request.spec.resource_requirement.resource_shape == "custom.gpu.1xh200"
    for owner, round_id, gpus in ((args[0], 2, 1), ("dagger-leo-online", 1, 1),
                                  (args[0], 1, 2), (args[0], 1, 8)):
        with pytest.raises(ValueError):
            operate(fake_client(), "plan", owner, round_id, args[2], gpus=gpus)


def test_wrong_owner_allocation_is_never_deleted():
    client = fake_client()
    name = name_for("dagger-beta-unit", 1)
    client.deployment.list_all.return_value = [SimpleNamespace(metadata=SimpleNamespace(id_=name))]
    client.deployment.get.side_effect = None
    client.deployment.get.return_value = SimpleNamespace(spec=SimpleNamespace(envs=[]))
    with pytest.raises((RuntimeError, AttributeError)):
        operate(client, "delete", "dagger-beta-unit", 1)
    client.deployment.delete.assert_not_called()


def test_named_leo_allocation_adopts_only_exact_existing_owner():
    from pipeline.deployment.h200.beta_pod import allocation_name
    owner = "dagger-leo-smoke-prebc-named-20261013"
    name = "isaac-sim-render-leo"
    assert allocation_name(owner, 1, 1, name) == name
    for values in [(owner, 1, 4, name), ("dagger-leo-other", 1, 1, name), (owner, 2, 1, name), (owner, 1, 1, "other")]:
        with pytest.raises(ValueError): allocation_name(*values)
    client = fake_client()
    with pytest.raises(RuntimeError, match="no longer exists"):
        operate(client, "create", owner, 1, gpus=1, existing_name=name)
    client.deployment.create.assert_not_called()
    dep = SimpleNamespace(metadata=SimpleNamespace(id_=name), spec=SimpleNamespace(
        envs=[SimpleNamespace(name="ROBORENDER_BETA_RUN_ID", value=owner),
              SimpleNamespace(name="ROBORENDER_BETA_ROUND_ID", value="1")],
        resource_requirement=SimpleNamespace(resource_shape="custom.gpu.1xh200"),
        reservation_config=SimpleNamespace(reservation_id="cosmos-solution-architect-xcwtffiq",
                                           allow_burst_to_other_reservations=False)))
    client.deployment.list_all.return_value = [dep]
    client.deployment.get.side_effect = None
    client.deployment.get.return_value = dep
    assert operate(client, "create", owner, 1, gpus=1, existing_name=name)["idempotent"]
    dep.spec.envs[0].value = "another-owner"
    with pytest.raises(RuntimeError):
        operate(client, "delete", owner, 1, gpus=1, existing_name=name)
    client.deployment.delete.assert_not_called()


def test_cpu_only_control_plan_has_no_renderer_or_learner():
    env, commands = launch_plan("/r", "/s", "/t", "/o", "cpu")
    assert [r[0] for r in commands] == ["coordinator", "registry", "evaluation"]
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    _, commands = launch_plan("/r", "/s", "/t", "/o", "gpu")
    assert len(commands) == 6 and commands[-1][-1] == [18620, 18621, 18622, 18623]


def test_owned_process_is_detached_idempotent_and_never_replayed(tmp_path):
    root = tmp_path / "runtime/campaign/child"
    marker = tmp_path / "marker"
    spec = tmp_path / "process-spec.json"
    value = {"argv": [sys.executable, "-c", "from pathlib import Path;import sys; p=Path(sys.argv[1]);p.open('a').write('once\\n')", str(marker)],
             "cwd": str(tmp_path), "ttl": 30, "env": {}, "owner": "dagger-beta-test"}
    spec.write_text(json.dumps(value))
    start(root, spec)
    deadline = time.monotonic() + 10
    while status(root)["state"] not in {"COMPLETE", "FAILED", "UNCERTAIN"}:
        if time.monotonic() > deadline: raise AssertionError("test subprocess did not complete")
        time.sleep(.05)
    assert status(root)["state"] == "COMPLETE"
    assert start(root, spec)["state"] == "COMPLETE"
    assert marker.read_text() == "once\n"
    value["argv"].append("different")
    spec.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="identity changed"): start(root, spec)
