import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.deployment.allocation_wait import (
    BOOT_TIMEOUT_SECONDS, PREPARE_TIMEOUT_SECONDS, QUEUE_TIMEOUT_SECONDS,
    wait_for_host_key,
)
from pipeline.deployment.h200.beta_pod import inspect_replicas, operate, name_for
from test.deployment.test_beta_infra import cloud_error, fake_client


def placement(node=None, replica="owned-pod-replica"):
    return {"name": "owned-pod", "present": True, "replicas": [
        {"replica_id": replica, "node_id": node, "reason": "Starting" if node else "Queueing"}]}


def runner(tmp_path, inspect, *, queue=10, boot=5):
    clock, records = [0.], []
    key = tmp_path / "host-key.pub"
    def poll():
        return inspect(clock[0], key)
    def run():
        return wait_for_host_key(poll, key, records.append,
            queue_timeout=queue, boot_timeout=boot, poll_seconds=1,
            clock=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0]+s))
    return run, clock, records


def test_long_queue_does_not_consume_boot_budget(tmp_path):
    def inspect(t, key):
        if t >= 1100:
            key.touch()
        return placement("node-a" if t >= 1000 else None)
    run, clock, records = runner(tmp_path, inspect, queue=3600, boot=900)
    assert run()["phase"] == "H200_HOST_KEY_READY"
    assert clock[0] == 1100 and records[-1]["phase_elapsed_seconds"] == 100
    assert {r["phase"] for r in records} == {"H200_QUEUED", "H200_BOOTING", "H200_HOST_KEY_READY"}
    assert not any(r["gpu_services_ready"] for r in records)


def test_stale_key_is_not_proof_of_allocation(tmp_path):
    def inspect(t, key):
        key.touch()
        return placement()
    run, clock, records = runner(tmp_path, inspect)
    with pytest.raises(TimeoutError, match="queue deadline"):
        run()
    assert clock[0] == 10 and records[-1]["phase"] == "H200_QUEUED_TIMEOUT"


def test_boot_has_its_own_bounded_deadline(tmp_path):
    run, clock, records = runner(tmp_path, lambda t, key: placement("node-a" if t >= 8 else None))
    with pytest.raises(TimeoutError, match="boot deadline"):
        run()
    assert clock[0] == 13 and records[-1]["phase"] == "H200_BOOTING_TIMEOUT"


@pytest.mark.parametrize("second", [placement(), placement("node-b"), placement("node-a", "new-replica")])
def test_assignment_loss_or_replacement_never_replays(tmp_path, second):
    run, _, _ = runner(tmp_path, lambda t, key: placement("node-a") if t == 0 else second)
    with pytest.raises(RuntimeError, match="assignment changed|lost its assigned"):
        run()


def test_missing_deployment_and_auth_fail_without_polling_forever(tmp_path):
    run, clock, _ = runner(tmp_path, lambda t, key: {"present": False})
    with pytest.raises(RuntimeError, match="disappeared"):
        run()
    assert clock[0] == 0
    def auth(t, key):
        raise PermissionError("denied")
    run, clock, _ = runner(tmp_path, auth)
    with pytest.raises(PermissionError):
        run()
    assert clock[0] == 0


def test_raw_replica_gate_preserves_scheduler_fields_and_redacts_message():
    raw = [{"metadata": {"id": "owned-pod-replica"}, "status": {
        "node": {"id": "node-a"}, "readiness_issue": {"reason": "Ready", "message": "do not expose"}}}]
    response = SimpleNamespace(raise_for_status=Mock(), json=lambda: raw)
    client = SimpleNamespace(_get=Mock(return_value=response))
    result = inspect_replicas(client, "owned-pod")
    assert result == [{"replica_id": "owned-pod-replica", "node_id": "node-a", "reason": "Ready"}]
    assert "do not expose" not in json.dumps(result)
    client._get.assert_called_once_with("/deployments/owned-pod/replicas", allow_redirects=False)
    raw[0]["metadata"]["id"] = "another-pod-replica"
    with pytest.raises(RuntimeError, match="replica_placement_identity_invalid"):
        inspect_replicas(client, "owned-pod")


def test_replica_reads_retry_only_bounded_transient_failures():
    response = SimpleNamespace(raise_for_status=Mock(), json=lambda: [])
    client = SimpleNamespace(_get=Mock(side_effect=[cloud_error(503), TimeoutError(), response]))
    sleep = Mock()
    assert inspect_replicas(client, "owned-pod", sleep=sleep) == []
    assert client._get.call_count == 3 and sleep.call_count == 2
    for status in (401, 403, 404, 409):
        client._get = Mock(side_effect=cloud_error(status))
        with pytest.raises(RuntimeError):
            inspect_replicas(client, "owned-pod", sleep=Mock())
        assert client._get.call_count == 1


def test_inspect_is_owner_attested_read_only_and_supports_absence():
    client = fake_client()
    assert operate(client, "inspect", "dagger-leo-unit", 1)["present"] is False
    name = name_for("dagger-leo-unit", 1)
    dep = SimpleNamespace(metadata=SimpleNamespace(id_=name), spec=SimpleNamespace(
        envs=[SimpleNamespace(name="ROBORENDER_BETA_RUN_ID", value="dagger-leo-unit"),
              SimpleNamespace(name="ROBORENDER_BETA_ROUND_ID", value="1")],
        resource_requirement=SimpleNamespace(resource_shape="custom.gpu.4xh200"),
        reservation_config=SimpleNamespace(reservation_id="cosmos-solution-architect-xcwtffiq",
                                           allow_burst_to_other_reservations=False)))
    client.deployment.get.side_effect = None
    client.deployment.get.return_value = dep
    client._get = Mock(return_value=SimpleNamespace(raise_for_status=Mock(), json=lambda: []))
    assert operate(client, "inspect", "dagger-leo-unit", 1)["replicas"] == []
    dep.spec.envs[0].value = "other-run"
    with pytest.raises(RuntimeError, match="owner_shape_reservation_mismatch"):
        operate(client, "inspect", "dagger-leo-unit", 1)
    client.deployment.create.assert_not_called()
    client.deployment.delete.assert_not_called()


def test_outer_prepare_budget_includes_queue_boot_and_render_startup():
    assert PREPARE_TIMEOUT_SECONDS >= QUEUE_TIMEOUT_SECONDS + BOOT_TIMEOUT_SECONDS + 1320 + 600
