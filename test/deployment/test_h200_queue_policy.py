"""CPU-only allocation-request tests; no SDK login, cloud mutation or GPU work."""
import copy
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.deployment.h200 import beta_pod
from test.deployment.test_beta_infra import fake_client


KEYS = ["ssh-ed25519 AAAA cpu", "ssh-ed25519 AAAA rtx"]
POLICY = {"priority_class": "high-9000", "can_preempt": True, "can_be_preempted": False}


@pytest.mark.parametrize("run,gpus", [("dagger-leo-bc-unit", 1), ("dagger-leo-queue-unit", 4)])
def test_new_owned_gpu_request_never_inherits_cpu_queue_defaults(run, gpus):
    client = fake_client()
    cpu = client.deployment.get("leo-cosmos-h200-access")
    before = copy.deepcopy(vars(cpu.spec.queue_config))
    result = beta_pod.operate(client, "create", run, 1, KEYS, gpus=gpus)
    request = client.deployment.create.call_args.args[0]
    assert vars(request.spec.queue_config) == POLICY
    assert result["queue_config"] == POLICY
    assert request.spec.queue_config is not cpu.spec.queue_config
    assert vars(cpu.spec.queue_config) == before == {
        "priority_class": "mid-4000", "can_preempt": None, "can_be_preempted": None}
    assert request.spec.reservation_config.reservation_id == beta_pod.RESERVATION
    assert request.spec.reservation_config.allow_burst_to_other_reservations is False
    assert request.spec.resource_requirement.resource_shape == f"custom.gpu.{gpus}xh200"
    assert request.spec.resource_requirement.min_replicas == request.spec.resource_requirement.max_replicas == 1
    client.deployment.delete.assert_not_called()


@pytest.mark.parametrize("inherited", [
    {"priority_class": "system-priority", "can_preempt": False, "can_be_preempted": True},
    {"priority_class": "high-7000", "can_preempt": None, "can_be_preempted": None},
])
def test_all_three_fields_are_replaced_not_merged(inherited):
    source = SimpleNamespace(queue_config=SimpleNamespace(**inherited, unrelated_cpu_field="not copied"))
    value = beta_pod.fresh_gpu_queue_config(source)
    assert vars(value) == POLICY
    assert vars(source.queue_config) == dict(inherited, unrelated_cpu_field="not copied")


@pytest.mark.parametrize("missing", [False, True])
def test_null_or_missing_template_queue_constructs_typed_sdk_policy(monkeypatch, missing):
    module = ModuleType("leptonai.api.v1.types.deployment")
    module.QueueConfig = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = SimpleNamespace() if missing else SimpleNamespace(queue_config=None)
    assert vars(beta_pod.fresh_gpu_queue_config(source)) == POLICY
    module.QueueConfig.assert_called_once_with(**POLICY)
    assert getattr(source, "queue_config", None) is None


@pytest.mark.parametrize("priority", beta_pod.GPU_PRIORITY_CLASSES)
def test_explicit_reviewed_priority_is_pinned_and_plan_never_allocates(priority):
    client = fake_client()
    plan = beta_pod.operate(client, "plan", "dagger-leo-queue-unit", 1, KEYS, priority_class=priority)
    assert plan["queue_config"] == dict(POLICY, priority_class=priority)
    assert plan["reservation"] == beta_pod.RESERVATION and plan["allow_burst"] is False
    client.deployment.create.assert_not_called()
    client.deployment.delete.assert_not_called()


@pytest.mark.parametrize("priority", [None, True, 9000, "mid-4000", "high-9999", "system-node-critical", "", ["high-9000"]])
def test_unreviewed_priority_rejected_before_any_cloud_read(priority):
    client = fake_client()
    with pytest.raises(ValueError, match="reviewed"):
        beta_pod.operate(client, "create", "dagger-leo-queue-unit", 1, KEYS, priority_class=priority)
    client.deployment.get.assert_not_called()
    client.deployment.create.assert_not_called()
    client.deployment.delete.assert_not_called()


def test_existing_owned_allocation_is_not_reconfigured_or_blocked_by_new_default():
    client = fake_client()
    run = "dagger-leo-queue-unit"
    beta_pod.operate(client, "create", run, 1, KEYS)
    old = client.deployment.create.call_args.args[0]
    old.metadata.id_ = old.metadata.name
    old.spec.queue_config = SimpleNamespace(priority_class="mid-4000", can_preempt=None, can_be_preempted=None)
    client.deployment.get.side_effect = None
    client.deployment.get.return_value = old
    client.deployment.create.reset_mock()
    assert beta_pod.operate(client, "status", run, 1)["present"] is True
    assert beta_pod.operate(client, "create", run, 1, KEYS)["idempotent"] is True
    assert vars(old.spec.queue_config) == {"priority_class": "mid-4000", "can_preempt": None, "can_be_preempted": None}
    client.deployment.create.assert_not_called()
    retained = beta_pod.operate(client, "delete", run, 1)
    assert retained["retained"] is True and retained["delete_requested"] is False
    client.deployment.delete.assert_not_called()
    assert beta_pod.operate(client, "delete", run, 1, confirm_release=old.metadata.name)["delete_requested"] is True
    client.deployment.delete.assert_called_once_with(old.metadata.name)


@pytest.mark.parametrize("confirmation", ["another-pod", "*", True, ""])
def test_release_requires_exact_owned_pod_confirmation(confirmation):
    client = fake_client()
    with pytest.raises(ValueError, match="exact owned pod"):
        beta_pod.operate(client, "delete", "dagger-leo-queue-unit", 1, confirm_release=confirmation)
    client.deployment.get.assert_not_called()
    client.deployment.delete.assert_not_called()


def test_release_confirmation_cannot_be_embedded_in_another_operation():
    client = fake_client()
    run = "dagger-leo-queue-unit"
    with pytest.raises(ValueError, match="accompany delete"):
        beta_pod.operate(client, "create", run, 1, KEYS,
                         confirm_release=beta_pod.name_for(run, 1))
    client.deployment.get.assert_not_called()
    client.deployment.create.assert_not_called()


def test_cli_default_plan_exposes_requested_queue_policy_without_allocating(monkeypatch, capsys):
    client = fake_client()
    monkeypatch.setattr(beta_pod, "workspace_client", lambda: client)
    monkeypatch.setattr(sys, "argv", ["allocator", "plan", "--run-id", "dagger-leo-queue-unit",
                                    "--round-id", "1", "--public-key", KEYS[0], "--public-key", KEYS[1]])
    beta_pod.main()
    assert json.loads(capsys.readouterr().out)["queue_config"] == POLICY
    client.deployment.create.assert_not_called()


def test_cli_cannot_request_an_arbitrary_priority(monkeypatch):
    client = Mock()
    monkeypatch.setattr(beta_pod, "workspace_client", client)
    monkeypatch.setattr(sys, "argv", ["allocator", "plan", "--run-id", "dagger-leo-queue-unit",
                                    "--round-id", "1", "--priority-class", "high-9999"])
    with pytest.raises(SystemExit) as error:
        beta_pod.main()
    assert error.value.code == 2
    client.assert_not_called()
