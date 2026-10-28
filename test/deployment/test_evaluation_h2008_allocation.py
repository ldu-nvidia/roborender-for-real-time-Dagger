"""CPU-only exact eight-H200 evaluation allocation and retention admission."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.deployment.h200 import beta_pod as allocator
from test.deployment.test_beta_infra import fake_client, cloud_error

RUN = allocator.SCALED_EVALUATION_RUN_ID
KEYS = ['ssh-ed25519 AAAA cpu', 'ssh-ed25519 AAAA leo']


def allocated():
    client = fake_client()
    allocator.operate(client, 'create', RUN, 1, KEYS, gpus=8)
    request = client.deployment.create.call_args.args[0]
    request.metadata.id_ = request.metadata.name
    client.deployment.get.side_effect = None
    client.deployment.get.return_value = request
    client.deployment.create.reset_mock()
    return client, request


def test_exact_eight_gpu_request_one_node_no_burst_or_autoscale():
    client = fake_client()
    source = client.deployment.get('leo-cosmos-h200-access')
    before = deepcopy(vars(source.spec))
    result = allocator.operate(client, 'create', RUN, 1, KEYS, gpus=8)
    request = client.deployment.create.call_args.args[0]
    assert result['shape'] == request.spec.resource_requirement.resource_shape == 'custom.gpu.8xh200'
    assert request.spec.resource_requirement.min_replicas == request.spec.resource_requirement.max_replicas == 1
    assert vars(request.spec.queue_config) == allocator.gpu_queue_policy('high-9000')
    assert request.spec.reservation_config.reservation_id == allocator.RESERVATION
    assert request.spec.reservation_config.allow_burst_to_other_reservations is False
    assert request.spec.auto_scaler is None and request.spec.api_tokens is None and request.spec.is_pod is True
    assert vars(source.spec) == before
    env = {row.name: row.value for row in request.spec.envs}
    assert env['ROBORENDER_BETA_RUN_ID'] == RUN and env['ROBORENDER_BETA_ROUND_ID'] == '1'
    client.deployment.create.assert_called_once()
    client.deployment.delete.assert_not_called()


@pytest.mark.parametrize('overrides', [dict(run_id='dagger-leo-other'), dict(round_id=2), dict(gpus=4),
    dict(gpus=1), dict(gpus=8.), dict(existing_name='isaac-sim-render-leo'),
    dict(priority_class='high-7000'), dict(infrastructure_attempt='a'*32)])
def test_new_shape_cannot_escape_exact_isolated_authorization(overrides):
    client = fake_client()
    args = dict(run_id=RUN, round_id=1, gpus=8, public_keys=KEYS)
    args.update(overrides)
    with pytest.raises(ValueError):
        allocator.operate(client, 'create', **args)
    client.deployment.get.assert_not_called()
    client.deployment.create.assert_not_called()
    client.deployment.delete.assert_not_called()


def test_idempotent_create_never_resizes_or_recreates():
    client, request = allocated()
    before = deepcopy(vars(request.spec))
    result = allocator.operate(client, 'create', RUN, 1, KEYS, gpus=8)
    assert result == dict(name=allocator.name_for(RUN, 1), present=True, idempotent=True)
    assert vars(request.spec) == before
    client.deployment.create.assert_not_called()
    client.deployment.delete.assert_not_called()


@pytest.mark.parametrize('fault', ['owner', 'shape', 'reservation', 'burst', 'replicas', 'autoscale', 'queue'])
def test_wrong_or_drifted_eight_gpu_owner_fails_closed(fault):
    client, request = allocated()
    if fault == 'owner': request.spec.envs[1].value = 'dagger-leo-someone-else'
    elif fault == 'shape': request.spec.resource_requirement.resource_shape = 'custom.gpu.4xh200'
    elif fault == 'reservation': request.spec.reservation_config.reservation_id = 'other'
    elif fault == 'burst': request.spec.reservation_config.allow_burst_to_other_reservations = True
    elif fault == 'replicas': request.spec.resource_requirement.max_replicas = 2
    elif fault == 'autoscale': request.spec.auto_scaler = SimpleNamespace(scale_down=0)
    else: request.spec.queue_config.can_be_preempted = True
    with pytest.raises(allocator.AllocationInvariantError):
        allocator.operate(client, 'create', RUN, 1, KEYS, gpus=8)
    client.deployment.create.assert_not_called()
    client.deployment.delete.assert_not_called()


def test_default_cleanup_retains_eight_gpus_and_wrong_confirmation_cannot_delete():
    client, request = allocated()
    result = allocator.operate(client, 'delete', RUN, 1, gpus=8)
    assert result['retained'] and not result['delete_requested']
    with pytest.raises(ValueError, match='exact owned'):
        allocator.operate(client, 'delete', RUN, 1, gpus=8,
            confirm_release=allocator.name_for('dagger-leo-bc-rendered-eval116-20261023', 1))
    client.deployment.delete.assert_not_called()


def test_uncertain_create_ack_is_not_automatically_retried():
    client = fake_client()
    client.deployment.create.side_effect = TimeoutError('simulated lost ACK')
    with pytest.raises(TimeoutError):
        allocator.operate(client, 'create', RUN, 1, KEYS, gpus=8)
    client.deployment.create.assert_called_once()
    client.deployment.delete.assert_not_called()


def test_plan_never_mutates_and_existing_one_gpu_identity_is_distinct():
    client = fake_client()
    result = allocator.operate(client, 'plan', RUN, 1, KEYS, gpus=8)
    assert result['created'] is False
    assert result['name'] != allocator.name_for('dagger-leo-bc-rendered-eval116-20261023', 1)
    client.deployment.create.assert_not_called()
    client.deployment.delete.assert_not_called()
