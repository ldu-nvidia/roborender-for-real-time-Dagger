"""CPU contract tests, not native phase/capacity certification."""
from copy import deepcopy

import pytest

from dagger_rtx.phase_modes import (
    SCHEMA, ROUTES, capabilities, mode_for, normalize_command, resolve, validate_completion,
)
from dagger_rtx.resident_service import ResidentOwner, SCHEMA as OWNER_SCHEMA


def selector(mode='STUDENT_TEST', operation='evaluate_scaled_rendered_panel', **parameters):
    if mode == 'STUDENT_TEST':
        parameters.setdefault('arm', 'mixed' if operation == 'evaluate_collection_mixed' else 'bc')
    if mode == 'TEACHER_TEST':
        parameters.setdefault('arm', 'teacher')
    return dict(schema=SCHEMA, mode=mode, operation=operation, parameters=parameters)


@pytest.mark.parametrize('mode,operation', [
    (mode, operation) for mode, operations in ROUTES.items() for operation in operations
])
def test_explicit_routes_preserve_parameters(mode, operation):
    body = selector(mode, operation)
    actual_mode, actual_operation, parameters = resolve(body)
    assert (actual_mode, actual_operation) == (mode, operation)
    assert parameters == body['parameters'] and parameters is not body['parameters']
    assert mode_for('run_phase', body) == mode


@pytest.mark.parametrize('mutation', [
    dict(schema='v0'), dict(mode='BC_TRAIN'), dict(mode='BC_RENDER'),
    dict(mode='TEACHER_TEST'), dict(operation='shutdown'), dict(operation='collect_round'),
    dict(parameters={'arm': 'teacher'}), dict(parameters={'arm': 'raytraced'}),
    dict(parameters=None), dict(extra=True),
])
def test_no_wrong_policy_legacy_or_offline_route(mutation):
    with pytest.raises(ValueError):
        resolve({**selector(), **mutation})


def test_no_default_benchmark_and_no_force_idle():
    body = selector()
    del body['operation']
    with pytest.raises(ValueError):
        resolve(body)
    with pytest.raises(ValueError):
        resolve(selector('IDLE', 'idle', force=True))
    assert mode_for('collect_mse_validation', {}) == 'LEGACY'


def test_normalization_preserves_admitted_owner_identity_and_freezes_nested_values():
    body = dict(schema=OWNER_SCHEMA, worker_instance_id='a'*32, command_id='student-1',
                operation='run_phase', payload=selector(nested={'seed': 12}))
    normalized = normalize_command(body)
    assert normalized['worker_instance_id'] == body['worker_instance_id']
    assert normalized['command_id'] == 'student-1'
    assert normalized['operation'] == 'evaluate_scaled_rendered_panel'
    normalized['payload']['nested']['seed'] = 999
    assert body['payload']['parameters']['nested']['seed'] == 12


@pytest.mark.parametrize('mode,receipt', [
    ('TEACHER_TEST', dict(complete=True, purpose='test', optimizer_steps=0)),
    ('STUDENT_TEST', dict(complete=True, purpose='test', optimizer_steps=0)),
    ('BC_COLLECT', dict(complete=True, purpose='bc_collection', optimizer_steps=0)),
    ('DAGGER_TRAIN', dict(complete=True, purpose='training')),
    ('IDLE', dict(complete=True)),
])
def test_completion_requires_mode_correct_evidence(mode, receipt):
    validate_completion(mode, receipt)
    with pytest.raises(RuntimeError):
        validate_completion(mode, {**receipt, 'complete': False})
    with pytest.raises(RuntimeError):
        validate_completion(mode, {**receipt, 'shutdown_requested': True})
    if mode in ('BC_COLLECT', 'STUDENT_TEST', 'TEACHER_TEST'):
        for invalid_steps in (1, False, None):
            with pytest.raises(RuntimeError):
                validate_completion(mode, {**receipt, 'optimizer_steps': invalid_steps})
        with pytest.raises(RuntimeError):
            validate_completion(mode, {**receipt, 'purpose': 'validation'})


def test_capabilities_never_claim_bc_adapter_or_live_qualification():
    value = capabilities()
    assert value['modes']['BC_COLLECT']['available'] is False
    assert value['native_certified'] is False
    assert value['compute_release_supported'] is False
    assert value['offline_h200_phases'] == ['BC_RENDER', 'BC_TRAIN']


class FakeBackend:
    """Owner scheduling only: no simulated evidence of an Isaac rollout."""
    def __init__(self):
        self.calls = []

    def validate(self, operation, payload):
        resolve(payload)

    def validate_command_identity(self, command):
        self.normalized = normalize_command(command)

    def run(self, operation, payload, report):
        self.calls.append(deepcopy(payload))
        if payload['parameters'].get('fail'):
            raise RuntimeError('native state unknown')
        return dict(complete=True)


def envelope(owner, name, payload):
    return dict(schema=OWNER_SCHEMA, worker_instance_id=owner.state['worker_instance_id'],
                command_id=name, operation='run_phase', payload=payload)


def test_sequential_modes_idempotency_and_no_mutable_command_leak(tmp_path):
    backend = FakeBackend()
    owner = ResidentOwner(tmp_path, backend, gpu_uuid='GPU-test', source_sha256='a'*64)
    body = envelope(owner, 'student-1', selector(nested={'seed': 12}))
    pristine = deepcopy(body)
    accepted = owner.submit(body)
    body['payload']['parameters']['nested']['seed'] = 999
    assert owner.status()['mode'] == 'STUDENT_TEST'
    with pytest.raises(ValueError, match='busy'):
        owner.submit(envelope(owner, 'idle-1', selector('IDLE', 'idle')))
    owner.run_next()
    assert backend.calls[0]['parameters']['nested']['seed'] == 12
    assert owner.submit(pristine) == accepted
    assert len(backend.calls) == 1
    assert owner.status()['mode'] == 'IDLE'
    owner.submit(envelope(owner, 'teacher-1', selector('TEACHER_TEST', 'evaluate_scaled_teacher_panel')))
    owner.run_next()
    assert len(backend.calls) == 2
    assert backend.normalized['operation'] == 'evaluate_scaled_teacher_panel'


def test_fault_never_returns_to_idle_or_replays(tmp_path):
    backend = FakeBackend()
    owner = ResidentOwner(tmp_path, backend, gpu_uuid='GPU-test', source_sha256='a'*64)
    body = envelope(owner, 'student-1', selector(fail=True))
    accepted = owner.submit(body)
    with pytest.raises(RuntimeError, match='unknown'):
        owner.run_next()
    assert owner.status()['phase'] == 'CRASH_UNCERTAIN'
    assert owner.status()['mode'] == 'STUDENT_TEST'
    assert owner.submit(body) == accepted
    with pytest.raises(ValueError, match='uncertain'):
        owner.submit(envelope(owner, 'idle-1', selector('IDLE', 'idle')))
    assert len(backend.calls) == 1
