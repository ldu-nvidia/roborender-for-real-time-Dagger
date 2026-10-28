"""Metadata-only fixed-final selection; these fixtures are not native evidence."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from pipeline.bc_training import plan_success_test as success
from pipeline.bc_training.handoff_common import binding, bound, canonical, digest
from pipeline.bc_training.stopping_gate import fixed_final_contract
from test.pipeline.test_bc_extension_success_handoff import fixture as legacy_fixture


def replace(path, value):
    path = Path(path)
    path.write_bytes(canonical(value) + b'\n')
    return binding(path)


def fixed_fixture(tmp_path, monkeypatch):
    args, called = legacy_fixture(tmp_path, monkeypatch)
    config = bound(args['training_config'])
    root = Path(config['output'])
    config.update(training_protocol=fixed_final_contract(3, budget_explicit=True),
                  max_epochs=3, epoch_budget_explicit=True, chunk_batch_size=16, wandb_mode='disabled')
    args['training_config'] = replace(args['training_config']['path'], config)
    checkpoints = root / 'epoch_checkpoints'
    checkpoints.mkdir()
    selected = checkpoints / 'epoch_0003_step_000006.pt'
    selected.write_bytes(b'synthetic final completed epoch, not loaded as a policy')
    os.link(selected, root / 'last.pt')
    report = bound(args['data_admission'])
    state = dict(schema='roborender.bc-fixed-final-state.v1', status='READY_FOR_TEST',
        stop_reason='explicit_epoch_budget', epoch=3, global_step=6, training_epoch_mse=.15,
        selected_checkpoint=str(selected), selected_checkpoint_sha256=success.checkpoint_identity(selected)['sha256'],
        training_protocol=config['training_protocol'], validation_enabled=False,
        optimizer_state_saved=True, rng_state_saved=True, test_execution_started=False, wandb_run_sha256=None,
        training_config_sha256=args['training_config']['raw_sha256'],
        data_admission_sha256=args['data_admission']['raw_sha256'],
        train_split_sha256=report['split_files']['train']['raw_sha256'],
        validation_split_sha256=report['split_files']['validation']['raw_sha256'],
        training_batch_contract=dict(schema='raven_full_bxt_16x81_activation_checkpoint_v1',
            chunks_per_update=16, frames_per_chunk=81, policy_forward_population=1296,
            drop_last=True, training_chunk_count=32, updates_per_epoch=2,
            index_safe_max_pools=0, activation_checkpointing=True))
    state['training_source_identity'] = {name: 'a' * 64 for name in (
        'pipeline/bc_training/train_bc_h200.py', 'pipeline/bc_training/stopping_gate.py',
        'pipeline/dagger/services/raven_learner_worker.py')}
    state['completion_operation_id'] = digest(canonical({key: state[key] for key in (
        'training_config_sha256', 'selected_checkpoint_sha256', 'epoch', 'global_step',
        'training_protocol', 'training_source_identity', 'wandb_run_sha256')}))
    replace(root / 'training_state.json', state)
    replace(root / 'training-finalization-intent.json', state)
    args['training_state'] = replace(root / 'training-complete.json', state)
    return args, called


def test_fixed_final_selects_sealed_last_epoch_not_validation_or_test(tmp_path, monkeypatch):
    args, called = fixed_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('planner launched a process'))
    import torch
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('planner loaded a checkpoint'))
    result = success.plan(**args, output=tmp_path / 'plan.json')
    doc = bound(result['plan'])
    assert called[0]['allow_training_extension'] is True
    assert doc['schema'] == 'roborender.bc-selected-success-test-plan.v2'
    assert doc['selection_metric'] == 'last_completed_epoch'
    assert doc['selected_epoch'] == 3 and doc['selected_training_mse'] == .15
    assert doc['selected_validation_mse'] is None and doc['validation_enabled'] is False
    assert doc['completion_proof'] == args['training_state']
    assert doc['checkpoint']['path'].endswith('/epoch_checkpoints/epoch_0003_step_000006.pt')
    assert doc['command']['payload']['checkpoint_sha256'] == result['checkpoint_sha256']
    assert doc['optimizer_steps'] == 0 and doc['test_case_count'] == 295
    assert doc['execution_profile'] == 'legacy_single_owner_b2_explicit_panel'
    assert doc['scaled_production_ready'] is False and doc['production_execution_authorized'] is False
    assert doc['executed'] is False and doc['success_test_completed'] is False


@pytest.mark.parametrize('size', [16, 32])
@pytest.mark.parametrize('wrong_schema', [False, True])
def test_versioned_replay_batch_handoff(size, wrong_schema, tmp_path, monkeypatch):
    from pipeline.bc_training.train_bc_h200 import student_lifecycle_batch_fields, REPLAY_STUDENT_LIFECYCLE
    args, _ = fixed_fixture(tmp_path, monkeypatch)
    config = bound(args['training_config']); root = Path(config['output'])
    config.update(chunk_batch_size=size, student_lifecycle=REPLAY_STUDENT_LIFECYCLE)
    args['training_config'] = replace(args['training_config']['path'], config)
    state = bound(args['training_state']); state['training_config_sha256'] = args['training_config']['raw_sha256']
    state['training_batch_contract'].update(student_lifecycle_batch_fields(REPLAY_STUDENT_LIFECYCLE, size),
        chunks_per_update=size, policy_forward_population=size * 81, training_chunk_count=2 * size)
    if wrong_schema: state['training_batch_contract']['schema'] = 'raven_full_bxt_16x81_activation_checkpoint_v1'
    state['completion_operation_id'] = digest(canonical({k: state[k] for k in (
        'training_config_sha256', 'selected_checkpoint_sha256', 'epoch', 'global_step',
        'training_protocol', 'training_source_identity', 'wandb_run_sha256')}))
    for name in ('training_state.json', 'training-finalization-intent.json'): replace(root / name, state)
    args['training_state'] = replace(root / 'training-complete.json', state)
    if wrong_schema:
        with pytest.raises(ValueError): success.plan(**args, output=tmp_path / 'plan.json')
    else:
        result = success.plan(**args, output=tmp_path / 'plan.json')
        assert not bound(result['plan'])['executed']


@pytest.mark.parametrize('change', [
    'partial_status', 'early_epoch', 'wrong_step', 'validation_enabled', 'optimizer_missing',
    'rng_missing', 'test_started', 'config_hash', 'admission_hash', 'train_split', 'validation_split',
    'checkpoint_hash', 'checkpoint_path', 'batch_size', 'updates_per_epoch', 'nan_mse',
    'wrong_completion_path', 'changed_current_state', 'changed_alias', 'checkpoint_symlink',
    'unknown_protocol', 'legacy_downgrade', 'changed_budget', 'inferred_budget',
    'missing_intent', 'changed_intent', 'missing_source', 'changed_operation', 'missing_logging_identity'])
def test_fixed_final_rejects_incomplete_mismatched_or_relabelled_proof(tmp_path, monkeypatch, change):
    args, _ = fixed_fixture(tmp_path, monkeypatch)
    state = bound(args['training_state'])
    config = bound(args['training_config'])
    root = Path(config['output'])
    if change == 'partial_status': state['status'] = 'TRAINING'
    elif change == 'early_epoch': state['epoch'] = 2
    elif change == 'wrong_step': state['global_step'] = 5
    elif change == 'validation_enabled': state['validation_enabled'] = True
    elif change == 'optimizer_missing': state['optimizer_state_saved'] = False
    elif change == 'rng_missing': state['rng_state_saved'] = False
    elif change == 'test_started': state['test_execution_started'] = True
    elif change == 'config_hash': state['training_config_sha256'] = '0' * 64
    elif change == 'admission_hash': state['data_admission_sha256'] = '0' * 64
    elif change == 'train_split': state['train_split_sha256'] = '0' * 64
    elif change == 'validation_split': state['validation_split_sha256'] = '0' * 64
    elif change == 'checkpoint_hash': state['selected_checkpoint_sha256'] = '0' * 64
    elif change == 'checkpoint_path': state['selected_checkpoint'] = str(root / 'last.pt')
    elif change == 'batch_size': state['training_batch_contract']['chunks_per_update'] = 2
    elif change == 'updates_per_epoch': state['training_batch_contract']['updates_per_epoch'] = 3
    elif change == 'nan_mse': state['training_epoch_mse'] = float('nan')
    elif change == 'missing_source': state.pop('training_source_identity')
    elif change == 'changed_operation': state['completion_operation_id'] = '0' * 64
    elif change == 'missing_logging_identity': state.pop('wandb_run_sha256')
    elif change == 'changed_alias':
        (root / 'last.pt').unlink()
        (root / 'last.pt').write_bytes(b'wrong alias')
    elif change == 'checkpoint_symlink':
        selected = Path(state['selected_checkpoint'])
        selected.unlink()
        selected.symlink_to(root / 'last.pt')
    elif change in {'unknown_protocol', 'legacy_downgrade', 'changed_budget', 'inferred_budget'}:
        if change == 'unknown_protocol': config['training_protocol'] = {'schema': 'unrecognized'}
        elif change == 'legacy_downgrade': config['training_protocol'] = 'legacy_validation'
        elif change == 'changed_budget': config['max_epochs'] = 4
        else: config['epoch_budget_explicit'] = False
        args['training_config'] = replace(args['training_config']['path'], config)
        state['training_config_sha256'] = args['training_config']['raw_sha256']
    # Non-finite metadata can be encoded by unrelated writers; the gate must
    # reject it even when the caller supplies an exact raw hash.
    if change == 'nan_mse':
        raw = json.dumps(state).encode()
        (root / 'training-complete.json').write_bytes(raw)
        (root / 'training_state.json').write_bytes(raw)
        (root / 'training-finalization-intent.json').write_bytes(raw)
        args['training_state'] = binding(root / 'training-complete.json')
    else:
        args['training_state'] = replace(root / 'training-complete.json', state)
        replace(root / 'training_state.json', state)
        replace(root / 'training-finalization-intent.json', state)
    if change == 'wrong_completion_path': args['training_state'] = replace(root / 'other-completion.json', state)
    if change == 'changed_current_state': replace(root / 'training_state.json', {**state, 'status': 'TRAINING'})
    if change == 'missing_intent': (root / 'training-finalization-intent.json').unlink()
    if change == 'changed_intent': replace(root / 'training-finalization-intent.json', {**state, 'global_step': 5})
    with pytest.raises((ValueError, FileNotFoundError)):
        success.plan(**args, output=tmp_path / 'plan.json')
    assert not (tmp_path / 'plan.json').exists()


def test_legacy_validation_selection_is_preserved_and_never_claims_scaled_readiness(tmp_path, monkeypatch):
    args, _ = legacy_fixture(tmp_path, monkeypatch)
    result = success.plan(**args, output=tmp_path / 'legacy-plan.json')
    doc = bound(result['plan'])
    assert doc['schema'] == 'roborender.bc-selected-success-test-plan.v1'
    assert doc['selection_metric'] == 'minimum_complete_validation_mse'
    assert doc['selected_epoch'] == 1 and doc['selected_validation_mse'] == .2
    assert doc['execution_profile'] == 'legacy_single_owner_b2_explicit_panel'
    assert doc['scaled_production_ready'] is False
    assert doc['executed'] is False


@pytest.mark.parametrize('fault', [None, 'changed_record', 'missing_hash'])
def test_fixed_final_binds_online_logging_record_without_contacting_wandb(tmp_path, monkeypatch, fault):
    args, _ = fixed_fixture(tmp_path, monkeypatch)
    config = bound(args['training_config'])
    root = Path(config['output'])
    config.update(wandb_mode='online', wandb_project='dagger')
    args['training_config'] = replace(args['training_config']['path'], config)
    state = bound(args['training_state'])
    state['training_config_sha256'] = args['training_config']['raw_sha256']
    logging = dict(mode='online', project='dagger', entity='fixture', id='fixture-run')
    state['wandb_run_sha256'] = replace(root / 'wandb-run.json', logging)['raw_sha256']
    if fault == 'missing_hash': state['wandb_run_sha256'] = None
    state['completion_operation_id'] = digest(canonical({key: state[key] for key in (
        'training_config_sha256', 'selected_checkpoint_sha256', 'epoch', 'global_step',
        'training_protocol', 'training_source_identity', 'wandb_run_sha256')}))
    for name in ('training_state.json', 'training-finalization-intent.json'):
        replace(root / name, state)
    args['training_state'] = replace(root / 'training-complete.json', state)
    if fault == 'changed_record': replace(root / 'wandb-run.json', {**logging, 'id': 'another-run'})
    if fault:
        with pytest.raises(ValueError): success.plan(**args, output=tmp_path / 'plan.json')
    else:
        doc = bound(success.plan(**args, output=tmp_path / 'plan.json')['plan'])
        assert doc['executed'] is False
        assert bound(doc['completion_proof'])['wandb_run_sha256'] == state['wandb_run_sha256']


def test_actual_cpu_trainer_completion_is_consumable_without_loading_checkpoint(tmp_path, monkeypatch):
    from test.pipeline.test_bc_fixed_final import FixedFinalBCTest
    trainer_test = FixedFinalBCTest()
    args, protocol, spec, paths = trainer_test._fixture(tmp_path)
    args.validation_split.write_bytes(b'{}')
    admission = replace(tmp_path / 'admission.json', {'purpose': 'cpu-schema-fixture'})
    args.data_admission_json = Path(admission['path'])
    root = trainer_test._run(args, protocol, spec, paths)
    training_state = binding(root / 'training-complete.json')
    training_config = binding(root / 'run_config.json')
    import torch
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('planner loaded a checkpoint'))
    identity, selection = success._fixed_final_selection(bound(training_state), bound(training_config),
        training_state=training_state, training_config=training_config, data_admission=admission, root=root)
    assert selection['selected_epoch'] == 2 and selection['selected_validation_mse'] is None
    assert identity['sha256'] == bound(training_state)['selected_checkpoint_sha256']
