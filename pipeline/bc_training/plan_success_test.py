"""Prepare, never submit, the frozen full-population success test for selected BC.

New fixed-final runs select the last complete epoch; historical validation
selection remains an explicit compatibility path. No test outcome selects a
checkpoint. This prepares the legacy single-owner B2 submission topology, NOT
the separately qualified paired/scaled executor. No pickle/model/CUDA load or
workload submission is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from pipeline.bc_training.handoff_common import bound, canonical, cli_binding, digest, publish
from pipeline.bc_training.data_admission import validate_training_admission
from pipeline.legacy.bc_collection.artifact_roles import validate_artifact_contract
from pipeline.deployment.rtx_actor.production_success_contract import evaluation_options, protocol_identity

def checkpoint_identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 4 << 30:
        raise ValueError('selected checkpoint must be a bounded regular file')
    before = path.stat()
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, name) != getattr(after, name) for name in fields):
        raise ValueError('checkpoint changed during identity verification')
    return dict(path=str(path), sha256=value.hexdigest(), bytes=before.st_size,
                file_identity={name: getattr(before, name) for name in fields})


def _fixed_final_selection(state, config, *, training_state, training_config, data_admission, root):
    """Verify the trainer's sealed complete-epoch handoff without loading torch."""
    from pipeline.bc_training.stopping_gate import fixed_final_contract
    protocol = fixed_final_contract(config.get('max_epochs'),
        budget_explicit=config.get('epoch_budget_explicit') is True,
        epoch_target=config.get('stop_training_epoch_mse_below'),
        eval_target=config.get('stop_training_eval_mse_below'))
    if (canonical(config.get('training_protocol')) != canonical(protocol)
            or canonical(state.get('training_protocol')) != canonical(protocol)
            or state.get('schema') != 'roborender.bc-fixed-final-state.v1'
            or state.get('status') != 'READY_FOR_TEST' or state.get('stop_reason') != 'explicit_epoch_budget'
            or state.get('validation_enabled') is not False
            or state.get('optimizer_state_saved') is not True or state.get('rng_state_saved') is not True
            or state.get('test_execution_started') is not False
            or state.get('training_config_sha256') != training_config['raw_sha256']
            or state.get('data_admission_sha256') != data_admission['raw_sha256']):
        raise ValueError('fixed_final requires the exact sealed no-validation completion proof')
    epoch, step, mse = state.get('epoch'), state.get('global_step'), state.get('training_epoch_mse')
    batch = state.get('training_batch_contract', {})
    if not isinstance(batch, dict):
        raise ValueError('fixed_final requires the complete training batch contract')
    updates, chunks = batch.get('updates_per_epoch'), batch.get('training_chunk_count')
    from pipeline.bc_training import cyclic_sampling
    sampling = state.get('training_sampling_contract', {})
    cyclic = sampling.get('schema') == cyclic_sampling.SCHEMA
    size = config.get('chunk_batch_size')
    lifecycle = config.get('student_lifecycle', 'direct_supervised_v1')
    if lifecycle == 'direct_supervised_v1' and size == 16:
        expected_batch = {'schema': 'raven_full_bxt_16x81_activation_checkpoint_v1'}
    elif lifecycle == 'raven_offline_replay_v1' and type(size) is int and size in (16, 32):
        expected_batch = dict(schema=('raven_offline_replay_16x81_activation_checkpoint_v2' if size == 16
            else 'raven_offline_replay_32x81_activation_checkpoint_v1'), student_lifecycle=lifecycle,
            replay_train_mode_forwards=81, student_normal_sample_calls_per_update=82,
            lane_permutation='torch.randperm_on_training_device_after_replay',
            object_prediction_loss_coefficient=0.0, object_head_trainable_with_zero_gradients=True)
    else:
        raise ValueError('fixed_final requires its exact versioned16/32 lifecycle identity')
    if cyclic:
        if (sampling.get('chunk_count') != chunks or sampling.get('batch_size') != size
                or sampling.get('mode') != 'raven_sorted_cyclic'
                or sampling.get('updates_per_reporting_epoch') != updates
                or sampling.get('reporting_epoch') != 'ceil_N_over_B_full_updates_cursor_never_reset'
                or sampling.get('partial_batch') != 'wrap_sorted_pool'
                or sampling.get('cursor_commit') != 'accepted_optimizer_updates_only'
                or batch.get('partial_batch') != 'wrap_sorted_pool'
                or config.get('experimental_shuffle', False)):
            raise ValueError('fixed_final cyclic sampling identity differs')
        cyclic_sampling.validate_cursor(state.get('training_cursor'), chunks, size, step)
        expected_updates = cyclic_sampling.updates_per_epoch(chunks, size)
    else:
        expected_updates = chunks // size if type(chunks) is int and chunks >= size and not chunks % size else None
    if (type(epoch) is not int or epoch != protocol['max_epochs']
            or type(step) is not int or type(updates) is not int or updates < 1
            or updates != expected_updates
            or step != epoch * updates or any(batch.get(k) != v for k, v in expected_batch.items())
            or batch.get('chunks_per_update') != size or batch.get('frames_per_chunk') != 81
            or batch.get('policy_forward_population') != size * 81
            or batch.get('drop_last') is not (not cyclic) or batch.get('activation_checkpointing') is not True
            or type(mse) not in (int, float) or not math.isfinite(mse) or mse < 0):
        raise ValueError('fixed_final requires a complete final epoch and exact training batch boundary')
    selected = Path(state['selected_checkpoint'])
    expected = root / 'epoch_checkpoints' / f'epoch_{epoch:04d}_step_{step:06d}.pt'
    if (selected != expected or not selected.resolve().is_relative_to(root)
            or Path(training_state['path']).resolve() != root / 'training-complete.json'
            or Path(training_config['path']).resolve() != root / 'run_config.json'):
        raise ValueError('fixed_final completion/config/checkpoint is not the selected run output')
    source = state.get('training_source_identity')
    expected_sources = {'pipeline/bc_training/train_bc_h200.py', 'pipeline/bc_training/stopping_gate.py',
                        'pipeline/dagger/services/raven_learner_worker.py'}
    if cyclic:
        expected_sources.add('pipeline/bc_training/cyclic_sampling.py')
    if (not isinstance(source, dict) or set(source) != expected_sources
            or any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
                   for value in source.values())):
        raise ValueError('fixed_final completion requires its exact source identity')
    mode, logging_sha = config.get('wandb_mode'), state.get('wandb_run_sha256')
    if mode == 'disabled':
        if 'wandb_run_sha256' not in state or logging_sha is not None or (root / 'wandb-run.json').exists():
            raise ValueError('fixed_final disabled logging identity differs')
    elif mode in ('online', 'offline'):
        if not isinstance(logging_sha, str) or not re.fullmatch('[0-9a-f]{64}', logging_sha):
            raise ValueError('fixed_final requires its exact logging record hash')
        logging = bound(dict(path=str(root / 'wandb-run.json'), raw_sha256=logging_sha))
        if logging.get('mode') != mode or logging.get('project') != config.get('wandb_project'):
            raise ValueError('fixed_final logging record differs from its run configuration')
    else:
        raise ValueError('fixed_final requires an explicit logging mode')
    operation = {key: state[key] for key in ('training_config_sha256', 'selected_checkpoint_sha256',
        'epoch', 'global_step', 'training_protocol', 'training_source_identity', 'wandb_run_sha256')}
    if state.get('completion_operation_id') != digest(canonical(operation)):
        raise ValueError('fixed_final completion operation identity differs')
    # The final publication occurs only after checkpoint/hash/logging success.
    # A resumed/changed live state cannot be silently paired with an old seal.
    for name in ('training_state.json', 'training-finalization-intent.json'):
        if bound(dict(path=str(root / name), raw_sha256=training_state['raw_sha256'])) != state:
            raise ValueError('fixed_final current state/finalization differs from completion proof')
    identity = checkpoint_identity(selected)
    if identity['sha256'] != state.get('selected_checkpoint_sha256'):
        raise ValueError('fixed_final final epoch checkpoint bytes changed')
    alias = checkpoint_identity(root / 'last.pt')
    if alias['sha256'] != identity['sha256']:
        raise ValueError('fixed_final last alias differs from the final epoch checkpoint')
    return identity, dict(selection_metric='last_completed_epoch', selected_epoch=epoch,
        selected_training_mse=mse, selected_validation_mse=None,
        training_protocol=protocol, validation_enabled=False,
        completion_proof=training_state)


def validate_union_admission(state, config, reference):
    """Recheck the admitted 2880 union without changing its frozen validator.

    Historical test metadata remains provenance, not permission to execute its
    ray-traced protocol. Native testing needs its separate rendered admission.
    """
    from pipeline.bc_training import train_union
    report = bound(reference)
    protocol = config.get('training_protocol')
    if (report.get('schema') != train_union.SCHEMA
            or not isinstance(protocol, dict) or protocol.get('schema') != 'roborender.bc-fixed-final.v1'
            or config.get('data_admission_json') != reference['path']
            or config.get('require_production_data_admission') is not True
            or any(config.get(key, False) for key in
                   ('allow_training_extension', 'allow_legacy_reuse', 'allow_qualification_data'))):
        raise ValueError('merged BC handoff requires fixed-final production admission without bypasses')
    actual = train_union.validate_training_admission(Path(reference['path']),
        Path(config['train_split']), Path(config['validation_split']), Path(config['dataset_root']))
    if (actual != report or report.get('production_admitted') is not True
            or report.get('validation_enabled') is not False
            or report.get('original_chunk_counts') != {'train': 2560, 'validation': 320}
            or report.get('chunk_count') != 2880 or report.get('frame_count') != 2880 * 81
            or any(state.get(role + '_split_sha256') != report['split_files'][role]['sha256']
                   for role in ('train', 'validation'))):
        raise ValueError('merged BC handoff differs from the exact admitted 2560+320 training union')
    return report


def plan(*, training_state, training_config, data_admission, output, command_id,
         resident_source_sha256, resident_gpu_uuid, resident_port, resident_token_file,
         resident_source_root, resident_jobs_root, resident_assets_root, panel_relative_path,
         rendered_observation_config=None):
    if Path(output).exists():
        raise FileExistsError('submission plan output already exists')
    if (not re.fullmatch(r'bc-success-[a-z0-9-]{1,80}', command_id)
            or not re.fullmatch('[0-9a-f]{64}', resident_source_sha256)
            or not re.fullmatch(r'GPU-[A-Za-z0-9-]+', resident_gpu_uuid)
            or type(resident_port) is not int or not 1024 <= resident_port < 65536):
        raise ValueError('explicit resident/source/command identity required')
    relative = Path(panel_relative_path)
    if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.json':
        raise ValueError('panel destination must be a safe relative JSON path')
    for path in (resident_token_file, resident_source_root, resident_jobs_root, resident_assets_root):
        if not Path(path).is_absolute():
            raise ValueError('resident destination paths must be absolute')
    state, config = bound(training_state), bound(training_config)
    report = bound(data_admission)
    from pipeline.bc_training import legacy_reuse, training_extension, train_union
    legacy = report.get('schema') == legacy_reuse.ADMISSION_SCHEMA
    extension = report.get('schema') == training_extension.ADMISSION_SCHEMA
    union = report.get('schema') == train_union.SCHEMA
    protocol = config.get('training_protocol')
    fixed = isinstance(protocol, dict) and protocol.get('schema') == 'roborender.bc-fixed-final.v1'
    if not fixed and protocol not in (None, 'legacy_validation'):
        raise ValueError('unknown BC selection protocol; historical selection is not an implicit fallback')
    if ((not fixed and (state.get('status') != 'complete'
                       or state.get('stop_reason') not in ('validation_saturation', 'max_epochs')))
            or config.get('data_admission_json') != data_admission['path']
            or (not (legacy or extension or union) and
                (report.get('purpose') != 'production' or report.get('production_admitted') is not True))):
        raise ValueError('completed production BC with exact data admission is required')
    roles = report['split_files']
    if union:
        actual = validate_union_admission(state, config, data_admission)
    elif legacy or extension:
        module = training_extension if extension else legacy_reuse
        opt_in = {'allow_training_extension': config.get('allow_training_extension') is True} if extension else {
            'allow_legacy_reuse': config.get('allow_legacy_reuse') is True}
        actual = module.validate_training_admission(data_admission['path'], config['train_split'],
            config['validation_split'], **opt_in,
            run_name=config['run_name'], bootstrap_path=config['bootstrap'], policy_spec_path=config['policy_spec'],
            resume_checkpoint=config.get('resume_checkpoint'), resume_validation_state=config.get('resume_validation_state'),
            dataset_root=config['dataset_root'])
    else:
        actual = validate_training_admission(Path(data_admission['path']), Path(config['train_split']),
                                            Path(config['validation_split']))
    digest_key = 'raw_sha256' if legacy or extension else 'sha256'
    if actual != report or any(state.get(role + '_split_sha256') != roles[role][digest_key]
                              for role in ('train', 'validation')):
        raise ValueError('training state binds a different admitted population')
    root = Path(config['output']).resolve(strict=True)
    if fixed:
        identity, selection = _fixed_final_selection(state, config, training_state=training_state,
            training_config=training_config, data_admission=data_admission, root=root)
    else:
        selected = Path(state['selected_checkpoint'])
        if (selected != root / 'best_validation.pt'
                or Path(training_state['path']).resolve() != root / 'validation_state.json'
                or Path(training_config['path']).resolve() != root / 'run_config.json'):
            raise ValueError('training state/checkpoint is not the selected run output')
        history = state.get('history', [])
        if (not history or any(type(r.get('epoch')) is not int or not isinstance(r.get('mse'), (int, float))
                               or not math.isfinite(r['mse']) or r['mse'] < 0 for r in history)
                or len({r['epoch'] for r in history}) != len(history)):
            raise ValueError('finite unambiguous complete-validation history required')
        best = min(history, key=lambda r: (r['mse'], r['epoch']))
        if state.get('best_epoch') != best['epoch'] or state.get('best_mse') != best['mse']:
            raise ValueError('selected state is not minimum complete-validation MSE')
        original = Path(best['checkpoint'])
        if not original.resolve().is_relative_to(root) or original.parent.name != 'validation_checkpoints':
            raise ValueError('best checkpoint does not belong to this validation history')
        identity = checkpoint_identity(selected)
        stat = original.stat()
        if (stat.st_dev, stat.st_ino) != (identity['file_identity']['st_dev'], identity['file_identity']['st_ino']):
            if checkpoint_identity(original)['sha256'] != identity['sha256']:
                raise ValueError('best_validation alias differs from selected epoch bytes')
        selection = dict(selection_metric='minimum_complete_validation_mse',
                         selected_epoch=best['epoch'], selected_validation_mse=best['mse'])
    if union:
        test = report['protected_historical_test']
    elif legacy or extension:
        test = report['closed_loop_test']
    else:
        contract = report['artifact_roles']
        validate_artifact_contract(report)
        test = contract['closed_loop_test']
    panel_binding = test['panel']
    panel = bound(panel_binding)
    if test.get('episode_count') != 295 or len(test.get('motion_bindings', {})) != 295:
        raise ValueError('this production handoff requires frozen equal-motion full295 test')
    source_panel_binding = panel_binding
    if panel.get('observation_mode') == 'raytraced_3view':
        # Old immutable data admission keeps its historical panel reference.
        # Only the newly executed observation mode/label change; cases and all
        # reset assignments remain byte-for-byte identical.
        from pipeline.deployment.rtx_actor.production_success_contract import build_rendered_panel
        panel = build_rendered_panel(panel)
        panel_binding = publish(Path(output).with_name(Path(output).stem+'.rendered-panel.json'), panel)
    options = evaluation_options(panel, raw_sha256=panel_binding['raw_sha256'])
    from pipeline.deployment.rendered_success_dispatch import artifact, payload_fields
    rendered_config, _, rendered_sha=artifact(rendered_observation_config,panel,identity['sha256'])
    rendered_path=Path(output).with_name(Path(output).stem+'.rendered-config.json')
    rendered_binding=publish(rendered_path,rendered_config)
    if rendered_binding['raw_sha256']!=rendered_sha:raise ValueError('rendered configuration serialization differs')
    rendered_relative=Path('rendered-success')/(rendered_sha+'.json')
    # validate_training_admission independently rechecks test identity, support,
    # process-seed isolation, balanced counts, and exact protocol binding.
    command = dict(command_id=command_id, operation='evaluate_explicit_panel', payload=dict(
        panel_file=str(relative), panel_sha256=panel_binding['raw_sha256'],
        checkpoint_sha256=identity['sha256'], arm='bc', output_name=command_id, **options,
        **payload_fields(str(rendered_relative),rendered_sha)))
    command_path = Path(output).with_name(Path(output).stem + '.command.json')
    command_binding = publish(command_path, command)
    remote_command = str(Path(resident_jobs_root) / 'commands' / command_path.name)
    argv = ['python3', str(Path(resident_source_root) / 'pipeline/deployment/rtx_actor/resident_isaac_client.py'),
            '--port', str(resident_port), '--token-file', str(resident_token_file),
            '--source-sha256', resident_source_sha256, '--gpu-uuid', resident_gpu_uuid,
            '--command', remote_command, '--wait']
    document = dict(schema='roborender.bc-selected-success-test-plan.v2' if fixed else
        'roborender.bc-selected-success-test-plan.v1', executed=False,
        production_training_completed=not (legacy or extension), legacy_reuse_training_completed=legacy,
        training_extension_completed=extension,
        success_test_completed=False,
        **selection, training_state=training_state,
        training_config=training_config, data_admission=data_admission, checkpoint=identity,
        test_panel=panel_binding, test_panel_canonical_sha256=digest(canonical(panel)),
        source_data_admission_test_panel=source_panel_binding,
        test_case_count=295, motion_count=295, optimizer_steps=0, command=command,
        success_test_protocol=protocol_identity(panel),
        rendered_observation_config=rendered_binding,
        command_file=command_binding, submission_argv_on_resident_host=argv,
        required_staging=[dict(source=identity['path'], destination=str(Path(resident_assets_root) / (identity['sha256']+'.pt')),
                               sha256=identity['sha256'], bytes=identity['bytes']),
                          dict(source=panel_binding['path'], destination=str(Path(resident_jobs_root) / relative),
                               sha256=panel_binding['raw_sha256']),
                          dict(source=rendered_binding['path'],destination=str(Path(resident_jobs_root)/rendered_relative),
                               sha256=rendered_sha),
                          dict(source=command_binding['path'], destination=remote_command,
                               sha256=command_binding['raw_sha256'])],
        resident=dict(source_sha256=resident_source_sha256, gpu_uuid=resident_gpu_uuid,
                      port=resident_port, token_file=str(resident_token_file)),
        execution_profile='legacy_single_owner_b2_explicit_panel',
        scaled_production_ready=False, production_execution_authorized=False,
        remaining_gates=['stage exact checkpoint and panel bytes', 'attest resident source/GPU/readiness',
                         'qualify actual resident B2/B1 tail, host memory and declared geometry coverage',
                         'paired scaled execution requires a separate source/topology/checkpoint-bound scaled plan and qualification',
                         'submit command explicitly', 'require completed 295-case result; do not infer success from plan'])
    result = publish(output, document)
    return dict(plan=result, executed=False, selected_epoch=selection['selected_epoch'], checkpoint_sha256=identity['sha256'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('training-state', 'training-config', 'data-admission', 'rendered-observation-config'):
        parser.add_argument('--'+key, nargs=2, required=True, metavar=('PATH', 'RAW_SHA256'))
    for key in ('command-id', 'resident-source-sha256', 'resident-gpu-uuid', 'resident-token-file', 'resident-source-root',
                'resident-jobs-root', 'resident-assets-root', 'panel-relative-path'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--resident-port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args())
    for key in ('training_state', 'training_config', 'data_admission', 'rendered_observation_config'):
        args[key] = cli_binding(args[key])
    print(json.dumps(plan(**args), sort_keys=True))

if __name__ == '__main__':
    main()
