"""CPU metadata integration, not native qualification or trained-policy proof.

The real union validator and trainer completion publisher are exercised. Only
the pre-existing expensive historical native/render audit uses its shared test
double. Checkpoint bytes are intentionally not torch payloads: no loader may
unpickle or launch a process during selection.
"""
from test.deployment.source_paths import REPO_ROOT
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from pipeline.bc_training import final_selection, plan_success_test, train_bc_h200 as trainer, train_union
from pipeline.bc_training.handoff_common import binding, bound, canonical, publish
from pipeline.bc_training.stopping_gate import fixed_final_contract
from pipeline.dagger import coverage_queue
from pipeline.deployment import bc_success_dispatch
from pipeline.deployment.rtx_actor import production_success_contract as original, scaled_success_contract as scaled
from pipeline.deployment.rtx_actor.rendered_student_success_contract import make_sharded_config
from test.pipeline.test_bc_train_union import base as base_fixture, sha, write_json


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    base = base_fixture.__wrapped__(tmp_path, monkeypatch)
    root = REPO_ROOT
    historical = binding(root / 'runtime/production-audits/20261015-raven-faithful/'
                         'plan-only-native295-b2-20261015.success-panel.json')
    legacy = bound(historical)
    roles = deepcopy(base['manifest']['artifact_roles'])
    roles['offline_chunk_counts'] = {'train': 2560, 'validation': 320}
    roles['closed_loop_test'].update(panel=historical,
        motion_bindings={r['motion_id']: r['sha256'] for c in legacy['cohorts'] for r in c['assignment']['lanes']})
    rows, indexes = [], {}
    for role, count in roles['offline_chunk_counts'].items():
        prototype = next(r for r in base['manifest']['chunks'] if r['split'] == role)
        items = []
        for i in range(count):
            cid = f'fixture-{role}-{i:04d}'
            row = deepcopy(prototype)
            row.update(chunk_id=cid, isaac_npz=f'isaac/{cid}.npz', rendered_npz=f'rendered/{cid}.npz')
            for field, size in (('isaac_npz', 'source_size_bytes'), ('rendered_npz', 'rendered_size_bytes')):
                (base['root'] / row[field]).write_bytes(b'x')
                row[size] = 1
            items.append({key: row[key] for key in ('chunk_id', 'isaac_npz', 'rendered_npz', 'stream_id')})
            rows.append(row)
        indexes[role] = dict(schema='roborender.bc_split.v1', split=role, chunk_count=count,
                             source_manifest=str(base['source']), items=items)
    source = {**base['manifest'], 'chunks': rows, 'chunk_count': 2880,
              'partition_counts': roles['offline_chunk_counts'], 'artifact_roles': roles}
    write_json(base['source'], source)
    write_json(base['source_pool'], dict(schema='cpu-only-native-pool', chunk_count=2880, chunks=rows))
    protocol = json.loads(base['protocol'].read_bytes())
    protocol['source_pool'] = binding(base['source_pool'])
    write_json(base['protocol'], protocol)
    report = deepcopy(base['report'])
    report.update(artifact_roles=roles, source_manifest_sha256=sha(base['source']), protocol=binding(base['protocol']))
    inventory = dict(schema='roborender.motion-inventory.v1', motion_count=295,
        motions=[dict(motion_id=r['motion_id'], sha256=r['sha256'])
                 for c in legacy['cohorts'] for r in c['assignment']['lanes']])
    sampling = dict(schema='roborender.coverage-sampling-plan.v3',
        partitions={'train': {'cohorts': [dict(process_seed=40)]},
                    'validation': {'cohorts': [dict(process_seed=41)]}},
        selection_uses_labels_rewards_or_loss=False,
        motion_inventory_sha256=coverage_queue.digest(inventory))
    report.update(closed_loop_test_executed=False, warnings=[])
    report['protocol'].update(sampling_plan_canonical_sha256=coverage_queue.digest(sampling),
                             motion_inventory_canonical_sha256=coverage_queue.digest(inventory))
    for role, index in indexes.items():
        index['source_manifest_sha256'] = sha(base['source'])
        path = Path(report['split_files'][role]['path'])
        write_json(path, index)
        report['split_files'][role]['sha256'] = sha(path)
        report['splits'][role] = dict(chunks=len(index['items']), frames=len(index['items']) * 81)
    write_json(base['report_path'], report)
    union = train_union.build_union(base['report_path'], base['root'], base['output'])
    panel = scaled.build_panel(original.build_rendered_panel(legacy), batch_size=16)
    from test.deployment.rtx_actor.test_rendered_endpoint_pool import pool_config
    from test.deployment.rtx_actor.test_rendered_student_success_contract import rendering_config
    rendering = make_sharded_config(rendering_config(original.build_rendered_panel(legacy)), panel,
                                    'a' * 64, pool_config(8)['endpoints'])
    return dict(base=base, union=union, data_admission=binding(base['output'] / 'data-admission.json'),
                bc_sampling_plan=publish(tmp_path / 'sampling.json', sampling),
                inventory=publish(tmp_path / 'inventory.json', inventory),
                test_panel=publish(tmp_path / 'scaled-test.json', panel),
                rendered_observation_config=publish(tmp_path / 'rendering.json', rendering))


def seal(admitted, directory, batch_size):
    """Use the real CPU-only publisher, with the trainer's emitted field names."""
    directory.mkdir()
    bootstrap = directory / 'initial.pt'; bootstrap.write_bytes(b'CPU-only saved bootstrap fixture')
    spec = publish(directory / 'policy-spec.json', dict(checkpoint_sha256=sha(bootstrap), cpu_fixture=True))
    report = admitted['union']; updates = 2880 // batch_size
    _,paths=trainer.load_split(Path(report['split_files']['train']['path']), 'train', admitted['base']['root'])
    sampling=trainer.training_sampling_contract(sorted(paths), admitted['base']['root'], shuffle=False,
                                                cyclic_batch_size=batch_size)
    config = dict(output=str(directory), run_name=directory.name, bootstrap=str(bootstrap),
        policy_spec=spec['path'], dataset_root=str(admitted['base']['root']),
        data_admission_json=admitted['data_admission']['path'],
        train_split=report['split_files']['train']['path'], validation_split=report['split_files']['validation']['path'],
        require_production_data_admission=True, chunk_batch_size=batch_size,
        student_lifecycle='raven_offline_replay_v1', experimental_shuffle=False,
        max_epochs=1, epoch_budget_explicit=True, training_protocol=fixed_final_contract(1, budget_explicit=True),
        wandb_mode='disabled',training_sampling_contract=sampling)
    config_ref = publish(directory / 'run_config.json', config)
    checkpoint = directory / 'epoch_checkpoints' / f'epoch_0001_step_{updates:06d}.pt'
    checkpoint.parent.mkdir(); checkpoint.write_bytes(b'CPU fixture final checkpoint: never unpickle')
    os.link(checkpoint, directory / 'last.pt')
    state = dict(schema='roborender.bc-fixed-final-state.v1', status='TRAINING', epoch=1, global_step=updates,
        training_epoch_mse=.25, selected_checkpoint=str(checkpoint), selected_checkpoint_sha256=sha(checkpoint),
        training_protocol=config['training_protocol'], validation_enabled=False,
        training_config_sha256=config_ref['raw_sha256'], data_admission_sha256=admitted['data_admission']['raw_sha256'],
        bootstrap_sha256=sha(bootstrap), policy_spec_sha256=spec['raw_sha256'], wandb_run_sha256=None,
        train_split_sha256=report['split_files']['train']['sha256'],
        validation_split_sha256=report['split_files']['validation']['sha256'],
        training_source_identity=trainer.training_source_identity(),
        training_sampling_contract=sampling,training_cursor=trainer.cyclic_sampling.cursor_state(2880,batch_size,updates),
        training_batch_contract=dict(**trainer.student_lifecycle_batch_fields('raven_offline_replay_v1', batch_size),
            chunks_per_update=batch_size, frames_per_chunk=81, policy_forward_population=batch_size * 81,
            training_chunk_count=2880, updates_per_epoch=updates, drop_last=False,
            partial_batch='wrap_sorted_pool',epoch_is_exact_dataset_pass=True,
            index_safe_max_pools=0, activation_checkpointing=True))
    publish(directory / 'training_state.json', state)
    trainer.publish_fixed_final_completion(directory, elapsed_seconds=0., recovery=False)
    return dict(training_state=binding(directory / 'training-complete.json'), training_config=config_ref,
        **{key: admitted[key] for key in ('data_admission', 'test_panel', 'rendered_observation_config')})


@pytest.mark.parametrize('batch_size', [16, 32])
def test_actual_publisher_union_selection_and_dispatch_gate(admitted, tmp_path, monkeypatch, batch_size):
    args = seal(admitted, tmp_path / f'new-bc-{batch_size}', batch_size)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('selection launched a process'))
    monkeypatch.setattr(trainer.torch, 'load', lambda *a, **k: pytest.fail('selection unpickled a checkpoint'))
    value = final_selection.build_selection(**args)
    assert final_selection.validate_selection(value) == value
    assert bc_success_dispatch.validate_selection(value) == bound(args['test_panel'])
    assert value['schema'] == final_selection.SCHEMA
    assert value['selected_global_step'] == 2880 // batch_size
    assert value['selection'] == 'last_completed_epoch' and value['selected_validation_mse'] is None
    assert not value['validation_enabled'] and not value['success_test_executed']
    assert not value['production_execution_authorized']
    assert value['training_batch_contract']['chunks_per_update'] == batch_size


def test_selection_cannot_relabel_diagnostic_or_change_any_bound_fields(admitted, tmp_path):
    args = seal(admitted, tmp_path / 'new-bc', 32)
    value = final_selection.build_selection(**args)
    changes = [dict(schema='roborender.bc32-lifecycle-qualification.v1'),
        dict(selection='minimum_complete_validation_mse'), dict(selected_global_step=2),
        dict(selected_epoch=0), dict(validation_enabled=True), dict(selected_validation_mse=.1),
        dict(optimizer_state_preserved=False), dict(rng_state_preserved=False),
        dict(success_test_executed=True), dict(expected_test_episodes=32),
        dict(production_execution_authorized=True), dict(checkpoint={**value['checkpoint'], 'sha256': 'f' * 64})]
    for change in changes:
        with pytest.raises(ValueError): final_selection.validate_selection({**value, **change})
    checkpoint = Path(value['checkpoint']['path']); checkpoint.write_bytes(b'changed selected bytes')
    with pytest.raises(ValueError): final_selection.validate_selection(value)


def test_new_selection_rejects_b2_qualification_and_shared_renderer_pool(admitted, tmp_path):
    args = seal(admitted, tmp_path / 'new-bc', 16)
    panel = bound(args['test_panel']); source = panel['scaled_protocol']['source_panel']
    for i, replacement in enumerate((source, scaled.build_panel(source, qualification=True),
                                      scaled.build_panel(source, batch_size=8))):
        ref = publish(tmp_path / f'wrong-panel-{i}.json', replacement)
        with pytest.raises(ValueError): final_selection.build_selection(**{**args, 'test_panel': ref})
    rendering = bound(args['rendered_observation_config'])
    rendering['renderer']['actor_shards'][1]['endpoint_slots'][0] = 3
    ref = publish(tmp_path / 'overlap.json', rendering)
    with pytest.raises(ValueError): final_selection.build_selection(**{**args, 'rendered_observation_config': ref})


def test_existing_planner_accepts_union_but_stays_explicitly_legacy_topology(admitted, tmp_path):
    args = seal(admitted, tmp_path / 'new-bc', 32)
    from test.deployment.rtx_actor.test_rendered_student_success_contract import rendering_config
    panel = bound(args.pop('test_panel'))['scaled_protocol']['source_panel']
    args['rendered_observation_config'] = publish(tmp_path / 'single-dependencies.json', rendering_config(panel))
    result = plan_success_test.plan(**args, output=tmp_path / 'legacy-plan.json', command_id='bc-success-new-final',
        resident_source_sha256='a' * 64, resident_gpu_uuid='GPU-fixture', resident_port=18640,
        resident_source_root='/source', resident_jobs_root='/jobs', resident_assets_root='/assets',
        resident_token_file='/private/token', panel_relative_path='tests/panel.json')
    value = bound(result['plan'])
    assert value['selection_metric'] == 'last_completed_epoch' and value['selected_validation_mse'] is None
    assert value['production_training_completed'] is True
    assert value['execution_profile'] == 'legacy_single_owner_b2_explicit_panel'
    assert value['scaled_production_ready'] is False and value['production_execution_authorized'] is False


def test_queue_requires_actual_paired_evidence_not_raw_outcome_flag(admitted, tmp_path, monkeypatch):
    args = seal(admitted, tmp_path / 'new-bc', 32)
    selected = final_selection.build_selection(**args)
    success = publish(tmp_path / 'actual-evidence.json', dict(schema='cpu-evidence-placeholder'))
    queue = dict(schema=coverage_queue.CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA,
        bc_selection_path=publish(tmp_path / 'selected.json', selected)['path'],
        bc_success_path=success['path'], panel=args['test_panel'], bc_data_admission=args['data_admission'],
        policy_spec_sha256=coverage_queue.digest(bound(selected['policy_spec'])), source_sha256='a' * 64)
    calls = []
    module = ModuleType('pipeline.deployment.paired_scaled_evaluation')
    def checked(evidence, **kwargs):
        calls.append((evidence, kwargs))
        return dict(completed_episodes=295, arm='bc', qualification_only=False)
    module.validate_final_evidence = checked
    monkeypatch.setitem(sys.modules, module.__name__, module)
    online = ModuleType('pipeline.deployment.paired_bc_success')
    online_calls=[]
    online.validate_online_handoff=lambda *args,**kwargs:online_calls.append((args,kwargs))
    monkeypatch.setitem(sys.modules, online.__name__, online)
    assert coverage_queue.validate_bc_handoff(queue) == selected
    assert online_calls[0][1]['selection_ref']==binding(queue['bc_selection_path'])
    assert online_calls[0][1]['result']==dict(completed_episodes=295,arm='bc',qualification_only=False)
    from pipeline.deployment.rendered_success_dispatch import configuration
    assert calls[0][1] == dict(panel=bound(args['test_panel']), checkpoint_sha256=selected['checkpoint']['sha256'],
        source_sha256='a' * 64, arm='bc', expected_rendered_config=configuration(
            args['rendered_observation_config'], bound(args['test_panel']), selected['checkpoint']['sha256']))
    online.validate_online_handoff=lambda *a,**k:(_ for _ in ()).throw(ValueError('missing online readback'))
    with pytest.raises(ValueError,match='online readback'):coverage_queue.validate_bc_handoff(queue)
    online.validate_online_handoff=lambda *a,**k:None
    for bad in (dict(completed_episodes=32, arm='bc', qualification_only=True),
                dict(completed_episodes=295, arm='teacher', qualification_only=False)):
        module.validate_final_evidence = lambda *a, **k: bad
        with pytest.raises(ValueError): coverage_queue.validate_bc_handoff(queue)
    module.validate_final_evidence = lambda *a, **k: (_ for _ in ()).throw(ValueError('missing actual joint proof'))
    with pytest.raises(ValueError, match='joint proof'): coverage_queue.validate_bc_handoff(queue)


def test_fresh_queue_uses_original_union_sampling_provenance(admitted, tmp_path):
    args = seal(admitted, tmp_path / 'new-bc', 32)
    selected = final_selection.build_selection(**args)
    files = [dict(path=name, sha256=sha, bytes=1, executable=False)
             for name, sha in sorted(trainer.training_source_identity().items())]
    source = dict(schema='roborender.source-snapshot.v1', files=files, file_count=len(files),
                  source_sha256=coverage_queue.digest(files))
    queue = dict(schema=coverage_queue.CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA,
        bc_selection_path=publish(tmp_path / 'selected.json', selected)['path'],
        source_manifest=publish(tmp_path / 'source.json', source), source_sha256=source['source_sha256'],
        panel=args['test_panel'], bc_data_admission=args['data_admission'],
        inventory=admitted['inventory'], bc_sampling_plan=admitted['bc_sampling_plan'],
        excluded_process_seeds=[40, 41])
    sampling, admission = coverage_queue._fresh_inputs(queue)
    assert sampling == bound(admitted['bc_sampling_plan']) and admission == admitted['union']
    with pytest.raises(ValueError): coverage_queue._fresh_inputs({**queue, 'excluded_process_seeds': [40]})
    with pytest.raises(ValueError): coverage_queue._fresh_inputs({**queue, 'schema': coverage_queue.SCALED_FRESH_SCHEMA})
    base_ref = admitted['union']['base_admission']
    path = Path(base_ref['path']); changed = json.loads(path.read_bytes())
    changed['protocol']['sampling_plan_canonical_sha256'] = 'f' * 64
    path.write_bytes(canonical(changed) + b'\n')
    with pytest.raises(ValueError): coverage_queue._fresh_inputs(queue)
