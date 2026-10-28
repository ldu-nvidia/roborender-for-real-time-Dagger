"""CPU-only provider integration; these fixtures never certify native capacity."""
from test.deployment.source_paths import REPO_ROOT
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pipeline.bc_training.handoff_common import binding, bound, publish
from pipeline.dagger.beta_campaign import immutable_json
from pipeline.dagger.coverage_production import digest
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import LEO_CONTINUOUS_FIXED100_SCHEMA, LEO_GPU_UUIDS
from pipeline.deployment import continuous_production_collection as module
from test.pipeline.test_dagger_coverage_production import production
from test.pipeline.test_dagger_continuous_plan import continuous_plan, continuous_evidence


@pytest.fixture
def provider(continuous_evidence, tmp_path):
    e = continuous_evidence
    cfg = dict(schema=LEO_CONTINUOUS_FIXED100_SCHEMA, run_id=e['plan']['run_id'], round_id=1,
        source_sha256=e['plan']['source_sha256'], coverage_plan=binding(tmp_path/'plan.json'),
        student=dict(initial_checkpoint_sha256='d'*64),
        rtx_actors=[dict(physical_gpu_index=g, concurrent_environments=16) for g in (0, 1)],
        logging=dict(wandb_run_id=e['plan']['rounds'][0]['wandb_run_id']))
    directory = tmp_path/'round'/'continuous'
    directory.mkdir(parents=True)
    commands = []; identities = []; prefixes = []; terminals = []
    for gpu, (pr, tr) in enumerate(zip(e['proof']['native_prefix_receipts'], e['terminals'])):
        row = bound(pr); terminal = bound(tr)
        for obj in (row, terminal):
            obj['actor_state'].update(source_sha256=cfg['source_sha256'], config_sha256=digest(cfg))
        prefix_ref = publish(tmp_path/f'provider-prefix-{gpu}.json', row)
        e['proof']['native_prefix_receipts'][gpu] = prefix_ref
        payload = dict(output_name=row['command_id'], motion_assignment_sha256=row['motion_assignment_sha256'])
        commands.append(dict(gpu=gpu, command_id=row['command_id'], operation='collect_production_segment', payload=payload))
        identities.append(dict(worker_instance_id=row['worker_instance_id'], native_pid=row['native_pid'],
            gpu_uuid=LEO_GPU_UUIDS[gpu], source_sha256=cfg['source_sha256']))
        prefixes.append(dict(actor_state=row['actor_state'], resource_guard=row['resource_guard'], episode_assignments=[]))
        terminals.append(dict(status='COMPLETE', receipt=dict(complete=True,
            segment_sha256=row['segment_sha256'], actor_state=terminal['actor_state'],
            resource_guard=terminal['resource_guard'])))
    site = SimpleNamespace(config=cfg, run=cfg['run_id'], rid=1, round=tmp_path/'round',
        jobs=tmp_path/'jobs', token=tmp_path/'token', pool=dict(workers=[]),
        source=REPO_ROOT)
    site.token.write_text('CPU-only-test-secret')
    intent = dict(commands=commands, owners=identities,
        segment_path=str(site.jobs/'round'/'segment.json'), segment_sha256=row['segment_sha256'],
        input_checkpoint_sha256='d'*64, config_sha256=digest(cfg), plan_sha256=digest(e['plan']))
    immutable_json(directory/'intent.json', intent)
    context = dict(status='selected_pending_test', iteration_id=100, stop_reason='fixed_100_updates',
        checkpoint_sha256=e['final'], selection=dict(checkpoint_selection='final_update_100',
            best_checkpoint_sha256=e['final'],
            history=[dict(iteration=100, result=dict(checkpoint_sha256=e['final']))]))
    barrier = dict(owned_actors_stopped=False,
        native_phase_barrier=dict(schema='roborender.isaac-phase-barrier.v1', workers=[
            dict(physical_gpu=g, gpu_uuid=i['gpu_uuid'], worker_instance_id=i['worker_instance_id'],
                source_sha256=cfg['source_sha256'], phase='IDLE') for g, i in enumerate(identities)]))
    site.barrier = lambda: barrier
    state = dict(polls=0, actions=[])
    def pool(action, cmds=None):
        state['actions'].append(action)
        if action == 'status':
            terminal = state['polls'] >= 2
            return dict(workers=[dict(i, phase='IDLE' if terminal else 'RUNNING',
                active_command_id=None if terminal else c['command_id']) for c, i in zip(commands, identities)])
        assert action == 'result', 'reattachment must never submit/stop/restart any actor'
        state['polls'] += 1
        return dict(results=terminals if state['polls'] >= 2 else [dict(status='RUNNING')]*2)
    site.pool_action = pool
    def get(route):
        if '/jobs/' in route:
            index = int(route.rsplit('-', 1)[-1]) - 1
            value = bound(binding(tmp_path/f'learner-{index}.json'))
            return dict(job=dict(status='succeeded', result=value))
        if '/mse-rounds/' in route: return dict(mse_round=context)
        return dict(iterations=e['iterations'])
    return e, site, intent, prefixes, state, get


def test_existing_command_continues_through_gate_without_resubmit(provider, monkeypatch):
    e, site, intent, prefixes, state, get = provider
    monkeypatch.setattr(module, 'ServiceClient', lambda *a, **kw: SimpleNamespace(get=get))
    monkeypatch.setattr(module, '_read_prefix', lambda s, c: prefixes[c['gpu']])
    calls = []
    def admit(s, plan, saved, ps, learner, directory):
        assert saved == intent and ps == prefixes
        calls.append('qualification')
        immutable_json(directory/'qualification.json', e['proof'])
        return binding(directory/'qualification.json')
    monkeypatch.setattr(module, 'admit_prefix', admit)
    result = module.collect(site, sleep=lambda _: None)
    assert calls == ['qualification']
    assert result['completed_updates'] == 100
    assert result['native_lifecycles_per_actor'] == [1, 1]
    assert not {'submit', 'shutdown', 'abort', 'wait'} & set(state['actions'])
    assert len(result['native_terminal_receipts']) == 2


@pytest.mark.parametrize('fault', ['owner', 'native_pid', 'source', 'different_command', 'crash'])
def test_unknown_native_state_never_restarts(provider, monkeypatch, fault):
    _, site, intent, _, state, _ = provider
    actual = [dict(i, phase='RUNNING', active_command_id=c['command_id'])
              for c, i in zip(intent['commands'], intent['owners'])]
    if fault == 'owner': actual[0]['worker_instance_id'] = 'f'*32
    elif fault == 'native_pid': actual[0]['native_pid'] += 1
    elif fault == 'source': actual[0]['source_sha256'] = 'f'*64
    elif fault == 'different_command': actual[0]['active_command_id'] = 'another-command'
    else: actual[0]['phase'] = 'CRASH_UNCERTAIN'
    site.pool_action = lambda action: dict(workers=actual)
    with pytest.raises(ValueError):
        module.reconcile_owners(site, intent['commands'], intent['owners'])


def test_gate_uses_native_mount_and_atomic_publication(provider, monkeypatch):
    e, site, intent, _, _, _ = provider
    ref = publish(site.round/'qualification.json', e['proof'])
    writes = {}; commands = []
    def publish_remote(path, raw, host, mode):
        assert host == 'beta' and mode == 0o444
        writes[str(path)] = json.loads(raw)
    site.publish = publish_remote
    site.command = lambda argv, host: commands.append((argv, host))
    module.publish_gate(site, intent, ref)
    gate_path = str(Path(intent['segment_path']).parent/'continuous-qualification-gate.prepared.json')
    gate = writes[gate_path]
    assert gate['proof']['path'].startswith('/resident/jobs/')
    assert gate['config_sha256'] == digest(site.config)
    assert gate['native_pids'] == {'0': 100, '1': 101}
    assert len(commands) == 1 and 'os.link(src,dst)' in commands[0][0][2]


def test_prefix_file_is_not_mistaken_for_a_terminal_receipt(provider):
    _, site, intent, prefixes, _, _ = provider
    state = dict(prefixes[0]['actor_state'], resource_guard=prefixes[0]['resource_guard'], episode_assignments=[])
    site.pool['workers'] = [dict(runtime_root='/cpu-only/g0')]
    site.command = lambda *args: json.dumps(state).encode()
    result = module._read_prefix(site, intent['commands'][0])
    assert result['actor_state']['phase'] == 'WAITING_FOR_QUALIFICATION'
    assert 'complete' not in result


def test_admission_builds_strict_proof_from_four_jobs_and_both_owners(provider, monkeypatch):
    e, site, intent, prefixes, state, get = provider
    site.verify_retained = lambda: dict(gpu_uuids=e['proof']['h200_uuids'])
    monkeypatch.setattr(module, 'copy_native_assignments',
        lambda *a: e['proof']['native_assignment_receipts'])
    monkeypatch.setattr(module, 'online_prefix', lambda *a: bound(e['proof']['online_metrics_receipt']))
    gates = []
    monkeypatch.setattr(module, 'publish_gate', lambda s, i, ref: gates.append(bound(ref)))
    proof_ref = module.admit_prefix(site, e['plan'], intent, prefixes,
        SimpleNamespace(get=get), site.round/'continuous')
    proof = bound(proof_ref)
    assert len(gates) == 1 and gates[0] == proof
    assert len(proof['learner_receipts']) == 4
    assert len(proof['native_prefix_receipts']) == 2
    assert len(proof['native_assignment_receipts']) == 32
    assert state['actions'] == ['status']


def test_prefix_resource_raw_evidence_is_checked_on_owner(provider, tmp_path):
    _, site, intent, prefixes, _, _ = provider
    runtime = tmp_path/'native-runtime'
    phase = runtime/'phases'/intent['commands'][0]['payload']['output_name']
    phase.mkdir(parents=True)
    site.pool['workers'] = [dict(runtime_root=str(runtime))]
    guard = dict(prefixes[0]['resource_guard'])
    # A summary can look healthy while the bound raw file has changed.
    raw = dict(guard, interval_s=.25)
    resource = phase/'resources.json'
    ref = publish(resource, raw)
    guard.update(path=str(resource), raw_sha256=ref['raw_sha256'])
    state = dict(prefixes[0]['actor_state'], resource_guard=guard, episode_assignments=[])
    publish(phase/'continuous-prefix-state.json', state)
    def remote(argv, host):
        assert host == 'beta'
        return subprocess.check_output([sys.executable, *map(str, argv[1:])], cwd=site.source,
                                       stderr=subprocess.PIPE)
    site.command = remote
    assert module._read_prefix(site, intent['commands'][0])['resource_guard'] == guard
    resource.write_text('{}')
    with pytest.raises(subprocess.CalledProcessError):
        module._read_prefix(site, intent['commands'][0])


def test_first_dispatch_persists_exact_intent_before_submit(provider, tmp_path):
    e, site, old, _, _, _ = provider
    directory = tmp_path/'fresh-intent'
    site.remote = site.jobs/'round'
    cfg_ref = publish(tmp_path/'config.json', site.config)
    protocol = tmp_path/'protocol.json'
    publish(protocol, dict(cpu_only_protocol=True))
    site.request = dict(round_config_path=cfg_ref['path'], mse_protocol_path=str(protocol),
        mse_protocol_file_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest())
    site.read = lambda *args: {}
    writes = []
    site.publish = lambda path, raw, *args: writes.append((path, raw))
    submits = []
    def pool(action, commands=None):
        if action == 'status':
            return dict(workers=[dict(row, phase='IDLE', active_command_id=None) for row in old['owners']])
        assert action == 'submit'
        saved = json.loads((directory/'intent.json').read_bytes())
        assert commands == saved['commands']
        submits.append(commands)
    site.pool_action = pool
    intent = module._stage(site, e['plan'], directory)
    assert len(submits) == 1 and len(submits[0]) == 2
    assert all(c['operation'] == 'collect_production_segment' for c in submits[0])
    segment = next(json.loads(raw) for path, raw in writes if path.name == 'segment.json')
    assert segment['window_count'] == 100 and segment['start_iteration_index'] == 0
    assert segment['qualification_after_updates'] == 4
    assert intent['segment_sha256'] == digest(segment)
    from pipeline.dagger.rtx.code.src.dagger_rtx.training_curriculum import bootstrap_request
    assert [c['payload']['training_curriculum'] for c in submits[0]] == [bootstrap_request()] * 2


def test_later_production_round_cannot_bootstrap_missing_curriculum(provider, tmp_path):
    evidence, site, old, _, _, _ = provider
    site.rid = 2
    site.request = {}
    site.pool_action = lambda _: dict(workers=[dict(row, phase='IDLE', active_command_id=None) for row in old['owners']])
    with pytest.raises(ValueError, match='predecessor curriculum'):
        module._stage(site, evidence['plan'], tmp_path/'unsubmitted-round2')


def test_production_context_carries_exact_prior_actor_exports(provider, tmp_path, monkeypatch):
    from pipeline.dagger.coverage_campaign import CoverageMSECampaign, MSECampaign
    from pipeline.dagger.rtx.code.src.dagger_rtx.training_curriculum import build_restore_request, digest as cdigest
    from test.pipeline.test_training_curriculum import binding as curriculum_binding, trained
    from pipeline.dagger.rtx.code.src.dagger_rtx.training_curriculum import export_training_curriculum
    evidence, site, _, _, _, _ = provider
    monkeypatch.setattr(MSECampaign, 'operation_context', lambda _: {})
    campaign = object.__new__(CoverageMSECampaign)
    campaign.coverage = evidence['plan']
    refs, wrappers = [], []
    for gpu in (0, 1):
        data = export_training_curriculum(trained(), curriculum_binding(gpu, production=True), phase='training')
        wrapped = dict(physical_gpu=gpu, training_curriculum=data, training_curriculum_sha256=cdigest(data),
                       native_terminal_receipt_sha256='b'*64)
        wrappers.append(wrapped)
        refs.append(publish(tmp_path/f'curriculum-previous-{gpu}.json', wrapped))
    campaign.state = dict(phase='COLLECT_TRAIN', round_id=2, config=site.config,
        training_curriculum_predecessor=dict(schema='roborender.dagger-continuous-collection-result.v2',
                                            round_id=1, native_terminal_receipts=refs))
    requests = campaign.operation_context()['training_curriculum']
    assert requests == [build_restore_request(row) for row in wrappers]
    del campaign.state['training_curriculum_predecessor']
    with pytest.raises(ValueError, match='prior curriculum-enabled round'):
        campaign.operation_context()


def test_learner_later_iteration_identity_is_not_ignored(provider):
    _, site, _, _, _, get = provider
    def wrong_job(route):
        result = get(route)
        if route.endswith('-0005'):
            result['job']['result']['iteration_id'] = 4
        return result
    with pytest.raises(ValueError, match='lineage'):
        module.learner_prefix(site, SimpleNamespace(get=wrong_job), site.round/'jobs', 5)
