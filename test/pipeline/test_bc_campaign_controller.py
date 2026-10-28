"""No GPUs: synthetic child lifecycle, source and receipt boundary tests."""
from test.deployment.source_paths import PIPELINE_ROOT
from copy import deepcopy
import json
import importlib.util
import itertools
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from pipeline.legacy.bc_collection import campaign_controller as c
from pipeline.legacy.bc_collection.stage_campaign import build_campaign

REAL_PRODUCER = PIPELINE_ROOT / 'deployment/rtx_actor/bc_execution_evidence.py'


def fixture(tmp_path, count=2):
    source = tmp_path / 'source'
    actor = source / c.ACTOR
    actor.mkdir(parents=True)
    records = []
    for name in c.MODULES:
        path = actor / name
        path.write_bytes(REAL_PRODUCER.read_bytes() if name == 'bc_execution_evidence.py' else b'# synthetic test source\n')
        records.append(dict(path=f'{c.ACTOR}/{name}', sha256=c.digest(path.read_bytes()), bytes=path.stat().st_size, executable=False))
    manifest = dict(schema='roborender.source-snapshot.v1', files=records, file_count=len(records), source_sha256=c.digest(c.canonical(records)))
    c.publish(tmp_path / 'source-manifest.json', manifest)
    receipt = dict(schema='roborender.native-bc-capacity.v2', verdict='PASS', native_runtime_tested=True,
        num_envs=1, concurrent_collectors=1, cgroup_limit_bytes=43 * 2**30)
    c.publish(tmp_path / 'capacity.json', receipt)
    sessions = []
    for index in range(count):
        folder = tmp_path / f'spec-{index}'
        request = dict(schema='roborender.bc-native-collection-request.v2', cohort_id=f'train-{index}',
            rank=0, run_id='bc-leo-fixture', sampling_plan_sha256='a' * 64, motions=[{}],
            output_dir='/dagger-runtime/bc_isaac', max_windows=1)
        c.publish(folder / 'requests/cohort.json', request)
        session = dict(schema='roborender.bc-native-session.v1', session_id=f'bc-session-leo-fixture-{index}',
            requests=[dict(path='/bc-session-requests/cohort.json', sha256=c.binding(folder / 'requests/cohort.json')['raw_sha256'])])
        c.publish(folder / 'session.json', session)
        sessions.append(dict(session_id=session['session_id'], session=c.binding(folder / 'session.json'),
            requests_dir=str(folder / 'requests'), runtime=str(tmp_path / 'sessions' / session['session_id']),
            capacity_receipt=c.binding(tmp_path / 'capacity.json'), timeout_seconds=60))
    doc = dict(schema=c.SCHEMA, campaign_id='bc-campaign-test', source_root=str(source),
        source_manifest=c.binding(tmp_path / 'source-manifest.json'), physical_gpu=0, memory_gib=43, sessions=sessions,
        storage_guard=dict(schema=c.STORAGE_SCHEMA, root=str(tmp_path), reserve_bytes=64*c.GIB,
            bytes_per_emission_budget=512<<20, session_metadata_budget_bytes=c.GIB, poll_seconds=1))
    c.publish(tmp_path / 'campaign.json', doc)
    return doc, c.binding(tmp_path / 'campaign.json'), manifest


def validator(tmp_path):
    return lambda doc: c.validate_campaign(doc, source_prefix=str(tmp_path), runtime_prefix=str(tmp_path / 'sessions'))


def options(tmp_path):
    return dict(_validate=validator(tmp_path), _state_prefix=str(tmp_path / 'states'), _hostname=lambda: 'leo-rtx-6k',
        _storage_snapshot=lambda policy, runtime, state: dict(root=policy['root'], device=1, inode=2, available_bytes=1<<50))


def synthetic_completion(doc, row, manifest):
    return dict(session_id=row['session_id'], runtime=row['runtime'], verified='synthetic-only')


class FakeChild:
    calls = []
    code = 0
    timeout = False
    on_start = None
    def __init__(self, command, **kwargs):
        self.command = command; self.pid = 12345; self.returncode = None
        self.calls.append(command)
        if self.on_start: self.on_start(command)
    def wait(self, timeout):
        if self.timeout: raise subprocess.TimeoutExpired(self.command, timeout)
        self.returncode = self.code
        return self.code
    def poll(self): return self.returncode
    def terminate(self): self.returncode = -15


@pytest.fixture(autouse=True)
def reset_child():
    FakeChild.calls = []; FakeChild.code = 0; FakeChild.timeout = False; FakeChild.on_start = None


def test_plan_only_has_no_state_or_child(tmp_path):
    doc, ref, _ = fixture(tmp_path)
    result = c.run(doc, ref, tmp_path / 'states/run', max_sessions=1, wall_budget=300, **options(tmp_path))
    assert result['execute'] is False and len(result['commands']) == 2
    assert not (tmp_path / 'states').exists() and FakeChild.calls == []
    assert result['commands'][0][-1] == '--execute'


def test_single_cohort_capacity_cannot_admit_two_scene_lifetime(tmp_path):
    doc, _, _ = fixture(tmp_path)
    path = Path(doc['sessions'][0]['session']['path'])
    session = json.loads(path.read_bytes())
    session['requests'].append(deepcopy(session['requests'][0]))
    c.publish(path, session, replace=True)
    doc['sessions'][0]['session'] = c.binding(path)
    with pytest.raises(ValueError, match='session identity'):
        validator(tmp_path)(doc)


def test_intent_exists_before_child_and_completed_only_resume(tmp_path):
    doc, ref, _ = fixture(tmp_path)
    state_dir = tmp_path / 'states/run'
    def before(command):
        state = json.loads(c.read_raw(state_dir / 'state.json'))
        assert state['phase'] == 'ACTIVE'
        assert c.bound(state['active_intent'])['command'] == command
    FakeChild.on_start = staticmethod(before)
    first = c.run(doc, ref, state_dir, max_sessions=1, wall_budget=300, execute=True,
                  _popen=FakeChild, _verify=synthetic_completion, **options(tmp_path))
    assert first['phase'] == 'READY' and first['completed_sessions'] == 1
    second = c.run(doc, ref, state_dir, max_sessions=1, wall_budget=300, execute=True,
                   _popen=FakeChild, _verify=synthetic_completion, **options(tmp_path))
    assert second['phase'] == 'COMPLETE' and len(FakeChild.calls) == 2
    third = c.run(doc, ref, state_dir, max_sessions=1, wall_budget=300, execute=True,
                  _popen=FakeChild, _verify=synthetic_completion, **options(tmp_path))
    assert third['launched_this_invocation'] == 0 and len(FakeChild.calls) == 2


@pytest.mark.parametrize('fault', ['nonzero', 'receipt', 'spawn', 'timeout'])
def test_failure_blocks_without_automatic_retry(tmp_path, fault):
    doc, ref, _ = fixture(tmp_path)
    state_dir = tmp_path / 'states/run'
    verifier = synthetic_completion
    if fault == 'nonzero': FakeChild.code = 1
    elif fault == 'receipt':
        def verifier(*args): raise ValueError('receipt failed')
    elif fault == 'spawn':
        def fail(*args): raise OSError('spawn failed')
        FakeChild.on_start = staticmethod(fail)
    else: FakeChild.timeout = True
    clock = itertools.count().__next__
    with pytest.raises((ValueError, RuntimeError, OSError, subprocess.TimeoutExpired)):
        c.run(doc, ref, state_dir, max_sessions=2, wall_budget=300, execute=True,
              _popen=FakeChild, _verify=verifier, _clock=clock, **options(tmp_path))
    state = json.loads(c.read_raw(state_dir / 'state.json'))
    assert state['phase'] == 'BLOCKED' and state['completed'] == []
    count = len(FakeChild.calls)
    with pytest.raises(ValueError, match='cannot be replayed'):
        c.run(doc, ref, state_dir, max_sessions=2, wall_budget=300, execute=True,
              _popen=FakeChild, _verify=synthetic_completion, **options(tmp_path))
    assert len(FakeChild.calls) == count


@pytest.mark.parametrize('phase', ['ACTIVE', 'UNKNOWN', 'BLOCKED'])
def test_file_presence_never_recovers_uncertain_phase(tmp_path, phase):
    doc, ref, _ = fixture(tmp_path)
    state_dir = tmp_path / 'states/run'
    c.publish(state_dir / 'state.json', dict(schema=c.STATE_SCHEMA, campaign=ref,
        controller_sha256=c.digest(c.read_raw(c.__file__)), phase=phase, next_index=0, completed=[]))
    Path(doc['sessions'][0]['runtime']).mkdir(parents=True)
    c.publish(Path(doc['sessions'][0]['runtime']) / 'session-execution.json', dict(verdict='PASS'))
    with pytest.raises(ValueError, match='cannot be replayed'):
        c.run(doc, ref, state_dir, max_sessions=2, wall_budget=300, execute=True, _popen=FakeChild, **options(tmp_path))
    assert not FakeChild.calls


def test_insufficient_wall_budget_stops_at_safe_boundary(tmp_path):
    doc, ref, _ = fixture(tmp_path)
    result = c.run(doc, ref, tmp_path / 'states/run', max_sessions=1, wall_budget=119, execute=True,
                   _popen=FakeChild, **options(tmp_path))
    assert result['phase'] == 'READY' and result['launched_this_invocation'] == 0


def test_changed_completed_exit_is_not_accepted(tmp_path):
    doc, ref, _ = fixture(tmp_path)
    folder = tmp_path / 'states/run'
    c.run(doc, ref, folder, max_sessions=1, wall_budget=300, execute=True,
          _popen=FakeChild, _verify=synthetic_completion, **options(tmp_path))
    (folder / 'session-0000.exit.json').write_text('{}')
    with pytest.raises(ValueError, match='immutable metadata changed'):
        c.run(doc, ref, folder, max_sessions=1, wall_budget=300, execute=True,
              _popen=FakeChild, _verify=synthetic_completion, **options(tmp_path))
    assert len(FakeChild.calls) == 1


@pytest.mark.parametrize('fault', ['gpu', 'cap', 'source', 'repeated', 'tail', 'request', 'plan', 'oversize'])
def test_campaign_launch_identity_validation(tmp_path, fault):
    doc, ref, _ = fixture(tmp_path)
    if fault == 'gpu': doc['physical_gpu'] = 1
    elif fault == 'cap': doc['memory_gib'] = 22
    elif fault == 'source': (Path(doc['source_root']) / c.ACTOR / 'bc_memory_guard.py').write_text('#changed')
    elif fault == 'repeated': doc['sessions'][1] = deepcopy(doc['sessions'][0])
    elif fault == 'tail':
        path = Path(doc['sessions'][0]['capacity_receipt']['path'])
        data = json.loads(path.read_bytes()); data['num_envs'] = 6
        path.write_bytes(c.canonical(data))
        for row in doc['sessions']: row['capacity_receipt'] = c.binding(path)
    elif fault in ('request', 'plan'):
        path = Path(doc['sessions'][1]['requests_dir']) / 'cohort.json'
        data = json.loads(path.read_bytes()); data['sampling_plan_sha256'] = 'b' * 64
        path.write_bytes(c.canonical(data))
        if fault == 'plan':
            spath = Path(doc['sessions'][1]['session']['path']); session = json.loads(spath.read_bytes())
            session['requests'][0]['sha256'] = c.binding(path)['raw_sha256']; spath.write_bytes(c.canonical(session))
            doc['sessions'][1]['session'] = c.binding(spath)
    else:
        doc['sessions'][0]['timeout_seconds'] = 50000
    with pytest.raises(ValueError): validator(tmp_path)(doc)


def test_stager_requires_explicit_small_gpu0_schedule(tmp_path):
    doc, ref, _ = fixture(tmp_path)
    rows = [dict(session_id=r['session_id'], session_path=r['session']['path'], session_sha256=r['session']['raw_sha256'],
        requests_dir=r['requests_dir'], physical_gpu=0, num_envs=1) for r in doc['sessions']]
    schedule = dict(schema='roborender.bc-native-schedule.v1', sessions=rows)
    result = build_campaign(schedule, campaign_id=doc['campaign_id'], source_root=doc['source_root'],
        source_manifest=doc['source_manifest']['path'], capacity_receipts=[doc['sessions'][0]['capacity_receipt']['path']],
        timeout_seconds=60, runtime_prefix=tmp_path / 'sessions')
    assert result == doc
    rows[0]['physical_gpu'] = 1
    with pytest.raises(ValueError):
        build_campaign(schedule, campaign_id=doc['campaign_id'], source_root=doc['source_root'],
            source_manifest=doc['source_manifest']['path'], capacity_receipts=[doc['sessions'][0]['capacity_receipt']['path']])


def publish_completed_fixture(doc, row, manifest):
    """Use the real frozen producer; all child execution is still synthetic."""
    runtime = Path(row['runtime'])
    session = c.bound(row['session'])
    request = json.loads(c.read_raw(Path(row['requests_dir']) / 'cohort.json'))
    cohort = dict(schema='roborender.bc-native-cohort.v1', status='complete', teacher_only=True,
        optimizer_steps=0, native_payload_writer_unchanged=True, request=request,
        request_sha256=session['requests'][0]['sha256'], vector_steps=81)
    c.publish(runtime / 'bc_isaac/cohort-manifest.json', cohort)
    native = dict(status='complete', failure=None, native_initializations=1, scene_initializations=1,
        session_id=session['session_id'], cohorts=[dict(cohort_id=request['cohort_id'],
        request_sha256=cohort['request_sha256'], output_dir='/dagger-runtime/bc_isaac', vector_steps=81)])
    host = dict(passed=True, exit_code=0, failure=None)
    files = {r['path']: r['sha256'] for r in manifest['files']}
    launch = dict(qualification_only=False, session_sha256=row['session']['raw_sha256'],
        source_sha256=manifest['source_sha256'], source_manifest_raw_sha256=doc['source_manifest']['raw_sha256'],
        capacity_binding=row['capacity_receipt'], guard_sha256=files[f'{c.ACTOR}/bc_capacity_guard.py'],
        host_guard_sha256=files[f'{c.ACTOR}/bc_memory_guard.py'], launcher_sha256=files[f'{c.ACTOR}/run_guarded_bc_capacity.py'],
        evidence_producer_sha256=files[f'{c.ACTOR}/bc_capacity_evidence.py'],
        execution_topology=dict(host_id='leo-rtx-6k', concurrent_collectors=1,
            cgroup_limit_bytes_per_collector=43 * 2**30, minimum_host_reserve_bytes=8 * 2**30),
        collector_execution=dict(runtime_contract_sha256='d' * 64),
        session_canonical_sha256=c.digest(c.canonical(session)),
        request_identities={cohort['request_sha256']: c.digest(c.canonical(request))}, port_manifest_raw_sha256='e' * 64)
    for name, value in [('launch.json', launch), ('host-result.json', host), ('bc-session-result.json', native)]:
        c.publish(runtime / name, value)
    spec = importlib.util.spec_from_file_location('fixture_execution_producer', REAL_PRODUCER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    sr, cr = module.build_session_receipts(launch=launch, launch_raw_sha256=c.binding(runtime / 'launch.json')['raw_sha256'],
        session=session, result=native, result_raw_sha256=c.binding(runtime / 'bc-session-result.json')['raw_sha256'],
        host_result=host, host_result_raw_sha256=c.binding(runtime / 'host-result.json')['raw_sha256'],
        cohort_envelopes=[dict(document=cohort, raw_sha256=c.binding(runtime / 'bc_isaac/cohort-manifest.json')['raw_sha256'])])
    c.publish(runtime / 'session-execution.json', sr)
    c.publish(runtime / 'bc_isaac/execution-receipt.json', cr[0])


def test_real_producer_reconstructed_before_completion_and_on_resume(tmp_path):
    doc, ref, manifest = fixture(tmp_path, count=1)
    row = doc['sessions'][0]
    FakeChild.on_start = staticmethod(lambda command: publish_completed_fixture(doc, row, manifest))
    result = c.run(doc, ref, tmp_path / 'states/run', max_sessions=1, wall_budget=300, execute=True,
        _popen=FakeChild, **options(tmp_path))
    assert result['phase'] == 'COMPLETE'
    again = c.run(doc, ref, tmp_path / 'states/run', max_sessions=1, wall_budget=300, execute=True,
        _popen=FakeChild, **options(tmp_path))
    assert again['launched_this_invocation'] == 0


@pytest.mark.parametrize('name', ['launch.json', 'host-result.json', 'bc-session-result.json',
    'session-execution.json', 'bc_isaac/cohort-manifest.json', 'bc_isaac/execution-receipt.json'])
def test_completed_chain_drift_fails_before_next_session(tmp_path, name):
    doc, ref, manifest = fixture(tmp_path)
    FakeChild.on_start = staticmethod(lambda command: publish_completed_fixture(doc, doc['sessions'][0], manifest))
    c.run(doc, ref, tmp_path / 'states/run', max_sessions=1, wall_budget=300, execute=True,
          _popen=FakeChild, **options(tmp_path))
    path = Path(doc['sessions'][0]['runtime']) / name
    value = json.loads(path.read_bytes()); value['changed'] = True
    path.write_bytes(c.canonical(value))
    with pytest.raises(ValueError):
        c.run(doc, ref, tmp_path / 'states/run', max_sessions=1, wall_budget=300, execute=True,
              _popen=FakeChild, **options(tmp_path))
    assert len(FakeChild.calls) == 1


def test_existing_unrecorded_runtime_never_relaunched(tmp_path):
    doc, ref, _ = fixture(tmp_path)
    Path(doc['sessions'][0]['runtime']).mkdir(parents=True)
    with pytest.raises(ValueError, match='never replayed'):
        c.run(doc, ref, tmp_path / 'states/run', max_sessions=1, wall_budget=300, execute=True,
              _popen=FakeChild, **options(tmp_path))
    assert not FakeChild.calls


def test_first_session_timeout_override_preserves_all_native_requests_and_later_bounds(tmp_path):
    doc, _, _ = fixture(tmp_path)
    rows = [dict(physical_gpu=0, num_envs=1, session_id=r['session_id'],
                 session_path=r['session']['path'], session_sha256=r['session']['raw_sha256'],
                 requests_dir=r['requests_dir']) for r in doc['sessions']]
    schedule = dict(schema='roborender.bc-native-schedule.v1', sessions=rows)
    kwargs = dict(campaign_id=doc['campaign_id'], source_root=doc['source_root'],
        source_manifest=doc['source_manifest']['path'], capacity_receipts=[tmp_path/'capacity.json'],
        runtime_prefix=tmp_path/'sessions')
    default = build_campaign(schedule, **kwargs)
    bounded = build_campaign(schedule, timeout_seconds=10800, first_session_timeout_seconds=3600, **kwargs)
    assert [r['timeout_seconds'] for r in default['sessions']] == [7200, 7200]
    assert [r['timeout_seconds'] for r in bounded['sessions']] == [3600, 10800]
    for previous, current in zip(default['sessions'], bounded['sessions']):
        assert {k:v for k,v in previous.items() if k!='timeout_seconds'} == {
            k:v for k,v in current.items() if k!='timeout_seconds'}
    validator(tmp_path)(bounded)


@pytest.mark.parametrize('value',[0,-1,59,43201,True,1.5,'3600'])
def test_first_session_timeout_override_rejects_invalid_bounds_before_reading_files(value):
    with pytest.raises(ValueError,match='integer seconds'):
        build_campaign({}, campaign_id='unused', source_root='unused', source_manifest='unused',
                       capacity_receipts=[], first_session_timeout_seconds=value)


def test_storage_budget_counts_all_declared_emissions_not_selected_outputs(tmp_path):
    doc, _, _ = fixture(tmp_path)
    policy = doc['storage_guard']
    budget = c.session_storage_budget(policy, [dict(motions=[{}]*6, max_windows=8, chunks_per_env=[1]*6),
                                                dict(motions=[{}]*6, max_windows=4, chunks_per_env=[1]*6)])
    assert budget['maximum_emissions'] == 72
    assert budget['budget_bytes'] == 72*(512<<20)+c.GIB
    assert budget['required_available_bytes'] == budget['budget_bytes']+64*c.GIB
    assert budget['budget_is_guaranteed_payload_bound'] is False


@pytest.mark.parametrize('field,value', [('reserve_bytes', 7*c.GIB), ('reserve_bytes', True),
    ('bytes_per_emission_budget', 511<<20), ('session_metadata_budget_bytes', 0),
    ('poll_seconds', 2), ('poll_seconds', True), ('root', '/'), ('root', '/tmp/../escape')])
def test_storage_policy_invalid_before_launch(tmp_path, field, value):
    doc, _, _ = fixture(tmp_path)
    doc['storage_guard'][field] = value
    with pytest.raises(ValueError, match='storage'):
        validator(tmp_path)(doc)
    assert FakeChild.calls == []


@pytest.mark.parametrize('windows', [True, 0, -1, 1.5, '4', 1000001])
def test_storage_budget_requires_actual_integer_window_ceiling(tmp_path, windows):
    doc, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='window ceiling'):
        c.session_storage_budget(doc['storage_guard'], [dict(motions=[{}], max_windows=windows)])


def test_storage_reserve_covers_one_inflight_full_writer_batch(tmp_path):
    doc, _, _ = fixture(tmp_path)
    policy = dict(doc['storage_guard'], reserve_bytes=8*c.GIB, bytes_per_emission_budget=2*c.GIB)
    with pytest.raises(ValueError, match='in-flight'):
        c.session_storage_budget(policy, [dict(motions=[{}]*6, max_windows=1)])


def test_insufficient_disk_never_spawns_or_publishes_active_intent(tmp_path):
    doc, ref, _ = fixture(tmp_path)
    kwargs = options(tmp_path)
    kwargs['_storage_snapshot'] = lambda policy, *args: dict(root=policy['root'], device=1, inode=2,
        available_bytes=policy['reserve_bytes']+c.GIB)
    folder = tmp_path/'states/run'
    with pytest.raises(c.StorageRejected, match='maximum session budget'):
        c.run(doc, ref, folder, max_sessions=2, wall_budget=300, execute=True,
              _popen=FakeChild, **kwargs)
    state = json.loads((folder/'state.json').read_bytes())
    assert state['phase'] == 'READY' and state['next_index'] == 0
    assert not list(folder.glob('*.intent.json')) and not FakeChild.calls


class OwnedWaitingChild(FakeChild):
    terminated = []
    def wait(self, timeout):
        if self.returncode is not None:
            return self.returncode
        raise subprocess.TimeoutExpired(self.command, timeout)
    def terminate(self):
        self.terminated.append(self.pid)
        self.returncode = -15


@pytest.mark.parametrize('fault', ['low', 'identity', 'missing', 'io'])
def test_live_storage_failure_stops_only_owned_child_and_blocks_replay(tmp_path, fault):
    doc, ref, _ = fixture(tmp_path)
    count = itertools.count()
    OwnedWaitingChild.terminated = []
    def observe(policy, *args):
        value = dict(root=policy['root'], device=1, inode=2, available_bytes=1<<50)
        if next(count) >= 2:
            if fault == 'low': value['available_bytes'] = policy['reserve_bytes']-1
            elif fault == 'identity': value['inode'] = 3
            elif fault == 'missing': value.pop('available_bytes')
            else: raise OSError('filesystem unavailable')
        return value
    kwargs = options(tmp_path); kwargs['_storage_snapshot'] = observe
    folder = tmp_path/'states/run'
    with pytest.raises((c.StorageRejected, OSError)):
        c.run(doc, ref, folder, max_sessions=2, wall_budget=300, execute=True,
              _popen=OwnedWaitingChild, _verify=synthetic_completion, **kwargs)
    state = json.loads((folder/'state.json').read_bytes())
    assert state['phase'] == 'BLOCKED' and state['next_index'] == 0
    assert OwnedWaitingChild.terminated == [12345] and len(FakeChild.calls) == 1
    failure = json.loads((folder/'session-0000.storage-failure.json').read_bytes())
    assert failure['passed'] is False and failure['owned_child_still_running'] is False
    with pytest.raises(ValueError, match='cannot be replayed'):
        c.run(doc, ref, folder, max_sessions=2, wall_budget=300, execute=True,
              _popen=OwnedWaitingChild, _verify=synthetic_completion, **kwargs)
    assert len(FakeChild.calls) == 1


def test_post_exit_disk_failure_does_not_signal_completed_child(tmp_path):
    doc, ref, _ = fixture(tmp_path)
    count = itertools.count()
    def observe(policy, *args):
        return dict(root=policy['root'], device=1, inode=2,
                    available_bytes=1<<50 if next(count)<2 else 0)
    class CompletedChild(FakeChild):
        def terminate(self): raise AssertionError('already exited child must not be signalled')
    kwargs = options(tmp_path); kwargs['_storage_snapshot'] = observe
    folder = tmp_path/'states/run'
    with pytest.raises(c.StorageRejected):
        c.run(doc, ref, folder, max_sessions=2, wall_budget=300, execute=True,
              _popen=CompletedChild, _verify=synthetic_completion, **kwargs)
    assert json.loads((folder/'state.json').read_bytes())['phase'] == 'BLOCKED'
    assert len(FakeChild.calls) == 1


def test_storage_observation_is_pinned_on_completed_resume(tmp_path):
    doc, ref, _ = fixture(tmp_path)
    folder = tmp_path/'states/run'
    c.run(doc, ref, folder, max_sessions=1, wall_budget=300, execute=True,
          _popen=FakeChild, _verify=synthetic_completion, **options(tmp_path))
    (folder/'session-0000.storage.json').write_text('{}')
    with pytest.raises(ValueError, match='immutable metadata changed'):
        c.run(doc, ref, folder, max_sessions=1, wall_budget=300, execute=True,
              _popen=FakeChild, _verify=synthetic_completion, **options(tmp_path))
    assert len(FakeChild.calls) == 1


def test_storage_filesystem_cannot_change_at_completed_session_boundary(tmp_path):
    doc, ref, _ = fixture(tmp_path)
    folder = tmp_path/'states/run'
    c.run(doc, ref, folder, max_sessions=1, wall_budget=300, execute=True,
          _popen=FakeChild, _verify=synthetic_completion, **options(tmp_path))
    kwargs = options(tmp_path)
    kwargs['_storage_snapshot'] = lambda policy, *args: dict(root=policy['root'], device=2,
        inode=3, available_bytes=1<<50)
    with pytest.raises(c.StorageRejected, match='between campaign sessions'):
        c.run(doc, ref, folder, max_sessions=1, wall_budget=300, execute=True,
              _popen=FakeChild, _verify=synthetic_completion, **kwargs)
    state = json.loads((folder/'state.json').read_bytes())
    assert state['phase'] == 'READY' and state['next_index'] == 1
    assert len(FakeChild.calls) == 1 and not (folder/'session-0001.intent.json').exists()


def test_statvfs_uses_unprivileged_available_blocks_and_checks_owned_paths(tmp_path, monkeypatch):
    doc, _, _ = fixture(tmp_path)
    monkeypatch.setattr(c.os, 'statvfs', lambda path: SimpleNamespace(f_bavail=5, f_bfree=500, f_frsize=4096))
    result = c.storage_snapshot(doc['storage_guard'], tmp_path/'sessions/new', tmp_path/'states/new')
    assert result['available_bytes'] == 5*4096
    assert result['device'] == tmp_path.stat().st_dev and result['inode'] == tmp_path.stat().st_ino
    with pytest.raises(c.StorageRejected, match='escaped'):
        c.storage_snapshot(doc['storage_guard'], tmp_path.parent/'outside', tmp_path/'states/new')
    (tmp_path/'alias').symlink_to(tmp_path/'source', target_is_directory=True)
    with pytest.raises(c.StorageRejected, match='escaped'):
        c.storage_snapshot(doc['storage_guard'], tmp_path/'alias/new', tmp_path/'states/new')
    with pytest.raises(c.StorageRejected, match='non-aliased'):
        c.storage_snapshot(dict(doc['storage_guard'], root=str(tmp_path/'alias')),
                           tmp_path/'alias/new', tmp_path/'alias/state')


def test_other_filesystem_under_guarded_root_is_rejected(tmp_path, monkeypatch):
    doc, _, _ = fixture(tmp_path)
    mounted = tmp_path/'mounted'; mounted.mkdir()
    original = Path.stat
    def changed(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == mounted:
            fields = list(value); fields[2] = value.st_dev+1
            return os.stat_result(fields)
        return value
    monkeypatch.setattr(Path, 'stat', changed)
    with pytest.raises(c.StorageRejected, match='unguarded filesystem'):
        c.storage_snapshot(doc['storage_guard'], mounted/'new', tmp_path/'states/new')


def test_legacy_campaign_remains_readable_verifiable_but_cannot_execute(tmp_path):
    doc, _, manifest = fixture(tmp_path)
    doc['schema'] = c.LEGACY_SCHEMA; doc.pop('storage_guard')
    path = tmp_path/'legacy.json'; c.publish(path, doc); ref = c.binding(path)
    result = c.run(doc, ref, tmp_path/'states/run', max_sessions=1, wall_budget=300, **options(tmp_path))
    assert result['execution_supported'] is False and result['storage_budgets'] is None
    row = doc['sessions'][0]
    publish_completed_fixture(doc, row, manifest)
    assert c.verify_completed(doc, row, manifest)['session_id'] == row['session_id']
    with pytest.raises(ValueError, match='historical campaign v1 is read-only'):
        c.run(doc, ref, tmp_path/'states/run', max_sessions=1, wall_budget=300, execute=True,
              _popen=FakeChild, **options(tmp_path))
    assert not FakeChild.calls and not (tmp_path/'states').exists()


def test_stager_emits_explicit_v2_storage_policy_without_disk_query(tmp_path, monkeypatch):
    doc, _, _ = fixture(tmp_path)
    schedule = dict(schema='roborender.bc-native-schedule.v1', sessions=[dict(physical_gpu=0, num_envs=1,
        session_id=r['session_id'], session_path=r['session']['path'], session_sha256=r['session']['raw_sha256'],
        requests_dir=r['requests_dir']) for r in doc['sessions']])
    monkeypatch.setattr(c.os, 'statvfs', lambda path: (_ for _ in ()).throw(AssertionError('plan only')))
    result = build_campaign(schedule, campaign_id=doc['campaign_id'], source_root=doc['source_root'],
        source_manifest=doc['source_manifest']['path'], capacity_receipts=[tmp_path/'capacity.json'],
        runtime_prefix=tmp_path/'sessions', storage_root=tmp_path, storage_reserve_bytes=80*c.GIB,
        storage_bytes_per_emission_budget=c.GIB, storage_session_metadata_budget_bytes=2*c.GIB)
    assert result['schema'] == c.SCHEMA and result['storage_guard']['reserve_bytes'] == 80*c.GIB
    assert result['storage_guard']['bytes_per_emission_budget'] == c.GIB
    validator(tmp_path)(result)
