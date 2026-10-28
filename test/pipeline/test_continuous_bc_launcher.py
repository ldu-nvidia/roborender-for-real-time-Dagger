"""CPU-only executable wiring and fail-closed guards, not native qualification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.bc_collection import run_continuous as launch
from pipeline.bc_collection.continuous_plan import build_plan, build_request
from pipeline.deployment.rtx_actor import continuous_bc_guard as guard
from pipeline.deployment.source_snapshot import canonical, file_record

GIB = 2**30


def test_detach_is_one_shot_and_session_independent(tmp_path):
    import os
    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=os.getpid())
    directory = tmp_path / 'launcher'
    result = launch.detach_once(['python', 'exact.py', '--execute'], directory, popen=popen)
    assert result['submitted_once'] and result['replay_allowed'] is False
    assert calls[0][1]['start_new_session'] is True
    assert json.loads((directory / 'owner.json').read_bytes())['pid'] == os.getpid()
    with pytest.raises(FileExistsError):
        launch.detach_once(['python', 'exact.py', '--execute'], directory, popen=popen)
    assert len(calls) == 1


def test_failed_detach_keeps_intent_and_forbids_retry(tmp_path):
    directory = tmp_path / 'launcher'
    def fail(*args, **kwargs):
        raise OSError('spawn response unavailable')
    with pytest.raises(OSError):
        launch.detach_once(['exact'], directory, popen=fail)
    assert (directory / 'intent.json').is_file()
    with pytest.raises(FileExistsError):
        launch.detach_once(['exact'], directory, popen=fail)


def request(b=32, windows=4, run_id='continuous-test'):
    motions = [dict(motion_id=f'motion-{i}', path=f'/motions/motion-{i}.npz',
                    sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(max(40, b))]
    return build_request(build_plan(motions, run_id=run_id, max_windows=windows,
                                   output_dir='/dagger-runtime/bc_isaac', num_envs=b))


def memory():
    return dict(memory_current=4 * GIB, memory_peak=5 * GIB, memory_max=43 * GIB,
        **{'memory_swap.current': 0}, events=dict(max=0, oom=0, oom_kill=0),
        stat=dict(anon=3 * GIB, file=GIB, shmem=0, kernel=GIB // 4))


def trainer(b):
    depth, rgb = SimpleNamespace(shape=(b, 480, 832, 1)), SimpleNamespace(shape=(b, 480, 832, 3))
    views = {key: dict(depth=depth, robot_rgb=rgb, rgb=None) for key in ('head_color', 'hand_left', 'hand_right')}
    return SimpleNamespace(env=SimpleNamespace(num_envs=b), vg_mode='collect', vg_rollout_action_source='teacher',
        vg_save_raw_depth=True, vg_save_raw_depth_native=True, vg_save_sim_rgb=False, vg_control_res=[240, 416],
        _depth_chunk=[depth] * 81, _robot_rgb_chunk=[rgb] * 81, _views_chunk=[views] * 81,
        _bc_ledger=SimpleNamespace(vector_steps=81, emitted_counts=[1] * b))


def test_guard_original_writer_once_at_b32_without_copies_or_reset():
    model, events, calls = trainer(32), [], []
    buffers = (model._depth_chunk, model._robot_rgb_chunk, model._views_chunk)
    assert guard.guarded_save(model, lambda: calls.append(1) or 'saved', snapshot=memory,
        host_available=lambda: 60 * GIB, num_envs=32, cap_bytes=43 * GIB,
        report=lambda phase, value: events.append(dict(phase=phase, **value))) == 'saved'
    assert calls == [1] and events[-1]['completed_original_save'] is True
    assert buffers == (model._depth_chunk, model._robot_rgb_chunk, model._views_chunk)


@pytest.mark.parametrize('saving', [False, True])
@pytest.mark.parametrize('mutation', ['swap', 'oom', 'missing', 'host', 'headroom'])
def test_capture_and_save_guard_fail_before_allocation(mutation, saving):
    snapshot, available = memory(), 60 * GIB
    if mutation == 'swap': snapshot['memory_swap.current'] = 1
    if mutation == 'oom': snapshot['events']['oom'] = 1
    if mutation == 'missing': snapshot.pop('memory_current')
    if mutation == 'host': available = 8 * GIB
    if mutation == 'headroom': snapshot['memory_current'] = 42 * GIB
    with pytest.raises(RuntimeError):
        guard.memory_gate(snapshot, available, num_envs=32, cap_bytes=43 * GIB, saving=saving)


def test_quota_exception_is_not_successful_continuous_completion():
    class _QuotaReached(Exception):
        pass
    events = []
    def write(): raise _QuotaReached()
    with pytest.raises(_QuotaReached):
        guard.guarded_save(trainer(4), write, snapshot=memory, host_available=lambda: 60 * GIB,
            num_envs=4, cap_bytes=43 * GIB, report=lambda phase, value: events.append(value))
    assert events[-1]['completed_original_save'] is False


def test_old_capacity_guard_is_not_broadened():
    from pipeline.deployment.rtx_actor.bc_capacity_guard import projected_save_gate, CapacityRejected
    with pytest.raises(CapacityRejected):
        projected_save_gate(num_envs=32, current_bytes=GIB, limit_bytes=43 * GIB,
                            host_available_bytes=60 * GIB, swap_bytes=0, events=dict(max=0, oom=0, oom_kill=0))


def test_full_inventory_terminal_receipt_has_explicit_bound_without_widening_legacy(tmp_path):
    from pipeline.deployment.rtx_actor.run_throughput_pair import owned_memory_sample
    receipt = tmp_path / 'cohort-manifest.json'
    receipt.write_text(json.dumps(dict(status='complete', motion_inventory='x' * (2 << 20))))
    process = SimpleNamespace(poll=lambda: 0)
    options = dict(completion_field='status', completion_value='complete')
    with pytest.raises(RuntimeError, match='bounded PASS receipt'):
        owned_memory_sample('exact-owner', process, receipt, **options)
    observed = owned_memory_sample('exact-owner', process, receipt, receipt_max_bytes=32 << 20, **options)
    assert observed['lifecycle'] == 'completed_attested'
    assert observed['completion_receipt_sha256'] == launch.digest(receipt.read_bytes())
    with pytest.raises(ValueError, match='at most32MiB'):
        owned_memory_sample('exact-owner', process, receipt, receipt_max_bytes=33 << 20, **options)


def test_default_launch_is_one_b32_exact_budget_not_independent_b16():
    value = request()
    assert launch.validate_launch(value, qualification=True, expected_num_envs=32,
                                  physical_gpu=0, memory_gib=43, timeout=3600) == value
    with pytest.raises(ValueError):
        launch.validate_launch(request(16), qualification=True, expected_num_envs=32,
                               physical_gpu=0, memory_gib=43, timeout=3600)
    with pytest.raises(ValueError):
        launch.validate_launch(request(windows=100), qualification=True, expected_num_envs=32,
                               physical_gpu=0, memory_gib=43, timeout=3600)


def test_command_is_genuine_guarded_original_collector():
    image = 'sha256:' + 'a' * 64
    calls = []
    def base(name, runtime, overlay, code, b, **kwargs):
        calls.append(dict(b=b, **kwargs))
        return ['docker', 'run', '--name', name, '--memory', '40g', '--memory-swap', '40g',
                '--env', 'PYTHONPATH=/isaac6-port', image, 'python', 'train.py',
                'trainer._target_=old', 'algo.config.num_steps_per_env=81']
    name, cmd = launch.build_command(request=request(), request_path='/requests/request.json',
        request_sha256='b' * 64, source=Path('/frozen'), runtime=Path('/runtime'), overlay=Path('/overlay'),
        physical_gpu=0, memory_gib=43, guard_sha256='c' * 64, base_builder=base, image_id=image)
    assert name.startswith('bc-cohort-leo-cont-') and calls[0]['b'] == 32
    assert calls[0]['mode'] == 'bc-collect' and calls[0]['physical_gpu'] == 0
    assert 'trainer._target_=continuous_bc_guard.GuardedContinuousBCTrainer' in cmd
    assert 'algo.trl.num_total_batches=4' in cmd and 'algo.config.num_learning_epochs=0' in cmd
    assert cmd[cmd.index('--memory') + 1] == cmd[cmd.index('--memory-swap') + 1] == '43g'
    assert cmd.count('org.systemd.property.MemorySwapMax=uint64 0') == 1
    assert 'ROBORENDER_CONTINUOUS_BC_GUARD_SHA256=' + 'c' * 64 in cmd
    assert any('dst=/bc-production-source,readonly' in arg for arg in cmd)
    assert 'type=bind,src=/frozen/pipeline/bc_collection,dst=/bc-stage/pipeline/bc_collection,readonly' in cmd
    assert any(arg.startswith('PYTHONPATH=/bc-stage:') for arg in cmd)
    assert cmd[cmd.index('--cidfile') + 1] == '/runtime/container.cid'
    assert any(arg.startswith('env.config.reset_from_dataset.motion_files=') for arg in cmd)
    assert not any('restart' in arg or 'H200' in arg for arg in cmd)


def fixture_capacity():
    from pipeline.bc_collection.collector import implementation_receipt
    r = request(4)
    sha = 'a' * 64
    models = {key: 'b' * 64 for key in ('teacher', 'homie_walk', 'homie_stand')}
    execution = dict(source_sha256='c' * 64, model_sha256=models)
    contract = launch.capacity_contract(execution, r, gpu_uuid='GPU-test', memory_gib=43)
    native = dict(schema='roborender.bc-native-continuous-manifest.v1', status='complete', request=r,
        request_sha256=sha, completion_reason='vector_window_budget', accepted_windows=4, vector_steps=324,
        initial_reset_all_calls=1, optimizer_steps=0, teacher_only=True, native_payload_writer_unchanged=True,
        collection_implementation=implementation_receipt(),
        init_at_random_ep_len=True, emitted_per_lane=[4] * 4, emitted_chunks=16, selected_chunks=16,
        overcollection_chunks=0, model_assets={key: dict(sha256=value) for key, value in models.items()},
        initial_state_attestation=dict(initial_reset_completion=dict(schema='roborender.initial-reset-completion.v1',
            generation=1, num_envs=4, boundary='after_reset_all_writes_and_observations', cleared_lanes=4,
            clear_calls=1, native_steps_added=0, ordinary_done_resets_modified=False)))
    plan = dict(capacity_contract=contract, request=r, request_sha256=sha, qualification_only=True,
                collector_execution=execution)
    host = dict(passed=True, exit_code=0, failure=None, started_monotonic=0, finished_monotonic=5,
                container_id='d' * 64)
    receipt = dict(schema=launch.SCHEMA, verdict='PASS', native_runtime_tested=True, qualification_only=True,
        request_sha256=sha, request_canonical_sha256=launch.digest(canonical(r)),
        collector_execution=execution, actual_vector_steps=324, actual_num_envs=4, initial_reset_all_calls=1,
        native_manifest_sha256=sha, host_result_sha256=sha, launch_sha256=sha, container_id='d' * 64)
    events, samples = [], []
    for window in range(1, 5):
        events.extend([dict(phase='presave', admitted=True, num_envs=4, memory_max_bytes=43 * GIB,
            memory=memory(), vector_steps=window * 81), dict(phase='postsave', completed_original_save=True,
            memory=memory(), vector_steps=window * 81, emitted_chunks_per_lane=[window] * 4,
            host_available_bytes=60 * GIB, memory_current_bytes=4 * GIB, memory_peak_bytes=5 * GIB,
            anon_bytes=3 * GIB, shmem_bytes=0, kernel_bytes=GIB // 4)])
        samples.append(dict(monotonic_seconds=window, host_available_kib=60 * 1024**2,
            simultaneous_native_image_owners=1,
            gpu=dict(uuid='GPU-test', total_bytes=96 * GIB, used_bytes=10 * GIB),
            container=dict(pid=123, memory_peak=5 * GIB, oom_killed=False,
                **{'memory_swap.current': 0}, events=dict(max=0, oom=0, oom_kill=0))))
    evidence = dict.fromkeys(('launch', 'host', 'native', 'save_events', 'memory_samples', 'execution'), sha)
    return dict(launch=plan, host=host, native=native, save_events=events,
                memory_samples=samples, execution_receipt=receipt, evidence=evidence)


def test_capacity_requires_actual_complete_full_buffers_and_is_not_production_admission():
    values = fixture_capacity()
    proof = launch.build_capacity(**values)
    assert proof['native_runtime_tested'] is True and proof['production_admitted'] is False
    assert proof['long_campaign_execution_proven'] is False
    contract = deepcopy(proof['contract'])
    contract['num_envs'] = 32
    with pytest.raises(ValueError): launch.validate_capacity(proof, contract)
    contract = deepcopy(proof['contract'])
    contract['collector_execution']['source_sha256'] = 'd' * 64
    with pytest.raises(ValueError): launch.validate_capacity(proof, contract)


@pytest.mark.parametrize('fault', ['early', 'reset', 'quota', 'missing_flush', 'bad_flush', 'no_live',
                                   'oom', 'swap', 'owner', 'gap', 'gpu', 'teacher', 'source', 'bad_hash',
                                   'container', 'presave_oom'])
def test_no_fake_capacity_from_incomplete_or_unsafe_evidence(fault):
    values = fixture_capacity()
    if fault == 'early': values['native']['vector_steps'] -= 81
    if fault == 'reset': values['native']['initial_reset_all_calls'] = 2
    if fault == 'quota': values['native']['selected_chunks'] -= 1
    if fault == 'missing_flush': values['save_events'].pop()
    if fault == 'bad_flush': values['save_events'][0]['admitted'] = False
    if fault == 'no_live': values['memory_samples'] = []
    if fault == 'oom': values['memory_samples'][0]['container']['events']['oom'] = 1
    if fault == 'swap': values['memory_samples'][0]['container']['memory_swap.current'] = 1
    if fault == 'container': values['execution_receipt']['container_id'] = 'e' * 64
    if fault == 'presave_oom': values['save_events'][0]['memory']['events']['oom'] = 1
    if fault == 'owner': values['memory_samples'][0]['simultaneous_native_image_owners'] = 2
    if fault == 'gap': values['host']['finished_monotonic'] = 100
    if fault == 'gpu': values['memory_samples'][0]['gpu']['used_bytes'] = 95 * GIB
    if fault == 'teacher': values['native']['model_assets']['teacher']['sha256'] = 'f' * 64
    if fault == 'source': values['execution_receipt']['collector_execution'] = {'wrong': True}
    if fault == 'bad_hash': values['execution_receipt']['native_manifest_sha256'] = 'f' * 64
    with pytest.raises((ValueError, RuntimeError)): launch.build_capacity(**values)


def test_capacity_recipe_is_independent_of_run_id_seed_and_budget_only():
    first = request()
    second = request(windows=100, run_id='new-run')
    second['process_seed'] = 73
    assert launch.capacity_contract({}, first, gpu_uuid='GPU-test', memory_gib=43) == \
        launch.capacity_contract({}, second, gpu_uuid='GPU-test', memory_gib=43)
    second['motion_inventory'].reverse()
    assert launch.capacity_contract({}, first, gpu_uuid='GPU-test', memory_gib=43) != \
        launch.capacity_contract({}, second, gpu_uuid='GPU-test', memory_gib=43)


def test_storage_budgets_every_emission_not_selected_quota():
    policy = dict(schema=launch.STORAGE_SCHEMA, root='/data1/workspaces/roborender/runs',
        reserve_bytes=64 * GIB, bytes_per_emission_budget=512 << 20,
        session_metadata_budget_bytes=GIB, poll_seconds=1)
    result = launch.storage_budget(policy, request())
    assert result['maximum_emissions'] == 128 and result['required_available_bytes'] == 129 * GIB
    policy['reserve_bytes'] = 8 * GIB
    with pytest.raises(ValueError): launch.storage_budget(policy, request())


def source_fixture(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    for name in launch.REQUIRED_SOURCE:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n' if path.suffix == '.json' else '# fixture\n')
    registry = Path(__file__).resolve().parents[2] / 'configs/storage_paths.json'
    (root / 'configs/storage_paths.json').write_bytes(registry.read_bytes())
    rows = [file_record(root, name) for name in sorted(launch.REQUIRED_SOURCE)]
    manifest = dict(schema='roborender.source-snapshot.v1', file_count=len(rows), files=rows,
                    source_sha256=launch.digest(canonical(rows)))
    return root, manifest


def test_source_snapshot_requires_new_launcher_guard_plan_and_ingest(tmp_path):
    root, manifest = source_fixture(tmp_path)
    assert launch.verify_source(root, manifest)[launch.SELF]
    (root / launch.SELF).write_text('# changed\n')
    with pytest.raises(ValueError): launch.verify_source(root, manifest)


def test_cpu_staging_does_not_materialize_or_launch_or_promote(tmp_path, monkeypatch, capsys):
    root, manifest = source_fixture(tmp_path)
    req_file, manifest_file = tmp_path / 'request.json', tmp_path / 'source.json'
    req_file.write_bytes(canonical(request()) + b'\n')
    manifest_file.write_bytes(canonical(manifest) + b'\n')
    def forbidden(*args, **kwargs): raise AssertionError('staging attempted an external/native action')
    monkeypatch.setattr(launch.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(launch.subprocess, 'check_output', forbidden)
    launch.main(['--request', str(req_file), '--request-sha256', launch.digest(req_file.read_bytes()),
        '--source-root', str(root), '--source-manifest', str(manifest_file),
        '--runtime', '/data1/workspaces/roborender/runs/bc-continuous/cpu-unit-never-execute',
        '--storage-root', '/data1/workspaces/roborender/runs', '--qualification'])
    report = json.loads(capsys.readouterr().out)
    assert report['execute'] is False and report['production_admitted'] is False
    assert report['num_envs'] == 32 and report['logical_scenes'] == 1
    assert report['required_vector_steps'] == 324 and report['capacity_supplied'] is False


def test_rederive_capacity_rejects_a_standalone_pass_json(tmp_path):
    path = tmp_path / 'capacity-receipt.json'
    path.write_bytes(canonical(launch.build_capacity(**fixture_capacity())) + b'\n')
    with pytest.raises((ValueError, OSError)):
        launch.read_capacity(path, launch.digest(path.read_bytes()))


def test_rederive_capacity_from_linked_raw_evidence_and_reject_later_mutation(tmp_path):
    values = fixture_capacity()
    (tmp_path / 'bc_isaac').mkdir()
    evidence = {}
    for key, name in (('launch', 'launch.json'), ('host', 'host-result.json'),
                      ('native', 'bc_isaac/cohort-manifest.json')):
        raw = canonical(values[key]) + b'\n'
        (tmp_path / name).write_bytes(raw)
        evidence[key] = launch.digest(raw)
    values['execution_receipt'].update(launch_sha256=evidence['launch'], host_result_sha256=evidence['host'],
                                       native_manifest_sha256=evidence['native'])
    raw = canonical(values['execution_receipt']) + b'\n'
    (tmp_path / 'continuous-execution.json').write_bytes(raw)
    evidence['execution'] = launch.digest(raw)
    for key, name in (('save_events', 'capacity-save-events.jsonl'), ('memory_samples', 'memory-samples.jsonl')):
        raw = b''.join(canonical(row) + b'\n' for row in values[key])
        (tmp_path / name).write_bytes(raw)
        evidence[key] = launch.digest(raw)
    values['evidence'] = evidence
    capacity = launch.build_capacity(**values)
    path = tmp_path / 'capacity-receipt.json'
    path.write_bytes(canonical(capacity) + b'\n')
    expected = launch.digest(path.read_bytes())
    assert launch.read_capacity(path, expected) == capacity
    values['host']['exit_code'] = 137
    (tmp_path / 'host-result.json').write_bytes(canonical(values['host']))
    with pytest.raises(ValueError): launch.read_capacity(path, expected)


def test_legacy_multicohort_session_rejects_continuous_protocol():
    from pipeline.deployment.rtx_actor.isaac6_bc_session import validate_session
    raw = canonical(request())
    session = dict(schema='roborender.bc-native-session.v1', session_id='bc-session-leo-test',
        requests=[dict(path='/bc-session-requests/one.json', sha256=launch.digest(raw))])
    with pytest.raises(ValueError, match='single-scene guarded'):
        validate_session(session, read_bytes=lambda _: raw)


def test_cleanup_without_acceptance_never_stops_by_name(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(launch.subprocess, 'run', lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError, match='acceptance is unknown'):
        launch.cleanup_exact('bc-cohort-leo-test', tmp_path / 'absent.cid', 'sha256:' + 'a' * 64)
    assert calls == []


def test_cleanup_checks_exact_docker_identity_before_scoped_stop(tmp_path, monkeypatch):
    from pipeline.deployment.rtx_actor.qualify_graphics import LABEL
    cid, image, name = 'b' * 64, 'sha256:' + 'a' * 64, 'bc-cohort-leo-test'
    path = tmp_path / 'container.cid'
    path.write_text(cid)
    info = dict(Id=cid, Image=image, Name='/' + name, Config=dict(Labels={LABEL: name}), State=dict(Running=True))
    calls = []
    def output(cmd, **kwargs):
        return cid + '\n' if cmd[1] == 'ps' else json.dumps([info])
    monkeypatch.setattr(launch.subprocess, 'check_output', output)
    monkeypatch.setattr(launch.subprocess, 'run', lambda cmd, **kwargs: calls.append(cmd))
    assert launch.cleanup_exact(name, path, image) == cid
    assert calls == [['docker', 'stop', '--time', '10', cid]]
    calls.clear()
    info['Config']['Labels'][LABEL] = 'different-owner'
    with pytest.raises(RuntimeError, match='owner/image'):
        launch.cleanup_exact(name, path, image)
    assert calls == []
