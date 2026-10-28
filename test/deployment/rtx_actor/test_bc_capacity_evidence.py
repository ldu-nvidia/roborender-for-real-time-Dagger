"""Synthetic metadata-only producer tests, never real runtime certification."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

import bc_capacity_evidence as evidence
from bc_memory_guard import exclusive_native_owner
from pipeline.deployment.rtx_actor.bc_execution_evidence import build_session_receipts
from pipeline.deployment.rtx_actor.isaac6_bc_production import NATIVE_COLLECTOR_SHA256
from pipeline.bc_training.production_protocol import validate_bounded_capacity

GIB = 2**30


def fixture(b=6, counts=None):
    counts = counts or [[1] * b, [2] * b]
    steps = 81 * len(counts)
    request = dict(schema='roborender.bc-native-collection-request.v2', cohort_id='test-cohort',
        motions=[{}] * b, chunks_per_env=[1] * b, selected_chunk_ordinals_per_env=[[1]] * b,
        exposure_span_chunks_per_env=[2] * b, max_windows=4, split='validation')
    rsha = evidence.digest(evidence.canonical(request) + b'\n')
    session = dict(session_id='bc-session-leo-fixture', requests=[dict(sha256=rsha)])
    execution = dict(adapter_source_sha256='a' * 64, source_release_sha256='b' * 64,
        runtime_contract_sha256='c' * 64, container_image_id='sha256:' + 'd' * 64,
        native_runtime_version='isaac6-test', lifecycle_recipe='one_kit_fresh_seeded_scene_per_declared_cohort_v1',
        memory_buffer_recipe='original_raven_capture_lists_and_writer_v1')
    launch = dict(qualification_only=True, production_admitted=False, collector_execution=execution,
        execution_topology=dict(host_id='leo-rtx-6k', concurrent_collectors=1,
            cgroup_limit_bytes_per_collector=43 * GIB, minimum_host_reserve_bytes=8 * GIB),
        source_manifest_raw_sha256='1' * 64, port_manifest_raw_sha256='2' * 64, session_sha256='3' * 64,
        session_canonical_sha256=evidence.digest(evidence.canonical(session)),
        request_identities={rsha: evidence.digest(evidence.canonical(request))},
        evidence_producer_sha256=evidence.digest(Path(evidence.__file__).read_bytes()))
    cohort = dict(status='complete', request=request, request_sha256=rsha, vector_steps=steps,
        native_collector_sha256=NATIVE_COLLECTOR_SHA256, optimizer_steps=0, teacher_only=True,
        native_payload_writer_unchanged=True, selected_chunks=b, emitted_per_lane=counts[-1],
        model_assets={k: dict(sha256=str(i) * 64) for i, k in enumerate(('teacher', 'homie_walk', 'homie_stand'), 4)})
    native = dict(status='complete', failure=None, native_initializations=1, scene_initializations=1,
        session_id=session['session_id'], cohorts=[dict(cohort_id='test-cohort', request_sha256=rsha,
            vector_steps=steps, output_dir='/dagger-runtime/cohort')])
    host = dict(passed=True, failure=None, exit_code=0)
    bindings = {name: evidence.digest(evidence.canonical(value)) for name, value in
        [('launch.json', launch), ('host-result.json', host), ('bc-session-result.json', native), ('cohort-manifest.json', cohort)]}
    sr, cr = build_session_receipts(launch=launch, launch_raw_sha256=bindings['launch.json'], session=session,
        result=native, result_raw_sha256=bindings['bc-session-result.json'], host_result=host,
        host_result_raw_sha256=bindings['host-result.json'],
        cohort_envelopes=[dict(document=cohort, raw_sha256=bindings['cohort-manifest.json'])])
    memory = dict(memory_current=20 * GIB, memory_peak=23 * GIB, memory_max=43 * GIB,
        **{'memory_swap.current': 0}, events=dict(max=0, oom=0, oom_kill=0))
    saves = []
    for count, emitted in enumerate(counts, 1):
        saves.append(dict(phase='presave', admitted=True, num_envs=b, memory_max_bytes=43 * GIB,
                          memory=deepcopy(memory), vector_steps=81 * count, host_available_bytes=24 * GIB))
        saves.append(dict(phase='postsave', completed_original_save=True, memory=deepcopy(memory),
            vector_steps=81 * count, emitted_chunks_per_lane=emitted,
            memory_current_bytes=20 * GIB, memory_peak_bytes=23 * GIB, anon_bytes=14 * GIB,
            shmem_bytes=GIB, kernel_bytes=GIB, file_bytes=4 * GIB, host_available_bytes=24 * GIB))
    samples = [dict(container=dict(pid=123, oom_killed=False, **deepcopy(memory)),
                    simultaneous_native_image_owners=1, host_available_kib=24 * 1024**2)]
    return dict(launch=launch, host=host, native=native, cohort=cohort, session_execution=sr,
        cohort_execution=cr[0], save_events=saves, memory_samples=samples, bindings=bindings)


def extended_fixture(completed=6):
    args = fixture(6, counts=[[1] * 6] * (completed - 1) + [[2] * 6])
    request = args['cohort']['request']
    request['max_windows'] = 8
    rsha = evidence.digest(evidence.canonical(request) + b'\n')
    args['launch']['qualification_max_windows'] = 8
    args['launch']['request_identities'] = {rsha: evidence.digest(evidence.canonical(request))}
    args['cohort']['request_sha256'] = rsha
    args['native']['cohorts'][0]['request_sha256'] = rsha
    session = dict(session_id=args['native']['session_id'], requests=[dict(sha256=rsha)])
    args['launch']['session_canonical_sha256'] = evidence.digest(evidence.canonical(session))
    args['bindings'] = {name: evidence.digest(evidence.canonical(value)) for name, value in
        [('launch.json', args['launch']), ('host-result.json', args['host']),
         ('bc-session-result.json', args['native']), ('cohort-manifest.json', args['cohort'])]}
    sr, cr = build_session_receipts(launch=args['launch'], launch_raw_sha256=args['bindings']['launch.json'],
        session=session, result=args['native'], result_raw_sha256=args['bindings']['bc-session-result.json'],
        host_result=args['host'], host_result_raw_sha256=args['bindings']['host-result.json'],
        cohort_envelopes=[dict(document=args['cohort'], raw_sha256=args['bindings']['cohort-manifest.json'])])
    args['session_execution'], args['cohort_execution'] = sr, cr[0]
    return args


@pytest.mark.parametrize('completed', [2, 5, 8])
def test_explicit_longer_probe_keeps_original_success_gates(completed):
    result = evidence.build_capacity_receipt(**extended_fixture(completed))
    assert result['vector_steps'] == completed * 81
    assert result['maximum_vector_steps_declared'] == 648
    assert result['production_admitted'] is False


@pytest.mark.parametrize('fault', ['implicit_eight', 'mismatch', 'over648', 'flat_lane', 'pressure'])
def test_longer_probe_cannot_weaken_original_gates(fault):
    args = extended_fixture(9 if fault == 'over648' else 6)
    if fault == 'implicit_eight': args['launch'].pop('qualification_max_windows')
    if fault == 'mismatch': args['launch']['qualification_max_windows'] = 4
    if fault == 'flat_lane': args['cohort']['emitted_per_lane'][1] = 1
    if fault == 'pressure': args['memory_samples'][0]['container']['events']['max'] = 1
    with pytest.raises((ValueError, RuntimeError)):
        evidence.build_capacity_receipt(**args)


@pytest.mark.parametrize('bound', [True, False, 8.0, '8', 0, 5, 9])
def test_capacity_receipt_rejects_unapproved_declared_bound(bound):
    args = extended_fixture()
    args['launch']['qualification_max_windows'] = bound
    with pytest.raises((ValueError, RuntimeError)):
        evidence.build_capacity_receipt(**args)


@pytest.mark.parametrize('b', [1, 4, 6])
def test_genuine_receipt_producer_and_production_checker_metadata_handoff(b):
    args = fixture(b)
    receipt = evidence.build_capacity_receipt(**args)
    protocol = dict(collector_execution=args['launch']['collector_execution'],
        execution_topology=args['launch']['execution_topology'], capacity_equivalence=evidence.RULE,
        model_sha256=receipt['model_sha256'])
    validate_bounded_capacity(receipt, protocol)
    assert receipt['num_envs'] == b and receipt['full_chunks_saved'] == b * 2
    assert receipt['production_admitted'] is False and receipt['long_campaign_execution_proven'] is False


def test_asynchronous_episode_emissions_do_not_bias_full_writer_capacity_admission():
    args = fixture(6, counts=[[0, 1, 0, 1, 0, 1], [1, 2, 1, 2, 1, 2], [2, 2, 2, 2, 2, 2]])
    args['save_events'][0]['host_available_bytes'] = 17 * GIB
    args['save_events'][1]['host_available_bytes'] = 18 * GIB
    receipt = evidence.build_capacity_receipt(**args)
    assert receipt['vector_steps'] == 243
    assert receipt['runtime_min_mem_available_bytes'] == 17 * GIB


@pytest.mark.parametrize('fault', ['flat_lane', 'swap', 'oom', 'pressure', 'host_failed', 'partial_native',
    'cohort_changed', 'source_changed', 'producer_changed', 'missing_owner', 'another_owner', 'early_ordinal',
    'no_flush', 'bad_final_steps', 'bad_model', 'unbound_raw_hash'])
def test_incomplete_or_changed_evidence_cannot_publish_pass(fault):
    args = fixture()
    if fault == 'flat_lane': args['save_events'][-1]['emitted_chunks_per_lane'][0] = 1
    elif fault == 'swap': args['memory_samples'][0]['container']['memory_swap.current'] = 1
    elif fault == 'oom': args['memory_samples'][0]['container']['events']['oom'] = 1
    elif fault == 'pressure': args['save_events'][0]['memory']['events']['max'] = 1
    elif fault == 'host_failed': args['host']['passed'] = False
    elif fault == 'partial_native': args['native']['status'] = 'running'
    elif fault == 'cohort_changed': args['cohort_execution']['actual_num_envs'] = 1
    elif fault == 'source_changed': args['session_execution']['collector_execution'] = {'wrong': 'source'}
    elif fault == 'producer_changed': args['launch']['evidence_producer_sha256'] = 'a' * 64
    elif fault == 'missing_owner': args['memory_samples'][0].pop('simultaneous_native_image_owners')
    elif fault == 'another_owner': args['memory_samples'][0]['simultaneous_native_image_owners'] = 2
    elif fault == 'early_ordinal': args['cohort']['request']['selected_chunk_ordinals_per_env'] = [[0]] * 6
    elif fault == 'no_flush': args['save_events'] = args['save_events'][:2]
    elif fault == 'bad_final_steps': args['save_events'][-1]['vector_steps'] = 243
    elif fault == 'bad_model': args['cohort']['model_assets']['teacher']['sha256'] = 'unknown'
    else: args['bindings']['host-result.json'] = '7' * 64
    with pytest.raises((ValueError, RuntimeError)):
        evidence.build_capacity_receipt(**args)


def test_file_producer_reads_only_completed_metadata(tmp_path, monkeypatch):
    args = fixture(1)
    values = {'launch.json': args['launch'], 'host-result.json': args['host'],
        'bc-session-result.json': args['native'], 'session-execution.json': args['session_execution'],
        'cohort/cohort-manifest.json': args['cohort'], 'cohort/execution-receipt.json': args['cohort_execution']}
    for name, value in values.items():
        p = tmp_path / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(evidence.canonical(value))
    for name, value in [('capacity-save-events.jsonl', args['save_events']), ('memory-samples.jsonl', args['memory_samples'])]:
        (tmp_path / name).write_bytes(b''.join(evidence.canonical(row) + b'\n' for row in value))
    read = Path.read_bytes
    def only_metadata(path):
        assert path.suffix in ('.json', '.jsonl', '.py')
        return read(path)
    monkeypatch.setattr(Path, 'read_bytes', only_metadata)
    def publish(path, value):
        with path.open('xb') as stream:
            stream.write(evidence.canonical(value))
    binding = evidence.publish_capacity_receipt(tmp_path, publish)
    assert binding['raw_sha256'] == evidence.digest(read(Path(binding['path'])))
    with pytest.raises(FileExistsError):
        evidence.publish_capacity_receipt(tmp_path, publish)


def test_exclusive_owner_never_admits_or_stops_another_container():
    assert exclusive_native_owner([]) == 0
    assert exclusive_native_owner(['our-container'], 'our-container') == 1
    for names, owner in [(['other'], None), (['other'], 'our-container'), (['our-container', 'other'], 'our-container')]:
        with pytest.raises(RuntimeError):
            exclusive_native_owner(names, owner)
