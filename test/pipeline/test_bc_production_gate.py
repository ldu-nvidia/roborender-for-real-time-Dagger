"""CPU-only admission tests; these fixtures never certify a native workload."""
from copy import deepcopy
import hashlib
import json

import pytest

from pipeline.bc_collection import run_continuous as launcher
from pipeline.bc_collection.continuous_plan import build_plan, build_request, validate_plan
from pipeline.bc_collection.production_gate import is_production
from pipeline.bc_collection.episode_drain import EpisodeDrain
from pipeline.bc_collection.streaming_storage import policy
from pipeline.bc_collection.storage_guard import STORAGE_SCHEMA, GIB
from pipeline.deployment.rtx_actor.bounded_bc_storage import PROTOCOL
from test.pipeline.test_continuous_bc_launcher import fixture_capacity


def plan(**changes):
    motions = [dict(motion_id=f'm{i}', path=f'/motions/m{i}.npz',
                    sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(40)]
    options = dict(run_id='production-cpu-fixture', max_windows=2000, num_envs=32,
                   output_dir='/dagger-runtime/bc_isaac', collection_storage_protocol=PROTOCOL,
                   target_total_chunks=10000, production_collection=True)
    options.update(changes)
    return build_plan(motions, **options)


def test_explicit_production_shape_is_not_qualification_or_admission():
    value = plan()
    assert validate_plan(value) == value
    request = build_request(value)
    assert is_production(request) and value['production_admitted'] is False
    launcher.validate_launch(request, qualification=False, expected_num_envs=32,
        physical_gpu=0, memory_gib=43, timeout=3 * 86400)
    with pytest.raises(ValueError, match='relabeled'):
        launcher.validate_launch(request, qualification=True, expected_num_envs=32,
            physical_gpu=0, memory_gib=43, timeout=3600)


@pytest.mark.parametrize('changes', [dict(num_envs=40), dict(target_total_chunks=9999),
    dict(target_total_chunks=10001), dict(max_windows=16), dict(production_collection=1),
    dict(production_collection=False), dict(collection_storage_protocol=None)])
def test_production_shape_cannot_expand_or_silently_bypass(changes):
    with pytest.raises(ValueError):
        plan(**changes)


def test_streaming_admission_is_explicit_and_never_claims_whole_campaign_fit():
    request = build_request(plan())
    disk = dict(schema=STORAGE_SCHEMA, root='/data1', reserve_bytes=64 * GIB,
                bytes_per_emission_budget=512 << 20, session_metadata_budget_bytes=GIB, poll_seconds=1)
    actual = launcher.storage_budget(disk, request)
    assert actual['required_available_bytes'] == 81 * GIB
    assert actual['budget_bytes'] is None and actual['whole_campaign_fit_guaranteed'] is False
    assert actual['shortage_is_success'] is False
    with pytest.raises(ValueError):
        launcher.storage_budget({**disk, 'reserve_bytes': 32 * GIB}, request)


def drain_fixture(tmp_path):
    values = fixture_capacity()
    request = build_request(plan(production_collection=False, target_total_chunks=1, max_windows=16))
    execution = values['launch']['collector_execution']
    contract = launcher.capacity_contract(execution, request, gpu_uuid='GPU-test', memory_gib=43)
    native = values['native']
    ledger = EpisodeDrain(32, 1)
    ledger.accept_window(1, 32, [dict(episode_id=f'e{i}', episode_step_index=81) for i in range(32)])
    ledger.finish_transition(81, [f'e{i}' for i in range(32)], [True] * 32)
    for step in range(82, 162):
        ledger.finish_transition(step, [f'new{i}' for i in range(32)], [False] * 32)
    ledger.accept_window(2, 32, [dict(episode_id=f'new{i}', episode_step_index=80) for i in range(32)])
    native.update(request=request, completion_reason='total_chunks_and_episodes_drained',
        accepted_windows=2, vector_steps=162, native_payload_writer_unchanged=False,
        emitted_per_lane=[1] * 32, emitted_chunks=32, selected_chunks=32,
        unrecorded_rows_per_lane=[80] * 32, remaining_unsealed_tail_rows_per_lane=[0] * 32,
        episode_drain=ledger.receipt(),
        streaming_storage=dict(**policy(32), output_root=request['output_dir'], windows_admitted=2,
                               emissions_verified=32, minimum_available_bytes=90 * GIB),
        collection_storage=dict(protocol=PROTOCOL, completed_windows=2, num_envs=32,
            frames_per_window=81, logical_scenes=1, cpu_conversion_lanes=1, capture_device='cuda:0',
            production_admitted=False, writer_sha256='d' * 64))
    native['initial_state_attestation']['initial_reset_completion'].update(num_envs=32, cleared_lanes=32)
    launch = {**values['launch'], 'request': request, 'capacity_contract': contract}
    host = {**values['host'], 'replay_allowed': False}
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(json.dumps(value))
        return launcher.digest(path.read_bytes())
    receipt = {**values['execution_receipt'], 'production_admitted': False,
        'actual_num_envs': 32, 'actual_vector_steps': 162,
        'request_canonical_sha256': launcher.digest(launcher.canonical(request)),
        'launch_sha256': write('launch.json', launch),
        'host_result_sha256': write('host-result.json', host),
        'native_manifest_sha256': write('bc_isaac/cohort-manifest.json', native)}
    path = tmp_path / 'continuous-execution.json'
    sha = write(path.name, receipt)
    return path, sha, contract


def test_drain_proof_requires_exact_real_chain(tmp_path):
    path, sha, contract = drain_fixture(tmp_path)
    result = launcher.read_drain_proof(path, sha, contract)
    assert result['raw_sha256'] == sha
    changed = deepcopy(contract)
    changed['collector_execution']['source_sha256'] = 'e' * 64
    with pytest.raises(ValueError):
        launcher.read_drain_proof(path, sha, changed)
    host = tmp_path / 'host-result.json'
    value = json.loads(host.read_bytes())
    value['exit_code'] = 1
    host.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        launcher.read_drain_proof(path, sha, contract)


def test_standalone_pass_cannot_certify_drain(tmp_path):
    path = tmp_path / 'continuous-execution.json'
    path.write_text('{"verdict":"PASS"}')
    with pytest.raises(ValueError):
        launcher.read_drain_proof(path, launcher.digest(path.read_bytes()), {})


def test_ingest_accounts_excluded_rows_only_for_explicit_drained_protocol():
    from pipeline.bc_collection.continuous_ingest import _accounting
    receipt = dict(emitted_per_lane=[1, 2], discarded_reset_tail_rows_per_lane=[1, 0],
                   remaining_unsealed_tail_rows_per_lane=[0, 0], unrecorded_rows_per_lane=[80, 0])
    _accounting(receipt, width=2, window=2, drained=True)
    with pytest.raises(ValueError):
        _accounting(receipt, width=2, window=2)
    receipt['unrecorded_rows_per_lane'][0] = 79
    with pytest.raises(ValueError):
        _accounting(receipt, width=2, window=2, drained=True)


def test_native_production_rejects_even_recovered_memory_limit_events():
    from pipeline.deployment.rtx_actor.bounded_bc_guard import memory_gate
    from test.pipeline.test_continuous_bc_launcher import memory
    sample = memory()
    sample['events']['max'] = 1
    with pytest.raises(RuntimeError):
        memory_gate(sample, 60 * GIB, num_envs=32, cap_bytes=43 * GIB, strict_limit_events=True)


def test_standalone_zero_age_completion_is_not_live_drain_qualification(tmp_path):
    path, sha, contract = drain_fixture(tmp_path)
    native_path = tmp_path / 'bc_isaac/cohort-manifest.json'
    native = json.loads(native_path.read_bytes())
    ledger = EpisodeDrain(32, 1)
    ledger.accept_window(1, 0, [dict(episode_id=f'e{i}', episode_step_index=0) for i in range(32)])
    ledger.accept_window(2, 32, [dict(episode_id=f'e{i}', episode_step_index=0) for i in range(32)])
    native.update(episode_drain=ledger.receipt(), unrecorded_rows_per_lane=[0] * 32)
    native_path.write_text(json.dumps(native))
    receipt = json.loads(path.read_bytes())
    receipt['native_manifest_sha256'] = launcher.digest(native_path.read_bytes())
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='actually cross'):
        launcher.read_drain_proof(path, launcher.digest(path.read_bytes()), contract)
