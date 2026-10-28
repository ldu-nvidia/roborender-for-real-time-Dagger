"""CPU-only synthetic metadata tests; these are not native capacity proof."""
from copy import deepcopy
import hashlib
import json

import pytest

import bc_memory_guard
import run_guarded_bc_capacity as shared
import run_guarded_bc_session as production


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def request(index, b=6):
    return {'schema': 'roborender.bc-native-collection-request.v2', 'cohort_id': f'cohort-{index}',
        'run_id': 'owned-plan', 'rank': 0, 'sampling_plan_sha256': 'a' * 64,
        'motions': [{}] * b, 'chunks_per_env': [1] * b, 'selected_chunk_ordinals_per_env': [[1]] * b,
        'exposure_span_chunks_per_env': [2] * b, 'split': 'validation', 'max_windows': 4}


def test_production_and_qualification_use_same_entrypoint_and_command_guard():
    assert production.main is shared.main
    assert shared.augment_command is bc_memory_guard.augment_command
    assert shared.admission is bc_memory_guard.admission
    assert shared.runtime_guard is bc_memory_guard.runtime_guard
    for b in (1, 2, 6):
        rows = [request(i, b) for i in range(4)]
        assert shared.validate_requests(rows, qualification_only=False) == b
    assert shared.validate_requests([request(0)], qualification_only=True) == 6


@pytest.mark.parametrize('bound', [4, 8])
def test_qualification_bound_must_be_explicit_and_match_request(bound):
    row = request(0)
    row['max_windows'] = bound
    assert shared.validate_requests([row], qualification_only=True,
                                   qualification_max_windows=bound) == 6
    if bound == 8:
        with pytest.raises(ValueError):
            shared.validate_requests([row], qualification_only=True)


@pytest.mark.parametrize('bound', [True, False, 8.0, '8', 0, 5, 9])
def test_qualification_rejects_coerced_or_unapproved_bound(bound):
    with pytest.raises(ValueError):
        shared.validate_requests([request(0)], qualification_only=True,
                                 qualification_max_windows=bound)


def test_output_directory_is_host_owned_before_native_writer(tmp_path):
    import os
    rows = [{'output_dir': '/dagger-runtime/cohorts/one'},
            {'output_dir': '/dagger-runtime/cohorts/two'}]
    paths = shared.prepare_output_directories(tmp_path, rows)
    assert all(path.stat().st_uid == os.getuid() for path in paths)
    (paths[0] / 'native-result.json').write_text('{}')
    (paths[0] / 'execution-receipt.json').write_text('{}')
    with pytest.raises(ValueError):
        shared.prepare_output_directories(tmp_path, rows)


@pytest.mark.parametrize('paths', [
    ['/outside'], ['/dagger-runtime'], ['/dagger-runtime/../escape'],
    ['/dagger-runtime/one', '/dagger-runtime/one'],
    ['/dagger-runtime/one', '/dagger-runtime/one/child'],
])
def test_output_precreation_rejects_broad_nested_or_duplicate_paths(tmp_path, paths):
    with pytest.raises(ValueError):
        shared.prepare_output_directories(tmp_path, [{'output_dir': path} for path in paths])
    assert not list(tmp_path.iterdir())


def test_output_precreation_rejects_parent_symlink(tmp_path):
    owned, outside = tmp_path / 'owned', tmp_path / 'outside'
    owned.mkdir()
    outside.mkdir()
    (owned / 'linked').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        shared.prepare_output_directories(owned, [{'output_dir': '/dagger-runtime/linked/cohort'}])
    assert not list(outside.iterdir())


@pytest.mark.parametrize('rows', [[request(i) for i in range(5)], [request(0), request(1, 2)], [request(0, 7)]])
def test_production_rejects_unbounded_or_mixed_width_sessions(rows):
    with pytest.raises(ValueError):
        shared.validate_requests(rows, qualification_only=False)


def test_qualification_does_not_accept_production_quotas_or_implicit_tail_batches():
    for rows in ([request(0), request(1)], [request(0, 2)]):
        with pytest.raises(ValueError):
            shared.validate_requests(rows, qualification_only=True)


def test_tail_requires_explicit_actual_b_and_identical_late_ordinal_recipe():
    assert shared.validate_requests([request(0, 1)], qualification_only=True, qualification_num_envs=1) == 1
    assert shared.validate_requests([request(0, 4)], qualification_only=True, qualification_num_envs=4) == 4
    for rows, expected in (([request(0, 1)], 6), ([request(0, 6)], 1), ([request(0, 2)], 2),
                           ([request(0, 1)], True)):
        with pytest.raises(ValueError):
            shared.validate_requests(rows, qualification_only=True, qualification_num_envs=expected)
    early = request(0, 1)
    early['selected_chunk_ordinals_per_env'] = [[0]]
    with pytest.raises(ValueError):
        shared.validate_requests([early], qualification_only=True, qualification_num_envs=1)
    with pytest.raises(ValueError):
        shared.validate_requests([request(0, 4)], qualification_only=True)


def prior(tmp_path):
    root = tmp_path / 'owned' / 'prior'
    root.mkdir(parents=True)
    old = request(0)
    request_sha = digest(canonical(old) + b'\n')
    execution = {'runtime_contract_sha256': 'b' * 64}
    native = {'session_id': 'prior-session', 'status': 'complete', 'failure': None,
        'native_initializations': 1, 'cohorts': [{'cohort_id': old['cohort_id'],
            'request_sha256': request_sha, 'output_dir': '/dagger-runtime/bc_isaac'}]}
    host = {'passed': True, 'exit_code': 0, 'failure': None}
    receipt = {'schema': 'roborender.bc-session-execution.v1', 'verdict': 'PASS',
        'session_id': native['session_id'], 'collector_execution': execution, 'native_initializations': 1,
        'scene_initializations': 1, 'request_identities': {request_sha: digest(canonical(old))},
        'native_result_raw_sha256': digest(canonical(native)), 'host_result_raw_sha256': digest(canonical(host))}
    child = {'status': 'complete', 'optimizer_steps': 0, 'request': old,
             'request_sha256': request_sha, 'vector_steps': 243}
    cohort = {'schema': 'roborender.bc-cohort-execution.v1', 'verdict': 'PASS',
        'cohort_manifest_raw_sha256': digest(canonical(child)),
        'session_execution_canonical_sha256': digest(canonical(receipt)),
        'collector_execution': execution, 'actual_vector_steps': 243, 'actual_num_envs': 6}
    for name, value in [('bc-session-result.json', native), ('host-result.json', host),
                        ('session-execution.json', receipt), ('bc_isaac/cohort-manifest.json', child),
                        ('bc_isaac/execution-receipt.json', cohort)]:
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(canonical(value))
    return root, digest(canonical(receipt)), execution


def test_new_session_can_follow_exact_complete_boundary_without_native_resume(tmp_path):
    path, sha, execution = prior(tmp_path)
    result = shared.validate_prior_boundary(path, sha, [request(1)], execution, _owned_root=path.parent)
    assert result['completed_cohorts'] == ['cohort-0'] and result['native_state_resumed'] is False


@pytest.mark.parametrize('fault', ['partial', 'changed_receipt', 'same_cohort', 'changed_plan', 'changed_contract', 'changed_cohort'])
def test_partial_or_drifted_session_never_replayed_or_auto_admitted(tmp_path, fault):
    path, sha, execution = prior(tmp_path)
    upcoming = [request(1)]
    if fault == 'partial':
        (path / 'bc-session-result.json').write_bytes(canonical({'status': 'failed'}))
    elif fault == 'changed_receipt':
        (path / 'session-execution.json').write_bytes(b'{}')
    elif fault == 'same_cohort':
        upcoming = [request(0)]
    elif fault == 'changed_plan':
        upcoming[0]['sampling_plan_sha256'] = 'c' * 64
    elif fault == 'changed_contract':
        execution['runtime_contract_sha256'] = 'c' * 64
    else:
        child = json.loads((path / 'bc_isaac/cohort-manifest.json').read_bytes())
        child['request']['rank'] = 1
        (path / 'bc_isaac/cohort-manifest.json').write_bytes(canonical(child))
    with pytest.raises(ValueError):
        shared.validate_prior_boundary(path, sha, upcoming, execution, _owned_root=path.parent)


def test_tail_command_keeps_original_buffers_and_explicit_expected_b():
    command = ['docker', 'run', '--memory', '40g', '--memory-swap', '40g', '--env',
        'PYTHONPATH=/source', 'sha256:' + 'a' * 64, 'trainer._target_=isaac6_bc_session.Isaac6BCSessionTrainer']
    changed = shared.augment_command(command, guard_path='/guard.py', guard_sha256='b' * 64,
        memory_gib=43, image='sha256:' + 'a' * 64, num_envs=1)
    assert 'ROBORENDER_BC_GUARD_NUM_ENVS=1' in changed
    assert not any('save_raw' in item or 'rollout_steps' in item for item in changed)
    assert command[-1] == 'trainer._target_=isaac6_bc_session.Isaac6BCSessionTrainer'
