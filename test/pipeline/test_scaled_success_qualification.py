"""Synthetic CPU evidence only; no generated proof is deployed or saved."""
from test.deployment.source_paths import PIPELINE_ROOT
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

HERE = PIPELINE_ROOT / 'deployment/rtx_actor'
sys.path.insert(0, str(HERE))
from test.deployment.rtx_actor.test_scaled_success_contract import accepted, config_for, scaled_cohort
from pipeline.deployment.rtx_actor import scaled_success_contract as scaled
from pipeline.deployment.rtx_actor import explicit_motion_panel as explicit
from pipeline.evaluation.scaled_success_qualification import SCHEMA, validate_qualification, request_concurrency
from pipeline.evaluation.scaled_success_resources import GIB, summarize
from test.pipeline.test_scaled_success_resources import fixture as resource_fixture

SOURCE = 'a'*64
CACHE = {}


def _cohort(original, panel, config, index, arm):
    row = scaled_cohort(original, panel, config, index, arm)
    pool = row['rendered_observation_receipt']['renderer_endpoint_pool']
    # Deterministic synthetic concurrent intervals; not real GPU evidence.
    for step in pool['steps']:
        step.update(started_monotonic_s=100., finished_monotonic_s=110.)
        for lane in step['lanes']:
            start = 100.+2*(lane['lane']//4)
            lane.update(request_started_monotonic_s=start, request_finished_monotonic_s=start+1.)
    pool['trace_sha256'] = hashlib.sha256(b''.join(
        json.dumps(s, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        for s in pool['steps'])).hexdigest()
    return row


def _trial(original, size, *, tail=False, arm='bc'):
    panel = scaled.build_panel(original, batch_size=size, qualification=True,
        qualification_variant='tail7' if tail else 'paired')
    config = config_for(original, panel, sharded=True)
    owners = []
    for gpu in range(1 if tail else 2):
        part = explicit.run_partition(panel, explicit.make_partition(panel, [gpu]),
            execute_cohort=lambda _, i: _cohort(original, panel, config, i, arm),
            checkpoint_sha256=config['checkpoint_sha256'], arm=arm, **scaled.evaluation_options(panel))
        result = scaled.wrap_partition(panel, part, config['checkpoint_sha256'], config)
        owners.append(dict(physical_gpu=gpu, worker_instance_id=f'isaac{gpu}', gpu_uuid=f'rtx{gpu}',
            native_receipt=dict(complete=True, source_sha256=SOURCE, optimizer_steps=0,
                native_initializations=1, resource_guard=dict(passed=True), result=result)))
    samples, options = resource_fixture()
    native = [dict(owner_id=r['worker_instance_id'], gpu_uuid=r['gpu_uuid'], source_sha256=SOURCE) for r in owners]
    renderers = [dict(owner_id=e['server_instance'], gpu_uuid=e['gpu_uuid'], source_sha256=e['source_sha256'])
                 for e in config['renderer']['endpoints']]
    options.update(native_owners=native, renderer_owners=renderers)
    for sample in samples:
        sample['native'] = [dict(deepcopy(sample['native'][0]), **owner) for owner in native]
        for gpu, row in enumerate(sample['native']):
            row['safety_sample'].update(process_id=17+gpu, cgroup_inode=123+gpu,
                                       cgroup_path=f'/sys/fs/cgroup/test-fixture{gpu}')
        sample['renderers'] = [dict(sample['renderers'][0], **owner) for owner in renderers]
    return dict(panel=panel, config=config, owners=owners, resources=dict(samples=samples,
        renderer_owners=renderers, summary=summarize(samples, **options)),
        started_unix=options['started_unix'], ended_unix=options['ended_unix'])


@pytest.fixture
def evidence(accepted, monkeypatch):
    import production_success_contract as bare
    from pipeline.deployment.rtx_actor import production_success_contract as packaged
    for key in ('PANEL_SHA256', 'PANEL_RAW_SHA256', 'LEGACY_PANEL_SHA256', 'LEGACY_PANEL_RAW_SHA256'):
        monkeypatch.setattr(packaged, key, getattr(bare, key))
    if not CACHE:
        panel = scaled.build_panel(accepted)
        config = config_for(accepted, panel, sharded=True)
        trials = [_trial(accepted, n) for n in (4, 8, 16, 16)]
        proof = dict(schema=SCHEMA, source_sha256=SOURCE, batch_size=16, arm='bc',
            checkpoint_sha256=config['checkpoint_sha256'], compute_release_authorized=False,
            cpu_tests_passed=True, trials=trials, tail_trial=_trial(accepted, 16, tail=True),
            synthetic_cpu_test_only=True)
        CACHE.update(proof=proof, panel=panel, config=config)
    return deepcopy(CACHE)


def check(evidence):
    return validate_qualification(evidence['proof'], source_sha256=SOURCE,
                                  panel=evidence['panel'], config=evidence['config'])


def test_real_semantic_validators_accept_complete_synthetic_fixture(evidence):
    assert check(evidence) == evidence['proof']


@pytest.mark.parametrize('mutate', [
    lambda p: p.update(source_sha256='f'*64),
    lambda p: p.update(checkpoint_sha256='f'*64),
    lambda p: p.update(compute_release_authorized=True),
    lambda p: p.update(cpu_tests_passed=False),
    lambda p: p.update(arm='teacher'),
    lambda p: p.pop('tail_trial'),
    lambda p: p['trials'].pop(),
    lambda p: p['trials'][0]['owners'].pop(),
    lambda p: p['trials'][0]['owners'][0]['native_receipt'].update(optimizer_steps=1),
    lambda p: p['trials'][0]['resources']['summary'].update(resources_passed=False),
    lambda p: p['trials'][0]['resources']['samples'][1]['renderers'].pop(),
    lambda p: p['trials'][0]['owners'][0]['native_receipt']['resource_guard'].update(passed=False),
])
def test_no_boolean_hash_or_idle_device_can_substitute_for_complete_evidence(evidence, mutate):
    mutate(evidence['proof'])
    with pytest.raises((ValueError, KeyError)):
        check(evidence)


def test_only_target_warm_repeat_requires_eight_simultaneous_rpc_intervals(evidence):
    pools = [owner['native_receipt']['result']['partition_result']['cohort_receipts'][0]
             ['result']['rendered_observation_receipt']['renderer_endpoint_pool']
             for owner in evidence['proof']['trials'][-1]['owners']]
    assert request_concurrency(pools) == 8
    assert request_concurrency(pools[:1]) == 4
    pools[1]['host_boot_id'] = 'ffffffff-ffff-ffff-ffff-ffffffffffff'
    with pytest.raises(ValueError, match='boot'):
        request_concurrency(pools)


def test_reject_false_eight_way_overlap_from_duplicate_renderer_requests(evidence):
    pool = evidence['proof']['trials'][-1]['owners'][0]['native_receipt']['result']['partition_result']
    pool = pool['cohort_receipts'][0]['result']['rendered_observation_receipt']['renderer_endpoint_pool']
    pool['steps'][0]['lanes'][4]['request_started_monotonic_s'] = 100.5
    with pytest.raises(ValueError, match='overlapping requests'):
        request_concurrency([pool])


def test_generic_dagger_checkpoint_qualification_uses_the_actual_student_arm(evidence, accepted):
    proof = evidence['proof']
    proof.update(arm='dagger', trials=[_trial(accepted, n, arm='dagger') for n in (4, 8, 16, 16)],
                 tail_trial=_trial(accepted, 16, tail=True, arm='dagger'))
    assert check(evidence) == proof


def test_repeated_target_initial_state_must_actually_match(evidence):
    receipt = evidence['proof']['trials'][-1]['owners'][0]['native_receipt']['result']['partition_result']['cohort_receipts'][0]
    receipt['result']['scaled_initial_state_attestation']['initial_state']['robot_root']['sha256'] = 'e'*64
    receipt['result_sha256'] = scaled.digest(receipt['result'])
    with pytest.raises(ValueError, match='repeated initialization'):
        check(evidence)


def test_target_repeat_cannot_claim_parallel_capacity_from_serial_receipts(evidence):
    receipt = evidence['proof']['trials'][-1]['owners'][1]['native_receipt']['result']['partition_result']['cohort_receipts'][0]
    pool = receipt['result']['rendered_observation_receipt']['renderer_endpoint_pool']
    for step in pool['steps']:
        for name in ('started_monotonic_s', 'finished_monotonic_s'):
            step[name] += 100.
        for lane in step['lanes']:
            for name in ('request_started_monotonic_s', 'request_finished_monotonic_s'):
                lane[name] += 100.
    pool['trace_sha256'] = hashlib.sha256(b''.join(
        json.dumps(s, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        for s in pool['steps'])).hexdigest()
    receipt['result_sha256'] = scaled.digest(receipt['result'])
    with pytest.raises(ValueError, match='concurrency'):
        check(evidence)
