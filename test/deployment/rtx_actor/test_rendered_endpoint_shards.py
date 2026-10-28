"""Synthetic CPU tests for two disjoint four-renderer actor shards."""
from test.deployment.source_paths import ACTOR_ROOT
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(ACTOR_ROOT))
import rendered_endpoint_pool as pool
import rendered_student_success_contract as rendered
import rendered_success_observations as observations
import scaled_success_contract as scaled
import explicit_motion_panel as explicit
from test.deployment.rtx_actor.test_rendered_endpoint_pool import Client, pool_config
from test.deployment.rtx_actor.test_rendered_success_observations import fixture as observation_fixture
from test.deployment.rtx_actor.test_scaled_success_contract import accepted, config_for, scaled_cohort, native_admission
from test.deployment.rtx_actor.test_rendered_student_native_routing import evaluator_function


def renderer():
    return dict(pool_config(8), routing=pool.SHARDED_ROUTING, actor_shards=deepcopy(pool.ACTOR_SHARDS))


def clients_for(config, count, gpu, *, cohort_index=None, gate=None, tracker=None, fail=False):
    index = gpu if cohort_index is None else cohort_index
    return [Client(lane, config, endpoint_slot=pool.endpoint_slot_for_lane(config, lane,
        cohort_index=index, physical_gpu=gpu), channel_namespace=gpu*1000,
        gate=gate, tracker=tracker, fail=fail and lane == 0) for lane in range(count)]


@pytest.mark.parametrize('count', [4, 7, 8, 16])
@pytest.mark.parametrize('gpu', [0, 1])
def test_every_lane_has_fixed_global_slot_and_tail_has_no_fake_activity(count, gpu):
    config = renderer()
    clients = clients_for(config, count, gpu)
    executor = pool.RenderedEndpointPool(clients, config, cohort_index=gpu, physical_gpu=gpu)
    try:
        result = executor.render([{}]*count, 0)
        assert [int(value.views['head_color'][0, 0, 0]) for value in result] == list(range(count))
        receipt = executor.receipt()
        assert [row['endpoint_slot'] for row in receipt['lane_bindings']] == [gpu*4+i%4 for i in range(count)]
        assert receipt['execution_shard'] == dict(cohort_index=gpu, physical_gpu=gpu, endpoint_slots=list(range(gpu*4, gpu*4+4)))
        expected = [sum(gpu*4+i%4 == slot for i in range(count)) for slot in range(8)]
        assert [row['completed'] for row in receipt['steps'][0]['endpoints']] == expected
        assert pool.validate_pool_receipt(receipt, count=count, steps=1, renderer=config, cohort_index=gpu) == receipt
    finally:
        executor.close()


def test_both_actors_reach_eight_concurrent_requests_without_sharing_an_engine():
    config = renderer()
    tracker = dict(lock=threading.Lock(), active=[0]*8, max=[0]*8, parallel=0)
    barrier = threading.Barrier(8)
    executors = [pool.RenderedEndpointPool(clients_for(config, 16, gpu, gate=barrier, tracker=tracker),
        config, cohort_index=gpu, physical_gpu=gpu) for gpu in (0, 1)]
    try:
        with ThreadPoolExecutor(max_workers=2) as actors:
            futures = [actors.submit(executor.render, [{}]*16, 0) for executor in executors]
            assert all(len(future.result(timeout=10)) == 16 for future in futures)
        assert tracker['max'] == [1]*8 and tracker['parallel'] == 8
        receipts = [executor.receipt() for executor in executors]
        assert receipts[0]['host_boot_id'] == receipts[1]['host_boot_id']
        events = []
        for receipt in receipts:
            for lane in receipt['steps'][0]['lanes']:
                events.extend([(lane['request_started_monotonic_s'], 1), (lane['request_finished_monotonic_s'], -1)])
        active = peak = 0
        for _, change in sorted(events):
            active += change
            peak = max(peak, active)
        assert peak == 8 and active == 0
    finally:
        for executor in executors: executor.close()


@pytest.mark.parametrize('fault', ['overlap', 'reversed', 'missing', 'duplicate_gpu', 'v2_downgrade'])
def test_changed_global_shards_or_identities_are_rejected(fault):
    config = renderer()
    if fault == 'overlap': config['actor_shards'][1]['endpoint_slots'][0] = 3
    elif fault == 'reversed': config['actor_shards'].reverse()
    elif fault == 'missing': config['endpoints'].pop()
    elif fault == 'duplicate_gpu': config['endpoints'][7]['gpu_uuid'] = config['endpoints'][0]['gpu_uuid']
    else: config['routing'] = pool.ROUTING
    with pytest.raises(ValueError): pool.validate_pool_config(config)


@pytest.mark.parametrize('actual,index', [(1, 0), (0, 1), (None, 0), (True, 1)])
def test_wrong_native_owner_fails_before_any_request(actual, index):
    config = renderer()
    clients = clients_for(config, 16, index)
    with pytest.raises(ValueError, match='physical GPU'):
        pool.RenderedEndpointPool(clients, config, cohort_index=index, physical_gpu=actual)
    assert all(client.calls == 0 and client._server_instance is None for client in clients)


@pytest.mark.parametrize('fault', ['wrong_owner', 'wrong_cohort', 'clock', 'negative_interval', 'inactive_count', 'route'])
def test_shard_receipts_cannot_forge_placement_or_concurrency(fault):
    config = renderer()
    executor = pool.RenderedEndpointPool(clients_for(config, 7, 0), config, cohort_index=18, physical_gpu=0)
    try:
        executor.render([{}]*7, 0)
        value = executor.receipt()
    finally:
        executor.close()
    if fault == 'wrong_owner': value['execution_shard']['physical_gpu'] = 1
    elif fault == 'wrong_cohort': value['execution_shard']['cohort_index'] = 16
    elif fault == 'clock': value['clock'] = 'wall_time'
    elif fault == 'negative_interval': value['steps'][0]['lanes'][0]['request_started_monotonic_s'] = -1
    elif fault == 'inactive_count': value['steps'][0]['endpoints'][4]['completed'] = 1
    else: value['lane_bindings'][0]['endpoint_slot'] = 4
    value['trace_sha256'] = hashlib.sha256(b''.join(json.dumps(row, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode() for row in value['steps'])).hexdigest()
    with pytest.raises(ValueError): pool.validate_pool_receipt(value, count=7, steps=1, renderer=config, cohort_index=18)


@pytest.mark.parametrize('gpu', [0, 1])
@pytest.mark.parametrize('failure', [False, True])
def test_adapter_preserves_all_lane_barrier_and_no_fallback_on_both_shards(tmp_path, gpu, failure):
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    trainer, old, obs, _, calls, _ = observation_fixture(tmp_path, count=16)
    old.close()
    config = renderer()
    config['identity'] = old.renderer_identity
    clients = clients_for(config, 16, gpu, fail=failure)
    binding = dict(schema=rendered.BINDING_SCHEMA, checkpoint_sha256='a'*64, panel_sha256='b'*64,
        config_sha256='c'*64, cohort_index=gpu, motion_assignment_sha256='d'*64)
    adapter = observations.GeneratedSuccessObservations(trainer, client_factory=lambda lane: clients[lane],
        anchor_files=old.anchor_files, anchor_seed=123, renderer_identity=config['identity'],
        evaluation_binding=binding, endpoint_pool_config=config, endpoint_pool_physical_gpu=gpu)
    trainer._success_observation_transform = adapter
    try:
        if failure:
            with pytest.raises(RuntimeError, match='partial failure'):
                observations.apply_observation_transform(trainer, obs, torch.zeros(16), 0)
            assert calls['normalize'] == 0 and adapter.steps == 0
            with pytest.raises(RuntimeError): adapter.endpoint_pool.render([{}]*16, 0)
        else:
            result = observations.apply_observation_transform(trainer, obs, torch.zeros(16), 0)
            assert result['vision_obs'] is not obs['vision_obs'] and calls['normalize'] == 1
            rendered._validate_scaled_receipt(adapter.receipt(), count=16, steps=1, renderer=config)
    finally:
        adapter.close()
        torch.set_num_threads(saved)


def test_new_full_config_and_tail_qualification_do_not_relabel_historical_cases(accepted):
    panel = scaled.build_panel(accepted, qualification=True, qualification_variant='tail7')
    assert panel['episode_count'] == 7 and len(panel['cohorts']) == 1
    source = [lane for row in accepted['cohorts'] for lane in row['assignment']['lanes']][-7:]
    assert [row['motion_id'] for row in panel['cohorts'][0]['assignment']['lanes']] == [row['motion_id'] for row in source]
    full = scaled.build_panel(accepted)
    assert panel['cohorts'][0]['assignment']['simulator_seed'] != full['cohorts'][18]['assignment']['simulator_seed']
    config = config_for(accepted, panel, sharded=True)
    assert rendered.validate_config(config, panel) == config
    assert rendered.bind_observation_config(config, panel, config['checkpoint_sha256']) == config
    assert scaled._configuration(panel, config) == config
    bad = deepcopy(config)
    bad['schema'] = rendered.SCALED_CONFIG_SCHEMA
    with pytest.raises(ValueError): rendered.validate_config(bad, panel)
    row = scaled_cohort(accepted, panel, config)
    rendered.validate_cohort_observations(row, count=7, config=config, panel=panel, cohort_index=0)


@pytest.mark.parametrize('gpu', [0, 1])
def test_native_evaluator_uses_exact_shard_before_action(accepted, tmp_path, monkeypatch, gpu):
    panel = scaled.build_panel(accepted, qualification=True)
    config = config_for(accepted, panel, sharded=True)
    monkeypatch.setenv('RESIDENT_PHYSICAL_GPU', str(gpu))
    calls = []
    class Adapter:
        def __init__(self, trainer, **kwargs):
            assert kwargs['endpoint_pool_physical_gpu'] == gpu
            assert kwargs['endpoint_pool_config'] == config['renderer']
            assert [kwargs['client_factory'](lane) for lane in range(16)] == [18550+4*gpu+lane%4 for lane in range(16)]
        def close(self): pass
    class Evaluator:
        def evaluate_cohort(self, assignment, **kwargs):
            calls.append(assignment)
            return scaled_cohort(accepted, panel, config, gpu)
    monkeypatch.setattr(observations, 'GeneratedSuccessObservations', Adapter)
    monkeypatch.setitem(sys.modules, 'pipeline.dagger.services.render_student',
        SimpleNamespace(_Image3FLoopbackClient=lambda host, port, **kwargs: port))
    execute = evaluator_function(tmp_path, Evaluator)
    execute.__globals__['os'] = os
    result = execute(SimpleNamespace(), panel, root=tmp_path/'native', checkpoint_path='/selected.pt',
        checkpoint_sha256=config['checkpoint_sha256'], prepare_cohort=lambda assignment: None,
        report=lambda row: None, rendered_observation_config=config,
        _scaled_partition=explicit.make_partition(panel, [gpu]), **scaled.evaluation_options(panel))
    assert calls == [panel['cohorts'][gpu]['assignment']]
    assert result['partition_result']['completed_episodes'] == 16


def test_native_wrong_physical_owner_rejected_before_checkpoint(accepted, tmp_path, monkeypatch):
    owner, payload, _, _, checks = native_admission(accepted, tmp_path, monkeypatch, sharded=True)
    monkeypatch.setenv('RESIDENT_PHYSICAL_GPU', '1')
    with pytest.raises(ValueError, match='actual physical RTX'):
        owner._scaled_success_inputs(payload)
    assert not any(path.parent == Path('/diagnostic-assets') for path in checks)


def test_direct_wrong_actor_rejected_before_scene_or_output(accepted, tmp_path, monkeypatch):
    panel = scaled.build_panel(accepted, qualification=True)
    config = config_for(accepted, panel, sharded=True)
    monkeypatch.setenv('RESIDENT_PHYSICAL_GPU', '1')
    execute = evaluator_function(tmp_path, object)
    execute.__globals__['os'] = os
    prepared = []
    with pytest.raises(ValueError, match='before scene preparation'):
        execute(SimpleNamespace(), panel, root=tmp_path/'never', checkpoint_path='/unused',
            checkpoint_sha256=config['checkpoint_sha256'], prepare_cohort=prepared.append,
            report=lambda row: None, rendered_observation_config=config,
            _scaled_partition=explicit.make_partition(panel, [0]), **scaled.evaluation_options(panel))
    assert not prepared and not (tmp_path/'never').exists()
