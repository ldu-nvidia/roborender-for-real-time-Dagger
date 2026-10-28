"""Explicit user-authorized diagnostic; not a production collection profile."""
from copy import deepcopy
from types import SimpleNamespace, ModuleType
import sys

import pytest
import torch

from pipeline.bc_collection import collector, continuous_plan, run_continuous
from pipeline.deployment.rtx_actor import isaac6_bc_production as native
from pipeline.deployment.rtx_actor.bounded_bc_storage import PROTOCOL, raw_window_bytes


def plan(batch=50):
    motions = [dict(motion_id=f'm{i}', path=f'/motions/{i}.npz', sha256=f'{i+1:064x}') for i in range(60)]
    return continuous_plan.build_plan(motions, run_id=f'smoke{batch}', num_envs=batch,
        max_windows=40, output_dir='/dagger-runtime/bc_isaac',
        collection_storage_protocol=PROTOCOL, target_chunks_per_lane=20)


@pytest.mark.parametrize('batch', [40, 50])
def test_smoke_is_explicit_and_not_production_or_capacity_admission(batch):
    p = plan(batch)
    req = continuous_plan.build_request(p)
    assert native.validate_collection_request(req) == req
    assert len(req['motion_inventory']) == 60 and len(req['motions']) == batch
    assert raw_window_bytes(batch) == batch * 905748480
    assert run_continuous.validate_launch(req, qualification=True, expected_num_envs=batch,
        physical_gpu=0, memory_gib=43, timeout=43200) == req
    with pytest.raises(ValueError, match='qualification-only'):
        run_continuous.validate_launch(req, qualification=False, expected_num_envs=batch,
            physical_gpu=0, memory_gib=43, timeout=43200)
    with pytest.raises(ValueError):
        run_continuous.build_capacity(launch={'capacity_contract': {}, 'request': req,
            'request_sha256': 'a'*64}, host={}, native={}, save_events=[], memory_samples=[],
            execution_receipt={}, evidence={})


@pytest.mark.parametrize('mutation', ['missing_stop', 'target', 'budget', 'storage', 'loop', 'width'])
def test_smoke_cannot_widen_legacy_requests_or_bypass_contract(mutation):
    req = continuous_plan.build_request(plan())
    if mutation == 'missing_stop': req.pop('diagnostic_stop')
    if mutation == 'target': req['diagnostic_stop']['chunks_per_lane'] = 1
    if mutation == 'budget': req['max_windows'] = 50000
    if mutation == 'storage': req.pop('collection_storage_protocol')
    if mutation == 'loop': req.pop('collection_loop_protocol')
    if mutation == 'width': req['motions'] = req['motions'][:49]
    with pytest.raises(ValueError): native.validate_collection_request(req)


@pytest.mark.parametrize('slow_never_finishes', [False, True])
@pytest.mark.parametrize('batch', [40, 50])
def test_actual_loop_waits_for_slowest_lane_without_reset_or_discard(monkeypatch, slow_never_finishes, batch):
    module = ModuleType('gr00t.rl.envs.base_task.staged_task_base')
    module.StagedTaskBase = type('StagedTaskBase', (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    reset_calls, windows, rng_boundaries = [], [], []
    def reset():
        reset_calls.append(1)
        return {'obs': torch.zeros(batch, 1)}
    owner = SimpleNamespace(env=SimpleNamespace(num_envs=batch, config={}, reset_all=reset,
        episode_length_buf=torch.zeros(batch, dtype=torch.long), max_episode_length=1000),
        num_steps_per_env=81, args=SimpleNamespace(num_total_batches=40, num_ppo_epochs=0, batch_size=batch),
        vg_mode='collect', vg_rollout_action_source='teacher', vg_enabled=True,
        value_model=None, policy_model=SimpleNamespace(is_recurrent=False),
        accelerator=SimpleNamespace(num_processes=1, device='cpu', unwrap_model=lambda x: x),
        is_deepspeed_enabled=False, dataloader=None, schedule_dict=None, train_with_evaluating_env=False,
        config={'init_at_random_ep_len': True}, state=SimpleNamespace(global_step=0, episode=0, tot_timesteps=0),
        model=torch.nn.Linear(1, 1), _register_stats_buffer=lambda: None,
        _train_mode=lambda: rng_boundaries.append(1), ep_infos=[],
        episode_env_tensors=SimpleNamespace(mean_and_clear=lambda: None),
        _bc_ledger=SimpleNamespace(emitted_counts=[0]*batch))
    def window(actor, model, obs):
        i = len(windows) + 1
        actor._bc_ledger.emitted_counts = [i]*(batch-1) + [0 if slow_never_finishes else max(0, i-4)]
        windows.append(deepcopy(actor._bc_ledger.emitted_counts))
        return obs
    monkeypatch.setattr(collector, 'collect_window', window)
    monkeypatch.setattr(collector.gc, 'collect', lambda: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    if slow_never_finishes:
        with pytest.raises(RuntimeError, match='safety window ceiling'):
            collector.collect(owner, 40, target_chunks_per_lane=20)
        assert len(windows) == 40
    else:
        collector.collect(owner, 40, target_chunks_per_lane=20)
        assert len(windows) == 24 and windows[-1] == [24]*(batch-1) + [20]
        assert owner.state.tot_timesteps == 24*81*batch
    assert reset_calls == [1] and len(rng_boundaries) == len(windows)


def test_completion_cannot_claim_target_for_19_chunks():
    req = continuous_plan.build_request(plan())
    report = dict(accepted_windows=25, emitted_per_lane=[20]*49 + [19],
                  collection_implementation=collector.implementation_receipt())
    with pytest.raises(ValueError, match='target not reached'):
        run_continuous.validate_completion(report, req, 'a'*64)


def test_native_and_host_receipts_accept_target_before_ceiling_only_at_saved_barrier():
    from test.pipeline.test_continuous_bc_launcher import fixture_capacity
    req, counts = continuous_plan.build_request(plan()), [24]*49 + [20]
    journal = native.ContinuousCollectionJournal.__new__(native.ContinuousCollectionJournal)
    journal.ledger = SimpleNamespace(request=req, continuous=True, inflight=None, pending=lambda: [],
        vector_steps=24*81, emitted_counts=counts, discarded_tail_rows=[0]*49 + [4*81],
        buffers=[[] for _ in counts], unrecorded_rows=[0]*50, drain=None)
    journal.count, journal.accepted_windows, journal.published_per_lane = sum(counts), 24, counts
    journal.require_completed(native_saved_chunks=sum(counts), native_tail_rows=[0]*50, initial_reset_all_calls=1)
    report = fixture_capacity()['native']
    report.update(request=req, request_sha256='a'*64, completion_reason='all_lanes_chunk_target',
        accepted_windows=24, vector_steps=24*81, emitted_per_lane=counts,
        emitted_chunks=sum(counts), selected_chunks=sum(counts), native_payload_writer_unchanged=False,
        collection_storage=dict(protocol=PROTOCOL, completed_windows=24, num_envs=50,
            frames_per_window=81, logical_scenes=1, cpu_conversion_lanes=1, capture_device='cuda:0',
            production_admitted=False, writer_sha256='b'*64))
    reset = report['initial_state_attestation']['initial_reset_completion']
    reset.update(num_envs=50, cleared_lanes=50)
    assert run_continuous.validate_completion(report, req, 'a'*64) == report
    journal.ledger.inflight = [{}]
    with pytest.raises(RuntimeError, match='pending work'):
        journal.require_completed(native_saved_chunks=sum(counts), native_tail_rows=[0]*50, initial_reset_all_calls=1)
