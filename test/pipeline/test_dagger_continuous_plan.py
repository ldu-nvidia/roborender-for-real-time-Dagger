"""CPU-only continuous-round contracts; never native qualification evidence."""
from copy import deepcopy
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pipeline.bc_training.handoff_common import bound, publish
from pipeline.dagger.coverage_production import (
    CONTINUOUS_FIXED100_SCHEMA, FIXED100_SCHEMA, digest, make_plan,
    make_continuous_segment, make_segment, validate_plan,
    stage, validate_continuous_qualification,
)
from pipeline.dagger import coverage_queue
from pipeline.dagger.coverage_campaign import CoverageMSECampaign, validate_continuous_round_completion
from test.pipeline.test_dagger_coverage_production import production, rendering_dependencies
from test.pipeline.test_dagger_scaled_queue import scaled_queue


@pytest.fixture
def continuous_plan(production):
    _, old = production
    return make_plan(**{key: old[key] for key in (
        'run_id', 'inventory', 'panel', 'excluded_process_seeds', 'seed', 'motion_root',
        'source_sha256', 'teacher_sha256', 'policy_spec_sha256')},
        environments_per_actor=16, fixed100=True, continuous_round=True)


def test_continuous_plan_has_one_lifecycle_not_fifty_reset_cohorts(continuous_plan):
    plan = continuous_plan
    assert validate_plan(plan) == plan
    assert plan['schema'] == CONTINUOUS_FIXED100_SCHEMA
    assert plan['native_lifecycles_per_actor_per_round'] == 1
    assert plan['max_updates_per_round'] == plan['optimizer_updates_per_round'] == 100
    assert not {'stopping', 'maximum_coverage_epochs', 'updates_per_coverage_epoch',
                'cohorts_per_coverage_epoch', 'windows_per_native_cohort'} & set(plan)
    expected = [row['motion_id'] for row in bound(plan['inventory'])['motions'][:32]]
    assert plan['initial_motion_support'] == dict(selection='first32_inventory_ordered',
        motion_ids=expected, motion_count=32, full_inventory_coverage_guaranteed=False)
    seeds = []
    for rid, row in enumerate(plan['rounds'], 1):
        assert 'epochs' not in row and 'heldout' not in row
        native = row['native_round']
        assert native['window_count'] == 100 and native['start_iteration'] == 0
        assert native['qualification_after_updates'] == 4
        assert native['lifecycle_policy'] == 'native_episode_done_only'
        assert native['model_rollout_lifecycle'] == 'per_81_step_window'
        assert native['reset_at_window_boundary'] is False
        assert [lane['motion_id'] for a in native['assignments'] for lane in a['lanes']] == expected
        assert [len(a['lanes']) for a in native['assignments']] == [16, 16]
        seeds.extend(a['simulator_seed'] for a in native['assignments'])
        assert row['teacher_beta'] == [.75, .5, .25, 0][rid-1]
        assert row['success_test']['completed_episodes'] == 295
    assert len(set(seeds)) == 8 and not set(seeds) & set(plan['excluded_process_seeds'])
    assert plan['production_admitted'] is False


def test_legacy_fixed100_plan_preserves_two_window_recipe(continuous_plan):
    plan = make_plan(**{key: continuous_plan[key] for key in (
        'run_id', 'inventory', 'panel', 'excluded_process_seeds', 'seed', 'motion_root',
        'source_sha256', 'teacher_sha256', 'policy_spec_sha256')},
        environments_per_actor=16, fixed100=True)
    assert plan['schema'] == FIXED100_SCHEMA
    assert validate_plan(plan) == plan
    assert all(len(row['epochs']) == 5 and 'native_round' not in row for row in plan['rounds'])
    assert plan['windows_per_native_cohort'] == 2


@pytest.mark.parametrize('fault', ['reset', 'small_window', 'coverage_claim', 'different_support',
                                 'model_lifecycle', 'legacy_schema'])
def test_continuous_semantics_cannot_be_relabelled(continuous_plan, fault):
    bad = deepcopy(continuous_plan)
    native = bad['rounds'][0]['native_round']
    if fault == 'reset': native['reset_at_window_boundary'] = True
    elif fault == 'small_window': native['window_count'] = 2
    elif fault == 'coverage_claim': bad['initial_motion_support']['full_inventory_coverage_guaranteed'] = True
    elif fault == 'different_support': native['assignments'][0]['lanes'].reverse()
    elif fault == 'model_lifecycle': native['model_rollout_lifecycle'] = 'per_round'
    else: bad['schema'] = FIXED100_SCHEMA
    with pytest.raises(ValueError):
        validate_plan(bad)


def test_continuous_segment_is_one_hundred_windows_and_rejects_old_dispatch(continuous_plan, tmp_path):
    plan = continuous_plan
    ref = publish(tmp_path/'plan.json', plan)
    assignments = [dict(physical_gpu_index=gpu, path=f'/native/g{gpu}.json', sha256=digest(a))
                   for gpu, a in enumerate(plan['rounds'][0]['native_round']['assignments'])]
    segment = make_continuous_segment(plan, round_id=1, input_checkpoint_sha256='d'*64,
        mse_protocol_file_sha256='e'*64, motion_assignments=assignments,
        coverage_plan_sha256=ref['raw_sha256'])
    assert segment['schema'] == 'roborender.dagger-continuous-round.v1'
    assert segment['window_count'] == 100 and segment['qualification_after_updates'] == 4
    assert segment['coverage_plan_sha256'] == ref['raw_sha256'] != digest(plan)
    assert segment['episode_namespace'] == 'continuous-r1'
    assert not {'coverage_epoch', 'cohort_index'} & set(segment)
    with pytest.raises(ValueError, match='reset-cohort'):
        make_segment(plan, round_id=1, epoch=1, index=0, input_checkpoint_sha256='d'*64,
            mse_protocol_file_sha256='e'*64, motion_assignments=assignments,
            coverage_plan_sha256=ref['raw_sha256'])


def test_continuous_queue_is_distinct_and_not_legacy_qualification_only(scaled_queue):
    queue = deepcopy(scaled_queue)
    queue['schema'] = coverage_queue.CONTINUOUS_FIXED100_QUEUE_SCHEMA
    assert coverage_queue.validate_queue(queue) == queue
    with pytest.raises(ValueError, match='fresh scaled queue v3'):
        coverage_queue.admit_execution_scope(queue, 'a'*64, qualification_only=True)


def test_continuous_campaign_never_finishes_collect_at_qualification_prefix(continuous_plan, tmp_path):
    staged = stage(continuous_plan, output_root=tmp_path/'plan', checkpoint=tmp_path/'parent.pt',
        checkpoint_sha256='d'*64, rendered_observation_config=rendering_dependencies(tmp_path, continuous_plan))
    cfg, schedule = bound(staged['first_round']), bound(staged['mse_plan'])
    ops = Mock(); ops.start.return_value = 'same-continuous-command'; ops.poll.return_value = None
    controller = Mock(); controller.tick.return_value = {'iteration': 4, 'status': 'collecting'}
    campaign = CoverageMSECampaign(cfg, tmp_path/'campaign', ops, plan=schedule)
    campaign.save('COLLECT_TRAIN', config_path=staged['first_round']['path'])
    with patch.object(campaign, 'get_controller', return_value=controller):
        assert campaign.tick()['operation_id'] == 'same-continuous-command'
        assert campaign.tick()['operation_id'] == 'same-continuous-command'
    assert ops.start.call_count == 1
    context = ops.start.call_args.args[1]
    assert context['continuous_native_round'] is True and context['qualification_after_updates'] == 4
    assert 'production_cohort_limit' not in context
    assert campaign.state['phase'] == 'COLLECT_TRAIN'


@pytest.fixture
def continuous_evidence(continuous_plan, tmp_path):
    """Synthetic protocol fixtures only, never live qualification/readiness."""
    plan = continuous_plan
    ref = publish(tmp_path/'plan.json', plan)
    assignments = [dict(physical_gpu_index=gpu, path=f'/native/g{gpu}.json', sha256=digest(a))
                   for gpu, a in enumerate(plan['rounds'][0]['native_round']['assignments'])]
    segment = make_continuous_segment(plan, round_id=1, input_checkpoint_sha256='d'*64,
        mse_protocol_file_sha256='e'*64, motion_assignments=assignments, coverage_plan_sha256=ref['raw_sha256'])
    windows = []; learners = []; iterations = []; parent = 'd'*64
    for index in range(100):
        candidate = hashlib.sha256(f'CPU-only-candidate-{index}'.encode()).hexdigest()
        windows.append(dict(index=index, input_checkpoint_sha256=parent, output_checkpoint_sha256=candidate))
        learners.append(publish(tmp_path/f'learner-{index}.json', dict(
            schema='roborender.dagger_learner_candidate_receipt.v3',
            parent_checkpoint_sha256=parent, output_checkpoint_sha256=candidate,
            train_chunks=32, train_frames=2592, replay_manifest_sha256=None,
            batching_semantics='raven_exact_flattened_bxt', optimizer_state_preserved=True,
            parent_global_step=1000+index, global_step=1001+index, train_mse=.123, iteration_id=index+1)))
        iterations.append(dict(round_id=1, iteration_id=index+1, input_checkpoint_sha256=parent,
            candidate_checkpoint_sha256=candidate, status='advanced' if index<99 else 'selected_pending_test'))
        parent = candidate
    guard = dict(schema='roborender.native-phase-resource-guard.v1', passed=True, samples=100,
        minimum_host_available_bytes=9<<30, swap_peak_bytes=0, oom_events=0, limit_pressure_events=0)
    prefixes = []; terminals = []
    for gpu in (0, 1):
        row = dict(schema='roborender.dagger-continuous-native-prefix.v1', physical_gpu=gpu,
            worker_instance_id=str(gpu+1)*32, native_pid=100+gpu, command_id=f'continuous-g{gpu}',
            source_sha256=plan['source_sha256'], segment_sha256=digest(segment),
            motion_assignment_sha256=assignments[gpu]['sha256'], resource_guard=guard,
            actor_state=dict(phase='WAITING_FOR_QUALIFICATION', physical_gpu=gpu, native_pid=100+gpu,
                uncertain_reason=None, completed_windows=4, transition_batches=324,
                physical_transition_rows=5184, native_environment_initializations=1,
                model_rollout_initializations=4, history_reset_at_window_boundary=False,
                segment=segment, segment_file_sha256=digest(segment), windows=windows[:4]))
        prefixes.append(publish(tmp_path/f'prefix-{gpu}.json', row))
        terminal = deepcopy(row); terminal['schema'] = 'roborender.dagger-continuous-native-terminal.v1'
        terminal['actor_state'].update(phase='COMPLETE', completed_windows=100,
            transition_batches=8100, physical_transition_rows=129600,
            model_rollout_initializations=100, windows=windows)
        terminals.append(publish(tmp_path/f'terminal-{gpu}.json', terminal))
    assignments_refs = []
    for gpu, assignment in enumerate(plan['rounds'][0]['native_round']['assignments']):
        for lane in assignment['lanes']:
            episode = f'CPU-only-{gpu}-{lane["lane"]}'
            row = publish(tmp_path/f'{episode}.json', dict(schema='roborender.native-episode-assignment.v1',
                purpose='training', simulator_seed=assignment['simulator_seed'], motion_id=lane['motion_id'],
                motion_sha256=lane['sha256'], episode_id=episode))
            reset = publish(tmp_path/f'{episode}-reset.json', dict(schema='roborender.native-episode-reset-evidence.v1',
                episode_id=episode, assignment_sha256=digest(assignment)))
            assignments_refs.append(dict(**row, reset_evidence=reset))
    metrics = publish(tmp_path/'metrics.json', dict(schema='roborender.dagger-continuous-online-prefix.v1',
        run_id=plan['run_id'], round_id=1, wandb_run_id=plan['rounds'][0]['wandb_run_id'],
        source_sha256=plan['source_sha256'], checkpoint_sha256=windows[3]['output_checkpoint_sha256'],
        iteration_id=4, train_mse=.123, online_metrics_verified=True))
    proof = dict(schema='roborender.dagger-continuous-prefix-qualification.v1', purpose='qualification',
        run_id=plan['run_id'], round_id=1, plan_sha256=digest(plan), parent_checkpoint_sha256='d'*64,
        source_sha256=plan['source_sha256'], optimizer_updates=4, full_batch_environments=32,
        native_prefix_receipts=prefixes, learner_receipts=learners[:4], allocation_retained=True,
        h200_uuids=[f'CPU-only-H{i}' for i in range(4)], native_assignment_receipts=assignments_refs,
        online_metrics_verified=True, online_metrics_receipt=metrics)
    return dict(plan=plan, proof=proof, terminals=terminals, iterations=iterations, final=parent)


def test_actual_bound_prefix_required_not_old_two_cohort_certificate(continuous_evidence, tmp_path):
    e = continuous_evidence
    assert validate_continuous_qualification(e['plan'], e['proof'], 'd'*64) == e['proof']
    old = dict(e['proof'], schema='roborender.dagger-full295-two-cohort-qualification.v2')
    with pytest.raises(ValueError, match='legacy cohorts'):
        validate_continuous_qualification(e['plan'], old, 'd'*64)


@pytest.mark.parametrize('fault', ['owner', 'reset', 'missing_window', 'wrong_pixels_lineage',
                                 'no_initial_evidence', 'metrics', 'optimizer'])
def test_prefix_proof_fails_closed(continuous_evidence, tmp_path, fault):
    e = continuous_evidence; proof = deepcopy(e['proof'])
    if fault in ('owner', 'reset', 'missing_window', 'wrong_pixels_lineage'):
        row = bound(proof['native_prefix_receipts'][0])
        if fault == 'owner': row['native_pid'] += 1
        elif fault == 'reset': row['actor_state']['native_environment_initializations'] = 2
        elif fault == 'missing_window': row['actor_state']['windows'].pop()
        else: row['actor_state']['windows'][2]['input_checkpoint_sha256'] = 'c'*64
        proof['native_prefix_receipts'][0] = publish(tmp_path/'bad-prefix.json', row)
    elif fault == 'no_initial_evidence': proof['native_assignment_receipts'].pop()
    elif fault == 'metrics': proof['online_metrics_verified'] = False
    else: proof['learner_receipts'][2] = proof['learner_receipts'][1]
    with pytest.raises((ValueError, RuntimeError)):
        validate_continuous_qualification(e['plan'], proof, 'd'*64)


def test_final_proof_rejects_restarted_owner_and_preserves_full_lineage(continuous_evidence, tmp_path):
    from pipeline.dagger.coverage_production import make_first_config
    from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import actor_ids, LEO_GPU_UUIDS
    e = continuous_evidence; plan = e['plan']
    config = make_first_config(plan, coverage_plan=publish(tmp_path/'final-plan.json', plan),
        checkpoint=tmp_path/'parent.pt', checkpoint_sha256='d'*64,
        rendered_observation_config=rendering_dependencies(tmp_path, plan))
    context = dict(status='selected_pending_test', iteration_id=100, stop_reason='fixed_100_updates',
        checkpoint_sha256=e['final'], selection=dict(checkpoint_selection='final_update_100',
            best_checkpoint_sha256=e['final'], history=[dict(iteration=100,
                result=dict(checkpoint_sha256=e['final']))]))
    receipt = dict(schema='roborender.dagger-continuous-collection-result.v1', run_id=plan['run_id'], round_id=1,
        source_sha256=plan['source_sha256'], coverage_plan_sha256=digest(plan), completed_updates=100,
        native_lifecycles_per_actor=[1,1], model_rollout_initializations_per_actor=[100,100],
        history_reset_at_window_boundary=False, actor_ids=actor_ids(config),
        last_candidate_checkpoint_sha256=e['final'], selected_checkpoint_sha256=e['final'],
        allocation_retained=True, qualification=publish(tmp_path/'proof.json', e['proof']),
        native_terminal_receipts=e['terminals'], native_phase_barrier=dict(
            schema='roborender.isaac-phase-barrier.v1', workers=[dict(physical_gpu=gpu, gpu_uuid=uuid,
                worker_instance_id=str(gpu+1)*32, source_sha256=plan['source_sha256'], phase='IDLE')
                for gpu, uuid in LEO_GPU_UUIDS.items()]))
    result = validate_continuous_round_completion(config, receipt, e['iterations'], context, plan)
    assert result['completed_iterations'] == 100 and 'completed_coverage_epochs' not in result
    changed = bound(receipt['native_terminal_receipts'][0]); changed['worker_instance_id'] = 'f'*32
    receipt['native_terminal_receipts'][0] = publish(tmp_path/'changed-terminal.json', changed)
    with pytest.raises(ValueError, match='changed after continuous qualification'):
        validate_continuous_round_completion(config, receipt, e['iterations'], context, plan)
