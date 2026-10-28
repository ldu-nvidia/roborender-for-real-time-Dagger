"""Native-driver state-machine tests with CPU windows, never native qualification."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.bc_training.handoff_common import publish
from pipeline.dagger.coverage_production import make_first_config, make_segment, digest
from test.pipeline.test_dagger_coverage_production import production, rendering_dependencies
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import driver_config, canonical
from pipeline.dagger.rtx.code.src.dagger_rtx.qualification_driver import actor_ids
from pipeline.dagger.rtx.code.src.dagger_rtx.production_segment import (
    ProductionSegmentDriver, validate_segment, validate_assignment_document, read_segment,
)
from pipeline.dagger.rtx.code.src.dagger_rtx.window_runner import RolloutLifecycle


def checkpoint(iteration):
    return 'd' * 64 if iteration == 0 else hashlib.sha256(f'candidate-{iteration}'.encode()).hexdigest()


@pytest.fixture
def inputs(production, tmp_path):
    _, plan = production
    ref = publish(tmp_path / 'coverage.json', plan)
    config = make_first_config(plan, coverage_plan=ref, checkpoint=tmp_path / 'parent.pt',
        checkpoint_sha256=checkpoint(0), gateway_pids=[101, 102, 103, 104],
        rendered_observation_config=rendering_dependencies(tmp_path,plan))
    mse = dict(schema='roborender.dagger-mse-round-protocol.v1', run_id=config['run_id'], round_id=1,
        initial_checkpoint_sha256=checkpoint(0), validation_manifest_path='/shared/heldout.json',
        validation_manifest_sha256='f' * 64, interval_iterations=148, max_iterations=296,
        min_validation_epochs=2, patience=3, relative_improvement=.005, absolute_improvement=1e-6,
        final_test=dict(scenario_manifest_sha256=config['validation']['scenario_manifest_sha256'],
            scenario_ids=['full295-test'], observation_mode='raytraced'))
    return plan, ref, config, mse


def segment(inputs, *, epoch=2, index=0):
    plan, ref, _, mse = inputs
    cohort = plan['rounds'][0]['epochs'][epoch - 1]['cohorts'][index]
    refs = [dict(physical_gpu_index=g, path=f'/native/assignments/g{g}.json', sha256=digest(a))
            for g, a in enumerate(cohort['assignments'])]
    return make_segment(plan, round_id=1, epoch=epoch, index=index,
        input_checkpoint_sha256=checkpoint(cohort['start_iteration']),
        mse_protocol_file_sha256=hashlib.sha256(canonical(mse)).hexdigest(), motion_assignments=refs,
        coverage_plan_sha256=ref['raw_sha256'])


class Coordinator:
    def __init__(self, config, mse, start, *, stop=False, prior_best=0):
        self.config = driver_config(config, 'c' * 64)
        self.mse, self.start, self.stop, self.prior_best = mse, start, stop, prior_best
        self.rows = [dict(round_id=1, iteration_id=i, status='advanced',
            input_checkpoint_sha256=checkpoint(i - 1), candidate_checkpoint_sha256=checkpoint(i))
            for i in range(1, start + 1)]
        self.rows.append(dict(round_id=1, iteration_id=start + 1, status='collecting',
            input_checkpoint_sha256=checkpoint(start), candidate_checkpoint_sha256=None))
        self.parent = checkpoint(start)
        self.barriers = [self.barrier(i, actor) for i in range(1, start + 1) for actor in actor_ids(self.config)]
        self.posts = []
        self.state = 'active'
        self.reason = None
        self.bad_mse = False
        self.missing_barrier = False
        self.wrong_activated_parent = False
        self.history = [dict(iteration=0, result=dict(checkpoint_sha256=checkpoint(0), mse=1.0))]

    def barrier(self, iteration, actor):
        return dict(round_id=1, iteration_id=iteration, actor_id=actor,
            checkpoint_sha256=checkpoint(iteration - 1),
            chunk_ids=[f'd1-{actor}-window-{iteration-1:06d}'])

    def __call__(self, method, path, body=None):
        if path.startswith('/v1/mse-rounds/'):
            best = checkpoint(self.prior_best) if self.state == 'selected_pending_test' else checkpoint(0)
            return copy.deepcopy(dict(ok=True, mse_round=dict(protocol=self.mse,
                protocol_sha256='0' * 64 if self.bad_mse else hashlib.sha256(canonical(self.mse)).hexdigest(),
                run_id=self.config['run_id'], round_id=1, status=self.state, stop_reason=self.reason,
                iteration_id=self.rows[-1]['iteration_id'], checkpoint_sha256=self.rows[-1]['candidate_checkpoint_sha256'] or self.parent,
                selection=dict(history=self.history, validation_manifest_sha256=self.mse['validation_manifest_sha256'],
                               best_checkpoint_sha256=best))))
        if method == 'POST':
            assert path == '/v1/barriers'
            iteration = self.rows[-1]['iteration_id']
            assert body['checkpoint_sha256'] == self.parent
            assert body['chunk_ids'] == [f"d1-{body['actor_id']}-window-{iteration-1:06d}"]
            self.posts.append(copy.deepcopy(body))
            if len(self.posts) % 2 == 0:
                rows = [self.barrier(iteration, actor) for actor in actor_ids(self.config)]
                self.barriers.extend(rows[:-1] if self.missing_barrier else rows)
                current = self.rows[-1]
                current['candidate_checkpoint_sha256'] = checkpoint(iteration)
                if self.stop and iteration == self.start + 2:
                    current['status'] = 'selected_pending_test'
                    self.state = 'selected_pending_test'
                    self.reason = ('maximum_iterations' if iteration == self.mse['max_iterations']
                                   else 'validation_mse_plateau')
                    self.history += [dict(iteration=self.prior_best,
                        result=dict(checkpoint_sha256=checkpoint(self.prior_best), mse=.2)),
                        dict(iteration=iteration, result=dict(checkpoint_sha256=checkpoint(iteration), mse=.3))]
                else:
                    current['status'] = 'advanced'
                    self.parent = checkpoint(iteration)
                    self.rows.append(dict(round_id=1, iteration_id=iteration + 1, status='collecting',
                        input_checkpoint_sha256='a' * 64 if self.wrong_activated_parent else self.parent,
                        candidate_checkpoint_sha256=None))
            return dict(ok=True, expected=4, iteration_id=iteration)
        return copy.deepcopy(dict(ok=True, run=dict(config_sha256='c' * 64, actor_ids=actor_ids(self.config),
            current_round=1, beta=.75, rollout_steps=81, validation_interval_iterations=1480,
            status='selected_pending_test' if self.state == 'selected_pending_test' else 'collecting',
            active_checkpoint_sha256=self.parent), iterations=self.rows, barriers=self.barriers))


def driver(inputs, tmp_path, *, epoch=2, index=0, stop=False, prior_best=0):
    _, _, config, mse = inputs
    item = segment(inputs, epoch=epoch, index=index)
    calls, windows, offsets = [], [], []
    runner = SimpleNamespace(num_envs=2, actor_prefix=f"{config['run_id']}-g0",
        episode_namespace=item['episode_namespace'], lifecycle=RolloutLifecycle(
            begin=lambda: calls.append('begin'), finish=lambda: calls.append('finish')),
        phase_telemetry=SimpleNamespace(set_vector_step_offset=offsets.append))
    def run_window(i, manage_lifecycle):
        assert manage_lifecycle is False
        windows.append(i)
        return dict(transition_batches=81, physical_transition_rows=162)
    runner.run_window = run_window
    control = Coordinator(config, mse, item['start_iteration_index'], stop=stop, prior_best=prior_best)
    obj = ProductionSegmentDriver(runner, control, tmp_path / 'native-segment', config, 'c' * 64, 0,
        segment=item, segment_file_sha256='e' * 64, mse_protocol=mse,
        mse_protocol_file_sha256=item['mse_protocol_file_sha256'], sleep=lambda _: None)
    return obj, control, calls, windows, offsets


def test_nonzero_global_segment_two_windows_one_native_lifecycle(inputs, tmp_path):
    obj, control, calls, windows, offsets = driver(inputs, tmp_path, epoch=2, index=3)
    result = obj.run()
    assert windows == [154, 155] and offsets == [154 * 81, 155 * 81]
    assert calls == ['begin', 'finish'] and result['lifecycle_initializations'] == 1
    assert result['completed_windows'] == 2 and result['transition_batches'] == 162
    assert result['physical_transition_rows'] == 324 and result['history_reset_at_window_boundary'] is False
    assert result['windows'][0]['input_checkpoint_sha256'] == checkpoint(154)
    assert result['windows'][1]['input_checkpoint_sha256'] == checkpoint(155)
    assert result['final_candidate_checkpoint_sha256'] == checkpoint(156)
    assert len(control.posts) == 4


def test_prior_cohort_mse_best_is_valid_at_complete_coverage_boundary(inputs, tmp_path):
    obj, _, calls, windows, _ = driver(inputs, tmp_path, epoch=2, index=73, stop=True, prior_best=148)
    result = obj.run()
    assert windows == [294, 295] and calls == ['begin', 'finish']
    assert result['selection']['selected_checkpoint_sha256'] == checkpoint(148)
    assert result['final_candidate_checkpoint_sha256'] == checkpoint(296)


@pytest.mark.parametrize('fault', ['future', 'current_barrier', 'parent', 'previous_candidate', 'stale_mse'])
def test_stale_or_started_segment_rejected_before_native_begin(inputs, tmp_path, fault):
    obj, control, calls, windows, _ = driver(inputs, tmp_path)
    if fault == 'future': control.rows.append(copy.deepcopy(control.rows[-1]))
    if fault == 'current_barrier': control.barriers.append(control.barrier(149, actor_ids(control.config)[0]))
    if fault == 'parent': control.rows[-1]['input_checkpoint_sha256'] = checkpoint(0)
    if fault == 'previous_candidate': control.rows[-2]['candidate_checkpoint_sha256'] = checkpoint(0)
    if fault == 'stale_mse': control.bad_mse = True
    with pytest.raises(RuntimeError): obj.run()
    assert 'begin' not in calls and not windows and not control.posts
    assert obj.state['phase'] == 'CRASH_UNCERTAIN'


@pytest.mark.parametrize('fault', ['missing_barrier', 'wrong_activated_parent'])
def test_all_actor_lineage_required_before_second_window(inputs, tmp_path, fault):
    obj, control, calls, windows, _ = driver(inputs, tmp_path)
    setattr(control, fault, True)
    with pytest.raises(RuntimeError): obj.run()
    assert windows == [148] and calls == ['begin', 'finish']
    assert obj.state['phase'] == 'CRASH_UNCERTAIN'


def test_existing_segment_state_forbids_simulator_restart(inputs, tmp_path):
    obj, _, _, _, _ = driver(inputs, tmp_path)
    obj.run()
    with pytest.raises(RuntimeError, match='restart forbidden'): driver(inputs, tmp_path)


def test_exact_segment_assignment_and_raw_file_binding(inputs, tmp_path):
    plan, ref, config, _ = inputs
    item = segment(inputs, epoch=2, index=3)
    assert validate_segment(item, config, plan) == item
    path = tmp_path / 'segment.json'; raw = canonical(item) + b'\n'; path.write_bytes(raw)
    assert read_segment(path, hashlib.sha256(raw).hexdigest(), config, ref['path']) == item
    cohort = plan['rounds'][0]['epochs'][1]['cohorts'][3]
    assert validate_assignment_document(item, plan, 1, cohort['assignments'][1]) == cohort['assignments'][1]
    with pytest.raises(ValueError): validate_assignment_document(item, plan, 0, cohort['assignments'][1])
    path.write_bytes(raw + b' ')
    with pytest.raises(ValueError, match='bytes changed'):
        read_segment(path, hashlib.sha256(raw).hexdigest(), config, ref['path'])


@pytest.mark.parametrize('field,value', [('cohort_id', 'arbitrary-cohort'), ('start_iteration_index', 155),
                                      ('episode_namespace', 'prod-r1-e02-c004'), ('window_count', 1)])
def test_segment_identity_changes_rejected(inputs, field, value):
    plan, _, config, _ = inputs
    item = segment(inputs, epoch=2, index=3); item[field] = value
    with pytest.raises(ValueError): validate_segment(item, config, plan)
