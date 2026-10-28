"""CPU telemetry contracts; never simulator or capacity qualification."""
from test.deployment.source_paths import SUCCESS_RATE_ROOT
from test.deployment.source_paths import ACTOR_ROOT
import ast
from copy import deepcopy
import math
from pathlib import Path

import pytest

from pipeline.deployment.rtx_actor.success_cycle_timing import (
    METADATA, PHASES, SuccessCycleTimings, validate_receipt,
)


def completed(clock_values=(10., 11., 13., 16., 20., 25.)):
    ticks = iter(clock_values)
    timings = SuccessCycleTimings(clock=lambda: next(ticks))
    cycle = timings.start(0)
    for phase in PHASES[:-1]:
        cycle.mark(phase)
    row = timings.complete(cycle)
    return timings, row


def test_exclusive_host_wall_phases_telescope_without_cuda_or_extra_calls():
    timings, row = completed()
    assert row == dict(step=0, **dict(zip(PHASES, (1., 2., 3., 4., 5.))), total_seconds=15.)
    receipt = timings.receipt(expected_steps=1)
    assert receipt['totals'] == {key: row[key] for key in (*PHASES, 'total_seconds')}
    assert receipt['cuda_synchronization_added'] is False
    assert receipt['timing_kind'] == 'exclusive_host_wall_not_gpu_kernel_time'
    assert 'later phase' in receipt['asynchronous_cuda_caveat']
    assert 'camera work' in receipt['capture_control_scope']
    assert 'reference-teacher' in receipt['student_homie_scope']
    assert 'excludes timing telemetry overhead' in receipt['measurement_boundary']
    row['env_step_seconds'] = 99
    receipt['steps'][0]['total_seconds'] = 99
    assert timings.receipt(expected_steps=1)['steps'][0]['total_seconds'] == 15.


def test_two_sequential_cycles_do_not_include_between_cycle_gap():
    ticks = iter((0., 1., 2., 3., 4., 5., 100., 102., 104., 106., 108., 110.))
    timings = SuccessCycleTimings(clock=lambda: next(ticks))
    for step in range(2):
        cycle = timings.start(step)
        for phase in PHASES[:-1]:
            cycle.mark(phase)
        timings.complete(cycle)
    receipt = timings.receipt(expected_steps=2)
    assert receipt['totals']['total_seconds'] == 15.
    assert [row['total_seconds'] for row in receipt['steps']] == [5., 10.]


@pytest.mark.parametrize('invalid', [True, -1., math.nan, math.inf, '1'])
def test_invalid_host_clock_cannot_be_reported_as_measurement(invalid):
    with pytest.raises(RuntimeError, match='timing unavailable'):
        SuccessCycleTimings(clock=lambda: invalid).start(0)


def test_regressed_clock_and_reordered_or_duplicate_phases_fail():
    ticks = iter((2., 1.))
    cycle = SuccessCycleTimings(clock=lambda: next(ticks)).start(0)
    with pytest.raises(RuntimeError, match='regressed'):
        cycle.mark(PHASES[0])
    cycle = SuccessCycleTimings(clock=lambda: 0.).start(0)
    with pytest.raises(RuntimeError, match='ordered'):
        cycle.mark(PHASES[1])
    cycle.mark(PHASES[0])
    with pytest.raises(RuntimeError, match='ordered'):
        cycle.mark(PHASES[0])


def test_partial_duplicate_or_foreign_cycle_never_publishes_complete_receipt():
    timings = SuccessCycleTimings(clock=lambda: 0.)
    with pytest.raises(RuntimeError, match='sequential'):
        timings.start(1)
    cycle = timings.start(0)
    with pytest.raises(RuntimeError, match='sequential'):
        timings.start(0)
    with pytest.raises(RuntimeError, match='completed vector cycle'):
        timings.complete(cycle)
    with pytest.raises(RuntimeError, match='partial'):
        timings.receipt(expected_steps=1)
    for phase in PHASES[:-1]:
        cycle.mark(phase)
    foreign = SuccessCycleTimings(clock=lambda: 0.).start(0)
    with pytest.raises(RuntimeError, match='completed vector cycle'):
        timings.complete(foreign)
    timings.complete(cycle)
    with pytest.raises(RuntimeError, match='completed vector cycle'):
        timings.complete(cycle)
    with pytest.raises(RuntimeError, match='sequential'):
        timings.start(0)
    with pytest.raises(ValueError, match='scope'):
        timings.receipt(expected_steps=2)


@pytest.mark.parametrize('change', [
    lambda value: value.update(cuda_synchronization_added=True),
    lambda value: value.update(vector_steps_completed=2),
    lambda value: value['steps'][0].update(step=1),
    lambda value: value['steps'][0].update(total_seconds=16.),
    lambda value: value['steps'][0].update(env_step_seconds=-1.),
    lambda value: value['steps'][0].update(env_step_seconds=math.nan),
    lambda value: value['totals'].update(total_seconds=1.),
])
def test_changed_measurement_receipt_fails(change):
    timings, _ = completed()
    receipt = deepcopy(timings.receipt(expected_steps=1))
    change(receipt)
    with pytest.raises(ValueError):
        validate_receipt(receipt, expected_steps=1)


def test_actual_native_loop_keeps_one_render_policy_and_step_in_original_order():
    path = (SUCCESS_RATE_ROOT / "evaluation_loop.py")
    tree = ast.parse(path.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SuccessDiagnosticTrainer')
    train = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
    loop = next(node for node in ast.walk(train) if isinstance(node, ast.For)
                and ast.unparse(node.target) == 'step')
    # The explicit new pooled protocol owns a larger fail-incomplete budget;
    # fixed295's existing branch still evaluates to exactly450 vector steps.
    expression = compile(ast.Expression(body=loop.iter), '<native-loop-budget>', 'eval')
    assert len(eval(expression, {'pooled': None})) == 450
    assert len(eval(expression, {'pooled': {'max_vector_steps': 900}})) == 900
    calls = sorted((node for node in ast.walk(loop) if isinstance(node, ast.Call)),
                   key=lambda node: (node.lineno, node.col_offset))
    names = [ast.unparse(node.func) for node in calls]
    for name in ('apply_observation_transform', 'select_policy_actions', 'self.env.step'):
        assert names.count(name) == 1
    assert not any(name.endswith('.synchronize') or name in {
        'self.env.reset_all', 'self.env.reset_panel', 'self.env.reset',
        'self.env.reset_dataset.resample', 'torch.manual_seed', 'torch.cuda.manual_seed_all'}
        for name in names)
    observed = [(ast.unparse(node.func), node.args[0].value if ast.unparse(node.func) == 'cycle.mark' else None)
                for node in calls if ast.unparse(node.func) in {
                    'cycle_timings.start', 'cycle.mark', 'apply_observation_transform',
                    'select_policy_actions', 'self.env.step', 'trace.write', 'emit', 'cycle_timings.complete'}]
    assert observed == [
        ('cycle_timings.start', None), ('cycle.mark', PHASES[0]),
        ('apply_observation_transform', None), ('cycle.mark', PHASES[1]),
        ('select_policy_actions', None), ('cycle.mark', PHASES[2]),
        ('self.env.step', None), ('cycle.mark', PHASES[3]),
        ('trace.write', None), ('emit', None), ('cycle_timings.complete', None)]
    source = ast.unparse(train)
    assert "getattr(self, '_scaled_initialization_distribution', None) is not None" in source
    assert "result['scaled_cycle_timing'] = cycle_timings.receipt(expected_steps=step + 1)" in source
    helper = ast.parse(path.with_name('episode_metrics.py').read_text())
    imports = [node.module for node in ast.walk(helper) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(helper) if isinstance(node, ast.Import) for alias in node.names]
    # The consolidated module also owns the CPU-only report comparison CLI.
    # Keep the no-Torch/Isaac/network import boundary, including its stdlib I/O.
    assert set(imports) <= {'__future__', 'copy', 'math', 'time', 'dataclasses',
                           'typing', 'collections.abc', 'json', 'argparse', 'pathlib'}
    assert METADATA['policy_reset_rng_or_step_semantics_changed'] is False
