"""Bounded diagnostic plan/dispatch; no native or GPU imports."""
from test.deployment.source_paths import ACTOR_ROOT
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(ACTOR_ROOT))
from scene_pressure_diagnostic import SCHEMA, EARLY_CAPTURE_SCHEMA, canonical, pressure_snapshot, read_plan, validate_plan
from test.deployment.rtx_actor.test_warm_camera_diagnostic import plan as warm_plan
from warm_camera_diagnostic import run_stages


def plan():
    reference = warm_plan()
    stages = [dict(assignment={**deepcopy(reference['arms']['0'][0]['assignment']),
                              'simulator_seed': index + 100}, steps=3 if index < 19 else 324)
              for index in range(20)]
    return dict(schema=SCHEMA, inventory=reference['inventory'],
                arms={'0': stages, '1': deepcopy(stages)})


def test_exact_twenty_scenes_and_381_steps_per_actor():
    value = validate_plan(plan())
    assert [sum(s['steps'] for s in arm) for arm in value['arms'].values()] == [381, 381]
    reports = []
    result = run_stages(value['arms']['0'], switch=lambda _: None, report=reports.append,
        execute=lambda steps, report: dict(result='PASS', environment_step_calls=steps,
            physical_transition_rows=16*steps, optimizer_steps=0, chunks_published=0))
    assert len(result) == 20 and reports[-1]['completed_vector_steps'] == 381


def early_capture_plan():
    value = plan()
    value['schema'] = EARLY_CAPTURE_SCHEMA
    for stages in value['arms'].values():
        for stage in stages[:-1]:
            stage['steps'] = 10
    return value


def test_v2_covers_ten_early_captures_per_scene_without_changing_v1():
    value = validate_plan(early_capture_plan())
    assert [sum(s['steps'] for s in arm) for arm in value['arms'].values()] == [514, 514]
    assert sum(s['steps'] for s in validate_plan(plan())['arms']['0']) == 381
    calls = []
    results = run_stages(value['arms']['0'], switch=lambda _: None, report=lambda _: None,
        execute=lambda steps, report: calls.append(steps) or dict(result='PASS',
            environment_step_calls=steps, physical_transition_rows=16*steps,
            optimizer_steps=0, chunks_published=0))
    assert calls == [10]*19 + [324] and len(results) == 20


@pytest.mark.parametrize('steps', [3, 9, 11, 81, True])
def test_v2_cannot_silently_shorten_or_expand_early_capture_budget(steps):
    value = early_capture_plan(); value['arms']['1'][0]['steps'] = steps
    with pytest.raises(ValueError): validate_plan(value)


def test_v2_failure_does_not_retry_or_advance_scene():
    calls = []
    def execute(steps, report):
        calls.append(steps)
        raise RuntimeError('native pressure failure')
    with pytest.raises(RuntimeError):
        run_stages(early_capture_plan()['arms']['0'], switch=lambda _: None,
                   execute=execute, report=lambda _: None)
    assert calls == [10]


def test_pressure_snapshot_is_small_read_only_and_does_not_invent_counters(tmp_path):
    (tmp_path / 'pressure').mkdir()
    text = 'compact_stall 12\nallocstall_movable 3\npgscan_direct 45\nnr_free_pages 900\n'
    (tmp_path / 'vmstat').write_text(text)
    (tmp_path / 'pressure/memory').write_text(
        'some avg10=3.14 avg60=1.0 avg300=0.1 total=1234\n'
        'full avg10=2.0 avg60=1.0 avg300=0.1 total=1000\n')
    result = pressure_snapshot(tmp_path)
    assert result['vmstat'] == dict(compact_stall=12, allocstall_movable=3, pgscan_direct=45)
    assert result['host_memory_pressure']['some'] == dict(total_us=1234, avg10=3.14)
    assert result['monotonic_ns'] > 0
    assert (tmp_path / 'vmstat').read_text() == text
    (tmp_path / 'pressure/memory').write_text('some avg10=0 total=0\n')
    with pytest.raises(ValueError): pressure_snapshot(tmp_path)


@pytest.mark.parametrize('mutate', [
    lambda p: p.update(schema='unknown'),
    lambda p: p['arms']['0'].pop(),
    lambda p: p['arms']['0'][0].update(steps=10),
    lambda p: p['arms']['0'][-1].update(steps=3),
    lambda p: p['arms']['0'][0]['assignment'].update(purpose='test'),
    lambda p: p['arms']['0'][0]['assignment']['lanes'].pop(),
    lambda p: p['arms']['0'][0]['assignment']['lanes'][0].update(sha256='f'*64),
    lambda p: p['inventory']['motions'][0].update(bytes=5),
    lambda p: p['arms']['0'][1].update(assignment=deepcopy(p['arms']['0'][0]['assignment'])),
])
def test_changed_inputs_cannot_relax_diagnostic(mutate):
    value = plan(); mutate(value)
    with pytest.raises(ValueError): validate_plan(value)


def test_pinned_regular_plan(tmp_path):
    path = tmp_path/'plan.json'; path.write_bytes(canonical(plan()))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert read_plan(path, digest)['schema'] == SCHEMA
    with pytest.raises(ValueError): read_plan(path, 'f'*64)


def test_failed_stage_never_advances_or_retries():
    calls = []
    def execute(steps, report):
        calls.append(steps)
        raise RuntimeError('pressure guard failed')
    with pytest.raises(RuntimeError):
        run_stages(plan()['arms']['0'], switch=lambda _: None, execute=execute, report=lambda _: None)
    assert calls == [3]


def test_resident_dispatch_is_inside_standard_production_guard():
    import ast
    tree = ast.parse((ACTOR_ROOT / 'resident_isaac_trainer.py').read_text())
    backend = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ResidentRavenBackend')
    methods = {n.name: n for n in backend.body if isinstance(n, ast.FunctionDef)}
    # Scene pressure stays guarded. Continuous probes share the production
    # segment's independent guard; they are not a scene-pressure exemption.
    # IDLE and the separately guarded BC adapter now precede this branch.
    # Find the actual production guard rather than assuming it is statement0.
    branches = methods['_run_guarded'].body
    guarded = [node for node in branches if isinstance(node, ast.If)
               and 'RESIDENT_PRODUCTION_FULL295' in ast.unparse(node.test)]
    assert len(guarded) == 1
    before = branches[:branches.index(guarded[0])]
    assert [ast.unparse(node.test) for node in before] == [
        "operation == 'idle'", "operation == 'collect_bc_continuous'"]
    assert 'self.validate(operation, payload)' in ast.unparse(before[1])
    assert 'self.bc_collection_adapter.run(payload, report)' in ast.unparse(before[1])
    condition = guarded[0].test
    excluded = next(node for node in ast.walk(condition)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
        and node.left.id == 'operation' and isinstance(node.ops[0], ast.NotIn))
    exemptions = set(ast.literal_eval(excluded.comparators[0]))
    assert exemptions == {'shutdown', 'collect_production_segment', 'probe_continuous_training'}
    assert 'diagnose_scene_pressure' not in exemptions
    guard = ast.unparse(methods['_run_guarded'])
    assert 'PhaseResourceGuard' in guard
    assert "'diagnose_scene_pressure'" in ast.unparse(methods['_run_operation'])
    assert 'run_native' in ast.unparse(methods['_run_scene_pressure_diagnostic'])
