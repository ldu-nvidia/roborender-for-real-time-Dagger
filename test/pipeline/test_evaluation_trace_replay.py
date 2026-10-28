"""CPU helper tests; these synthetic events are not historical/live evidence."""
from copy import deepcopy
import json

import pytest

from test.deployment.rtx_actor import evaluation_trace_replay as replay
from test.deployment.source_paths import REPO_ROOT
from test.pipeline.test_scaled_teacher_dispatch import accepted, plan
from test.pipeline.test_scaled_teacher_benchmark import native_result


REFERENCE = REPO_ROOT / 'runtime/raven_gr00t_collection_94d7985/gr00t/rl'


@pytest.fixture(scope='module')
def oracle():
    return replay.load_oracle(REFERENCE)


def scripted_events():
    events = []
    for step, (goals, dones) in enumerate([
        ([False, True], [1, 0]), ([True, False], [1, 0]), ([True, False], [1, 1])
    ]):
        lanes = [dict(current_last_stage_completed_task_buf=goals[i], reset_buf=dones[i],
            time_out_buf=dones[i] == 1 and i == 0, episode_reward=float(2+i+step)) for i in range(2)]
        completed = []
        if step == 0:
            completed = [dict(lanes[0], lane=0, steps=1, native_success=False,
                              exported_success=False, timeout=True)]
        if step == 2:
            completed = [dict(lanes[1], lane=1, steps=3, native_success=True,
                              exported_success=True, timeout=False)]
        events.append(dict(step=step, arm='teacher', lanes=lanes, exported_goals=goals,
                           completed_outcomes=completed))
    return events


def test_real_original_callbacks_filter_first_failure_and_latch_earlier_goal(oracle):
    events = scripted_events()
    before = deepcopy(events)
    value = replay.replay_events(events, 2, oracle)
    assert value['vector_steps'] == 3
    assert value['original_completed'] == 2 and value['original_successes'] == 1
    assert value['original_completion_lane_order'] == [0, 1]
    assert [r['exported_success'] for r in value['accounting']['outcomes']] == [False, True]
    assert [r['episode_reward'] for r in value['accounting']['outcomes']] == [2., 5.]
    assert events == before


@pytest.mark.parametrize('fault', ['truncated', 'extra', 'missing_lane', 'step', 'arm',
    'goal_type', 'native_goal', 'done_type', 'timeout_type', 'nan_reward', 'summary', 'missing_raw_goal'])
def test_raw_trace_corruption_fails_closed(oracle, fault):
    events = scripted_events()
    if fault == 'truncated': events.pop()
    elif fault == 'extra': events.append(deepcopy(events[-1]))
    elif fault == 'missing_lane': events[0]['lanes'].pop()
    elif fault == 'step': events[1]['step'] = 7
    elif fault == 'arm': events[0]['arm'] = 'bc'
    elif fault == 'goal_type': events[0]['exported_goals'][0] = 0
    elif fault == 'native_goal': events[0]['lanes'][0]['current_last_stage_completed_task_buf'] = True
    elif fault == 'done_type': events[0]['lanes'][0]['reset_buf'] = True
    elif fault == 'timeout_type': events[0]['lanes'][0]['time_out_buf'] = 1
    elif fault == 'nan_reward': events[0]['lanes'][0]['episode_reward'] = float('nan')
    elif fault == 'summary': events[0]['completed_outcomes'][0]['exported_success'] = True
    elif fault == 'missing_raw_goal': events[0]['lanes'][0].pop('current_last_stage_completed_task_buf')
    with pytest.raises(ValueError): replay.replay_events(events, 2, oracle)


def test_oracle_hash_failure_cannot_fall_back_to_mutable_vendor(tmp_path):
    for relative in replay.PINS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('changed source')
    with pytest.raises(ValueError, match='pinned Raven source changed'):
        replay.load_oracle(tmp_path)


@pytest.mark.parametrize('text', ['{"goal":true,"goal":false}', '{"reward":NaN}'])
def test_ambiguous_json_rejected(text):
    with pytest.raises(ValueError): replay.parse(text)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


@pytest.fixture
def campaign(plan, tmp_path):
    """Full synthetic 295-member fixture, visibly separate from actual receipts."""
    dispatch_root, trace_root = tmp_path / 'dispatch', tmp_path / 'traces'
    write(dispatch_root / 'plan.json', plan)
    values = []
    for entry in plan['commands']:
        index = entry['cohort_index']
        owner = plan['owners'][entry['owner_index']]
        cid = entry['body']['command_id']
        raw = native_result((plan['panel'], plan['owners']), index)
        count = raw['expected']
        events = []
        for step in range(3):
            lanes = [dict(current_last_stage_completed_task_buf=i == 0 and step == 0,
                reset_buf=int(step == 2), time_out_buf=step == 2 and i != 0,
                episode_reward=float((i+1)*(step+1))) for i in range(count)]
            completed = [dict(row, lane=i, steps=3, native_success=i==0,
                exported_success=i==0, timeout=i!=0) for i,row in enumerate(lanes)] if step == 2 else []
            events.append(dict(step=step, arm='teacher', lanes=lanes,
                exported_goals=[r['current_last_stage_completed_task_buf'] for r in lanes],
                completed_outcomes=completed))
        raw['outcomes'] = events[-1]['completed_outcomes']
        raw['numeric_trace_sha256'] = 'f'*64
        raw['numeric_trace_steps_recorded'] = 3
        value = replay.teacher.wrap_result(plan['panel'], entry['artifacts'][0]['value'], raw)
        values.append(value)
        row = dict(status='COMPLETE', payload_sha256=entry['payload_sha256'], receipt=dict(
            complete=True, source_sha256=owner['source_sha256'], worker_instance_id=owner['worker_instance_id'],
            native_pid=owner['native_pid'], command_id=cid, result=value))
        commit = dict(schema=replay.dispatch.COMMIT_SCHEMA, plan_sha256=replay.dispatch.common.digest(plan),
            cohort_index=index, owner=owner, command_id=cid, payload_sha256=entry['payload_sha256'],
            native_command_receipt=row, result=value)
        write(dispatch_root / f'cohort-{index:03d}' / 'commit.json', commit)
        base = trace_root / f"gpu{entry['owner_index']}" / 'phases' / cid / f'cohort-{index:03d}'
        write(base / 'result.json', raw)
        write(base.parent / 'result.json', value)
        (base / 'steps.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    write(dispatch_root / 'result.json', replay.teacher.aggregate_results(plan['panel'], values))
    return dispatch_root, trace_root


def test_full_membership_and_all_failures_no_threshold_or_protocol_relabel(campaign):
    value = replay.replay_campaign(*campaign, REFERENCE)
    assert value['completed'] == 295 and value['successes'] == 19 and value['failures'] == 276
    assert len(value['cohorts']) == 19 and len(value['outcomes']) == 295
    assert [r['count'] for r in value['cohorts']] == [16]*18+[7]
    for key in ('full_trace_hashes_contemporaneously_sealed', 'independent_native_trajectory_tested',
                'policy_tested', 'renderer_tested', 'resource_requalified',
                'stage4_independent_live_Raven_evaluation_passed', 'compute_release_authorized'):
        assert value[key] is False
    assert value['protocol'] == 'matched_first_episode_per_lane_not_Raven_default_pooled'


@pytest.mark.parametrize('fault', ['missing_cohort', 'extra_trace', 'command', 'source', 'raw_result', 'wrapper', 'aggregate'])
def test_full_campaign_missing_extra_or_changed_bindings_rejected(campaign, fault):
    dispatch_root, trace_root = campaign
    first_trace = next(trace_root.rglob('steps.jsonl'))
    commit_path = dispatch_root / 'cohort-000/commit.json'
    if fault == 'missing_cohort': first_trace.unlink()
    elif fault == 'extra_trace': write(trace_root / 'unexpected/steps.jsonl', {})
    elif fault in ('command', 'source'):
        value = replay.read(commit_path)
        if fault == 'command': value['command_id'] = 'wrong'
        else: value['native_command_receipt']['receipt']['source_sha256'] = '0'*64
        write(commit_path, value)
    elif fault == 'raw_result':
        path = first_trace.with_name('result.json'); value = replay.read(path)
        value['vector_steps'] = 4; write(path, value)
    elif fault == 'wrapper':
        path = first_trace.parent.parent / 'result.json'; value = replay.read(path)
        value['cohort_result']['vector_steps'] = 4; write(path, value)
    elif fault == 'aggregate':
        path = dispatch_root / 'result.json'; value = replay.read(path)
        value['successes'] += 1; write(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        replay.replay_campaign(dispatch_root, trace_root, REFERENCE)
