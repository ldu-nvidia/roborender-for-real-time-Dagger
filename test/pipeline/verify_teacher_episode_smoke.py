"""Independent read-only validation of a completed32x2 teacher smoke.

Does not launch work, replay physics, release compute or certify student rendering.
"""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    read = lambda name: json.loads((root/name).read_text())
    result, status, host = read('result.json'), read('status.json'), read('host-result.json')
    spec = read('teacher-episodes.json')
    mixed = result['schema'] == 'roborender.collection-bound-mixed-episodes.v1'
    mode_based = mixed or result['schema'] == 'roborender.collection-bound-teacher-episodes.v4'
    assert result['schema'] in ('roborender.collection-bound-teacher-episodes.v3',
                                'roborender.collection-bound-teacher-episodes.v4',
                                'roborender.collection-bound-mixed-episodes.v1')
    repeats = 2 if mixed else 3 if mode_based else 2
    expected = 32*repeats
    assert result['spec'] == spec
    assert spec['initial_clocks'] == 'native_reset_episode0_stages1_full_episode_budget'
    assert status['phase'] == 'COMPLETE_IDLE' and result['physics_idle']
    assert result['complete'] and result['completed'] == expected
    assert spec['episodes_per_lane']==repeats and spec['expected_episodes']==expected
    assert result['completed_per_lane'] == [repeats]*32
    assert host['verdict'] == 'PASS' and host['warm_owner_retained'] and not host['compute_released']
    assert host['result_sha256'] == hashlib.sha256((root/'result.json').read_bytes()).hexdigest()
    assert not (root/'host-fault.json').exists()
    outcomes = result['outcomes']
    assert len(outcomes) == expected
    assert {(row['lane'],row['episode_index']) for row in outcomes} == {(i,j) for i in range(32) for j in range(repeats)}
    assert all(type(row['success']) is bool and type(row['timeout']) is bool and row['steps'] > 0 for row in outcomes)
    successes = sum(row['success'] for row in outcomes)
    assert result['successes'] == successes and result['failures'] == expected-successes
    assert result['success_rate'] == successes/expected
    transitions = [json.loads(line) for line in (root/'transitions.jsonl').read_text().splitlines()]
    assert [row['step'] for row in transitions] == list(range(result['vector_steps']))
    assert [x for row in transitions for x in row['completed']] == outcomes
    for lane in range(32):
        rows = sorted((r for r in outcomes if r['lane']==lane),key=lambda r:r['episode_index'])
        assert rows[0]['steps'] == rows[0]['terminal_vector_step']+1
        for prior, current in zip(rows,rows[1:]):
            assert current['steps'] == current['terminal_vector_step']-prior['terminal_vector_step']
    init = read('initial-reset.json')
    assert init['reset']['cleared_lanes'] == 32 and init['reset']['native_steps_added'] == 0
    assert init['clocks']['randomization_applied'] is False
    assert init['clocks']['clocks'] == dict(episode_length_buf=[0]*32,
        time_in_stage_buf=[1]*32,actual_time_in_stage_buf=[1]*32)
    canonical = lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    assert init['motion_binding']['ordered_motion_sha256'] == hashlib.sha256(canonical(spec['motions'])).hexdigest()
    assert init['motion_binding']['queried_lanes'] == list(range(32))
    assert result['spec_sha256'] == hashlib.sha256(canonical(spec)).hexdigest()
    proof = result['teacher_parameter_attestation']
    assert proof['strict_mean_parameters_and_rms_equality'] and not proof['rms_updates_enabled']
    assert all(p['strict_parameter_equality'] and not p['updates_enabled'] for p in proof['homie_parameter_attestations'].values())
    if mixed:
        from pipeline.success_rate.mixed_episodes import MixedModeBackend
        assert MixedModeBackend.valid_policy_result(None,result)
        counts = [0]*32
        for row in transitions:
            assert len(row['teacher_mask'])==32 and all(type(x) is bool for x in row['teacher_mask'])
            assert all(len(row[key])==32 for key in ('teacher26','student11','selected26'))
            for lane,take_teacher in enumerate(row['teacher_mask']):
                teacher,student,selected=[row[k][lane] for k in ('teacher26','student11','selected26')]
                assert len(teacher)==len(selected)==26 and len(student)==11
                assert selected[:11]==(teacher[:11] if take_teacher else student)
                assert selected[11:]==teacher[11:]
            counts = [a+int(b) for a,b in zip(counts,row['teacher_mask'])]
        assert counts == result['teacher_decisions_per_lane']
        assert 0 < sum(counts) < result['vector_steps']*32
    else:
        assert all(result[k] == 0 for k in ['optimizer_steps','student_forward_calls','renderer_calls'])
    peak = 0
    minimum_host = float('inf')
    for line in (root/'memory-samples.jsonl').read_text().splitlines():
        sample = json.loads(line);c = sample['container']
        assert c['container_id'] == host['container_id'] and c['pid'] == host['pid']
        assert c['memory_swap.current'] == 0 and not c['oom_killed']
        assert all(c['events'][key] == 0 for key in ['max','oom','oom_kill'])
        assert c['memory_max'] == 32*2**30 and c['memory_swap.max'] == 0
        peak = max(peak,c['memory_peak'])
        minimum_host = min(minimum_host,sample['host_available_bytes'])
    assert peak > 0 and minimum_host >= 8*2**30
    if mode_based:
        mode = read('mode-completion.json')
        assert mode['teacher']['status']=='COMPLETE' and mode['teacher']['receipt']['result']==result
        assert mode['teacher']['receipt']['mode']==('STUDENT_TEST' if mixed else 'TEACHER_TEST')
        assert mode['idle_after']['status']=='COMPLETE' and mode['idle_after']['receipt']['mode']=='IDLE'
        assert mode['owner']['phase']=='IDLE' and mode['owner']['mode']=='IDLE'
        assert mode['duplicate_returned_same_receipt'] is True
        initial=read('mode-initial-owner.json')
        assert initial['worker_instance_id']==mode['owner']['worker_instance_id']
        assert initial['native_pid']==mode['owner']['native_pid'] and mode['owner']['native_initializations']==1
    return dict(verdict='PASS',scope=f'native_{"mixed" if mixed else "teacher"}32_{repeats}_full_time_budget_episodes_per_lane',
        successes=successes,completed=expected,success_rate=successes/expected,
        repeat_successes=[sum(r['success'] for r in outcomes if r['episode_index']==i) for i in range(repeats)],
        vector_steps=result['vector_steps'],evaluation_seconds=result['seconds'],
        container_peak_gib=peak/2**30,minimum_host_available_gib=minimum_host/2**30,
        source_sha256=host['source_sha256'],result_sha256=host['result_sha256'],
        student_rendering_certified=mixed,compute_released=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root),indent=2))
