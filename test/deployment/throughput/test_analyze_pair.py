import json
import pytest
from analyze_pair import analyze


def evidence(tmp_path):
    (tmp_path/'pair-result.json').write_text(json.dumps({
        'passed':True,'run_id':'test','elapsed_seconds':100}))
    (tmp_path/'resources.jsonl').write_text(json.dumps({
        'mem_available_kib':8*1024**2,'swap_free_kib':10*1024**2})+'\n')
    for gpu in (1,2):
        root=tmp_path/f'gpu{gpu}'
        root.mkdir()
        (root/'native-service-result.json').write_text(json.dumps({'num_envs':2,'capacity_steps':3}))
        steps=[]
        for step in range(3):
            steps.append({'event':'vector_step_complete','time_ns':(10+step*2)*10**9,
                'step_keys':[{'rollout_step_index':step}]*2,
                'unique_prepare_receipts':2,'unique_capture_digests':2,
                'unique_selection_receipts':2,'unique_commit_receipts':2,
                'unique_remote_commit_receipts':2,
                'observation_source_counts':{'generated_controls':1,'raytraced':1},
                'selected_source_counts':{'teacher':1,'student':1},
                'timings_ns':{'vector_step_total':2*10**9,'local_prepare_barrier':400_000_000,
                    'native_capture_batch':200_000_000,'h200_prepare_render_student_select_barrier':10**9,
                    'local_commit_vector_barrier':300_000_000,'native_env_step_batch':200_000_000,
                    'h200_commit_barrier':200_000_000}})
        (root/'capacity-telemetry.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in steps))


def test_aggregate_uses_shared_wall_time_not_sum_of_rates(tmp_path):
    evidence(tmp_path)
    result=analyze(tmp_path)
    assert result['aggregate_warm_rows']==8
    assert result['aggregate_warm_seconds']==4
    assert result['actor_overlap_seconds']==4
    assert result['aggregate_committed_transitions_per_second']==2
    assert result['full_collection_wall_seconds_including_cold_step_and_seal']==6
    assert result['all_committed_rows']==12
    assert sum(result['actors'][0]['phase_mean_seconds'].values())==pytest.approx(2)
    assert sum(result['actors'][0]['phase_share_percent'].values())==pytest.approx(100)


def test_duplicate_vector_index_is_rejected(tmp_path):
    evidence(tmp_path)
    path=tmp_path/'gpu1/capacity-telemetry.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    rows[1]['step_keys'][0]['rollout_step_index']=0
    path.write_text(''.join(json.dumps(s)+'\n' for s in rows))
    with pytest.raises(RuntimeError,match='duplicate'):
        analyze(tmp_path)
