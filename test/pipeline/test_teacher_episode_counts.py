"""CPU-only accounting tests; not B32 native capacity certification."""
from copy import deepcopy
import pytest

from pipeline.success_rate.teacher_episodes import PerLaneEpisodes, make_spec
from test.pipeline.test_continuous_bc_launcher import request


def test_atomic_status_is_host_readable_after_every_replacement(tmp_path):
    import json
    import stat
    from pipeline.success_rate.teacher_episodes import publish_status
    path = tmp_path/'status.json'
    for phase in ['EVALUATING','COMPLETE_IDLE']:
        publish_status(path,{'phase':phase})
        assert stat.S_IMODE(path.stat().st_mode) == 0o644
        assert json.loads(path.read_text()) == {'phase':phase}


def test_pair_start_requires_exact_admission_and_publishes_start(tmp_path, monkeypatch):
    import hashlib
    import json
    from pathlib import Path
    import pipeline.success_rate.teacher_episodes as module
    pair = tmp_path/'pair'
    pair.mkdir()
    spec = dict(seed=20261029,bc_request_sha256='a'*64)
    plan = dict(seeds=[20261029,20261030],bc_request_sha256='a'*64)
    raw = json.dumps(plan).encode()
    (pair/'plan.json').write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setenv('ROBORENDER_TEACHER_PAIR_SHA256',digest)
    monkeypatch.setenv('ROBORENDER_TEACHER_PAIR_RANK','0')
    monkeypatch.setattr(module,'Path',lambda value:pair if value=='/teacher-pair' else Path(value))
    (pair/'start.json').write_text(json.dumps(dict(pair_sha256=digest,ranks=[0,1])))
    phases=[]
    module.wait_pair_start(tmp_path,spec,lambda phase,**kw:phases.append(phase))
    assert phases==['READY_PAIR']
    assert json.loads((tmp_path/'evaluation-start.json').read_text())['rank']==0
    (pair/'start.json').write_text(json.dumps(dict(pair_sha256='b'*64,ranks=[0,1])))
    with pytest.raises(RuntimeError,match='start receipt mismatch'):
        module.wait_pair_start(tmp_path,spec,lambda *args,**kwargs:None)
    monkeypatch.setenv('ROBORENDER_TEACHER_PAIR_RANK','1')
    with pytest.raises(RuntimeError,match='identity mismatch'):
        module.wait_pair_start(tmp_path,spec,lambda *args,**kwargs:None)


def test_full_episode_smoke_rejects_shortened_initial_clocks():
    from types import SimpleNamespace
    import numpy as np
    from pipeline.success_rate.teacher_episodes import require_full_initial_clocks
    names = ['episode_length_buf','time_in_stage_buf','actual_time_in_stage_buf']
    env = SimpleNamespace(num_envs=32,**{name:np.full(32,0 if name=='episode_length_buf' else 1,dtype=np.int64) for name in names})
    assert require_full_initial_clocks(env)['randomization_applied'] is False
    for name in names:
        before = getattr(env,name).copy()
        getattr(env,name)[7] = 123
        with pytest.raises(RuntimeError,match='native reset clocks'):
            require_full_initial_clocks(env)
        getattr(env,name)[:] = before


def test_fast_lanes_cannot_replace_slow_lane_repeats():
    account = PerLaneEpisodes(32, 2)
    for step in range(8):
        done = [True] * 31 + [step in (3, 7)]
        account.update(step, [True] * 31 + [step == 0], done, [False] * 32)
        assert account.complete is (step == 7)
    result = account.summary()
    assert result['completed'] == 64
    assert result['completed_per_lane'] == [2] * 32
    assert result['successes'] == 63 and result['failures'] == 1
    assert [row['steps'] for row in result['outcomes'] if row['lane'] == 31] == [4, 4]


def test_goal_latch_clears_only_on_that_lanes_terminal():
    account = PerLaneEpisodes(2, 2)
    account.update(0, [True, False], [False, True], [False, True])
    account.update(1, [False, True], [True, True], [False, False])
    account.update(2, [False, True], [True, True], [False, False])
    assert [r['success'] for r in account.outcomes if r['lane'] == 0] == [True, False]
    assert [r['success'] for r in account.outcomes if r['lane'] == 1] == [False, True]
    assert account.complete


def test_partial_never_claims_full_result_and_cannot_replay_transition():
    account = PerLaneEpisodes(32, 2)
    account.update(0, [False] * 32, [True] * 32, [False] * 32)
    assert not account.summary()['complete']
    assert account.summary()['completed'] == 32
    with pytest.raises(ValueError):
        account.update(0, [False] * 32, [True] * 32, [False] * 32)


def test_source_population_bound_to_full_collection_inventory():
    original = request()
    spec = make_spec(original, 'a' * 64)
    assert spec['motions'] == original['motions']
    assert spec['motion_inventory'] == original['motion_inventory']
    assert spec['expected_episodes'] == 64
    assert spec['optimizer_steps'] == spec['renderer_calls'] == spec['student_forward_calls'] == 0
    with pytest.raises(ValueError):
        make_spec(original, 'a' * 64, seed=original['process_seed'])
    altered = deepcopy(original)
    altered['motions'].reverse()
    with pytest.raises(ValueError):
        make_spec(altered, 'a' * 64)


@pytest.mark.parametrize('field', ['goals', 'dones', 'timeouts'])
def test_rejects_wrong_lane_count(field):
    inputs = dict(goals=[False] * 32, dones=[False] * 32, timeouts=[False] * 32)
    inputs[field].pop()
    with pytest.raises(ValueError):
        PerLaneEpisodes(32, 2).update(0, **inputs)


def test_teacher_launch_is_isolated_and_bound_to_bc(tmp_path):
    import hashlib
    import json
    import sys
    from pathlib import Path
    actor = Path(__file__).resolve().parents[2] / 'pipeline/deployment/rtx_actor'
    sys.path.insert(0, str(actor))
    from qualify_native_isaac6 import command, IMAGE
    bc = tmp_path/'bc.json'
    bc.write_text(json.dumps(request()))
    spec = make_spec(request(),hashlib.sha256(bc.read_bytes()).hexdigest())
    specfile = tmp_path/'spec.json'
    specfile.write_text(json.dumps(spec))
    overlay = tmp_path/'overlay'
    overlay.mkdir()
    (overlay/'port-manifest.json').write_text('{"files":[]}')
    kwargs = dict(mode='teacher-episodes', deployment_host='leo-rtx-6k',physical_gpu=1,
        memory_gib=32,compact_host_heap=True,seed=spec['seed'],
        teacher_episode_spec=specfile,teacher_bc_request=bc)
    cmd = command('teacher-test',tmp_path/'out',overlay,
                  tmp_path/'source/pipeline/dagger/rtx/code',32,**kwargs)
    assert 'num_envs=32' in cmd
    assert 'trainer._target_=pipeline.success_rate.teacher_episodes.TeacherEpisodeEvaluator' in cmd
    assert 'algo.config.num_learning_epochs=0' in cmd
    assert 'algo.config.init_at_random_ep_len=false' in cmd
    assert '--no-healthcheck' in cmd
    assert 'org.systemd.property.MemorySwapMax=uint64 0' in cmd
    assert not any('raytraced_mix_prob' in v or 'anchor_pool_dir=' in v for v in cmd)
    assert cmd.index('--no-healthcheck') < cmd.index(IMAGE)
    for gpu in (0,2):
        with pytest.raises(ValueError):
            command('teacher-test',tmp_path/'out',overlay,
                    tmp_path/'source/pipeline/dagger/rtx/code',32,**{**kwargs,'physical_gpu':gpu})
    pair = tmp_path/'pair'
    pair.mkdir()
    (pair/'plan.json').write_text(json.dumps(dict(schema='roborender.teacher32-pair.v1',
        seeds=[20261029,20261030],bc_request_sha256=hashlib.sha256(bc.read_bytes()).hexdigest(),
        expected_episodes=128)))
    for gpu in (0,1):
        seed = 20261029+gpu
        specfile.write_text(json.dumps(make_spec(request(),hashlib.sha256(bc.read_bytes()).hexdigest(),seed=seed)))
        paired = command('paired',tmp_path/'out',overlay,tmp_path/'source/pipeline/dagger/rtx/code',32,
            **{**kwargs,'physical_gpu':gpu,'seed':seed,'teacher_pair_dir':pair})
        assert f'ROBORENDER_TEACHER_PAIR_RANK={gpu}' in paired
        assert any('dst=/teacher-pair,readonly' in v for v in paired)
    (pair/'plan.json').write_text(json.dumps(dict(schema='roborender.teacher32-pair.v2',
        seeds=[20261031,20261101],bc_request_sha256=hashlib.sha256(bc.read_bytes()).hexdigest(),
        expected_episodes=192,source_sha256='b'*64)))
    for gpu,seed in enumerate([20261031,20261101]):
        specfile.write_text(json.dumps(make_spec(request(),hashlib.sha256(bc.read_bytes()).hexdigest(),
                                                seed=seed,mode_based=True)))
        paired=command('paired-mode',tmp_path/'out',overlay,tmp_path/'source/pipeline/dagger/rtx/code',32,
            **{**kwargs,'physical_gpu':gpu,'seed':seed,'teacher_pair_dir':pair})
        assert paired[paired.index('--network')+1]=='host'
        assert 'ROBORENDER_TEACHER_MODE_SOURCE_SHA256='+'b'*64 in paired


def test_three_episode_mode_spec_is_separate_from_historical_two():
    from pipeline.success_rate.teacher_episodes import MODE_SCHEMA
    old = make_spec(request(),'a'*64)
    new = make_spec(request(),'a'*64,seed=20261031,mode_based=True)
    assert old['schema'] != new['schema'] == MODE_SCHEMA
    assert new['expected_episodes']==96 and new['episodes_per_lane']==3
    assert old['expected_episodes']==64
    assert new['motions']==old['motions'] and new['motion_inventory']==old['motion_inventory']


def test_three_repeats_per_lane_ignore_fast_extras():
    account = PerLaneEpisodes(32,3)
    for step in range(12):
        account.update(step,[True]*32,[True]*31+[step in (3,7,11)],[False]*32)
    assert account.complete and account.summary()['completed']==96
    assert account.counts==[3]*32
    assert [r['steps'] for r in account.outcomes if r['lane']==31]==[4,4,4]


def test_teacher_mode_queue_keeps_kit_and_never_replays(tmp_path):
    from dagger_rtx.resident_service import ResidentOwner, SCHEMA
    from pipeline.success_rate.teacher_episodes import TeacherModeBackend, digest
    spec = make_spec(request(),'a'*64,seed=20261031,mode_based=True)
    calls=[]
    result=dict(complete=True,spec=spec,spec_sha256=digest(spec),completed=96,completed_per_lane=[3]*32,
                optimizer_steps=0,student_forward_calls=0,renderer_calls=0)
    backend=TeacherModeBackend(spec,evaluate=lambda:calls.append('evaluate') or result,
        pause=lambda:calls.append('pause'),paused=lambda:True,kit_identity=lambda:123,revalidate=lambda:spec)
    owner=ResidentOwner(tmp_path,backend,gpu_uuid='GPU-test',source_sha256='b'*64)
    def command(cid,mode,operation,parameters):
        return dict(schema=SCHEMA,worker_instance_id=owner.state['worker_instance_id'],command_id=cid,
            operation='run_phase',payload=dict(schema='roborender.isaac-phase.v1',mode=mode,
                operation=operation,parameters=parameters))
    owner.submit(command('before','IDLE','idle',{})); owner.run_next()
    body=command('teacher','TEACHER_TEST','evaluate_collection_teacher',dict(arm='teacher',spec_sha256=digest(spec)))
    acceptance=owner.submit(body); owner.run_next()
    assert owner.submit(body)==acceptance and calls.count('evaluate')==1
    with pytest.raises(ValueError,match='unused bound'):
        owner.submit({**body,'command_id':'another-test'})
    owner.submit(command('after','IDLE','idle',{})); owner.run_next()
    assert owner.state['phase']=='IDLE' and owner.state['mode']=='IDLE'
    assert owner.result('teacher')['receipt']['kit_identity']==owner.result('after')['receipt']['kit_identity']==123


@pytest.mark.parametrize('fault',['partial','optimizer','changed_spec','changed_kit','unpaused'])
def test_teacher_mode_fault_never_certifies_idle(fault):
    from pipeline.success_rate.teacher_episodes import TeacherModeBackend, digest
    spec=make_spec(request(),'a'*64,seed=20261031,mode_based=True)
    result=dict(complete=True,spec=spec,spec_sha256=digest(spec),completed=96,completed_per_lane=[3]*32,
                optimizer_steps=0,student_forward_calls=0,renderer_calls=0)
    if fault=='partial':result['completed']=95
    if fault=='optimizer':result['optimizer_steps']=1
    identity=[123]
    backend=TeacherModeBackend(spec,evaluate=lambda:result,pause=lambda:None,paused=lambda:fault!='unpaused',
        kit_identity=lambda:identity[0],revalidate=lambda:{} if fault=='changed_spec' else spec)
    if fault=='changed_kit':identity[0]=456
    with pytest.raises(RuntimeError):
        backend.run('run_phase',dict(schema='roborender.isaac-phase.v1',mode='TEACHER_TEST',
            operation='evaluate_collection_teacher',parameters=dict(arm='teacher',spec_sha256=digest(spec))),lambda *a,**k:None)
