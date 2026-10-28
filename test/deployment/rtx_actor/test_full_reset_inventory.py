"""Differential reset boundaries against hash-pinned Raven; CPU only."""
import ast
from copy import deepcopy
import hashlib
import importlib.machinery
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import numpy as np

from test.deployment.source_paths import REPO_ROOT
from test.deployment.rtx_actor.test_native_motion_assignment import fixture, key
from pipeline.dagger.rtx.code.src.dagger_rtx.reset_inventory import FullResetInventoryMixin
from pipeline.deployment.rtx_actor.native_motion_assignment import (
    configure_motion_environment, assert_motion_environment, explicit_scene_motion_binding,
    install_native_assignment_observer, NativeEpisodeAssignmentRecorder)
from pipeline.deployment.rtx_actor.native_artifact_io import canonical


def inventory_fixture(tmp_path):
    assignment,cfg=fixture(tmp_path)
    rows=[{k:r[k] for k in ('motion_id','path','sha256')} for r in assignment['lanes']]
    for i in (2,3):
        p=tmp_path/f'motion-{i}.npz';p.write_bytes(bytes([i+1])*3)
        rows.append(dict(motion_id=f'motion-{i}',path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    assignment['lanes']=[dict(lane=i,**row) for i,row in enumerate(rows[2:])]
    assignment['purpose']='training'
    configure_motion_environment(cfg,assignment,motion_inventory=rows,lane_offset=2)
    return assignment,cfg,rows


def test_second_actor_keeps_global_inventory_and_local_geometry(tmp_path,monkeypatch):
    assignment,cfg,rows=inventory_fixture(tmp_path)
    assert assert_motion_environment(cfg,assignment,motion_inventory=rows,lane_offset=2)==assignment
    scenario=ModuleType('scenario');scenario.__file__=str(tmp_path/'scenario.py')
    Path(scenario.__file__).write_text('# fixture\n')
    scenario.get_table_grid_meta_data=lambda *_:'original'
    scenario.table_top_material_prim_paths=[];scenario.leg_material_prim_paths=[]
    monkeypatch.setattr(importlib.machinery.SourceFileLoader,'exec_module',lambda *_:None)
    loader=importlib.machinery.SourceFileLoader('scenario',scenario.__file__)
    motion_loader=SimpleNamespace(from_file=lambda p,_:SimpleNamespace(table_custom_data=p))
    with explicit_scene_motion_binding(cfg,assignment,scenario,motion_loader,motion_inventory=rows,lane_offset=2) as proof:
        loader.exec_module(scenario)
        assert cfg.reset_from_dataset.motion_files==[r['path'] for r in rows]
        assert scenario.get_table_grid_meta_data(cfg,0,'cpu')==rows[2]['path']
        assert scenario.get_table_grid_meta_data(cfg,1,'cpu')==rows[3]['path']
        assert proof['reset_lane_offset']==2 and proof['reset_motion_count']==4
        assert proof['motion_inventory_sha256']==hashlib.sha256(canonical(rows)).hexdigest()
    with pytest.raises(ValueError):
        assert_motion_environment(cfg,assignment,motion_inventory=rows,lane_offset=0)
    with pytest.raises(ValueError):
        assert_motion_environment(cfg,assignment)


def original_sampler():
    p=REPO_ROOT/'runtime/raven_gr00t_collection_94d7985/gr00t/rl/envs/loco_manip/reset_from_dataset_wsppt.py'
    if not p.is_file():pytest.skip('pinned Raven source unavailable')
    raw=p.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='d42185ef7535e193bb5affcda6e7ad2fbac92e1c478feb9a846ed455f476ac94'
    cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='ResetFromDatasetWSPPT')
    method=deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_get_sample_idx'))
    namespace={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(p),'exec'),namespace)
    return type('RavenSampler',(),{'_get_sample_idx':namespace['_get_sample_idx']})


@pytest.mark.parametrize('lanes',[[0,1],[1],[]])
@pytest.mark.parametrize('mode',['random','curriculum','start'])
def test_mapped_sampler_matches_original_global_ids_and_rng(lanes,mode):
    Base=original_sampler()
    Current=type('Current',(FullResetInventoryMixin,Base),{})
    current,original=Current(),Base()
    for env in (current,original):
        env.config=SimpleNamespace(reset_from_dataset=SimpleNamespace(roborender_reset_lane_offset=2))
        env.device='cpu';env.num_envs=2
        env.reset_dataset=SimpleNamespace(motion_files=list(range(4)),
            sample_idx_based_on_demo_clip_number=[(0,9),(10,19),(20,29),(30,39)])
    kwargs={'processed_ratio':torch.ones(len(lanes))} if mode=='curriculum' else {'reset_from_teleop_start':mode=='start'}
    torch.manual_seed(77)
    expected=original._get_sample_idx(torch.tensor(lanes,dtype=torch.long)+2,**kwargs)
    expected_rng=torch.get_rng_state()
    torch.manual_seed(77)
    got=current._get_sample_idx(torch.tensor(lanes,dtype=torch.long),**kwargs)
    assert torch.equal(got,expected) and torch.equal(torch.get_rng_state(),expected_rng)
    current.reset_dataset.sample_idx_based_on_demo_clip_number.pop()
    with pytest.raises(ValueError,match='eligible'):
        current._get_sample_idx(torch.tensor(lanes,dtype=torch.long),**kwargs)


def test_observer_keeps_global_samples_and_accepts_native_inclusive_endpoint(tmp_path):
    assignment,cfg,rows=inventory_fixture(tmp_path)
    proof=dict(motion_inventory_sha256=hashlib.sha256(canonical(rows)).hexdigest(),reset_lane_offset=2,reset_motion_count=4)
    env=SimpleNamespace(config=cfg,reset_dataset=SimpleNamespace(motion_files=[r['path'] for r in rows],
        sample_idx_based_on_demo_clip_number=[(0,9),(10,19),(20,29),(30,39)]),
        simulator=SimpleNamespace(native_reset_inventory_binding=proof,
            native_motion_assignment_sha256=hashlib.sha256(canonical(assignment)).hexdigest()))
    def task(lanes):
        env.sample_idx=[29+10*i for i in lanes];env.sample_idx_in_seq=[2]*len(lanes)
    env._reset_task_states=task
    env.reset_envs_idx=lambda lanes:env._reset_task_states(lanes)
    trainer=SimpleNamespace(env=env)
    with install_native_assignment_observer(trainer,assignment,tmp_path/'observer',lambda:'a'*64,
            motion_inventory=rows,lane_offset=2) as recorder:
        env.reset_envs_idx([1])
        item=recorder.bind(1,key('lane1'),{'use_raytraced':False})
        audit=json.loads(Path(item['native_assignment_path']).with_suffix('.reset.json').read_bytes())
        assert audit['sample_idx']==39 and item['motion_id']=='motion-3'
        with pytest.raises(ValueError,match='declared lane'):
            recorder.observe_task_reset([1],[40],[2],[(20,29),(30,39)],'a'*64)


def test_original_periodic_resampling_consumes_full_inventory_including_unassigned(tmp_path):
    p=REPO_ROOT/'runtime/raven_gr00t_collection_94d7985/gr00t/rl/utils/teleop_reset_lib/dataset_wsppt.py'
    if not p.is_file():pytest.skip('pinned Raven source unavailable')
    raw=p.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='949cc147ad9abc32600e277fb69a0579ad5fd87e7f2086249785b966d88de1a3'
    cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='MotionDatasetWSPPT')
    method=deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='resample'))
    namespace={'torch':torch,'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(p),'exec'),namespace)

    class Motion:
        duration=4.0
        def __init__(self):self.starts=[]
        def sample(self,_,times):
            self.starts.append(float(times[0]));self.n=len(times)
            return tuple(torch.zeros(self.n,d) for d in (2,2,3,4,3,3))
        def objects(self):return tuple(torch.zeros(self.n,d) for d in (3,4,3,3))
        sample_table_grid=sample_bottle_table=sample_bottle_hand=sample_tray_table=objects
        def sample_homie_command(self):return tuple(torch.zeros(self.n,d) for d in (7,11,2))

    _,cfg,rows=inventory_fixture(tmp_path)
    motions=[Motion() for _ in cfg.reset_from_dataset.motion_files]
    dataset=SimpleNamespace(motion_loaders=motions,num_per_sample=2,sample_interval_s=.5)
    np.random.seed(37)
    namespace['resample'](dataset)
    namespace['resample'](dataset)
    expected=np.random.RandomState(37).rand(2,len(rows))
    assert np.array_equal(np.asarray([m.starts for m in motions]).T,expected)
    assert dataset.sample_idx_based_on_demo_clip_number==[[0,2],[3,5],[6,8],[9,11]]
    assert dataset.samples_robot[0].shape[0]==12
