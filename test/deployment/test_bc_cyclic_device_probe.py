"""CPU-only guards for a fresh, finite nondivisible CUDA BC diagnostic."""
import importlib.util
from pathlib import Path
import pytest
PATH=Path(__file__).resolve().parents[1]/'pipeline/native_bc_cyclic_probe.py'
spec=importlib.util.spec_from_file_location('cyclic_probe',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def plan():
    return dict(schema='roborender.bc-cyclic-device-probe.v1',chunk_count=63,batch_size=32,
        reference_updates=4,resume_from_update=2,actual_optimizer_operations=6,validation_enabled=False,
        diagnostic_only=True,production_promoted=False,compute_release_authorized=False,
        selected_chunks=[dict(path=f'/fresh/{i:03d}.npz') for i in range(63)])
def test_exact_scope():assert m.validate_plan(plan())==plan()
@pytest.mark.parametrize('key,value',[('batch_size',16),('chunk_count',64),('reference_updates',100),
    ('resume_from_update',0),('actual_optimizer_operations',4),('validation_enabled',True),
    ('production_promoted',True),('compute_release_authorized',True)])
def test_wrong_scope(key,value):
    p=plan();p[key]=value
    with pytest.raises(ValueError):m.validate_plan(p)
def test_reordering_or_duplicate_paths():
    p=plan();p['selected_chunks'].reverse()
    with pytest.raises(ValueError):m.validate_plan(p)
    p=plan();p['selected_chunks'][3]=p['selected_chunks'][2]
    with pytest.raises(ValueError):m.validate_plan(p)
def test_uses_actual_sampler_and_current_learner():
    source=PATH.read_text()
    assert 'cyclic.CyclicBatchSampler(63,32)' in source and 'sampler.start_after(accepted)' in source
    assert 'bc.backward_raven_offline_bc_mse' in source and 'base.assert_equal(reference[key],resumed[key])' in source

def test_preserves_stricter_joint_renderer_headroom():
    from pipeline.dagger.services import learner_memory
    policy=learner_memory.admission(free_bytes=120<<30,total_bytes=140<<30,reserved_bytes=0)
    assert policy['allocator_cap_bytes']==83<<30
    source=PATH.read_text()
    assert 'paired_memory=learner_memory.begin(torch.cuda)' in source
    assert "min(memory['allocator_cap_bytes'],paired_memory['policy']['allocator_cap_bytes'])" in source
    assert "learner_memory.observe(paired_memory,torch.cuda,'bc_cyclic_operation_boundary')" in source

def runtime():
    path=PATH.parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN/bc_cyclic_device_v1.py'
    spec=importlib.util.spec_from_file_location('cyclic_device_runtime',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def test_runtime_keeps_existing_renderers_and_single_gpu_scope():
    r=runtime();argv,env,output=r.command('train')
    assert argv[:3]==['bash','--noprofile','--norc'] and str(PROBE:=r.PROBE) in argv[-1]
    assert env['CUDA_VISIBLE_DEVICES']==r.read(r.base.FLEET/'deployment.json')['workers'][7]['gpu_uuid']
    assert output==r.OUTPUT and r.base.SOURCE_SHA==r.SOURCE
    with pytest.raises(ValueError):r.command('render')

def test_runtime_requires_actual_terminal_scale_proof_before_start():
    r=runtime();source=Path(r.__file__).read_text()
    assert 'c4.qualify(c,live)' in source and 'idle_barrier' in source
    assert 'existing accepted launch requires observation; no replay' in source
    assert 'kill(' not in source and 'shutdown(' not in source

def test_resource_polling_does_not_fake_learner_progress(monkeypatch,tmp_path):
    from types import SimpleNamespace
    r=runtime();store={str(r.ROOT/'progress.json'):dict(updated_unix=0,faults=[])}
    monkeypatch.setattr(r,'CONTROL',tmp_path/'control')
    io=SimpleNamespace(read=lambda p,default=None:store.get(str(p),default),
        save=lambda p,value:store.__setitem__(str(p),value.copy()))
    monkeypatch.setattr(r,'module',lambda *a,**k:io)
    monkeypatch.setattr(r,'read',lambda p:store[str(p)].copy())
    clock=[100];monkeypatch.setattr(r,'time',SimpleNamespace(time=lambda:clock[0]))
    r.status('RUNNING',operations=1)
    clock[0]=200;r.status('RUNNING',operations=1)
    assert store[str(r.ROOT/'progress.json')]['updated_unix']==100
    clock[0]=300;r.status('RUNNING',operations=2)
    assert store[str(r.ROOT/'progress.json')]['updated_unix']==300
    r.status('FAILED',fault='actual fault')
    assert store[str(r.ROOT/'progress.json')]['faults']==['actual fault']

def test_owned_diagnostic_has_fifteen_minute_bound(monkeypatch):
    from types import SimpleNamespace
    r=runtime();monkeypatch.setattr(r,'original_sample',lambda:{})
    monkeypatch.setattr(r,'read',lambda _:dict(started_unix=0))
    monkeypatch.setattr(r,'time',SimpleNamespace(time=lambda:901))
    monkeypatch.setattr(r,'status',lambda *a,**k:pytest.fail('a poll must not hide a stalled test'))
    with pytest.raises(TimeoutError,match='15minutes'):r.resource_sample()
