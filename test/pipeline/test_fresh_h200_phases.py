"""Fresh H200 launcher scope/no-replay tests, no GPU work."""
import importlib.util
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[2]/'runtime/raven-dagger-alignment-20261026-7pYLxI/fresh_h200_phases.py'
if not PATH.is_file():
    pytest.skip('optional run-local fresh-parent H200 launcher absent',allow_module_level=True)
spec=importlib.util.spec_from_file_location('fresh_h200_phases_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_render_scope_is_fresh64_independent_teacher_not_online():
    argv,env,output=m.command('render')
    assert argv[argv.index('--limit')+1]=='64'
    assert argv[argv.index('--gpus')+1]=='0,1,2,3,4,5,6,7'
    assert str(m.DATA/'manifest.json') in argv and '--execution-identity' in argv
    assert output==m.RENDER_RUN and env['PYTHONPATH']==str(m.CODE)


def test_training_uses_explicit_new_data_bootstrap_module_and_online_logging():
    argv,env,output=m.command('train')
    script=argv[-1]
    assert '--fresh-continuous-pool '+str(m.DATA) in script
    assert '--fresh-render-plan '+str(m.RENDER_RUN/'plan.json') in script
    assert '--fresh-rendered-root '+str(m.RENDERED) in script
    assert '--wandb-mode online' in script and '--min-free-gpu-gib 120' in script
    assert output==m.TRAIN_RUN and env['CUDA_VISIBLE_DEVICES'].startswith('GPU-')
    assert '--resume' not in script and 'bc116' not in script


def test_saved_intent_never_submits_duplicate(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    (tmp_path/'fresh-render-control').mkdir()
    monkeypatch.setattr(m,'host_check',lambda:None)
    monkeypatch.setattr(m,'source_check',lambda:828)
    monkeypatch.setattr(m,'command',lambda p:([],{},tmp_path/'output'))
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:pytest.fail('duplicate launched'))
    with pytest.raises(FileExistsError):m.start('render')


def test_unknown_phase_rejected():
    with pytest.raises(ValueError):m.command('production')


@pytest.mark.parametrize('bad',['gpu','host','missing','duplicate'])
def test_resource_floor_rejects_unsafe_or_missing_observation(bad):
    row=dict(gpus=[[g,120*1024,19000,0] for g in range(8)],host_available_bytes=64*2**30)
    assert m.healthy_sample(row)
    if bad=='gpu':row['gpus'][3][1]=16*1024-1
    elif bad=='host':row['host_available_bytes']=16*2**30-1
    elif bad=='missing':row['gpus'].pop()
    else:row['gpus'][7][0]=0
    assert not m.healthy_sample(row)
