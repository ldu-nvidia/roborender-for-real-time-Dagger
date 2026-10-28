"""State-machine fixtures only; native certification requires the real adapter."""
import json
from pathlib import Path
import pytest
from functools import partial
from pipeline.deployment.mode_execution import execute as execute_legacy_mode,PHASES
from pipeline.deployment.mode_execution import adopt_completed_bc
from pipeline.bc_training.handoff_common import binding
from pipeline.deployment.beta_operations import atomic

# These tests exercise the preserved historical v1 state machine deliberately.
execute = partial(execute_legacy_mode, allow_legacy_validation=True)


def test_new_mode_execution_rejects_validation_before_any_side_effect(tmp_path):
    paths=setup(tmp_path);adapter=Adapter()
    with pytest.raises(ValueError,match='MSE validation is disabled'):
        execute_legacy_mode(*paths,adapter=adapter)
    assert adapter.calls==[]
    assert not (tmp_path/'execution').exists()


def setup(tmp_path):
    plan=tmp_path/'plan.json';settings=tmp_path/'settings.json'
    atomic(plan,dict(schema='roborender.pipeline-mode.v1',mode='smoke',run_id='smoke-exec',output_root=str(tmp_path)))
    atomic(settings,dict(compute_release_authorized=False))
    return plan,settings


class Adapter:
    def __init__(self,fail=None):self.calls=[];self.fail=fail
    def __getattr__(self,phase):
        def run(completed):
            self.calls.append(phase)
            if phase==self.fail:raise RuntimeError('fixture failure')
            return dict(phase=phase,fixture=True)
        return run
    def validate(self,phase,receipt,completed):
        assert receipt==dict(phase=phase,fixture=True)


def test_all_phases_order_and_completed_state_not_reexecuted(tmp_path):
    paths=setup(tmp_path);a=Adapter()
    state=execute(*paths,adapter=a)
    assert state['status']=='COMPLETE' and a.calls==list(PHASES)
    assert execute(*paths,adapter=a)==state and a.calls==list(PHASES)
    assert state['compute_release_authorized'] is False


def test_failure_retains_intent_and_forbids_native_replay(tmp_path):
    paths=setup(tmp_path);a=Adapter('test_bc')
    with pytest.raises(RuntimeError):execute(*paths,adapter=a)
    state=json.loads((tmp_path/'execution/state.json').read_bytes())
    assert state['status']=='RECONCILE_REQUIRED' and len(state['completed'])==2
    with pytest.raises(RuntimeError,match='replay'):execute(*paths,adapter=a)
    assert a.calls==['prepare_bc','train_bc','test_bc']


def test_completed_receipt_tamper_rejected(tmp_path):
    paths=setup(tmp_path);execute(*paths,adapter=Adapter())
    atomic(tmp_path/'execution/test_bc-receipt.json',dict(fake=True))
    with pytest.raises(ValueError):execute(*paths,adapter=Adapter())


def recovery_fixture(tmp_path):
    old=tmp_path/'old';old.mkdir();plan,settings=setup(old)
    panel=tmp_path/'panel.json';atomic(panel,{'cases':2})
    p=json.loads(plan.read_bytes());p.update(settings={'seed':42,'source_sha256':'a'},
        source_sha256='a',test_panel=binding(panel))
    atomic(plan,p);atomic(settings,dict(compute_release_authorized=False,retained={'identity':'retained'}))
    with pytest.raises(RuntimeError):execute(plan,settings,adapter=Adapter('test_bc'))
    new=tmp_path/'new';new.mkdir();p.update(output_root=str(new),source_sha256='b',settings={'seed':42,'source_sha256':'b'})
    atomic(new/'plan.json',p);atomic(new/'settings.json',json.loads(settings.read_bytes()))
    workers=[]
    for gpu in (0,1):
        j=tmp_path/f'journal{gpu}.json';atomic(j,{'source_sha256':'a','commands':{}})
        workers.append(dict(physical_gpu=gpu,unit_state='inactive',journal=binding(j)))
    e=tmp_path/'evidence.json';atomic(e,dict(schema='roborender.pretest-recovery.v1',
        previous_state=binding(old/'execution/state.json'),controller_status={'state':'STOPPED'},
        bootstrap_status={'state':'FAILED'},workers=workers))
    return new/'plan.json',new/'settings.json',old/'execution/state.json',e


def test_recover_completed_bc_does_not_repeat_collection_or_learning(tmp_path):
    p,s,old,e=recovery_fixture(tmp_path);a=Adapter()
    result=adopt_completed_bc(p,s,previous_state=old,stopped_evidence=e,adapter=a)
    assert result['recovery']['native_state_reused'] is False
    result=execute(p,s,adapter=a)
    assert a.calls==['test_bc','dagger'] and result['status']=='COMPLETE'
    assert json.loads(old.read_bytes())['status']=='RECONCILE_REQUIRED'


@pytest.mark.parametrize('mutation',['live','test','seed','receipt'])
def test_recovery_rejects_unsafe_or_changed_boundary(tmp_path,mutation):
    p,s,old,e=recovery_fixture(tmp_path)
    evidence=json.loads(e.read_bytes())
    if mutation=='live':evidence['workers'][0]['unit_state']='active'
    elif mutation=='test':
        j=Path(evidence['workers'][0]['journal']['path'])
        atomic(j,{'source_sha256':'a','commands':{'x':{'command':{'operation':'evaluate_explicit_panel'}}}})
        evidence['workers'][0]['journal']=binding(j)
    elif mutation=='seed':
        value=json.loads(p.read_bytes());value['settings']['seed']=43;atomic(p,value)
    elif mutation=='receipt':atomic(old.parent/'train_bc-receipt.json',{'tampered':True})
    atomic(e,evidence)
    with pytest.raises((ValueError,AssertionError)):
        adopt_completed_bc(p,s,previous_state=old,stopped_evidence=e,adapter=Adapter())


def test_site_campaign_handoff_uses_same_canonical_writer_as_consumer():
    import ast
    from pipeline.deployment import mode_site
    tree=ast.parse(Path(mode_site.__file__).read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and n.args
        and isinstance(n.func,ast.Name) and n.func.id in ('atomic','immutable_json')
        and isinstance(n.args[0],ast.BinOp) and isinstance(n.args[0].right,ast.Constant)
        and n.args[0].right.value=='operations.json']
    assert len(calls)==1 and calls[0].func.id=='immutable_json'
