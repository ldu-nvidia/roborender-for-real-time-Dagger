import json
from pathlib import Path
import pytest
from pipeline.bc_training.handoff_common import binding,bound
from pipeline.deployment.beta_operations import atomic
from pipeline.deployment.completed_mode_recovery import finalize_completed_dagger

class Adapter:
    def __init__(self): self.validated=[]
    def validate(self,phase,receipt,completed):
        self.validated.append(phase)
        if phase=='dagger':
            assert [x['round_id'] for x in bound(receipt['campaign_state'])['completed_rounds']]==[1,2,3,4]

def setup(tmp_path):
    plan=tmp_path/'plan.json';settings=tmp_path/'settings.json'
    atomic(plan,dict(output_root=str(tmp_path),run_id='smoke-test'))
    atomic(settings,dict(compute_release_authorized=False))
    root=tmp_path/'execution';completed=[]
    for phase in ['prepare_bc','train_bc','test_bc']:
        p=root/(phase+'-receipt.json');atomic(p,{'ok':True})
        completed.append(dict(phase=phase,receipt=binding(p)))
    atomic(root/'state.json',dict(status='RECONCILE_REQUIRED',phase='dagger',plan=binding(plan),
        settings=binding(settings),completed=completed,started_at=1))
    atomic(tmp_path/'dagger/campaign/state.json',dict(run_id='dagger-leo-smoke-test',phase='COMPLETE',
        completed_rounds=[dict(round_id=i) for i in range(1,5)]))
    return plan,settings

def test_complete_campaign_finalized_without_phase_execution(tmp_path):
    paths=setup(tmp_path);adapter=Adapter()
    result=finalize_completed_dagger(*paths,adapter=adapter)
    assert result['status']=='COMPLETE' and len(result['completed'])==4
    assert adapter.validated==['prepare_bc','train_bc','test_bc','dagger']
    assert list((tmp_path/'execution/recoveries').glob('*/prior-state.json'))
    with pytest.raises(RuntimeError):finalize_completed_dagger(*paths,adapter=adapter)

@pytest.mark.parametrize('bad',['running','run','rounds','receipt','release'])
def test_incomplete_or_changed_evidence_rejected(tmp_path,bad):
    p,s=setup(tmp_path);c=tmp_path/'dagger/campaign/state.json';state=json.loads(c.read_bytes())
    if bad=='running':state['phase']='EVALUATE'
    elif bad=='run':state['run_id']='other'
    elif bad=='rounds':state['completed_rounds'].pop()
    elif bad=='receipt':atomic(tmp_path/'execution/train_bc-receipt.json',{'changed':True})
    elif bad=='release':atomic(s,dict(compute_release_authorized=True))
    atomic(c,state)
    with pytest.raises((ValueError,RuntimeError,AssertionError)):
        finalize_completed_dagger(p,s,adapter=Adapter())
