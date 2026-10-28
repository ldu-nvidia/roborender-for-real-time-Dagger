from copy import deepcopy
import pytest
from pipeline.deployment.evaluation_transfer_recovery import validate_absent_test

def fixture():
    request=dict(run_id='test',round_id=2,phase='evaluate')
    workers=[dict(physical_gpu=i,worker_instance_id=str(i)) for i in range(2)]
    pool=dict(source_sha256='a'*64,workers=workers)
    states=[dict(phase='IDLE',source_sha256='a'*64,worker_instance_id=str(i)) for i in range(2)]
    journals=deepcopy(states)
    for i,j in enumerate(journals):
        j['commands']={'train':dict(status='COMPLETE',accepted_at_ns=1,command=dict(command_id=f'test-r2-collect-g{i}'))}
    return request,pool,dict(quiescent=True,workers=states),journals

def test_unsubmitted_test_is_admitted():
    assert validate_absent_test(*fixture())=='test-r2-explicit-test'

@pytest.mark.parametrize('bad',['accepted','running','source','generation','latest','missing','missing_ready','incomplete','training'])
def test_uncertain_or_changed_native_state_rejected(bad):
    request,pool,status,journals=fixture()
    if bad=='accepted':journals[0]['commands']['test-r2-explicit-test']={}
    elif bad=='running':journals[0]['phase']='RUNNING'
    elif bad=='source':journals[0]['source_sha256']='b'*64
    elif bad=='generation':journals[0]['worker_instance_id']='other'
    elif bad=='latest':journals[0]['commands']['later']=dict(accepted_at_ns=2,command=dict(command_id='other'))
    elif bad=='missing':journals.pop()
    elif bad=='missing_ready':status['workers'].pop()
    elif bad=='incomplete':journals[0]['commands']['train']['status']='FAILED'
    elif bad=='training':request['phase']='collect'
    with pytest.raises(RuntimeError):validate_absent_test(request,pool,status,journals)
