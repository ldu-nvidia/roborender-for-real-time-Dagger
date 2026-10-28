from copy import deepcopy
import hashlib
from types import SimpleNamespace
import pytest
import torch

from test.deployment.rtx_actor.test_production_success_contract import accepted
from test.deployment.rtx_actor.test_explicit_motion_panel import result
from test.deployment.rtx_actor.test_native_production_success_admission import admission
from test.deployment.rtx_actor.test_rendered_student_success_contract import cohort_result, rendering_config
import rendered_student_success_contract as rendered
from explicit_motion_panel import summarize
from initial_robot_refresh_diagnostic import SMOKE_STUDENT_CLEAR_ONCE, clear_already_applied_initial_refresh, control_contract
import smoke_success_contract as contract
import production_success_contract as production


def corrected_result(panel, arm='bc'):
    config=rendering_config(panel)
    row=cohort_result(panel,arm=arm,config=config,successes=[True,False])
    value=summarize(panel,[row]);value['smoke_success_protocol']=contract.protocol_identity(panel)
    value['rendered_student_success_protocol']=rendered.protocol_identity(panel,config)
    return value


@pytest.mark.parametrize('arm',['bc','dagger'])
def test_corrected_smoke_and_production_are_separate(accepted,tmp_path,monkeypatch,arm):
    panel=contract.make_panel(accepted)
    call,checks,_,_=admission(tmp_path,monkeypatch,panel,production=False,arm=arm,
        options=contract.evaluation_options(panel))
    call()
    output=corrected_result(panel,arm)
    assert contract.validate_smoke_result(panel,output,'b'*64,
        rendered_config=rendering_config(panel))['success_rate']==.5
    with pytest.raises(ValueError): production.validate_production_result(panel,output,'b'*64)
    call,_,_,_=admission(tmp_path,monkeypatch,panel,production=True,arm=arm,
        options=contract.evaluation_options(panel))
    with pytest.raises(ValueError): call()


@pytest.mark.parametrize('fault',['reset','protocol','checkpoint','optimizer','count'])
def test_bad_smoke_receipt_rejected(accepted,fault):
    panel=contract.make_panel(accepted);value=corrected_result(panel)
    if fault=='reset':value['diagnostic_initial_robot_refresh']='raven_original'
    if fault=='protocol':value.pop('smoke_success_protocol')
    if fault=='checkpoint':value['checkpoint_sha256']='a'*64
    if fault=='optimizer':value['optimizer_steps']=1
    if fault=='count':value['completed_episodes']=1
    with pytest.raises(ValueError): contract.validate_smoke_result(panel,value,'b'*64)
