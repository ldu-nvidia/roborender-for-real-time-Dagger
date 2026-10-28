"""Process-local opt-in only; no CUDA or host-wide kernel changes."""
from test.deployment.source_paths import REPO_ROOT
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=REPO_ROOT
sys.path.insert(0,str(ROOT/'pipeline/deployment/rtx_actor'))
import native_thp_policy as thp
import run_production_resident_isaac as launcher
from bootstrap_production_resident_pool import validate_spec


def test_default_does_not_call_prctl():
    def forbidden(*args):raise AssertionError('default must be unchanged')
    value=thp.configure(thp.INHERITED,prctl=forbidden)
    assert thp.attest(value,prctl=forbidden)==value
    assert value['global_host_policy_modified'] is False
    assert 'disabled_after' not in value


def test_optin_sets_only_process_flag_and_attests_it():
    calls=[];state=0
    def call(option,value):
        nonlocal state
        calls.append((option,value))
        if option==thp.PR_GET_THP_DISABLE:return state
        assert option==thp.PR_SET_THP_DISABLE and value==1
        state=1;return 0
    receipt=thp.configure(thp.DISABLED,prctl=call)
    assert calls==[(42,0),(41,1),(42,0)]
    assert receipt['disabled_before']==0 and receipt['disabled_after']==1
    thp.attest(receipt,prctl=call)
    state=0
    with pytest.raises(RuntimeError,match='drifted'):thp.attest(receipt,prctl=call)
    assert calls[-1]==(42,0)  # No silent repair.


def test_already_disabled_does_not_set_again():
    calls=[]
    value=thp.configure(thp.DISABLED,prctl=lambda op,arg:calls.append((op,arg)) or 1)
    assert calls==[(42,0),(42,0)] and value['disabled_before']==1


@pytest.mark.parametrize('policy',[None,'disabled','never',True,1])
def test_bad_policy_never_changes_process(policy):
    with pytest.raises(ValueError):
        thp.configure(policy,prctl=lambda *_:pytest.fail('must reject before prctl'))


@pytest.mark.parametrize('returns',[[3],[0,-1],[0,0,0]])
def test_failed_or_unexpected_kernel_state_fails_closed(returns):
    values=iter(returns)
    with pytest.raises(RuntimeError):thp.configure(thp.DISABLED,prctl=lambda *_:next(values))


def test_receipt_cannot_be_reused_by_different_process():
    value=thp.configure(thp.INHERITED);value['native_pid']+=1
    with pytest.raises(RuntimeError,match='another process'):thp.attest(value)


def test_optin_environment_is_pinned_in_container_not_host(monkeypatch,tmp_path):
    base=['docker','run','--memory','21g','--memory-swap','21g',launcher.IMAGE,'num_envs=6']
    monkeypatch.setattr(launcher,'build_resident_command',lambda **_:list(base))
    kwargs=dict(motion_inventory_sha256='a'*64,jobs=tmp_path/'jobs')
    ordinary=launcher.production_command(**kwargs)
    assert not any(x.startswith(thp.ENV+'=') for x in ordinary)
    opted=launcher.production_command(**kwargs,native_thp_policy=thp.DISABLED)
    pos=opted.index(thp.ENV+'='+thp.DISABLED)
    assert opted[pos-1]=='--env' and pos<opted.index(launcher.IMAGE)
    assert opted[:pos-1]+opted[pos+1:]==ordinary


def test_native_applies_optin_before_importing_scene_dependencies():
    tree=ast.parse((ROOT/'pipeline/deployment/rtx_actor/resident_isaac_trainer.py').read_text())
    setting=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='NATIVE_THP_CONFIGURATION' for t in n.targets))
    torch=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Import)
        and any(a.name=='torch' for a in n.names))
    assert setting<torch


def test_bootstrap_spec_explicit_optin_and_unknown_field_rejection():
    value=dict(schema='roborender.production-b2-bootstrap.v1',source_root='/tmp/source',
        source_manifest='/tmp/manifest',source_sha256='a'*64,run_id='standing-isaac-leo-prod-test',
        root='/home/ldu/roborender/runtime/standing-isaac/standing-isaac-leo-prod-test',python='/usr/bin/python3',
        motion_inventory=dict(path='/tmp/inventory',raw_sha256='b'*64))
    assert validate_spec(value)==value
    assert validate_spec(dict(value,native_thp_policy=thp.DISABLED))['native_thp_policy']==thp.DISABLED
    with pytest.raises(ValueError):validate_spec(dict(value,native_thp_policy='relax-pressure'))
    with pytest.raises(ValueError):validate_spec(dict(value,native_thp_policy=thp.DISABLED,extra=True))


@pytest.mark.skipif(sys.platform!='linux',reason='Linux prctl only')
def test_real_optin_is_confined_to_short_lived_child_process():
    # Never change the pytest process. The child has no CUDA/Isaac imports.
    before=thp._prctl(thp.PR_GET_THP_DISABLE)
    code=('import sys,json;sys.path.insert(0,sys.argv[1]);import native_thp_policy as t;'
          'print(json.dumps(t.attest(t.configure(t.DISABLED))))')
    result=subprocess.run([sys.executable,'-c',code,str(ROOT/'pipeline/deployment/rtx_actor')],
                          capture_output=True,text=True,timeout=10,check=True)
    value=json.loads(result.stdout)
    assert value['native_pid']!=os.getpid() and value['disabled_after']==1
    assert thp._prctl(thp.PR_GET_THP_DISABLE)==before
