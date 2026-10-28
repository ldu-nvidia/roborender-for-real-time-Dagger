"""CPU settings/dispatch fixtures, not native memory qualification."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pipeline.deployment.mode_site import ModeSite, native_policy_from_settings


def base():
    return dict(schema='roborender.mode-site.v1',site={},retained={},bc_capacity={},compute_release_authorized=False)


def test_old_settings_default_is_unchanged():
    assert native_policy_from_settings(base())=='inherited'


@pytest.mark.parametrize('policy',['inherited','disable-v1'])
def test_explicit_versioned_policy(policy):
    assert native_policy_from_settings(dict(base(),schema='roborender.mode-site.v2',native_thp_policy=policy))==policy


@pytest.mark.parametrize('mutation',[
    {'native_thp_policy':'disable-v1'}, {'schema':'roborender.mode-site.v2'},
    {'schema':'roborender.mode-site.v2','native_thp_policy':'relax-watchdog'},
    {'schema':'roborender.mode-site.v2','native_thp_policy':'disable-v1','extra':True}])
def test_policy_requires_exact_versioned_keys(mutation):
    with pytest.raises(ValueError):native_policy_from_settings(dict(base(),**mutation))


@pytest.mark.parametrize('policy',['inherited','disable-v1'])
def test_bootstrap_and_pool_gate_bind_optin_only(tmp_path,policy):
    inventory=tmp_path/'inventory.json';inventory.write_bytes(b'{"motion_count":295}')
    obj=ModeSite.__new__(ModeSite);obj.native_thp_policy=policy
    obj.plan=dict(mode='production',source_sha256='a'*64,settings=dict(inventory=dict(
        path=str(inventory),raw_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest())))
    obj.site=dict(rtx_python='/python',rtx_source_manifest='/manifest')
    obj.remote=tmp_path/'remote';obj.source=tmp_path/'source';obj.poolrun='owned';obj.poolroot=tmp_path/'pool'
    obj.poolpath=obj.poolroot/'pool.json';obj.host=Mock();obj.native=Mock();obj.read=Mock(return_value={'test':True})
    obj._bootstrap_start()
    spec=json.loads(obj.host.publish.call_args_list[-1].args[1])
    assert spec.get('native_thp_policy','inherited')==policy
    assert ('native_thp_policy' in spec)==(policy!='inherited')
    obj.host.process.return_value={'state':'COMPLETE'}
    assert obj._pool()=={'test':True}
    if policy=='inherited':obj.native.assert_not_called()
    else:
        argv=obj.native.call_args.args[0]
        assert argv[-1]=='disable-v1' and argv[-2]==obj.poolpath
        assert 'validate_native_policy' in argv[2]
        obj.native.side_effect=RuntimeError('failed attestation')
        with pytest.raises(RuntimeError,match='attestation'):obj._pool()
