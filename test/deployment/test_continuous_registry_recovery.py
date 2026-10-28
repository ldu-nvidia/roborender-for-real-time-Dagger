import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
def module():
    spec=importlib.util.spec_from_file_location('pre_native_registry_recovery_test',ROOT/'continuous_registry_recovery.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

@pytest.mark.parametrize('mutation',['live','posts','diagnostic','unknown','other_failure'])
def test_only_exact_pre_native_rejection_can_resume(mutation):
    m=module();failure=dict(error_type='ServiceError',message='400 contract_error: checkpoint path must be below /safe')
    posts=[];diagnostic=False;identity=lambda _:None
    if mutation=='live':identity=lambda _:{'pid':1}
    if mutation=='posts':posts=['accepted.json']
    if mutation=='diagnostic':diagnostic=True
    if mutation=='unknown':failure['error_type']='ConnectionError'
    if mutation=='other_failure':failure['message']='400 other failure'
    with pytest.raises(ValueError,match='pre-native'):m.check_boundary(failure,{'identity':{'pid':1}},identity,posts,diagnostic)

def test_exited_exact_rejection_is_a_safe_pre_native_boundary():
    m=module();m.check_boundary(dict(error_type='ServiceError',message='400 contract_error: checkpoint path must be below /safe'),
        {'identity':{'pid':1}},lambda _:None,[],False)
    source=(ROOT/'continuous_registry_recovery.py').read_text()
    assert 'site.prepare(' not in source and '.supervisor(' not in source
    assert 'services_replaced=False' in source and 'native_replay_allowed=False' in source
    assert "controller.ensure_registered()" in source and "api.collect(site,sleep=drive)" in source
