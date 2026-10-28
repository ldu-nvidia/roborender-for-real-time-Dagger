import importlib.util
from pathlib import Path
import pytest
PATH=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN/native_launch_preflight_retry_v5.py'
spec=importlib.util.spec_from_file_location('preflight_retry_guard',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def absent():
    return dict(remote_launch_intent_absent=True,token_absent=True,runtime_absent=True,
        container_absent=True,port_bind_available=True,read_only=True,
        unit=dict(LoadState='not-found',MainPID='0'),physical_gpu=0,compute_release_authorized=False)

def test_exact_prelaunch_absence_passes():assert m.validate_absence(absent())==absent()

@pytest.mark.parametrize('key',['remote_launch_intent_absent','token_absent','runtime_absent','container_absent','port_bind_available','read_only'])
def test_uncertain_presence_blocks_retry(key):
    row=absent();row[key]=False
    with pytest.raises(ValueError,match='absence'):m.validate_absence(row)

def test_guard_never_deletes_original_intents_or_mutates_allocation():
    source=PATH.read_text()
    assert "if intent.exists():return transport.remote(parameters,'status')" in source
    assert 'unlink(' not in source and 'kill(' not in source and 'release(' not in source
    compile(m.ABSENT,'<readonly-preflight-absence>','exec')
