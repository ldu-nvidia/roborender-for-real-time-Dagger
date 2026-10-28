import importlib.util
from pathlib import Path

import pytest

PATH=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN/native_upgrade_v5.py'

@pytest.fixture
def m():
    spec=importlib.util.spec_from_file_location('test_event_upgrade',PATH)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def test_completed_journal_remote_not_empty_journal_relabel(m):
    assert m.u.m.REMOTE==m.teacher.REMOTE
    assert "assert q['last'] is None" not in m.u.m.REMOTE
    assert "q['expected_commands']" in m.u.m.REMOTE
    assert m.u.m.validate_unused_boundary==m.teacher.validate_boundary

def test_new_source_runtime_and_authority_are_distinct(m):
    assert m.u.ROOT==m.ROOT and m.u.SOURCE==m.SOURCE and m.SOURCE!=m.OLD_SOURCE
    assert m.u.__file__==str(PATH)
    assert m.u.RUN.endswith('-v5')
    native,old=m.old_pair()
    config=m.config_for(old)
    assert config['native_root']!=str(Path(native['jobs_root']).parent)
    assert config['source_sha256']==m.SOURCE
    assert config['checkpoint_sha256'] is None
    for key in ('compute_release_authorized','native_replay_allowed','automatic_restart_allowed','h200_workload_changes_authorized'):
        assert config[key] is False

def test_parameters_bind_actual_complete_history(m,monkeypatch):
    owner={'physical_gpu':0};commands={'real':{'payload_sha256':'a'*64,'receipt':{'complete':True}}}
    row={'ready':{'current_event':{'command_id':'real'}},'state':{'commands':commands}}
    monkeypatch.setattr(m,'bound',lambda ref:{'owners':[row]})
    monkeypatch.setattr(m,'validate_audit',lambda value,native:value)
    native={'workers':[owner],'source_root':'/frozen/v4'}
    result=m.parameters({'old_native':native,'old_boundary':{},'shutdown_ids':['new-v5-stop']})[0]
    assert result['expected_commands'] is commands
    assert result['last']==dict(command_id='real',payload_sha256='a'*64,receipt_sha256=m.sha(m.canonical({'complete':True})))
    assert result['shutdown_id']=='new-v5-stop'

def test_changed_owner_rejected_before_generic_audit(m):
    native,_=m.old_pair();native=dict(native,source_sha256='f'*64)
    with pytest.raises(ValueError,match='previous owners'):
        m.validate_audit({},native)

def test_actual_junit_is_bound_as_xml_not_parsed_as_json(m):
    value=m.cpu_gate()
    assert value['path'].endswith('frozen-v5-event-gate-ready.xml')
    assert value['raw_sha256']==m.sha(Path(value['path']).read_bytes())
