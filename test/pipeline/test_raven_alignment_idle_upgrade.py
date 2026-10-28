"""No SSH/GPU calls: exact-owner lifecycle admission and uncertain ACK tests."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

PATH=Path(__file__).resolve().parents[2]/'runtime/raven-dagger-alignment-20261026-7pYLxI/idle_upgrade.py'
spec=importlib.util.spec_from_file_location('alignment_idle_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def inputs():
    audit=m.read(PATH.parent/'idle-audit-before-upgrade.json')
    row=audit['owners'][0]
    p=dict(owner=row['owner'],source_root='/unused',source_sha256=m.OLD_SOURCE,
        last=None,expected_commands={},shutdown_id='test-shutdown')
    return row,p


@pytest.mark.parametrize('change',['work','identity','busy','process'])
def test_changed_idle_boundary_is_rejected(change):
    row,p=inputs();row=deepcopy(row)
    if change=='work':row['state']['commands']['other']={'status':'COMPLETE'}
    elif change=='identity':row['state']['worker_instance_id']='other'
    elif change=='busy':row['ready']['phase']='RUNNING'
    else:row['actual']['host_pid']+=1
    with pytest.raises(ValueError):m.validate_unused_boundary(row,p)


def test_actual_authority_and_remote_program_compile_without_execution():
    assert len(m.authority())==2
    compile(m.REMOTE,'<no-execution>','exec')
    row,p=inputs();assert m.validate_unused_boundary(row,p)==row


def test_lost_shutdown_ack_preserves_intent_and_never_resubmits(tmp_path):
    row,p=inputs()
    class Transport:
        def __init__(self):self.actions=[]
        def remote(self,parameters,action):
            self.actions.append(action)
            if action=='observe':return row
            raise TimeoutError('unknown, not dead')
    transport=Transport();directory=tmp_path/'shutdown'
    with pytest.raises(TimeoutError):m.shutdown_once(directory,p,transport)
    assert m.read(directory/'unknown.json')['replay_authorized'] is False
    assert m.read(directory/'intent.json')==m.n.n.shutdown_body(p)
    with pytest.raises(TimeoutError):m.shutdown_once(directory,p,transport)
    assert transport.actions==['observe','shutdown','exit']


def test_busy_owner_gets_no_shutdown_intent(tmp_path):
    row,p=inputs();row['ready']['active_command_id']='someone-else'
    class Transport:
        def remote(self,parameters,action):
            assert action=='observe';return row
    with pytest.raises(ValueError):m.shutdown_once(tmp_path/'shutdown',p,Transport())
    assert not (tmp_path/'shutdown/intent.json').exists()
