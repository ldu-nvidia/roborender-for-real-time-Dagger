"""Network-only admission preserves exact workloads and rejects ambiguous replay."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
spec=importlib.util.spec_from_file_location('relay_test',ROOT/'network_relay_resumed.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_two_hops_only_forward_exact_retained_ports():
    argv=m.build_argv()
    assert set(argv)=={'private_h200','reverse_rtx'}
    assert 'ProxyCommand=none' in argv['private_h200']
    assert 'HostName=10.50.176.97' in argv['private_h200']
    assert argv['private_h200'].count('-L')==9
    assert argv['reverse_rtx'].count('-R')==9
    for remote,local in zip(m.REMOTE_PORTS,m.LOCAL_PORTS):
        assert f'127.0.0.1:{remote}:127.0.0.1:{local}' in argv['reverse_rtx']
    assert '127.0.0.1:28568:127.0.0.1:2222' in argv['private_h200']
    assert all('ExitOnForwardFailure=yes' in row for row in argv.values())
    assert not any('0.0.0.0' in arg for row in argv.values() for arg in row)

def test_native_transport_has_no_mutation_api(monkeypatch):
    monkeypatch.setattr(m,'query',lambda *_a,**_k:pytest.fail('mutating request attempted'))
    for action in ('submit','shutdown','launch','stage'):
        with pytest.raises(ValueError,match='read-only'):m.native_call({},action)

def test_existing_relay_never_prepared_again(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'OUT',tmp_path)
    with pytest.raises(ValueError,match='scope exists'):m.prepare()

def test_existing_launch_never_replayed(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'OUT',tmp_path)
    monkeypatch.setattr(m,'load',lambda:dict(argv={'private_h200':['ssh']}))
    (tmp_path/'private_h200-intent.json').write_text('{}')
    monkeypatch.setattr(m.subprocess,'Popen',lambda *_a,**_k:pytest.fail('duplicate accepted route'))
    with pytest.raises(ValueError,match='already accepted'):m.start('private_h200')

def test_accepted_candidate_blocks_route_admission(monkeypatch):
    entry=dict(owner_index=0,body={'command_id':'already-accepted'})
    monkeypatch.setattr(m.s,'read',lambda _:dict(run={'trials':[{'plan':{'commands':[entry],'owners':[{}]}}]}))
    monkeypatch.setattr(m,'native_call',lambda owner,action,**kw:
        dict(phase='IDLE',active_command_id=None) if action=='ready' else {'status':'COMPLETE'})
    with pytest.raises(ValueError,match='already accepted'):m.absent_candidate()
