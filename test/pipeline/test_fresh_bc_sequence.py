"""Bounded fresh-parent sequencing, no remote/GPU work in these tests."""
import importlib.util
import json
from pathlib import Path

import pytest

PATH=Path(__file__).resolve().parents[2]/'runtime/raven-dagger-alignment-20261026-7pYLxI/fresh_bc_sequence.py'
if not PATH.is_file():
    pytest.skip('optional run-local fresh-parent controller absent',allow_module_level=True)
spec=importlib.util.spec_from_file_location('fresh_bc_sequence_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_existing_sequence_cannot_create_second_controller(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'CONTROL',tmp_path)
    monkeypatch.setattr(m,'wait_collector',lambda:pytest.fail('duplicate controller'))
    with pytest.raises(FileExistsError):m.run()


def test_accepted_or_unknown_command_never_replays(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'CONTROL',tmp_path)
    calls=[]
    def uncertain(*args,**kwargs):calls.append(args);raise TimeoutError('UNKNOWN')
    monkeypatch.setattr(m,'command',uncertain)
    with pytest.raises(TimeoutError):m.once('render',['exact-command'])
    with pytest.raises(FileExistsError):m.once('render',['exact-command'])
    assert len(calls)==1


@pytest.mark.parametrize('state',['failed','activating','deactivating'])
def test_no_native_launch_on_uncertain_collector_state(state,monkeypatch):
    monkeypatch.setattr(m,'command',lambda *a,**k:'ActiveState='+state+'\nMainPID=123\nSubState=unknown\nExecMainStatus=1')
    with pytest.raises(RuntimeError,match='reconcile'):m.wait_collector()


def test_clean_collector_exit_returns_without_replaying_collection(monkeypatch):
    calls=[]
    def observed(argv,**kwargs):
        calls.append(argv);return 'ActiveState=inactive\nMainPID=0\nSubState=dead\nExecMainStatus=0'
    monkeypatch.setattr(m,'command',observed)
    m.wait_collector()
    assert len(calls)==1 and 'systemctl --user show' in calls[0][-1]


@pytest.mark.parametrize('terminal',[{'exit_code':1,'failure':None},{'exit_code':0,'failure':{'type':'fault'}}])
def test_failed_phase_is_not_completion(tmp_path,monkeypatch,terminal):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    control=tmp_path/'fresh-render-control';control.mkdir()
    (control/'terminal.json').write_text(json.dumps(terminal))
    with pytest.raises(RuntimeError,match='failed'):m.wait_phase('render')


def test_successful_phase_requires_real_terminal_file(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    control=tmp_path/'fresh-render-control';control.mkdir()
    (control/'terminal.json').write_text(json.dumps({'exit_code':0,'failure':None}))
    assert m.wait_phase('render')=={'exit_code':0,'failure':None}
