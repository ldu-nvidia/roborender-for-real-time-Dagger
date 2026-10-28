"""Native stall detection is read-only and separate from CPU heartbeat."""
import importlib.util
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN/progress_probe_v4.py'
if not PATH.is_file():pytest.skip('dated observer absent',allow_module_level=True)
spec=importlib.util.spec_from_file_location('probe_v4_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def row(step=3,**kwargs):
    return dict(worker_instance_id='owner',phase='RUNNING',command_id='exact',
        progress=dict(phase='evaluation_rollout',vector_step=step,cohort_index=0,**kwargs))

def test_cpu_poll_or_trace_timestamp_cannot_hide_native_stall():
    state,faults=m.stalled([row(trace_mtime_ns=100)],{},100)
    assert not faults
    state,faults=m.stalled([row(trace_mtime_ns=9999)],state,1401)
    assert faults==['native_cursor_unchanged20minutes:owner:exact']

def test_actual_step_advancement_clears_stall_window():
    state,_=m.stalled([row()],{},100)
    state,faults=m.stalled([row(4)],state,1401)
    assert not faults and state['owner']['changed_unix']==1401

def test_phase_transition_or_new_command_is_not_replay_or_fault():
    state,_=m.stalled([row()],{},100)
    changed=row();changed['command_id']='new-exact'
    assert m.stalled([changed],state,2000)[1]==[]
    changed['progress']={'phase':'phase_changed_during_read'}
    assert m.stalled([changed],state,2000)==({},[])

def test_idle_is_not_stall_and_terminal_fault_is_not_restart():
    value=row();value['phase']='IDLE'
    assert m.stalled([value],{},9999)==({},[])
    value['phase']='CRASH_UNCERTAIN'
    assert m.stalled([value],{},9999)[1]==['native_phase:CRASH_UNCERTAIN:owner']
    assert "'/v1/commands'" not in m.REMOTE

def test_remote_tail_ignores_incomplete_record(tmp_path):
    import ast
    tree=ast.parse(m.REMOTE);func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tail')
    import json
    ns={'json':json};exec(compile(ast.Module(body=[func],type_ignores=[]),'readonly_tail','exec'),ns)
    path=tmp_path/'steps.jsonl';path.write_bytes(b'{"step":4}\n{"step":5')
    assert ns['tail'](path)['vector_step']==5
    path.write_bytes(b'{"step":4}\n{"step":5}\n')
    assert ns['tail'](path)['vector_step']==6
    link=tmp_path/'alias';link.symlink_to(path)
    with pytest.raises(ValueError):ns['tail'](link)
