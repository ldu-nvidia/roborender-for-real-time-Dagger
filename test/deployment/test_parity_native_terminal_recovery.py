"""No GPU actions: the new attempt cannot impersonate terminal failed work."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
PATH=ROOT/'native_recovery_v4.py'
if not PATH.is_file():pytest.skip('runtime adapter absent',allow_module_level=True)
spec=importlib.util.spec_from_file_location('parity_native_terminal_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_failed_commands_are_exact_not_synthetic_shutdown():
    rows=m.old_parameters()
    assert len(rows)==2 and [r['command_id'] for r in rows]==['live-smoke-parity-20261026-v2-b4-c000','live-smoke-parity-20261026-v2-b4-c001']
    assert all('shutdown_id' not in row for row in rows)
    assert "s['phase']=='CRASH_UNCERTAIN'" in m.CHECK
    assert "command['payload_sha256']==q['payload_sha256']" in m.CHECK
    assert "state.get('MainPID')=='0'" in m.CHECK
    assert "not pathlib.Path('/proc',str(o['host_pid'])).exists()" in m.CHECK

def test_no_replay_no_shutdown_api_and_fresh_scope():
    t=m.Transport()
    for action in ('shutdown','observe','reset','resume'):
        with pytest.raises(ValueError):t.remote({},action)
    old=m.read(m.u.ROOT/'upgrade-plan.json')['config'];new=m.config_for(old)
    assert new['run_id'].endswith('-v4') and new['native_root']!=old['native_root']
    assert new['compute_release_authorized'] is new['automatic_restart_allowed'] is new['native_replay_allowed'] is False
    assert new['checkpoint_sha256'] is None and new['checkpoint_loaded_at_startup'] is False

def test_fresh_launch_retains_source_memory_port_and_old_exit_guards():
    assert 'for q in p[\'old_owners\']:check_exit(q)' in m.REMOTE
    assert 'verify_source(str(source)' in m.REMOTE
    assert "float(used)<=2048" in m.REMOTE
    assert "listener.bind(('127.0.0.1',18640+gpu))" in m.REMOTE
    assert "not marker.exists()" in m.REMOTE
    compile(m.REMOTE,'<tested-native-recovery>','exec')

def test_new_work_still_respects_actual_eight_hour_authority(monkeypatch):
    assert m.authority()['path']==str(ROOT/'user-authorization.json')
    monkeypatch.setattr(m.u.time,'time',lambda:m.read(ROOT/'user-authorization.json')['deadline_unix'])
    with pytest.raises(ValueError,match='horizon'):m.authority(new_work=True)

def test_actual_xml_gate_is_bound_as_bytes_not_parsed_as_json():
    ref=m.cpu_gate()
    assert ref['path'].endswith('/frozen-v4-complete.xml')
    assert ref['raw_sha256']==m.sha(Path(ref['path']).read_bytes())
