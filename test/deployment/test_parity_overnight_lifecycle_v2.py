"""Fresh version namespace retains exact idle-owner and no-replay guards."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
PATH=ROOT/'native_upgrade_v2.py'
if not PATH.is_file():pytest.skip('runtime adapter not installed',allow_module_level=True)
spec=importlib.util.spec_from_file_location('parity_native_v2_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_fresh_namespace_preserves_unused_v1_and_exact_old_pair():
    assert m.ROOT==ROOT/'native-v2'
    native,old=m.old_pair()
    assert [r['worker_instance_id'] for r in native['workers']]==m.OLD_WORKERS
    cfg=m.config_for(old)
    assert cfg['source_sha256']==m.SOURCE!=m.OLD_SOURCE
    assert cfg['run_id'].endswith('-v2') and cfg['initial_num_envs']==2
    assert cfg['compute_release_authorized'] is cfg['automatic_restart_allowed'] is cfg['native_replay_allowed'] is False
    assert cfg['checkpoint_loaded_at_startup'] is False

def test_current_authority_and_deadline(monkeypatch):
    assert m.authority()['path']==str(ROOT/'user-authorization.json')
    monkeypatch.setattr(m.time,'time',lambda:m.read(ROOT/'user-authorization.json')['deadline_unix'])
    with pytest.raises(ValueError,match='horizon'):m.authority(new_work=True)
    assert m.authority()

def test_failed_frozen_gate_cannot_upgrade(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'AUTH_ROOT',tmp_path)
    (tmp_path/'frozen-v2-ready.xml').write_text('<testsuites><testsuite tests="436" failures="2" errors="0" skipped="0"/></testsuites>')
    with pytest.raises(ValueError,match='gate'):m.cpu_gate()

def test_wrong_gpu_cannot_launch(monkeypatch):
    _,old=m.old_pair()
    monkeypatch.setattr(m,'parameters',lambda _:['two exact exited owners'])
    for value in (2,-1,True):
        with pytest.raises(ValueError):m.launch_parameters({'config':m.config_for(old)},value)
    p=m.launch_parameters({'config':m.config_for(old)},1)
    assert p['gpu']==1 and p['old_owners']==['two exact exited owners']

def test_lifecycle_remains_exact_guarded_native_commands():
    source=PATH.read_text()
    assert 'm.shutdown_once(' in source and 'm.n.n.launch_once(' in source
    assert "m.validate_unused_boundary(t.remote(p,'observe'),p)" in source
    assert 'subprocess.Popen' not in source and 'release_allocation' not in source
    assert 'frozen-v2-ready.xml' in source
