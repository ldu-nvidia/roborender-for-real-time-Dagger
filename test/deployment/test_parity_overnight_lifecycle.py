"""Runtime-only overnight orchestration guard tests; never invoke SSH/CUDA."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

PATH=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN/native_upgrade.py'
if not PATH.is_file():
    pytest.skip('operational overnight helper not installed',allow_module_level=True)
spec=importlib.util.spec_from_file_location('overnight_lifecycle_tests',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_current_authority_has_no_release_or_replay_permission():
    assert m.authority()['path']==str(PATH.parent/'user-authorization.json')
    value=m.read(PATH.parent/'user-authorization.json')
    assert value['deadline_unix']-value['started_unix']==8*3600
    assert value['compute_release_authorized'] is False
    assert value['native_unknown_replay_authorized'] is False


def test_deadline_blocks_new_work_but_not_readonly_reconciliation(monkeypatch):
    deadline=m.read(PATH.parent/'user-authorization.json')['deadline_unix']
    monkeypatch.setattr(m.time,'time',lambda:deadline)
    with pytest.raises(ValueError,match='horizon'):m.authority(new_work=True)
    assert m.authority()


def test_fresh_release_and_existing_exact_pair_are_separate():
    native,old=m.old_pair();cfg=m.config_for(old)
    assert native['source_sha256']==m.OLD_SOURCE!=cfg['source_sha256']==m.SOURCE
    assert cfg['checkpoint_sha256'] is None and cfg['checkpoint_loaded_at_startup'] is False
    assert cfg['compute_release_authorized'] is cfg['automatic_restart_allowed'] is cfg['native_replay_allowed'] is False
    assert cfg['native_root']!=old['native_root'] and cfg['leo_source_root']!=old['leo_source_root']
    assert m.cpu_gate()['path']==str(PATH.parent/'frozen-focused.xml')


def test_failed_or_skipped_frozen_gate_rejects(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    (tmp_path/'frozen-focused.xml').write_text('<testsuites><testsuite tests="319" failures="1" errors="0" skipped="0"/></testsuites>')
    with pytest.raises(ValueError,match='gate'):m.cpu_gate()


def test_launch_parameters_cannot_change_gpu_or_replay_old_owner(monkeypatch):
    native,old=m.old_pair();cfg=m.config_for(old)
    monkeypatch.setattr(m,'parameters',lambda p:[{'old':'owned'}])
    plan={'config':cfg}
    p=m.launch_parameters(plan,1)
    assert p['gpu']==1 and p['old_owners']==[{'old':'owned'}]
    assert p['config_sha256']==m.sha(m.canonical(cfg))
    assert '--execute' in p['argv']
    for invalid in (2,True,-1):
        with pytest.raises(ValueError):m.launch_parameters(plan,invalid)


def test_actual_program_reuses_guarded_one_shot_primitives():
    compile(m.m.REMOTE,'<reviewed-idle-program>','exec')
    source=PATH.read_text()
    assert 'm.shutdown_once(' in source and 'm.n.n.launch_once(' in source
    assert 'release_allocation' not in source and 'subprocess.Popen' not in source
