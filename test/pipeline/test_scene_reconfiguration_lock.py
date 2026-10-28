"""Real OS-lock concurrency without Isaac, CUDA, or allocation operations."""
from test.deployment.source_paths import PIPELINE_ROOT
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Event

import pytest

sys.path.insert(0,str(PIPELINE_ROOT/'deployment/rtx_actor'))
from scene_reconfiguration_lock import prepare_scene_lock,scene_reconfiguration_lock


def test_two_owners_never_overlap_scene_creation(tmp_path):
    path=prepare_scene_lock(tmp_path)
    first_entered=Event();release_first=Event();second_entered=Event();records=[]
    def first():
        with scene_reconfiguration_lock(lambda event,**kw:records.append(('first',event)),path=path):
            first_entered.set()
            assert release_first.wait(3)
    def second():
        with scene_reconfiguration_lock(lambda event,**kw:records.append(('second',event)),path=path):
            second_entered.set()
    with ThreadPoolExecutor(2) as pool:
        a=pool.submit(first);assert first_entered.wait(2)
        b=pool.submit(second)
        assert not second_entered.wait(.2)
        release_first.set();a.result(timeout=3);b.result(timeout=3)
    assert second_entered.is_set()
    assert records.count(('first','scene_admission_acquired'))==1
    assert records.count(('second','scene_admission_acquired'))==1


def test_timeout_does_not_enter_or_delete_held_lock(tmp_path):
    path=prepare_scene_lock(tmp_path);inode=path.stat().st_ino
    with scene_reconfiguration_lock(lambda *a,**kw:None,path=path):
        with pytest.raises(TimeoutError,match='no native replay'):
            with scene_reconfiguration_lock(lambda *a,**kw:None,path=path,timeout=.05):
                pytest.fail('must not enter a held scene')
    assert path.stat().st_ino==inode
    with scene_reconfiguration_lock(lambda *a,**kw:None,path=path):pass


def test_exception_releases_lock_without_changing_file(tmp_path):
    path=prepare_scene_lock(tmp_path)
    with pytest.raises(ValueError):
        with scene_reconfiguration_lock(lambda *a,**kw:None,path=path):raise ValueError('native failed')
    with scene_reconfiguration_lock(lambda *a,**kw:None,path=path):pass
    assert path.read_bytes()==b''
    assert path.stat().st_mode&0o777==0o444


def test_legacy_no_flag_has_no_io_or_reports(monkeypatch):
    monkeypatch.delenv('RESIDENT_SCENE_LOCK_FILE',raising=False)
    with scene_reconfiguration_lock(lambda *a,**kw:pytest.fail('unexpected report')):pass


@pytest.mark.parametrize('fault',['writable','symlink','nonempty'])
def test_invalid_lock_refused(tmp_path,fault):
    path=tmp_path/'scene-reconfigure.lock'
    if fault=='symlink':
        target=tmp_path/'target';target.touch(mode=0o444);path.symlink_to(target)
    else:
        path.write_bytes(b'x' if fault=='nonempty' else b'')
        path.chmod(0o444 if fault=='nonempty' else 0o644)
    with pytest.raises((OSError,RuntimeError)):prepare_scene_lock(tmp_path)


def test_prepare_is_idempotent_same_inode(tmp_path):
    first=prepare_scene_lock(tmp_path);identity=first.stat().st_ino
    assert prepare_scene_lock(tmp_path).stat().st_ino==identity


def test_release_report_failure_preserves_native_failure(tmp_path):
    path=prepare_scene_lock(tmp_path)
    original=ValueError('native operation failed')
    def report(event,**kw):
        if event=='scene_admission_released':raise RuntimeError('report failed')
    with pytest.raises(ValueError) as caught:
        with scene_reconfiguration_lock(report,path=path):raise original
    assert caught.value is original
    with scene_reconfiguration_lock(lambda *a,**kw:None,path=path):pass
