"""Serialize host-heavy scene rebuilds, not physics steps or inference.

Both containers bind the same host-owned inode read-only. flock is released by
the kernel on owner exit; there is no stale lock-file deletion or native replay.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
import sys
import time


def prepare_scene_lock(pool_root):
    """Create one public empty lock inode before new native lifetimes launch."""
    root=Path(pool_root).resolve(strict=True)
    path=root/'scene-reconfigure.lock'
    try:
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o444)
    except FileExistsError:
        pass
    else:
        os.close(fd)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        value=os.fstat(fd)
        if (not stat.S_ISREG(value.st_mode) or value.st_size!=0
                or value.st_uid!=os.getuid() or value.st_mode&0o222):
            raise RuntimeError('shared scene lock must be empty, owned and read-only')
    finally:
        os.close(fd)
    return path


@contextmanager
def scene_reconfiguration_lock(report, *, path=None, timeout=120., clock=time.monotonic, sleep=time.sleep):
    selected=path if path is not None else os.environ.get('RESIDENT_SCENE_LOCK_FILE')
    if not selected:
        # Historical releases/isolated diagnostics have no shared scene lock.
        yield
        return
    if not 0<timeout<=120:
        raise ValueError('scene admission needs a bounded timeout')
    selected=Path(selected)
    if not selected.is_absolute():raise ValueError('absolute shared scene lock required')
    fd=os.open(selected,os.O_RDONLY|os.O_NOFOLLOW)
    locked=False; started=clock()
    try:
        value=os.fstat(fd)
        if not stat.S_ISREG(value.st_mode) or value.st_size!=0 or value.st_mode&0o222:
            raise RuntimeError('invalid shared scene lock inode')
        report('scene_admission_wait',timeout_seconds=timeout)
        while True:
            try:
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                locked=True
                break
            except BlockingIOError:
                elapsed=clock()-started
                if elapsed>=timeout:raise TimeoutError('shared scene admission timed out; no native replay')
                sleep(min(.1,timeout-elapsed))
        report('scene_admission_acquired',wait_seconds=clock()-started)
        yield
    finally:
        primary=sys.exc_info()[1]
        if locked:
            fcntl.flock(fd,fcntl.LOCK_UN)
        os.close(fd)
        if locked:
            try:report('scene_admission_released',elapsed_seconds=clock()-started)
            except Exception:
                if primary is None:raise
