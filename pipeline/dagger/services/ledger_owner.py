"""One-host ledger ownership; rollback journals avoid cross-host WAL mappings.

Remote diagnostics must use the owner's HTTP/SSH interface, never open the live
shared database. This guard covers application clients, not arbitrary sqlite3
commands. Existing unowned/WAL ledgers require an offline reconciled import.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import socket
import sqlite3

POLICY = 'roborender.sqlite-single-host-delete-full.v1'


def host_identity():
    return {'hostname': socket.gethostname(),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def expected_owner():
    return {'schema': POLICY, **host_identity()}


def check_owner(root, expected):
    path = Path(root) / 'ledger-owner.json'
    if path.is_symlink() or not path.is_file():
        raise RuntimeError('ledger owner binding missing or unsafe')
    if expected != expected_owner() or json.loads(path.read_text()) != expected:
        raise RuntimeError('live ledger belongs to another host/lifetime; use owner RPC')


def open_readonly(database, *, timeout=5):
    """Owner-only, query-only SQL with a lock-capable existing-file handle.

    The retained virtiofs mount rejects F_GETLK/F_WRLCK on O_RDONLY handles
    with EINVAL, exposed as SQLITE_IOERR_CHECKRESERVEDLOCK during a writer's
    rollback journal. mode=rw permits that lock probe without creating a DB;
    query_only forbids SQL writes. Do not use immutable/nolock on a live DB.
    """
    database = Path(database).resolve()
    check_owner(database.parent, expected_owner())
    connection = sqlite3.connect(database.as_uri() + '?mode=rw', uri=True, timeout=timeout)
    try:
        connection.execute('PRAGMA query_only=ON')
        return connection
    except BaseException:
        connection.close()
        raise


@contextmanager
def initialize_owner(root):
    root = Path(root)
    fd = os.open(root / 'ledger-owner.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        expected = expected_owner()
        marker = root / 'ledger-owner.json'
        database = root / 'state.sqlite3'
        if not marker.exists():
            if database.exists() or database.is_symlink():
                raise RuntimeError('unowned existing ledger requires offline reconciled import')
            data = json.dumps(expected, sort_keys=True, separators=(',', ':')).encode()
            out = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            try:
                with os.fdopen(out, 'wb') as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except BaseException:
                # Preserve an incomplete marker for explicit reconciliation.
                raise
        check_owner(root, expected)
        if database.is_symlink():
            raise RuntimeError('ledger database symlink forbidden')
        yield expected
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
