"""Owner-host binding and acknowledged multiprocess commit durability."""
from test.deployment.source_paths import REPO_ROOT
import concurrent.futures
import ast
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3

import pytest

from pipeline.dagger.services import ledger_owner
from pipeline.dagger.services.store import StateStore


def _writer(root, worker):
    store = StateStore(Path(root))
    receipts = []
    for index in range(40):
        key = f'{worker}:{index}'
        with store.transaction() as db:
            db.execute('INSERT INTO durability_test VALUES (?,?)', (key, index))
        # An ACK is exposed only after transaction context commit returns.
        receipts.append(key)
    return receipts


def _crash_before_ack(root):
    store = StateStore(Path(root))
    with store.transaction() as db:
        db.execute('INSERT INTO durability_test VALUES (?,?)', ('never-acknowledged', 1))
        os._exit(17)  # Isolated test child; exercise hot rollback-journal recovery.


def test_four_writers_every_ack_survives_reopen(tmp_path):
    store = StateStore(tmp_path)
    with store.transaction() as db:
        db.execute('CREATE TABLE durability_test (key TEXT PRIMARY KEY, value INTEGER)')
    with concurrent.futures.ProcessPoolExecutor(4, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_writer, str(tmp_path), i) for i in range(4)]
        acks = [key for future in futures for key in future.result(timeout=60)]
    with StateStore(tmp_path).connection() as db:
        assert db.execute('PRAGMA journal_mode').fetchone()[0] == 'delete'
        assert db.execute('PRAGMA synchronous').fetchone()[0] == 2
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert {row[0] for row in db.execute('SELECT key FROM durability_test')} == set(acks)
    assert len(acks) == len(set(acks)) == 160


def test_dead_writer_cannot_publish_unacknowledged_row(tmp_path):
    store = StateStore(tmp_path)
    with store.transaction() as db:
        db.execute('CREATE TABLE durability_test (key TEXT PRIMARY KEY, value INTEGER)')
        db.execute('INSERT INTO durability_test VALUES (?,?)', ('acknowledged', 0))
    process = multiprocessing.get_context('spawn').Process(target=_crash_before_ack, args=(str(tmp_path),))
    process.start()
    process.join(10)
    assert process.exitcode == 17
    with StateStore(tmp_path).connection() as db:
        assert [row[0] for row in db.execute('SELECT key FROM durability_test')] == ['acknowledged']


def test_owner_change_rejected_before_sqlite_open(tmp_path, monkeypatch):
    store = StateStore(tmp_path)
    monkeypatch.setattr(ledger_owner, 'host_identity', lambda: dict(hostname='foreign', boot_id='foreign'))
    monkeypatch.setattr(sqlite3, 'connect', lambda *a, **k: pytest.fail('must not open foreign ledger'))
    with pytest.raises(RuntimeError, match='another host/lifetime'):
        with store.connection():
            pass
    with pytest.raises(RuntimeError, match='another host/lifetime'):
        StateStore(tmp_path)
    with pytest.raises(RuntimeError, match='another host/lifetime'):
        ledger_owner.open_readonly(store.path)


def test_existing_unowned_wal_is_not_silently_converted(tmp_path):
    with sqlite3.connect(tmp_path / 'state.sqlite3') as db:
        assert db.execute('PRAGMA journal_mode=WAL').fetchone()[0] == 'wal'
        db.execute('CREATE TABLE untouched (value TEXT)')
    with pytest.raises(RuntimeError, match='offline reconciled import'):
        StateStore(tmp_path)
    assert not (tmp_path / 'ledger-owner.json').exists()
    with sqlite3.connect(tmp_path / 'state.sqlite3') as db:
        assert db.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'


def test_exception_rolls_back_and_marker_is_private(tmp_path):
    store = StateStore(tmp_path)
    marker = tmp_path / 'ledger-owner.json'
    assert marker.stat().st_mode & 0o077 == 0
    assert json.loads(marker.read_text()) == ledger_owner.expected_owner()
    with store.transaction() as db:
        db.execute('CREATE TABLE transaction_test (value INTEGER)')
    with pytest.raises(ValueError):
        with store.transaction() as db:
            db.execute('INSERT INTO transaction_test VALUES (1)')
            raise ValueError('no acknowledgement')
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM transaction_test').fetchone()[0] == 0


def test_symlink_marker_rejected(tmp_path):
    outside = tmp_path / 'outside.json'
    outside.write_text(json.dumps(ledger_owner.expected_owner()))
    (tmp_path / 'ledger-owner.json').symlink_to(outside)
    with pytest.raises(RuntimeError, match='missing or unsafe'):
        StateStore(tmp_path)


def test_query_only_reader_has_lock_capable_handle_and_rejects_writes(tmp_path, monkeypatch):
    store = StateStore(tmp_path)
    with store.transaction() as db:
        db.execute('CREATE TABLE reader_test (value INTEGER)')
        db.execute('INSERT INTO reader_test VALUES (7)')
    original = sqlite3.connect
    opened = []
    def observed(database, **kwargs):
        opened.append(database)
        return original(database, **kwargs)
    monkeypatch.setattr(ledger_owner.sqlite3, 'connect', observed)
    db = ledger_owner.open_readonly(store.path)
    try:
        assert opened == [store.path.resolve().as_uri() + '?mode=rw']
        assert db.execute('PRAGMA query_only').fetchone()[0] == 1
        assert db.execute('SELECT value FROM reader_test').fetchone()[0] == 7
        for sql in ('INSERT INTO reader_test VALUES (8)', 'DELETE FROM reader_test',
                    'CREATE TABLE forbidden (x INTEGER)', 'DROP TABLE reader_test'):
            with pytest.raises(sqlite3.OperationalError, match='readonly'):
                db.execute(sql)
    finally:
        db.close()


def test_query_only_reader_never_creates_missing_database(tmp_path):
    StateStore(tmp_path)
    missing = tmp_path / 'missing.sqlite3'
    with pytest.raises(sqlite3.OperationalError, match='unable to open'):
        ledger_owner.open_readonly(missing)
    assert not missing.exists()


@pytest.mark.parametrize('relative', [
    'pipeline/dagger/services/raven_learner_worker.py',
    'pipeline/deployment/h200/runtime_check.py',
    'pipeline/deployment/h200/migration_smoke_control.py',
    'pipeline/dagger/monitor_live_round.py',
    'pipeline/dagger/materialize_next_round.py',
    'pipeline/dagger/overnight_round_supervisor.py',
    'pipeline/dagger/export_heldout_manifest.py',
])
def test_live_readers_cannot_bypass_owner_binding(relative):
    source = (REPO_ROOT / relative).read_text()
    tree = ast.parse(source)
    assert 'open_readonly' in source
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and isinstance(node.func.value, ast.Name)
                   and node.func.value.id == 'sqlite3' and node.func.attr == 'connect'
                   for node in ast.walk(tree))
