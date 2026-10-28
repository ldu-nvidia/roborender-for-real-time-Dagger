import sqlite3

import pytest

from pipeline.deployment.h200.runtime_check import window_audit
from pipeline.dagger.services.ledger_owner import initialize_owner


@pytest.mark.parametrize("count,gpus", [(4, (1, 2)), (6, (1, 2)), (8, (1,))])
@pytest.mark.parametrize("missing_commit", [False, True])
def test_window_audit_uses_registered_environment_count(tmp_path, count, gpus, missing_commit):
    (tmp_path / "services").mkdir()
    with initialize_owner(tmp_path / 'services'):
        pass
    with sqlite3.connect(tmp_path / "services/state.sqlite3") as db:
        db.executescript('''
            CREATE TABLE step_prepares (idempotency_key TEXT, actor_id TEXT, run_id TEXT,
              rollout_window_id TEXT, rollout_step_index INTEGER, checkpoint_sha256 TEXT, status TEXT);
            CREATE TABLE step_commits (prepare_idempotency_key TEXT);
            CREATE TABLE chunk_reservations (chunk_id TEXT, status TEXT);
        ''')
        for gpu in gpus:
            for lane in range(count):
                actor = f"test-g{gpu}-lane-{lane:03d}"
                for step in range(81):
                    key = f"{actor}-{step}"
                    db.execute("INSERT INTO step_prepares VALUES (?,?,?,?,?,?,?)",
                               (key, actor, "test", "window-000000", step, "parent", "committed"))
                    if not (missing_commit and gpu == gpus[-1] and lane == 0 and step == 80):
                        db.execute("INSERT INTO step_commits VALUES (?)", (key,))
                db.execute("INSERT INTO chunk_reservations VALUES (?,?)",
                           (f"d1-{actor}-window-000000", "committed"))
    if missing_commit:
        with pytest.raises(RuntimeError, match="incomplete window"):
            window_audit(tmp_path, "test", 0, "parent", count, gpus)
    else:
        audit = window_audit(tmp_path, "test", 0, "parent", count, gpus)
        assert audit["committed_transitions"] == len(gpus) * count * 81
        assert len(audit["train_chunk_ids"]) == len(gpus) * count
