"""Whole-round restart preserves BC, but never adopts failed DAgger work."""
import fcntl
import json
from pathlib import Path
import sqlite3

import pytest

from pipeline.bc_training.handoff_common import binding, bound
from pipeline.dagger.services.store import SCHEMA
from pipeline.deployment.failed_round_restart import (
    EVIDENCE_SCHEMA, TAIL_EXCLUSION_SCHEMA, ledger_binding, restart_first_round,
)
from pipeline.deployment.mode_execution import execute
from test.pipeline.test_pretraining_mode_recovery import Adapter, fixture as pretraining_fixture,rename_recovery_plan
from pipeline.deployment.pretraining_mode_recovery import compact_recovery_prefix


def insert(db, table, values):
    columns = db.execute('PRAGMA table_info(' + table + ')').fetchall()
    row = {r[1]: (0 if r[2] in ('INTEGER', 'REAL') else '') for r in columns if r[3] and r[4] is None}
    row.update(values)
    db.execute('INSERT INTO ' + table + ' (' + ','.join(row) + ') VALUES (' +
               ','.join('?' for _ in row) + ')', tuple(row.values()))


def fixture(tmp_path):
    p, s, old, e = pretraining_fixture(tmp_path)
    evidence = bound(binding(e))
    campaign_path = Path(evidence['campaign_state']['path'])
    campaign = bound(evidence['campaign_state'])
    campaign['phase'] = 'COLLECT_TRAIN'
    campaign_path.write_text(json.dumps(campaign))
    evidence['campaign_state'] = binding(campaign_path)
    (campaign_path.parent / 'services').mkdir()
    path = Path(evidence['closed_ledger']['database']['path'])
    with sqlite3.connect(path) as db:
        for table in [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]:
            db.execute('DROP TABLE ' + table)
        db.executescript(SCHEMA)
        parent, candidate, run = 'd' * 64, 'e' * 64, 'dagger-leo-prod'
        for identity, active in ((run, candidate), ('heldout-' + run + '-r1', parent)):
            insert(db, 'runs', dict(run_id=identity, current_round=1, beta=.75,
                   initial_checkpoint_sha256=parent, active_checkpoint_sha256=active))
        for digest in (parent, candidate):
            insert(db, 'checkpoints', dict(sha256=digest))
        insert(db, 'round_iterations', dict(run_id=run, round_id=1, iteration_id=1,
               input_checkpoint_sha256=parent, candidate_checkpoint_sha256=candidate, status='advanced'))
        insert(db, 'round_iterations', dict(run_id=run, round_id=1, iteration_id=2,
               input_checkpoint_sha256=candidate, candidate_checkpoint_sha256=None, status='collecting'))
        insert(db, 'learner_jobs', dict(job_id='j1', run_id=run, round_id=1,
               input_checkpoint_sha256=parent, status='succeeded',
               result_json=json.dumps(dict(output_checkpoint_sha256=candidate))))
        insert(db, 'learner_job_iterations', dict(job_id='j1', run_id=run, round_id=1, iteration_id=1))
        insert(db, 'mse_round_protocols', dict(run_id=run, round_id=1, status='active'))
        insert(db, 'step_prepares', dict(idempotency_key='p1', run_id=run, round_id=1, status='committed'))
        insert(db, 'step_commits', dict(idempotency_key='c1', prepare_idempotency_key='p1'))
    for worker in evidence['workers']:
        journal_path = Path(worker['journal']['path'])
        journal = bound(worker['journal'])
        journal['commands']['x']['command']['operation'] = 'collect_production_segment'
        journal_path.write_text(json.dumps(journal))
        worker['journal'] = binding(journal_path)
    evidence.update(schema=EVIDENCE_SCHEMA, all_failed_round_data_excluded=True,
                    all_failed_round_candidates_excluded=True, native_state_reused=False,
                    closed_ledger=ledger_binding(path))
    e.write_text(json.dumps(evidence))
    return p, s, old, e


def test_restart_copies_only_bc_test_and_does_not_replay_failed_round(tmp_path):
    p, s, old, e = fixture(tmp_path)
    evidence = bound(binding(e))
    prior_bytes, db_bytes = old.read_bytes(), Path(evidence['closed_ledger']['path']).read_bytes()
    a = Adapter()
    result = restart_first_round(p, s, previous_state=old, reconciliation=e, adapter=a)
    assert a.calls == []
    assert [r['phase'] for r in result['completed']] == ['prepare_bc', 'train_bc', 'test_bc']
    recovery = result['recovery']
    assert recovery['exclusion_audit']['completed_updates'] == 1
    assert recovery['exclusion_audit']['excluded_tail_iteration'] == 2
    assert recovery['exclusion_audit']['parent_checkpoint_sha256'] == 'd' * 64
    assert recovery['failed_round_population_reused'] is False
    assert recovery['failed_round_candidates_reused'] is False
    assert recovery['native_state_reused'] is False
    assert recovery['bc_optimizer_updates_repeated'] == recovery['bc_test_episodes_repeated'] == 0
    assert result['compute_release_authorized'] is False
    state_path = p.parent / 'execution/state.json'
    recovered_bytes = state_path.read_bytes()
    with pytest.raises(ValueError, match='MSE validation is disabled'):
        execute(p, s, adapter=a)
    assert a.calls == [] and state_path.read_bytes() == recovered_bytes
    assert execute(p, s, adapter=a, allow_legacy_validation=True)['status'] == 'COMPLETE'
    assert a.calls == ['dagger']
    assert old.read_bytes() == prior_bytes
    assert Path(evidence['closed_ledger']['path']).read_bytes() == db_bytes
    with pytest.raises(FileExistsError):
        restart_first_round(p, s, previous_state=old, reconciliation=e, adapter=a)


def test_compact_name_whole_round_restart_preserves_exact_parent_and_exclusion(tmp_path):
    p,s,old,e=fixture(tmp_path)
    rename_recovery_plan(p,compact_recovery_prefix('prod')+'v16')
    prior=binding(old);result=restart_first_round(p,s,previous_state=old,reconciliation=e,adapter=Adapter())
    assert result['recovery']['previous_state']==prior
    assert result['recovery']['exclusion_audit']['completed_updates']==1
    assert result['recovery']['failed_round_population_reused'] is False


@pytest.mark.parametrize('fault', [
    'live_service', 'live_controller', 'live_native', 'changed_native', 'reuse_data', 'reuse_candidate',
    'reuse_state', 'different_parent', 'test_parent', 'round2', 'completed_round',
    'retention', 'release', 'seed', 'coverage', 'test_panel', 'changed_receipt',
    'running_job', 'job_parent', 'job_result', 'missing_job_link', 'orphan_job_link',
    'checkpoint_chain', 'unlearned_candidate', 'missing_iteration', 'extra_checkpoint',
    'promotion', 'test_started', 'uncommitted', 'orphan_commit', 'foreign_prepare',
    'foreign_run', 'wrong_beta', 'changed_ledger', 'journal_sidecar', 'wal',
    'selected_protocol', 'stopped_protocol', 'final_test_scheduled', 'foreign_protocol',
])
def test_unsafe_restart_is_rejected(tmp_path, fault):
    p, s, old, e = fixture(tmp_path)
    evidence, plan = bound(binding(e)), bound(binding(p))
    campaign_path = Path(evidence['campaign_state']['path'])
    campaign = bound(evidence['campaign_state'])
    path = Path(evidence['closed_ledger']['path'])
    if fault == 'live_service': evidence['services'][0]['status']['state'] = 'RUNNING'
    elif fault == 'live_controller': evidence['controller_status']['state'] = 'RUNNING'
    elif fault == 'live_native': evidence['workers'][0]['container_absent'] = False
    elif fault == 'changed_native': evidence['workers'][0]['worker_instance_id'] = 'different'
    elif fault == 'reuse_data': evidence['all_failed_round_data_excluded'] = False
    elif fault == 'reuse_candidate': evidence['all_failed_round_candidates_excluded'] = False
    elif fault == 'reuse_state': evidence['native_state_reused'] = True
    elif fault == 'different_parent': campaign['config']['student']['initial_checkpoint_sha256'] = 'e' * 64
    elif fault == 'test_parent':
        state = bound(binding(old)); receipt = Path(state['completed'][2]['receipt']['path'])
        value = bound(state['completed'][2]['receipt']); value['checkpoint_sha256'] = 'e' * 64
        receipt.write_text(json.dumps(value)); state['completed'][2]['receipt'] = binding(receipt)
        old.write_text(json.dumps(state)); evidence['previous_state'] = binding(old)
    elif fault == 'round2': campaign['round_id'] = 2
    elif fault == 'completed_round': campaign['completed_rounds'] = [1]
    elif fault == 'retention': evidence['retained_after'] = {'allocation_retained': False}
    elif fault == 'release': evidence['compute_release_authorized'] = True
    elif fault == 'seed': plan['settings']['seed'] += 1
    elif fault == 'coverage':
        coverage_path = Path(plan['settings']['coverage_plan']['path'])
        coverage = bound(plan['settings']['coverage_plan']); coverage['rounds'][0]['seed'] += 1
        coverage_path.write_text(json.dumps(coverage)); plan['settings']['coverage_plan'] = binding(coverage_path)
    elif fault == 'test_panel': plan['test_panel']['raw_sha256'] = '0' * 64
    elif fault == 'changed_receipt':
        Path(bound(binding(old))['completed'][1]['receipt']['path']).write_text('{}')
    elif fault == 'journal_sidecar': Path(str(path) + '-journal').write_bytes(b'uncertain')
    else:
        with sqlite3.connect(path) as db:
            sql = {
                'running_job': "UPDATE learner_jobs SET status='running'",
                'job_parent': "UPDATE learner_jobs SET input_checkpoint_sha256='wrong'",
                'job_result': "UPDATE learner_jobs SET result_json='{}'",
                'missing_job_link': 'DELETE FROM learner_job_iterations',
                'orphan_job_link': "INSERT INTO learner_job_iterations VALUES('orphan','dagger-leo-prod',1,2)",
                'checkpoint_chain': "UPDATE round_iterations SET input_checkpoint_sha256='wrong' WHERE iteration_id=2",
                'unlearned_candidate': "UPDATE round_iterations SET candidate_checkpoint_sha256='wrong' WHERE iteration_id=2",
                'missing_iteration': 'DELETE FROM round_iterations WHERE iteration_id=1',
                'extra_checkpoint': "UPDATE checkpoints SET sha256='wrong' WHERE sha256 LIKE 'e%'",
                'promotion': "INSERT INTO promotions VALUES('dagger-leo-prod',1,'wrong',0)",
                'uncommitted': "UPDATE step_prepares SET status='prepared'",
                'orphan_commit': "UPDATE step_commits SET prepare_idempotency_key='unknown'",
                'foreign_prepare': "UPDATE step_prepares SET run_id='different'",
                'foreign_run': "UPDATE runs SET run_id='different' WHERE run_id='dagger-leo-prod'",
                'wrong_beta': 'UPDATE runs SET beta=.5',
                'changed_ledger': "UPDATE learner_jobs SET status='failed'",
                'wal': 'PRAGMA journal_mode=WAL',
                'selected_protocol': "UPDATE mse_round_protocols SET status='selected'",
                'stopped_protocol': "UPDATE mse_round_protocols SET stop_reason='patience'",
                'final_test_scheduled': "UPDATE mse_round_protocols SET final_test_id='test'",
                'foreign_protocol': 'UPDATE mse_round_protocols SET round_id=2',
            }
            if fault == 'test_started':
                insert(db, 'success_evaluations', dict(evaluation_id='test'))
            else: db.execute(sql[fault])
        if fault != 'changed_ledger': evidence['closed_ledger'] = ledger_binding(path)
    campaign_path.write_text(json.dumps(campaign)); evidence['campaign_state'] = binding(campaign_path)
    e.write_text(json.dumps(evidence)); p.write_text(json.dumps(plan))
    with pytest.raises((ValueError, RuntimeError)):
        restart_first_round(p, s, previous_state=old, reconciliation=e, adapter=Adapter())
    assert not (p.parent / 'execution/state.json').exists()


def test_live_owner_lock_rejects_admission(tmp_path):
    p, s, old, e = fixture(tmp_path)
    campaign = Path(bound(binding(e))['campaign_state']['path']).parent
    with (campaign / 'services/bundle-owner.lock').open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            restart_first_round(p, s, previous_state=old, reconciliation=e, adapter=Adapter())


def tail_fixture(tmp_path):
    p, s, old, e = fixture(tmp_path)
    evidence = bound(binding(e)); path = Path(evidence['closed_ledger']['path'])
    with sqlite3.connect(path) as db:
        db.execute('UPDATE runs SET actor_ids_json=? WHERE run_id=?',
                   (json.dumps(['lane0', 'lane1']), 'dagger-leo-prod'))
        insert(db, 'step_prepares', dict(idempotency_key='tail-p0', request_sha256='a'*64,
            run_id='dagger-leo-prod', round_id=1, actor_id='lane0', rollout_window_id='window-000001',
            rollout_step_index=0, checkpoint_sha256='e'*64, status='prepared'))
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT idempotency_key,request_sha256,actor_id,'
            'rollout_window_id,rollout_step_index,checkpoint_sha256 FROM step_prepares '
            'WHERE idempotency_key="tail-p0"')]
    evidence.update(schema=TAIL_EXCLUSION_SCHEMA, closed_ledger=ledger_binding(path),
        uncommitted_tail_exclusion=dict(schema='roborender.uncommitted-tail-exclusion.v1',
            disposition='abandoned_never_replayed_or_learned',physical_outcome='unknown',rows=rows))
    e.write_text(json.dumps(evidence))
    return p, s, old, e


def test_explicit_tail_abandonment_preserves_unknown_outcome_without_replay(tmp_path):
    p, s, old, e = tail_fixture(tmp_path)
    path = Path(bound(binding(e))['closed_ledger']['path']); before = path.read_bytes()
    result = restart_first_round(p,s,previous_state=old,reconciliation=e,adapter=Adapter())
    audit = result['recovery']['exclusion_audit']['excluded_unknown_outcome_prepares']
    assert audit['count'] == 1 and audit['physical_outcome'] == 'unknown' and audit['replay_allowed'] is False
    assert path.read_bytes() == before
    assert [r['phase'] for r in result['completed']] == ['prepare_bc','train_bc','test_bc']
    assert result['recovery']['failed_round_candidates_reused'] is False


@pytest.mark.parametrize('fault', ['v1_waiver','missing_exclusion','known_outcome','changed_list',
    'learned_window','wrong_parent','foreign_actor','hole','not_last','step80','barrier','chunk',
    'status_commit_mismatch','missing_commit_marked_committed','duplicate_pending_actor'])
def test_tail_abandonment_cannot_relax_scientific_or_receipt_boundaries(tmp_path,fault):
    p,s,old,e=tail_fixture(tmp_path); evidence=bound(binding(e));path=Path(evidence['closed_ledger']['path'])
    exclusion=evidence['uncommitted_tail_exclusion']
    if fault=='v1_waiver':evidence['schema']=EVIDENCE_SCHEMA
    elif fault=='missing_exclusion':del evidence['uncommitted_tail_exclusion']
    elif fault=='known_outcome':exclusion['physical_outcome']='not_stepped'
    elif fault=='changed_list':exclusion['rows'][0]['request_sha256']='f'*64
    else:
        with sqlite3.connect(path) as db:
            if fault=='learned_window':db.execute('UPDATE step_prepares SET rollout_window_id="window-000000" WHERE idempotency_key="tail-p0"')
            elif fault=='wrong_parent':db.execute('UPDATE step_prepares SET checkpoint_sha256=? WHERE idempotency_key="tail-p0"',('d'*64,))
            elif fault=='foreign_actor':db.execute('UPDATE step_prepares SET actor_id="foreign" WHERE idempotency_key="tail-p0"')
            elif fault=='hole':db.execute('UPDATE step_prepares SET rollout_step_index=2 WHERE idempotency_key="tail-p0"')
            elif fault=='step80':db.execute('UPDATE step_prepares SET rollout_step_index=80 WHERE idempotency_key="tail-p0"')
            elif fault=='missing_commit_marked_committed':db.execute('UPDATE step_prepares SET status="committed" WHERE idempotency_key="tail-p0"')
            elif fault=='status_commit_mismatch':db.execute('UPDATE step_prepares SET status="prepared" WHERE idempotency_key="p1"')
            elif fault=='barrier':insert(db,'iteration_barriers',dict(run_id='dagger-leo-prod',round_id=1,iteration_id=2,actor_id='lane0'))
            elif fault=='chunk':insert(db,'chunk_reservations',dict(chunk_id='d1-lane0-window-000001'))
            elif fault in ('not_last','duplicate_pending_actor'):
                insert(db,'step_prepares',dict(idempotency_key='tail-p1',request_sha256='b'*64,
                    run_id='dagger-leo-prod',round_id=1,actor_id='lane0',rollout_window_id='window-000001',
                    rollout_step_index=1,checkpoint_sha256='e'*64,
                    status='committed' if fault=='not_last' else 'prepared'))
                if fault=='not_last':insert(db,'step_commits',dict(idempotency_key='tail-c1',prepare_idempotency_key='tail-p1'))
            db.row_factory=sqlite3.Row
            exclusion['rows']=[dict(r) for r in db.execute('SELECT p.idempotency_key,p.request_sha256,p.actor_id,'
                'p.rollout_window_id,p.rollout_step_index,p.checkpoint_sha256 FROM step_prepares p '
                'LEFT JOIN step_commits c ON c.prepare_idempotency_key=p.idempotency_key '
                'WHERE c.idempotency_key IS NULL ORDER BY p.idempotency_key')]
        evidence['closed_ledger']=ledger_binding(path)
    e.write_text(json.dumps(evidence))
    with pytest.raises((ValueError,RuntimeError)):
        restart_first_round(p,s,previous_state=old,reconciliation=e,adapter=Adapter())
    assert not (p.parent/'execution/state.json').exists()
    assert not (p.parent / 'execution/state.json').exists()
