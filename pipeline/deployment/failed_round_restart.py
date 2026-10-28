"""Explicit whole-round-1 restart; never a simulator or partial-window resume.

Only the completed BC/test prefix is copied. All prior DAgger data, candidates
and optimizer updates remain in the abandoned namespace. The caller must first
stop/reconcile exact old owners, preserving the allocated compute.
"""
from contextlib import ExitStack, closing
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3

from pipeline.bc_training.handoff_common import binding, bound
from pipeline.deployment.beta_operations import atomic
from pipeline.deployment.mode_execution import SCHEMA
from pipeline.deployment.pretraining_mode_recovery import compare_plans

EVIDENCE_SCHEMA = 'roborender.failed-first-round-reconciliation.v1'
TAIL_EXCLUSION_SCHEMA = 'roborender.failed-first-round-reconciliation.v2'
TERMINAL = {'STOPPED', 'FAILED', 'COMPLETE', 'ABSENT'}


def inspect_uncommitted_tail(db, run, iteration, checkpoint, exclusion):
    """Account for abandoned selections, not for unobserved physical outcomes.

    All simulator owners must already be stopped by validate_reconciliation.
    No row is updated, acknowledged, replayed or admitted to another dataset.
    Only a fresh whole-round restart from BC may consume this audit.
    """
    fields = {'schema', 'disposition', 'physical_outcome', 'rows'}
    if (not isinstance(exclusion, dict) or set(exclusion) != fields
            or exclusion['schema'] != 'roborender.uncommitted-tail-exclusion.v1'
            or exclusion['disposition'] != 'abandoned_never_replayed_or_learned'
            or exclusion['physical_outcome'] != 'unknown'):
        raise ValueError('explicit unknown-outcome tail exclusion required')
    expected = [dict(row) for row in db.execute(
        'SELECT p.idempotency_key,p.request_sha256,p.actor_id,p.rollout_window_id,'
        'p.rollout_step_index,p.checkpoint_sha256 FROM step_prepares p '
        'LEFT JOIN step_commits c ON c.prepare_idempotency_key=p.idempotency_key '
        'WHERE c.idempotency_key IS NULL ORDER BY p.idempotency_key')]
    if not expected or len(expected) > 128 or exclusion['rows'] != expected:
        raise ValueError('tail exclusion differs from exact closed-ledger selections')
    actors = json.loads(db.execute('SELECT actor_ids_json FROM runs WHERE run_id=?', (run,)).fetchone()[0])
    if not isinstance(actors, list) or not 1 <= len(actors) <= 128 or len(set(actors)) != len(actors):
        raise ValueError('registered actor population required for tail exclusion')
    window = f'window-{iteration-1:06d}'
    seen = set()
    for row in expected:
        prepare = db.execute('SELECT run_id,round_id,status FROM step_prepares WHERE idempotency_key=?',
                             (row['idempotency_key'],)).fetchone()
        if (tuple(prepare) != (run, 1, 'prepared') or row['actor_id'] not in actors
                or row['actor_id'] in seen or row['rollout_window_id'] != window
                or row['checkpoint_sha256'] != checkpoint
                or type(row['rollout_step_index']) is not int or not 0 <= row['rollout_step_index'] < 80):
            raise ValueError('unmatched selection is not a unique unsealed tail lane')
        seen.add(row['actor_id'])
        indices = [r[0] for r in db.execute(
            'SELECT rollout_step_index FROM step_prepares WHERE run_id=? AND round_id=1 '
            'AND actor_id=? AND rollout_window_id=? ORDER BY rollout_step_index',
            (run, row['actor_id'], window))]
        if indices != list(range(row['rollout_step_index']+1)):
            raise ValueError('unmatched prepare is not the final contiguous lane step')
    if db.execute('SELECT COUNT(*) FROM step_prepares p JOIN step_commits c '
                  'ON c.prepare_idempotency_key=p.idempotency_key WHERE p.status!="committed"').fetchone()[0]:
        raise ValueError('commit/status inconsistency is not an abandoned selection')
    if db.execute('SELECT COUNT(*) FROM iteration_barriers WHERE run_id=? AND round_id=1 AND iteration_id=?',
                  (run, iteration)).fetchone()[0]:
        raise ValueError('tail has learner barriers; different recovery required')
    for actor in actors:
        chunk_id = f'd1-{actor}-{window}'
        for table in ('chunk_reservations', 'chunk_provenance'):
            if db.execute('SELECT 1 FROM '+table+' WHERE chunk_id=?', (chunk_id,)).fetchone():
                raise ValueError('tail has sealed/published chunks; different recovery required')
    return dict(count=len(expected), physical_outcome='unknown', replay_allowed=False,
                rows_sha256=hashlib.sha256(json.dumps(expected, sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest())


def ledger_binding(path):
    """Stream only the closed metadata ledger, never trajectory payloads."""
    path = Path(path).absolute()
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > 2 << 30
            or any(Path(str(path) + suffix).exists() for suffix in ('-wal', '-shm', '-journal'))):
        raise ValueError('closed regular rollback ledger required')
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('closed ledger changed during binding')
    return dict(path=str(path), size_bytes=after.st_size, sha256=digest.hexdigest())


def inspect_closed_ledger(ref, run, parent, *, tail_exclusion=None):
    """Accept only complete advanced updates followed by one unlearned window."""
    path = Path(ref['path'])
    if ledger_binding(path) != ref:
        raise ValueError('closed ledger binding changed')
    # immutable=1 can report journal_mode=delete even for a WAL-format file.
    # Require the actual on-disk read/write versions to be rollback format.
    with path.open('rb') as stream:
        header = stream.read(20)
    if header[:16] != b'SQLite format 3\0' or header[18:20] != b'\x01\x01':
        raise ValueError('uncertain WAL is not admitted by this restart path')
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        db.row_factory = sqlite3.Row
        if db.execute('PRAGMA journal_mode').fetchone()[0] != 'delete':
            raise ValueError('uncertain WAL is not admitted by this restart path')
        for table in ('promotions', 'success_evaluations', 'success_evaluation_outcomes',
                      'success_evaluation_iterations'):
            if db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]:
                raise ValueError('round selection/test already started; different recovery required')
        runs = [dict(r) for r in db.execute(
            'SELECT run_id,current_round,beta,initial_checkpoint_sha256,active_checkpoint_sha256 FROM runs')]
        if ({r['run_id'] for r in runs} != {run, 'heldout-' + run + '-r1'}
                or any(r['current_round'] != 1 or r['beta'] != .75
                       or r['initial_checkpoint_sha256'] != parent for r in runs)):
            raise ValueError('only round1 and its BC-parent held-out run may exist')
        if db.execute('SELECT COUNT(*) FROM round_iterations WHERE round_id!=1').fetchone()[0]:
            raise ValueError('later round work exists')
        protocols = [tuple(r) for r in db.execute(
            'SELECT run_id,round_id,status,stop_reason,final_test_id FROM mse_round_protocols')]
        if protocols != [(run, 1, 'active', None, None)]:
            raise ValueError('only an unselected active first-round MSE protocol may exist')
        iterations = [dict(r) for r in db.execute(
            'SELECT iteration_id,input_checkpoint_sha256,candidate_checkpoint_sha256,status '
            'FROM round_iterations WHERE run_id=? ORDER BY iteration_id', (run,))]
        if len(iterations) < 2 or len(iterations) > 201:
            raise ValueError('bounded learned prefix plus unlearned tail required')
        checkpoints = [parent]
        for index, row in enumerate(iterations, 1):
            if row['iteration_id'] != index or row['input_checkpoint_sha256'] != checkpoints[-1]:
                raise ValueError('non-contiguous checkpoint chain')
            candidate = row['candidate_checkpoint_sha256']
            if index == len(iterations):
                if row['status'] != 'collecting' or candidate is not None:
                    raise ValueError('last iteration must be unlearned collecting work')
            else:
                if row['status'] != 'advanced' or not isinstance(candidate, str) or len(candidate) != 64 or candidate in checkpoints:
                    raise ValueError('completed prefix must have unique advanced checkpoints')
                checkpoints.append(candidate)
        for row in runs:
            if row['active_checkpoint_sha256'] != (checkpoints[-1] if row['run_id'] == run else parent):
                raise ValueError('active checkpoint does not match prefix')
        jobs = [dict(r) for r in db.execute(
            'SELECT j.job_id,j.run_id,j.round_id,j.input_checkpoint_sha256,j.status,j.result_json,'
            'i.iteration_id,i.run_id AS linked_run,i.round_id AS linked_round '
            'FROM learner_jobs j LEFT JOIN learner_job_iterations i ON i.job_id=j.job_id')]
        if (len(jobs) != len(checkpoints) - 1
                or {j['iteration_id'] for j in jobs} != set(range(1, len(checkpoints)))
                or db.execute('SELECT COUNT(*) FROM learner_job_iterations').fetchone()[0] != len(jobs)):
            raise ValueError('learner/iteration cardinalities differ')
        for job in jobs:
            index = job['iteration_id']
            result = json.loads(job['result_json'] or '{}')
            if (job['status'] != 'succeeded' or job['run_id'] != run or job['linked_run'] != run
                    or job['round_id'] != 1 or job['linked_round'] != 1
                    or job['input_checkpoint_sha256'] != checkpoints[index - 1]
                    or result.get('output_checkpoint_sha256') != checkpoints[index]):
                raise ValueError('unfinished or inconsistent learner job')
        if {r[0] for r in db.execute('SELECT sha256 FROM checkpoints')} != set(checkpoints):
            raise ValueError('unaccounted checkpoint present')
        uncommitted = db.execute(
            'SELECT COUNT(*) FROM step_prepares p LEFT JOIN step_commits c '
            'ON c.prepare_idempotency_key=p.idempotency_key '
            'WHERE p.status!="committed" OR c.idempotency_key IS NULL').fetchone()[0]
        orphaned = db.execute(
            'SELECT COUNT(*) FROM step_commits c LEFT JOIN step_prepares p '
            'ON c.prepare_idempotency_key=p.idempotency_key WHERE p.idempotency_key IS NULL').fetchone()[0]
        if orphaned or db.execute(
                'SELECT COUNT(*) FROM step_prepares WHERE round_id!=1 OR run_id NOT IN (?,?)',
                (run, 'heldout-' + run + '-r1')).fetchone()[0]:
            raise ValueError('unreconciled or foreign step records')
        abandoned = None
        if uncommitted:
            if tail_exclusion is None:
                raise ValueError('unreconciled or foreign step records')
            abandoned = inspect_uncommitted_tail(db, run, len(iterations), checkpoints[-1], tail_exclusion)
        elif tail_exclusion is not None:
            raise ValueError('tail-exclusion recovery requires actual unmatched selections')
        result = dict(completed_updates=len(jobs), excluded_checkpoints=checkpoints[1:],
                      excluded_tail_iteration=len(iterations), parent_checkpoint_sha256=parent,
                      excluded_prepares=db.execute('SELECT COUNT(*) FROM step_prepares').fetchone()[0],
                      excluded_commits=db.execute('SELECT COUNT(*) FROM step_commits').fetchone()[0])
        if abandoned is not None:
            result['excluded_unknown_outcome_prepares'] = abandoned
    if ledger_binding(path) != ref:
        raise ValueError('closed ledger changed during inspection')
    return result


def validate_reconciliation(old, old_ref, evidence):
    if (old.get('mode') != 'production' or old.get('status') != 'RECONCILE_REQUIRED' or old.get('phase') != 'dagger'
            or [r['phase'] for r in old.get('completed', [])] != ['prepare_bc', 'train_bc', 'test_bc']
            or evidence.get('schema') not in {EVIDENCE_SCHEMA, TAIL_EXCLUSION_SCHEMA} or evidence.get('previous_state') != old_ref
            or evidence.get('compute_release_authorized') is not False
            or evidence.get('controller_status', {}).get('state') not in {'STOPPED', 'FAILED'}
            or evidence.get('all_failed_round_data_excluded') is not True
            or evidence.get('all_failed_round_candidates_excluded') is not True
            or evidence.get('native_state_reused') is not False):
        raise ValueError('explicit failed first-round exclusion required')
    prior = bound(old['plan'])
    campaign = bound(evidence['campaign_state'])
    run = 'dagger-leo-' + prior['run_id']
    train = bound(old['completed'][1]['receipt'])
    test = bound(old['completed'][2]['receipt'])
    parent = train['checkpoint']['sha256']
    if (campaign.get('run_id') != run or campaign.get('round_id') != 1
            or campaign.get('phase') != 'COLLECT_TRAIN' or campaign.get('completed_rounds') != []
            or campaign['config']['student']['initial_checkpoint_sha256'] != parent
            or test['checkpoint_sha256'] != parent):
        raise ValueError('unchanged completed BC/test parent and unfinished round1 required')
    if (evidence.get('retained_before') != evidence.get('retained_after')
            or evidence['retained_after'].get('allocation_retained') is not True):
        raise ValueError('same retained allocation required')
    services = evidence.get('services', [])
    if (len(services) != 3 or {(r['host'], r['label']) for r in services} !=
            {('beta', 'tunnels'), ('local', 'tunnels'), ('gpu', 'gpu-services')}
            or any(r['status'].get('state') not in TERMINAL for r in services)):
        raise ValueError('all former service/tunnel owners must be stopped')
    workers = evidence.get('workers', [])
    if len(workers) != 2 or {r['physical_gpu'] for r in workers} != {0, 1}:
        raise ValueError('both old native owners required')
    for worker in workers:
        journal = bound(worker['journal'])
        if (worker.get('container_absent') is not True or worker.get('unit_state') not in ('inactive', 'failed')
                or journal['worker_instance_id'] != worker['worker_instance_id']
                or journal['source_sha256'] != prior['source_sha256']):
            raise ValueError('native owner not reconciled')
    if evidence['schema'] == EVIDENCE_SCHEMA:
        if 'uncommitted_tail_exclusion' in evidence:
            raise ValueError('legacy reconciliation cannot waive unmatched prepares')
        exclusion = None
    else:
        if 'uncommitted_tail_exclusion' not in evidence:
            raise ValueError('version2 requires explicit unmatched-tail exclusion')
        exclusion = evidence['uncommitted_tail_exclusion']
    return inspect_closed_ledger(evidence['closed_ledger'], run, parent, tail_exclusion=exclusion)


def restart_first_round(plan_path, settings_path, *, previous_state, reconciliation, adapter=None):
    """Publish READY with BC/test only. Does not launch or mutate old state."""
    evidence_ref = binding(reconciliation)
    evidence = bound(evidence_ref)
    campaign_root = Path(evidence['campaign_state']['path']).parent
    with ExitStack() as stack:
        for path in (Path(previous_state).parent / 'controller.lock', campaign_root / 'scheduler.lock',
                     campaign_root / 'services/bundle-owner.lock'):
            if not path.parent.is_dir():
                raise ValueError('existing owner-lock directory required')
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old_ref = binding(previous_state)
        old = bound(old_ref)
        plan_ref, settings_ref = binding(plan_path), binding(settings_path)
        plan, settings = bound(plan_ref), bound(settings_ref)
        compare_plans(bound(old['plan']), plan)
        audit = validate_reconciliation(old, old_ref, evidence)
        if (settings.get('compute_release_authorized') is not False
                or bound(old['settings'])['retained'] != settings['retained']):
            raise ValueError('retained allocation binding changed')
        if adapter is None:
            from pipeline.deployment.mode_site import ModeSite
            adapter = ModeSite(plan, settings)
        for row in old['completed']:
            adapter.validate(row['phase'], bound(row['receipt']), old['completed'])
        root = Path(plan['output_root']) / 'execution'
        if root.resolve() == Path(previous_state).parent.resolve():
            raise ValueError('fresh output namespace required')
        root.mkdir(parents=True, exist_ok=True)
        with (root / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = root / 'state.json'
            if target.exists():
                raise FileExistsError('restart already admitted')
            state = dict(schema=SCHEMA, plan=plan_ref, settings=settings_ref, run_id=plan['run_id'],
                         mode=plan['mode'], status='READY', completed=old['completed'],
                         compute_release_authorized=False,
                         recovery=dict(kind='whole_first_round_restart', previous_state=old_ref,
                                       reconciliation=evidence_ref, exclusion_audit=audit,
                                       native_state_reused=False, failed_round_population_reused=False,
                                       failed_round_candidates_reused=False,
                                       bc_optimizer_updates_repeated=0, bc_test_episodes_repeated=0))
            atomic(target, state)
    return state
