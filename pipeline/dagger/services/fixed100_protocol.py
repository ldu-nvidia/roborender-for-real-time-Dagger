"""Explicit final-update selection, using the existing durable round ledger.

The historical ``mse_round_protocols`` table and endpoints are storage/transport
names only. This schema never creates held-out data, inference or MSE receipts.
"""
from __future__ import annotations

import json

from .contracts import canonical_json, identifier, sha256_digest

SCHEMA = 'roborender.dagger-fixed100-round-protocol.v1'
CAMPAIGN_SCHEMA = 'roborender.dagger-fixed100-campaign-plan.v1'
UPDATES = 100


def is_fixed100(value):
    return isinstance(value, dict) and value.get('schema') == SCHEMA


def validate_protocol(value):
    fields = {'schema', 'run_id', 'round_id', 'initial_checkpoint_sha256',
              'interval_iterations', 'max_iterations', 'final_test',
              'full_batch_environments', 'rollout_steps', 'checkpoint_selection'}
    if (set(value) != fields or not is_fixed100(value)
            or type(value['round_id']) is not int or value['round_id'] not in (1, 2, 3, 4)
            or type(value['max_iterations']) is not int or value['max_iterations'] != UPDATES
            or value['interval_iterations'] != UPDATES
            or value['full_batch_environments'] != 32 or value['rollout_steps'] != 81
            or value['checkpoint_selection'] != 'final_update_100'):
        raise ValueError('fixed100 requires exactly100 full32x81 updates and final selection')
    identifier(value['run_id'], 'run_id')
    sha256_digest(value['initial_checkpoint_sha256'], 'round parent')
    test = value['final_test']
    if (set(test) != {'scenario_manifest_sha256', 'scenario_ids', 'observation_mode'}
            or test['observation_mode'] != 'generated'
            or len(test['scenario_ids']) != 295 or len(set(test['scenario_ids'])) != 295):
        raise ValueError('fixed100 requires the rendered fixed295 final test')
    sha256_digest(test['scenario_manifest_sha256'], 'test manifest')
    for scenario in test['scenario_ids']:
        identifier(scenario, 'test scenario')
    return value


def initial_selection(protocol):
    return {'checkpoint_selection': 'final_update_100',
            'best_checkpoint_sha256': protocol['initial_checkpoint_sha256'], 'history': []}


def accept_final_candidate(db, job, result, now, *, commit=True):
    """Commit final selection in the successful learner-result transaction.

    No iteration101 is created. Repeated learner acknowledgements retain the
    existing idempotent job path; this does not launch or repeat optimization.
    """
    from .mse_protocol import get_protocol
    row = get_protocol(db, job['run_id'], job['round_id'])
    if row is None or not is_fixed100(json.loads(row['protocol_json'])):
        return False
    validate_protocol(json.loads(row['protocol_json']))
    iid = job['iteration_id']
    if (type(iid) is not int or not 1 <= iid <= UPDATES
            or result.get('train_chunks') != 32 or result.get('train_frames') != 32 * 81
            or result.get('batching_semantics') != 'raven_exact_flattened_bxt'
            or result.get('optimizer_state_preserved') is not True
            or result.get('global_step') != result.get('parent_global_step', -2) + 1
            or result.get('replay_manifest_sha256') is not None):
        raise ValueError('fixed100 learner result is not one accepted full32x81 AdamW update')
    current = db.execute('SELECT input_checkpoint_sha256 FROM round_iterations WHERE run_id=? AND round_id=? AND iteration_id=?',
                         (job['run_id'], job['round_id'], iid)).fetchone()
    if (current is None or result.get('parent_checkpoint_sha256') != current['input_checkpoint_sha256']
            or result.get('output_checkpoint_sha256') == current['input_checkpoint_sha256']):
        raise ValueError('fixed100 learner receipt does not bind the actual current parent')
    if iid > 1:
        previous = db.execute('SELECT j.result_json FROM learner_jobs j JOIN learner_job_iterations i ON i.job_id=j.job_id '
                              'WHERE i.run_id=? AND i.round_id=? AND i.iteration_id=? AND j.status="succeeded"',
                              (job['run_id'], job['round_id'], iid - 1)).fetchall()
        if (len(previous) != 1
                or json.loads(previous[0]['result_json'])['global_step'] != result['parent_global_step']):
            raise ValueError('fixed100 AdamW global steps are discontinuous across accepted updates')
    if iid < UPDATES:
        return False
    rows = db.execute('SELECT * FROM round_iterations WHERE run_id=? AND round_id=? ORDER BY iteration_id',
                      (job['run_id'], job['round_id'])).fetchall()
    protocol = json.loads(row['protocol_json'])
    parent = protocol['initial_checkpoint_sha256']
    if [r['iteration_id'] for r in rows] != list(range(1, UPDATES + 1)):
        raise ValueError('fixed100 requires an exact contiguous100-update history')
    for prior in rows[:-1]:
        if prior['status'] != 'advanced' or prior['input_checkpoint_sha256'] != parent:
            raise ValueError('fixed100 accepted update lineage is incomplete')
        parent = prior['candidate_checkpoint_sha256']
    if rows[-1]['input_checkpoint_sha256'] != parent or result['parent_checkpoint_sha256'] != parent:
        raise ValueError('fixed100 final AdamW parent differs')
    candidate = sha256_digest(result['output_checkpoint_sha256'], 'final candidate')
    if row['status'] in ('selected_pending_test', 'complete'):
        if json.loads(row['selection_json'])['best_checkpoint_sha256'] != candidate:
            raise ValueError('fixed100 final candidate changed on acknowledgement retry')
        return True
    if row['status'] != 'active':
        raise ValueError('fixed100 learner result cannot reopen a terminal round')
    selection = {'checkpoint_selection': 'final_update_100', 'best_checkpoint_sha256': candidate,
                 'history': [{'iteration': UPDATES, 'result': {'checkpoint_sha256': candidate}}]}
    if commit:
        db.execute("UPDATE mse_round_protocols SET selection_json=?,status='selected_pending_test',"
                   "stop_reason='fixed_100_updates',updated_at=? WHERE run_id=? AND round_id=?",
                   (canonical_json(selection), now, job['run_id'], job['round_id']))
    return True
