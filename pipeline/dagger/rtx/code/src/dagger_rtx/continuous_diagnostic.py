"""Explicit four-update diagnostic scope, never a shortened production round.

The underlying production config and optimizer protocol stay fixed100. This
separate owner-bound capability permits a clean terminal return only at the
genuine first-four-update qualification barrier. No production selection is
made, and this diagnostic is never a physical-state resume boundary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .contracts import canonical_json, require_identifier, require_sha256

SCHEMA = 'roborender.continuous-four-update-diagnostic.v1'
CURRICULUM_SCHEMA = 'roborender.continuous-four-update-diagnostic.v2'
GATE_SCHEMA = 'roborender.continuous-four-update-diagnostic-gate.v1'
RESULT_SCHEMA = 'roborender.continuous-four-update-diagnostic-result.v1'
ACK = 'I_ACKNOWLEDGE_FOUR_REAL_B32X81_OPTIMIZER_UPDATES_DIAGNOSTIC_ONLY'


def validate_scope(value, *, config_sha256, segment_sha256, segment, source_sha256):
    fields = {'schema', 'purpose', 'run_id', 'round_id', 'source_sha256', 'config_sha256',
        'segment_sha256', 'coverage_plan_sha256', 'mse_protocol_file_sha256',
        'parent_checkpoint_sha256', 'optimizer_updates', 'environments_per_actor',
        'full_batch_environments', 'rollout_steps', 'checkpoint_selection',
        'terminal_action', 'owners', 'acknowledgement'}
    curriculum = isinstance(value, dict) and value.get('schema') == CURRICULUM_SCHEMA
    if curriculum:
        fields.add('training_curriculum')
    if (not isinstance(value, dict) or set(value) != fields or value['schema'] not in {SCHEMA, CURRICULUM_SCHEMA}
            or value['purpose'] != 'diagnostic_only' or value['acknowledgement'] != ACK
            or segment.get('schema') not in ('roborender.dagger-continuous-round.v1','roborender.dagger-continuous-round.v2')
            or segment.get('window_count') != 100 or segment.get('start_iteration_index') != 0
            or segment.get('qualification_after_updates') != 4
            or value['run_id'] != segment['run_id'] or value['round_id'] != segment['round_id']
            or type(value['round_id']) is not int or value['round_id'] != 1
            or value['source_sha256'] != source_sha256 or value['config_sha256'] != config_sha256
            or value['segment_sha256'] != segment_sha256
            or value['coverage_plan_sha256'] != segment['coverage_plan_sha256']
            or value['mse_protocol_file_sha256'] != segment['mse_protocol_file_sha256']
            or value['parent_checkpoint_sha256'] != segment['input_checkpoint_sha256']
            or value['checkpoint_selection'] != 'none_diagnostic_candidate_only'
            or value['terminal_action'] != 'finish_diagnostic_after_genuine_prefix_qualification'
            or any(type(value[k]) is not int or value[k] != n for k, n in
                [('optimizer_updates', 4), ('environments_per_actor', 16),
                 ('full_batch_environments', 32), ('rollout_steps', 81)])):
        raise ValueError('exact current-source four-update continuous diagnostic scope required')
    for key in ('source_sha256', 'config_sha256', 'segment_sha256', 'coverage_plan_sha256',
                'mse_protocol_file_sha256', 'parent_checkpoint_sha256'):
        require_sha256(value[key], key)
    owners = value['owners']
    if (not isinstance(owners, list) or len(owners) != 2
            or [r.get('physical_gpu') for r in owners] != [0, 1]
            or any(type(r['physical_gpu']) is not int for r in owners)):
        raise ValueError('diagnostic requires the exact two native owners')
    import re
    for row in owners:
        if (set(row) != {'physical_gpu', 'gpu_uuid', 'worker_instance_id', 'native_pid', 'command_id'}
                or not isinstance(row['worker_instance_id'], str)
                or not re.fullmatch('[0-9a-f]{32}', row['worker_instance_id'])
                or type(row['native_pid']) is not int or row['native_pid'] <= 0
                or not isinstance(row['gpu_uuid'], str) or not row['gpu_uuid'].startswith('GPU-')):
            raise ValueError('diagnostic native identity is invalid')
        require_identifier(row['command_id'], 'diagnostic command')
    for key in ('gpu_uuid', 'worker_instance_id', 'command_id'):
        if len({row[key] for row in owners}) != 2:
            raise ValueError('diagnostic owners must be distinct')
    if curriculum:
        from .training_curriculum import validate_request
        requests = value['training_curriculum']
        if not isinstance(requests, list) or len(requests) != 2:
            raise ValueError('both actors require explicit training curriculum requests')
        for gpu, request in enumerate(requests):
            validate_request(request)
            if request['predecessor'] is not None and request['predecessor']['binding']['physical_gpu'] != gpu:
                raise ValueError('diagnostic curriculum actor order changed')
    canonical_json(value)
    return value


def read_scope(path, raw_sha256, **bindings):
    require_sha256(raw_sha256, 'diagnostic scope bytes')
    path = Path(path)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.stat().st_size > 1 << 20):
        raise ValueError('bounded regular continuous diagnostic scope required')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != raw_sha256:
        raise ValueError('continuous diagnostic scope bytes changed')
    value = json.loads(raw)
    if canonical_json(value) != raw:
        raise ValueError('continuous diagnostic scope must use exact canonical bytes')
    return validate_scope(value, **bindings)


def require_owner(scope, state, *, gpu, native_pid, command_id=None):
    owner = scope['owners'][gpu]
    if (state.get('schema') != 'roborender.isaac-resident.v1'
            or state.get('source_sha256') != scope['source_sha256']
            or state.get('native_initializations') != 1
            or any(state.get(k) != owner[k] for k in ('worker_instance_id', 'native_pid', 'gpu_uuid'))
            or native_pid != owner['native_pid']
            or (command_id is not None and command_id != owner['command_id'])
            or state.get('phase') not in {'IDLE', 'RUNNING'}
            or state.get('active_command_id') != (owner['command_id'] if state['phase'] == 'RUNNING' else None)):
        raise ValueError('continuous diagnostic owner/command/source differs')
    return owner
