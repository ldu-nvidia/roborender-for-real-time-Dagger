"""Explicit BC32-smoke -> four-update-smoke permission; never production admission."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline.dagger.rtx.code.src.dagger_rtx.continuous_diagnostic import validate_scope
from pipeline.dagger.rtx.code.src.dagger_rtx.contracts import canonical_json

ADMISSION_SCHEMA = 'roborender.bc32-diagnostic-parent-continuation.v1'
BC_SCHEMA = 'roborender.bc32-lifecycle-qualification.v1'
CHECKPOINT_SCHEMA = 'roborender.continuous-diagnostic-student-checkpoint.v1'
PARAMETER = 'bc32_diagnostic_parent_admission'


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def validate_admission(value, request):
    if (not isinstance(value, dict)
            or set(value) != {'schema', 'scope', 'scope_sha256', 'segment'}
            or value['schema'] != ADMISSION_SCHEMA
            or digest(value['scope']) != value['scope_sha256']):
        raise ValueError('exact BC32 diagnostic parent admission required')
    scope, segment = value['scope'], value['segment']
    validate_scope(scope, config_sha256=scope['config_sha256'], segment_sha256=digest(segment),
                   segment=segment, source_sha256=scope['source_sha256'])
    parameters = request['parameters']
    iteration = parameters.get('iteration_id')
    if (request.get('run_id') != scope['run_id'] or request.get('round_id') != 1
            or type(iteration) is not int or not 1 <= iteration <= 4
            or parameters.get('metric_protocol') != 'fixed100_final_success'
            or parameters.get('raven_exact_full_batch') is not True
            or parameters.get('microbatch_environments') != 32
            or len(request.get('train_chunk_ids', [])) != 32
            or parameters.get('replay_manifest') is not None
            or (iteration == 1 and request.get('input_checkpoint_sha256') != scope['parent_checkpoint_sha256'])):
        raise ValueError('BC32 diagnostic parent permits only four bound full32x81 round1 updates')
    return value


def controller_admission(scope_path: Path, config: dict, request: dict):
    """Read the provider's immutable local scope/intent before admitting a job."""
    def read(path):
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or path.stat().st_size > 1 << 20):
            raise ValueError('bounded physical diagnostic scope/intent required')
        return json.loads(path.read_bytes())
    scope = read(scope_path)
    intent = read(scope_path.parent / 'intent.json')
    if (scope.get('config_sha256') != digest(config)
            or scope.get('source_sha256') != config.get('source_sha256')
            or scope.get('parent_checkpoint_sha256') != config['student']['initial_checkpoint_sha256']
            or intent.get('schema') != 'roborender.continuous-diagnostic-intent.v1'
            or intent.get('diagnostic_scope', {}).get('path') != str(scope_path)
            or intent['diagnostic_scope']['raw_sha256'] != digest(scope)
            or scope_path.read_bytes() != canonical_json(scope)):
        raise ValueError('BC32 continuation scope differs from the actual native diagnostic intent/config')
    value = dict(schema=ADMISSION_SCHEMA, scope=scope, scope_sha256=digest(scope), segment=intent['segment'])
    return validate_admission(value, request)
