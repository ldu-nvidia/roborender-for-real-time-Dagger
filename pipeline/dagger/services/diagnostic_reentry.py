"""Explicit learner-only train/test/train capability; never production promotion.

Historical native-v1 scopes retain their separately released learner/control
plane. New curriculum-v2 scopes use an explicitly versioned unified release;
equal source hashes must never masquerade as a historical split-source overlay.
It authorizes either the original BC32 diagnostic leg or exactly one return
leg from its actual fourth candidate after that candidate's scaled test.
It does not launch services, inspect a live ledger, or restore simulator state.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from . import bc32_diagnostic_parent as original

PARAMETER = 'diagnostic_learner_chain_admission'
SCHEMA = 'roborender.diagnostic-learner-chain-admission.v1'
CHECKPOINT_SCHEMA = 'roborender.continuous-diagnostic-reentry-checkpoint.v1'
RNG_PARAMETER = 'learner_rng_protocol'
RNG_PROTOCOL = 'roborender.learner-rng-continuation.v1'
EXECUTION_FIELDS = {'native_source_sha256', 'learner_source_sha256',
                    'worker_sha256', 'server_sha256', 'controller_sha256'}
UNIFIED_EXECUTION_SCHEMA = 'roborender.diagnostic-unified-execution.v1'
CURRICULUM_SCOPE_SCHEMA = 'roborender.continuous-four-update-diagnostic.v2'
SELECTION = 'diagnostic_candidate_only_no_production_selection'
OWNER_FIELDS = ('physical_gpu', 'gpu_uuid', 'worker_instance_id', 'native_pid')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
            'exact SHA-256 identity required')
    return value


def read_bound(ref):
    """Read immutable JSON evidence, never a live SQLite database or a URL."""
    require(isinstance(ref, dict) and set(ref) == {'path', 'raw_sha256'},
            'exact evidence path/raw SHA binding required')
    sha(ref['raw_sha256'])
    path = Path(ref['path'])
    require(path.is_absolute() and path.resolve() == path and path.is_file()
            and 0 < path.stat().st_size <= 64 << 20, 'bounded physical evidence file required')
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            and hashlib.sha256(raw).hexdigest() == ref['raw_sha256'], 'evidence bytes changed')
    return json.loads(raw)


def _owners(scope):
    return [{k: row[k] for k in OWNER_FIELDS} for row in scope['owners']]


def _rng_receipt(value, *, global_step, worker_sha256):
    require(isinstance(value, dict) and value.get('schema') == RNG_PROTOCOL
            and type(value.get('global_step')) is int and value['global_step'] == global_step
            and value.get('worker_file_sha256') == worker_sha256
            and value.get('train_generator_usage') == 'preserved_unused_by_dagger',
            'actual source-bound learner RNG receipt required; inherited BC RNG is insufficient')
    for key in ('parent_checkpoint_sha256', 'input_rng_sha256', 'output_rng_sha256'):
        sha(value[key])
    return value


def validate_execution(execution, native_source):
    require(isinstance(execution, dict), 'versioned native/learner/control identities required')
    unified = execution.get('schema') == UNIFIED_EXECUTION_SCHEMA
    expected = EXECUTION_FIELDS | {'schema'} if unified else EXECUTION_FIELDS
    require(set(execution) == expected, 'exact split-source or unified execution fields required')
    for key in EXECUTION_FIELDS:
        sha(execution[key])
    require(execution['native_source_sha256'] == native_source, 'native source differs from execution')
    if unified:
        require(execution['learner_source_sha256'] == native_source,
                'unified execution requires one actual frozen native/learner source')
    else:
        require(execution['learner_source_sha256'] != native_source,
                'learner overlay must not relabel the native frozen source')
    return execution


def validate_runtime_files(admission, *, worker_file=None, server_file=None, controller_file=None):
    """Bind the actually loaded/selected files, not an environment's claim."""
    observed = {}
    for name, file in (('worker', worker_file), ('server', server_file), ('controller', controller_file)):
        if file is None:
            continue
        path = Path(file)
        require(path.is_absolute() and path.resolve() == path and path.is_file()
                and 0 < path.stat().st_size <= 4 << 20, 'bounded physical implementation file required')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == admission['execution'][name + '_sha256'], name + ' implementation differs from admission')
        observed[name + '_sha256'] = digest
    require(bool(observed), 'at least one actual implementation file must be attested')
    return observed


def validate_curriculum_terminal(row, scope, gpu, checkpoint_sha256):
    """Re-derive the training-only handoff from its actual completed actor."""
    from pipeline.dagger.rtx.code.src.dagger_rtx import training_curriculum as curriculum
    exported = curriculum.validate_export(row.get('training_curriculum'))
    binding = exported['binding']; owner = scope['owners'][gpu]
    expected = dict(run_id=scope['run_id'], round_id=scope['round_id'],
        actor_id=f'rtx-g{gpu}', physical_gpu=gpu, num_envs=16, completed_updates=4,
        boundary='diagnostic_update4', final_checkpoint_sha256=checkpoint_sha256,
        source_sha256=scope['source_sha256'], diagnostic_scope_sha256=original.digest(scope))
    expected.update({k: owner[k] for k in ('gpu_uuid', 'worker_instance_id', 'native_pid', 'command_id')})
    require(all(binding.get(k) == v for k, v in expected.items()),
            'curriculum export must belong to this exact completed training leg')
    initialized = row.get('training_curriculum_initialization') or {}
    requested = scope['training_curriculum'][gpu]
    curriculum.validate_request(requested)
    require(initialized.get('mode') == requested['mode']
            and initialized.get('initial_reset_calls') == 1
            and initialized.get('restored_before_initial_reset') is True
            and initialized.get('physical_or_rng_state_restored') is False
            and isinstance(initialized.get('before_initial_reset'), dict)
            and isinstance(initialized.get('after_initial_reset'), dict),
            'curriculum requires the actual clean training initialization observation')
    if requested['predecessor'] is not None:
        require(initialized.get('export_sha256') == requested['predecessor']['export_sha256'],
                'training restored a different curriculum predecessor')
    return curriculum.build_restore_request(row)


def validate_predecessor(value, scope, execution):
    """Re-derive the previous four updates, actual candidate test and IDLE pair."""
    require(isinstance(value, dict) and set(value) ==
            {'diagnostic_result', 'scaled_test_result', 'post_test_idle'}, 'exact preceding diagnostic/test evidence required')
    result = read_bound(value['diagnostic_result'])
    prior_scope = read_bound(result['diagnostic_scope'])
    require(result.get('schema') == 'roborender.continuous-four-update-diagnostic-result.v1'
            and result.get('complete') is True and result.get('diagnostic_only') is True
            and result.get('production_round_complete') is False and result.get('round_id') == 1
            and result.get('completed_updates') == 4 and result.get('full_batch_environments') == 32
            and result.get('frames_per_update') == 2592 and result.get('selected_checkpoint_sha256') is None
            and result.get('optimizer_job5_absent') is True
            and result.get('native_lifecycles_per_actor') == [1, 1]
            and result.get('model_rollout_initializations_per_actor') == [4, 4]
            and result.get('source_sha256') == scope['source_sha256']
            and result.get('run_id') == prior_scope.get('run_id') and result['run_id'] != scope['run_id']
            and prior_scope.get('source_sha256') == scope['source_sha256']
            and _owners(prior_scope) == _owners(scope)
            and all(a['command_id'] != b['command_id'] for a, b in zip(prior_scope['owners'], scope['owners']))
            and prior_scope.get('optimizer_updates') == 4,
            'return requires a completed four-update leg on the same two native owners and a fresh namespace')
    refs = result.get('learner_receipts', [])
    require(len(refs) == 4 and len(result.get('native_terminal_receipts', [])) == 2,
            'four learner and both native terminal receipts required')
    parent = prior_scope['parent_checkpoint_sha256']
    previous_rng = None
    prior_admission_sha = None
    for iteration, ref in enumerate(refs, 1):
        row = read_bound(ref)
        require(row.get('schema') == 'roborender.dagger_learner_candidate_receipt.v3'
                and row.get('iteration_id') == iteration and row.get('parent_checkpoint_sha256') == parent
                and row.get('parent_global_step') == iteration + 1 and row.get('global_step') == iteration + 2
                and row.get('train_chunks') == 32 and row.get('train_frames') == 2592
                and row.get('batching_semantics') == 'raven_exact_flattened_bxt'
                and row.get('optimizer_state_preserved') is True and row.get('optimizer_parameter_states') == 78
                and row.get('auxiliary_object_head_zero_moments') is True
                and row.get('replay_manifest_sha256') is None and row.get('diagnostic_only') is True
                and row.get('production_promoted') is False
                and row.get('diagnostic_scope_sha256') == original.digest(prior_scope)
                and row.get('output_checkpoint_schema') == original.CHECKPOINT_SCHEMA
                and row.get('learner_execution') == execution
                and type(row.get('train_mse')) in (int, float) and math.isfinite(row['train_mse'])
                and row['train_mse'] >= 0,
                'preceding learner policy/78-slot AdamW/source lineage differs')
        sha(row['diagnostic_learner_chain_admission_sha256'])
        require(prior_admission_sha in (None, row['diagnostic_learner_chain_admission_sha256']),
                'preceding learner jobs changed their chain admission')
        prior_admission_sha = row['diagnostic_learner_chain_admission_sha256']
        rng = _rng_receipt(row.get('learner_rng_receipt'), global_step=iteration + 2,
                           worker_sha256=execution['worker_sha256'])
        require(rng['parent_checkpoint_sha256'] == parent
                and (previous_rng is None or rng['input_rng_sha256'] == previous_rng),
                'preceding learner RNG states are not an actual continuation chain')
        previous_rng = rng['output_rng_sha256']
        parent = sha(row['output_checkpoint_sha256'])
    require(parent == result.get('last_diagnostic_candidate_checkpoint_sha256')
            and parent == scope['parent_checkpoint_sha256'], 'return parent must be exactly previous update4')
    for gpu, ref in enumerate(result['native_terminal_receipts']):
        row = read_bound(ref)
        state = row.get('actor_state', {})
        require(row.get('schema') == 'roborender.continuous-diagnostic-native-terminal.v1'
                and all(row.get(k) == scope['owners'][gpu][k] for k in OWNER_FIELDS if k != 'gpu_uuid')
                and row.get('source_sha256') == scope['source_sha256']
                and row.get('command_id') == prior_scope['owners'][gpu]['command_id']
                and state.get('phase') == 'DIAGNOSTIC_COMPLETE' and state.get('completed_windows') == 4
                and state.get('transition_batches') == 324 and state.get('physical_transition_rows') == 5184
                and state.get('fifth_window_started') is False and state.get('selection') is None
                and state.get('final_candidate_checkpoint_sha256') == parent,
                'prior diagnostic did not finish cleanly on the same warm owners')
        if execution.get('schema') == UNIFIED_EXECUTION_SCHEMA:
            require(prior_scope.get('schema') == CURRICULUM_SCOPE_SCHEMA
                    and scope['training_curriculum'][gpu] ==
                    validate_curriculum_terminal(row, prior_scope, gpu, parent),
                    'return must restore the exact previous training curriculum, never test state')
    qualification = read_bound(result['qualification'])
    require(qualification.get('source_sha256') == scope['source_sha256']
            and qualification.get('parent_checkpoint_sha256') == prior_scope['parent_checkpoint_sha256']
            and qualification.get('optimizer_updates') == 4 and qualification.get('learner_receipts') == refs
            and qualification.get('online_metrics_verified') is True, 'preceding genuine prefix/W&B proof differs')

    tested = read_bound(value['scaled_test_result'])
    plan = read_bound(tested['plan'])
    selection = tested.get('checkpoint_selection', {})
    require(tested.get('schema') == 'roborender.paired-scale-phase-smoke.v1.result'
            and tested.get('status') == 'DIAGNOSTIC_COMPLETE' and tested.get('online_readback_verified') is True
            and tested.get('warm_native_owners_retained') is True and tested.get('optimizer_steps') == 0
            and tested.get('diagnostic_candidate_tested') is True
            and tested.get('compute_release_authorized') is False
            and tested.get('production_campaign_completed') is False
            and tested.get('sampler_status', {}).get('state') == 'STOPPED_IDLE'
            and selection.get('arm') == 'dagger' and selection.get('checkpoint_sha256') == parent
            and selection.get('diagnostic_result') == value['diagnostic_result']
            and selection == plan.get('checkpoint_selection')
            and plan.get('checkpoint_sha256') == parent and plan.get('source_sha256') == scope['source_sha256'],
            'actual generated candidate-specific scaled PASS is required before returning to training')
    proof = read_bound(tested['qualification_proof'])
    require(proof.get('arm') == 'dagger' and proof.get('checkpoint_sha256') == parent
            and proof.get('batch_size') == 16, 'candidate-specific B16x2 scaled proof required')
    from pipeline.evaluation.scaled_success_qualification import validate_qualification
    target = proof['trials'][2]
    validate_qualification(proof, source_sha256=scope['source_sha256'], panel=target['panel'], config=target['config'])
    for trial in [*proof['trials'], proof['tail_trial']]:
        for row in trial['owners']:
            owner = scope['owners'][row['physical_gpu']]
            require(all(row.get(k) == owner[k] for k in ('physical_gpu', 'worker_instance_id', 'gpu_uuid')),
                    'candidate was tested on another native owner')
    require(value['post_test_idle']['path'] == str(Path(plan['output']) / 'final-idle.json'),
            'IDLE evidence must be the actual candidate-test terminal barrier')
    idle = read_bound(value['post_test_idle'])
    require(isinstance(idle, list) and len(idle) == 2, 'both post-test IDLE receipts required')
    for row, owner in zip(idle, scope['owners']):
        require(row.get('source_sha256') == scope['source_sha256'] and row.get('phase') == 'IDLE'
                and row.get('active_command_id') is None and row.get('native_initializations') == 1
                and all(row.get(k) == owner[k] for k in OWNER_FIELDS if k != 'physical_gpu'),
                'post-test native owner/source/lifetime changed or is not IDLE')
    return dict(result=result, scope=prior_scope, final_receipt=read_bound(refs[-1]))


def validate_admission(value, request):
    require(isinstance(value, dict) and set(value) ==
            {'schema', 'mode', 'scope_admission', 'execution', 'predecessor'}
            and value['schema'] == SCHEMA and value['mode'] in {'initial_bc32', 'post_test_return'},
            'exact versioned diagnostic learner chain required')
    old = original.validate_admission(value['scope_admission'], request)
    require(request['parameters'].get(original.PARAMETER) == old
            and request['parameters'].get(RNG_PARAMETER) == RNG_PROTOCOL,
            'chain requires both the unchanged native scope and explicit exact learner RNG protocol')
    validate_execution(value['execution'], old['scope']['source_sha256'])
    if value['execution'].get('schema') == UNIFIED_EXECUTION_SCHEMA:
        require(old['scope'].get('schema') == CURRICULUM_SCOPE_SCHEMA,
                'unified diagnostic requires explicit training-curriculum scope v2')
    if value['mode'] == 'initial_bc32':
        require(value['predecessor'] is None, 'initial diagnostic cannot claim post-test predecessor evidence')
    else:
        validate_predecessor(value['predecessor'], old['scope'], value['execution'])
    return value


def validate_parent_metadata(checkpoint, request):
    """Only lineage; the worker must still validate/load every tensor/AdamW slot."""
    admission = validate_admission(request['parameters'].get(PARAMETER), request)
    iteration = request['parameters']['iteration_id']
    initial = admission['mode'] == 'initial_bc32'
    parent_step = (2 if initial else 6) + iteration - 1
    expected = (original.BC_SCHEMA if iteration == 1 else original.CHECKPOINT_SCHEMA) if initial else (
        original.CHECKPOINT_SCHEMA if iteration == 1 else CHECKPOINT_SCHEMA)
    require(checkpoint.get('schema') == expected and checkpoint.get('global_step') == parent_step,
            'exact diagnostic parent schema/global optimizer step required')
    if initial and iteration == 1:
        return  # Existing BC32 metadata/tensor validator remains mandatory.
    prior = checkpoint.get(PARAMETER)
    if not initial and iteration == 1:
        evidence = validate_predecessor(admission['predecessor'], admission['scope_admission']['scope'], admission['execution'])
        require(isinstance(prior, dict) and prior.get('mode') == 'initial_bc32'
                and prior.get('execution') == admission['execution']
                and prior.get('scope_admission', {}).get('scope') == evidence['scope']
                and original.digest(prior) == evidence['final_receipt']['diagnostic_learner_chain_admission_sha256']
                and checkpoint.get('learner_rng_receipt') == evidence['final_receipt']['learner_rng_receipt']
                and checkpoint.get('dagger_run_id') == evidence['result']['run_id']
                and checkpoint.get('dagger_iteration_id') == 4,
                'return parent differs from the exact tested prior diagnostic candidate')
    else:
        require(prior == admission and checkpoint.get('dagger_run_id') == request['run_id']
                and checkpoint.get('dagger_iteration_id') == iteration - 1,
                'diagnostic successor changed its admitted run/iteration/chain')
    require(checkpoint.get('dagger_round_id') == 1
            and checkpoint.get('production_checkpoint') is False and checkpoint.get('production_promoted') is False
            and checkpoint.get('selection') == SELECTION,
            'diagnostic checkpoint cannot be promoted or relabelled as production')
    _rng_receipt(checkpoint.get('learner_rng_receipt'), global_step=parent_step,
                 worker_sha256=admission['execution']['worker_sha256'])


def checkpoint_metadata(request):
    admission = validate_admission(request['parameters'].get(PARAMETER), request)
    return {PARAMETER: admission,
            'schema': original.CHECKPOINT_SCHEMA if admission['mode'] == 'initial_bc32' else CHECKPOINT_SCHEMA,
            'production_checkpoint': False, 'production_promoted': False, 'selection': SELECTION,
            'diagnostic_origin_checkpoint_sha256': admission['scope_admission']['scope']['parent_checkpoint_sha256']}


def controller_admission(path, scope_path, config, request):
    """Lazily bind native scope after the existing provider has staged it."""
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file() and path.stat().st_size <= 1 << 20,
            'physical bounded learner-chain plan required')
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict) and set(value) == {'schema', 'mode', 'execution', 'predecessor'},
            'exact learner-chain controller plan required')
    value['scope_admission'] = original.controller_admission(Path(scope_path), config, request)
    request['parameters'][original.PARAMETER] = value['scope_admission']
    request['parameters'][RNG_PARAMETER] = RNG_PROTOCOL
    return validate_admission(value, request)
