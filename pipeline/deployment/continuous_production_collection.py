"""One native command per actor/round, including the live four-window gate.

The provider may reattach to the SAME accepted commands, never resubmit an
intent, reconstruct physics from a checkpoint, or rotate motion cohorts. All
publications are immutable. GPU allocation and native shutdown are absent.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from pipeline.bc_training.handoff_common import binding, bound
from pipeline.dagger.beta_campaign import immutable_json
from pipeline.dagger.coverage_production import (
    CONTINUOUS_FIXED100_SCHEMA, is_continuous_plan, is_eight_h200_plan, digest, load_plan, make_continuous_segment, require,
    validate_continuous_qualification,
)
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import actor_ids, canonical
from pipeline.dagger.services.client import ServiceClient
from pipeline.deployment.resident_production_collection import copy_native_assignments


def reconcile_owners(site, commands, identities, *, terminal=False):
    status = site.pool_action('status')
    require(len(status.get('workers', [])) == 2, 'both exact continuous owners required')
    for command, expected, actual in zip(commands, identities, status['workers']):
        require(all(actual.get(k) == expected[k] for k in
                    ('worker_instance_id', 'native_pid', 'gpu_uuid', 'source_sha256')),
                'continuous native owner identity changed; checkpoint cannot restore Isaac')
        if terminal:
            require(actual.get('phase') == 'IDLE' and actual.get('active_command_id') is None,
                    'terminal continuous owners must be genuinely IDLE')
        else:
            require(actual.get('phase') in {'QUEUED', 'RUNNING', 'IDLE'} and
                    actual.get('active_command_id') in {command['command_id'], None},
                    'continuous owner fault or another command owns the simulator')
    return status


def learner_prefix(site, learner, directory, count):
    """Reconcile actual accepted jobs, with no optimizer or model-side action."""
    parent = site.config['student']['initial_checkpoint_sha256']
    previous_step = None
    refs = []
    for iid in range(1, count + 1):
        job = learner.get(f'/v1/jobs/{site.run}-dagger-{site.rid}-iteration-{iid:04d}')['job']
        row = job.get('result') or {}
        require(job.get('status') == 'succeeded'
                and row.get('schema') == 'roborender.dagger_learner_candidate_receipt.v3'
                and row.get('iteration_id') == iid
                and row.get('parent_checkpoint_sha256') == parent
                and type(row.get('train_mse')) in (int, float) and math.isfinite(row['train_mse'])
                and row['train_mse'] >= 0
                and row.get('train_chunks') == 32 and row.get('train_frames') == 2592
                and row.get('batching_semantics') == 'raven_exact_flattened_bxt'
                and row.get('optimizer_state_preserved') is True
                and row.get('replay_manifest_sha256') is None
                and type(row.get('global_step')) is int
                and row['global_step'] == row.get('parent_global_step', -2) + 1
                and (previous_step is None or row['parent_global_step'] == previous_step),
                'continuous accepted learner/checkpoint/optimizer lineage differs')
        path = directory / f'learner-{iid:04d}.json'
        immutable_json(path, row)
        refs.append(binding(path))
        parent, previous_step = row['output_checkpoint_sha256'], row['global_step']
    return refs


def online_prefix(site, result):
    import wandb
    from pipeline.deployment.wandb_preflight import verify_remote_summary
    run = site.config['logging']['wandb_run_id']
    verify_remote_summary(wandb.Api(), 'nvidia', run,
                          {'last_iteration_id': 4, 'last_train_mse': result['train_mse']})
    return dict(schema='roborender.dagger-continuous-online-prefix.v1', run_id=site.run,
        wandb_run_id=run, round_id=site.rid, source_sha256=site.config['source_sha256'],
        checkpoint_sha256=result['output_checkpoint_sha256'], iteration_id=4,
        train_mse=result['train_mse'], online_metrics_verified=True)


def native_wrapper(site, command, identity, segment_sha, receipt, *, terminal=False, diagnostic_scope=None):
    state = receipt['actor_state']
    require(state.get('physical_gpu') == command['gpu'] and
            state.get('native_pid') == identity['native_pid'] and
            state.get('segment_file_sha256') == segment_sha and
            state.get('source_sha256') == site.config['source_sha256'] and
            state.get('config_sha256') == digest(site.config),
            'continuous actor snapshot identity differs')
    wrapped = dict(schema=('roborender.dagger-continuous-native-terminal.v1' if terminal else
                        'roborender.dagger-continuous-native-prefix.v1'),
        physical_gpu=command['gpu'], worker_instance_id=identity['worker_instance_id'],
        native_pid=identity['native_pid'], command_id=command['command_id'],
        source_sha256=site.config['source_sha256'], segment_sha256=segment_sha,
        motion_assignment_sha256=command['payload']['motion_assignment_sha256'],
        actor_state=state, resource_guard=receipt['resource_guard'])
    if site.config.get('population_curriculum_protocol') is not None:
        from pipeline.dagger.rtx.code.src.dagger_rtx.population_curriculum import validate_progress
        updates = 100 if terminal else 4
        if diagnostic_scope is not None:
            from pipeline.dagger.rtx.code.src.dagger_rtx.continuous_diagnostic import validate_scope
            validate_scope(diagnostic_scope, config_sha256=digest(site.config), segment_sha256=segment_sha,
                segment=state['segment'], source_sha256=site.config['source_sha256'])
            require(terminal and receipt.get('diagnostic_only') is True
                and receipt.get('production_round_complete') is False
                and receipt.get('diagnostic_scope_sha256') == digest(diagnostic_scope)
                and state.get('phase') == 'DIAGNOSTIC_COMPLETE' and state.get('completed_windows') == 4,
                'four-window terminal requires its exact admitted diagnostic scope')
            updates = 4
        progress=receipt.get('population_curriculum') if terminal else state.get('population_curriculum')
        validate_progress(progress,updates)
        wrapped['population_curriculum']=progress
    if site.config.get('reset_inventory_protocol') is not None:
        from pipeline.dagger.rtx.code.src.dagger_rtx.reset_inventory import validate_binding
        proof=receipt.get('reset_inventory') if terminal else state.get('reset_inventory')
        validate_binding(proof,lane_offset=command['gpu']*16)
        require(proof==state.get('reset_inventory'),'native reset inventory changed during command')
        wrapped['reset_inventory']=proof
    if terminal and 'training_curriculum' in command['payload']:
        from pipeline.dagger.rtx.code.src.dagger_rtx.training_curriculum import validate_export, digest as curriculum_digest
        export = validate_export(receipt.get('training_curriculum'))
        binding_value = export['binding']
        require(receipt.get('training_curriculum_sha256') == curriculum_digest(export)
            and all(binding_value.get(k) == identity[k] for k in ('worker_instance_id','native_pid','gpu_uuid','source_sha256'))
            and binding_value['command_id'] == command['command_id']
            and binding_value['physical_gpu'] == command['gpu']
            and binding_value['run_id'] == site.run and binding_value['round_id'] == site.rid
            and binding_value['final_checkpoint_sha256'] == state['final_candidate_checkpoint_sha256']
            and binding_value['config_sha256'] == digest(site.config)
            and binding_value['motion_assignment_sha256'] == command['payload']['motion_assignment_sha256'],
            'terminal training curriculum differs from actual native/checkpoint lineage')
        initialized = receipt.get('training_curriculum_initialization') or {}
        require(initialized.get('mode') == command['payload']['training_curriculum']['mode']
            and initialized.get('initial_reset_calls') == 1
            and initialized.get('restored_before_initial_reset') is True
            and initialized.get('physical_or_rng_state_restored') is False
            and isinstance(initialized.get('before_initial_reset'), dict)
            and isinstance(initialized.get('after_initial_reset'), dict),
            'terminal curriculum lacks its actual initial-reset observation')
        requested = command['payload']['training_curriculum']
        if requested['predecessor'] is not None:
            require(initialized.get('export_sha256') == requested['predecessor']['export_sha256'],
                    'native initialized from a different training curriculum export')
        wrapped.update(schema='roborender.dagger-continuous-native-terminal.v2',
            training_curriculum=export, training_curriculum_sha256=curriculum_digest(export),
            native_terminal_receipt_sha256=curriculum_digest(receipt), training_curriculum_initialization=initialized)
    return wrapped


def _read_prefix(site, command):
    worker = site.pool['workers'][command['gpu']]
    path = Path(worker['runtime_root']) / 'phases' / command['payload']['output_name'] / 'continuous-prefix-state.json'
    # Bounded metadata only. Absence means initialization/collection has not yet
    # reached the gate; transport/permission errors propagate as UNKNOWN.
    code = '''import json,pathlib,sys
sys.path.insert(0,sys.argv[2])
from pipeline.deployment.rtx_actor.production_resident_pool import validate_guard
p=pathlib.Path(sys.argv[1])
assert not p.is_symlink()
b=p.read_bytes() if p.exists() else b"null"
assert len(b)<=1048576
state=json.loads(b)
if state is not None:
    # Validate the actual hash-bound resource file on its owning host, not
    # just a copied summary. This is the same validator as terminal results.
    validate_guard(state['resource_guard'],json.loads(sys.argv[3]))
sys.stdout.buffer.write(b)
'''
    state = json.loads(site.command(['python3', '-c', code, path, site.source,
                                    canonical(worker).decode()], 'beta'))
    if state is None:
        return None
    return dict(actor_state=state, resource_guard=state['resource_guard'],
                episode_assignments=state['episode_assignments'])


def admit_prefix(site, plan, intent, prefixes, learner, directory):
    commands, identities = intent['commands'], intent['owners']
    reconcile_owners(site, commands, identities)
    require(all(p and p['actor_state'].get('phase') == 'WAITING_FOR_QUALIFICATION' for p in prefixes),
            'both live native prefixes must wait at the four-window gate')
    native_refs = []
    for command, identity, receipt in zip(commands, identities, prefixes):
        wrapped = native_wrapper(site, command, identity, intent['segment_sha256'], receipt)
        path = directory / f'prefix-g{command["gpu"]}.json'
        immutable_json(path, wrapped)
        native_refs.append(binding(path))
    jobs = learner_prefix(site, learner, directory, 4)
    final = bound(jobs[-1])
    metrics = directory / 'online-prefix.json'
    immutable_json(metrics, online_prefix(site, final))
    assignments = copy_native_assignments(site, {'results': [{'receipt': p} for p in prefixes]},
        commands, plan['rounds'][site.rid - 1]['native_round'], directory / 'assignments')
    retained = site.verify_retained()
    proof = dict(schema='roborender.dagger-continuous-prefix-qualification.v1', purpose='qualification',
        run_id=site.run, round_id=site.rid, plan_sha256=digest(plan),
        source_sha256=site.config['source_sha256'],
        parent_checkpoint_sha256=site.config['student']['initial_checkpoint_sha256'],
        optimizer_updates=4, full_batch_environments=32, native_prefix_receipts=native_refs,
        learner_receipts=jobs, native_assignment_receipts=assignments,
        allocation_retained=True, h200_uuids=retained['gpu_uuids'],
        online_metrics_verified=True, online_metrics_receipt=binding(metrics))
    if is_eight_h200_plan(plan):
        attachment=directory/'renderer-attachment.json'
        immutable_json(attachment,site.renderer_attachment(running=True))
        proof.update(runtime_config=binding(site.request['round_config_path']),
                     h200_gateway_identities=site.config['h200']['gateway_identities'],
                     renderer_attachment=binding(attachment))
    validate_continuous_qualification(plan, proof, proof['parent_checkpoint_sha256'], round_id=site.rid)
    proof_path = directory / 'qualification.json'
    immutable_json(proof_path, proof)
    publish_gate(site, intent, binding(proof_path))
    return binding(proof_path)


def publish_gate(site, intent, proof_ref):
    """Publish the same validated decision; never generate a second command."""
    proof = bound(proof_ref)
    plan = load_plan(site.config['coverage_plan'])
    validate_continuous_qualification(plan, proof, intent['input_checkpoint_sha256'], round_id=site.rid)
    last = bound(proof['learner_receipts'][-1])['output_checkpoint_sha256']
    remote = Path(intent['segment_path']).parent
    raw = canonical(proof)
    remote_proof = remote / 'continuous-qualification-proof.json'
    site.publish(remote_proof, raw, 'beta', 0o444)
    # Inside the native container, the jobs directory is mounted at this path.
    container_proof = Path('/resident/jobs') / remote_proof.relative_to(site.jobs)
    gate = dict(schema='roborender.dagger-continuous-gate.v1', run_id=site.run,
        round_id=site.rid, source_sha256=site.config['source_sha256'],
        config_sha256=digest(site.config), segment_sha256=intent['segment_sha256'],
        parent_checkpoint_sha256=intent['input_checkpoint_sha256'], checkpoint_sha256=last,
        optimizer_updates=4, approved=True,
        native_pids={str(i): row['native_pid'] for i, row in enumerate(intent['owners'])},
        proof=dict(path=str(container_proof), raw_sha256=hashlib.sha256(raw).hexdigest()))
    diagnostic = intent.get('diagnostic_scope')
    if diagnostic is not None:
        from pipeline.dagger.rtx.code.src.dagger_rtx.continuous_diagnostic import GATE_SCHEMA
        _require_diagnostic_intent(site, intent)
        gate.update(schema=GATE_SCHEMA, diagnostic_scope_sha256=diagnostic['raw_sha256'],
                    decision='finish_diagnostic')
    gate_path = remote / ('continuous-diagnostic-gate.json' if diagnostic else 'continuous-qualification-gate.json')
    gate_raw = canonical(gate)
    prepared = remote / 'continuous-qualification-gate.prepared.json'
    site.publish(prepared, gate_raw, 'beta', 0o444)
    # The actors poll concurrently: expose only the fully written, fsynced
    # gate, never a partially written JSON from O_EXCL publication.
    code = ('import os,pathlib,sys; src=pathlib.Path(sys.argv[1]); dst=pathlib.Path(sys.argv[2]); '
            'assert src.parent==dst.parent and not src.is_symlink() and not dst.is_symlink(); '
            'f=src.open("rb"); os.fsync(f.fileno()); f.close(); '
            'existing=dst.exists(); assert not existing or dst.read_bytes()==src.read_bytes(); '
            'None if existing else os.link(src,dst); '
            'fd=os.open(dst.parent,os.O_DIRECTORY); os.fsync(fd); os.close(fd)')
    site.command(['python3', '-c', code, prepared, gate_path], 'beta')


def _stage(site, plan, directory, *, diagnostic=False):
    require(not site.request.get('production_cohort_limit'), 'continuous rounds cannot use reset-cohort qualification')
    site.barrier()
    owners = site.pool_action('status')['workers']
    require(len(owners) == 2 and all(row.get('phase') == 'IDLE'
            and row.get('active_command_id') is None
            and row.get('source_sha256') == site.config['source_sha256'] for row in owners),
            'fresh continuous round requires two IDLE owners')
    from pipeline.dagger.rtx.code.src.dagger_rtx.training_curriculum import bootstrap_request, validate_request
    curriculum_requests = getattr(site, 'training_curriculum_requests', site.request.get('training_curriculum'))
    if curriculum_requests is None and not diagnostic:
        require(site.rid == 1, 'later production rounds require exact predecessor curriculum receipts')
        curriculum_requests = [bootstrap_request(), bootstrap_request()]
    if curriculum_requests is not None:
        require(isinstance(curriculum_requests, list) and len(curriculum_requests) == 2,
                'both native actors require explicit curriculum requests')
        for gpu, request in enumerate(curriculum_requests):
            validate_request(request)
            predecessor = request['predecessor']
            if not diagnostic and site.rid > 1:
                require(predecessor is not None and predecessor['binding']['boundary'] == 'production_update100'
                    and predecessor['binding']['round_id'] == site.rid - 1,
                    'production curriculum must carry the prior completed round update100')
            if predecessor is not None:
                prior = predecessor['binding']
                require(prior['physical_gpu'] == gpu
                    and prior['final_checkpoint_sha256'] == site.config['student']['initial_checkpoint_sha256']
                    and all(prior[k] == owners[gpu][k] for k in ('native_pid','worker_instance_id','gpu_uuid','source_sha256')),
                    'curriculum predecessor differs from exact warm owner and parent checkpoint')
    remote = site.remote / ('continuous-diagnostic' if diagnostic else 'continuous-round')
    config_path = site.remote / 'config.json'
    site.publish(config_path, Path(site.request['round_config_path']).read_bytes(), 'beta', 0o444)
    protocol = Path(site.request['mse_protocol_path']).read_bytes()
    mse_sha = site.request['mse_protocol_file_sha256']
    require(hashlib.sha256(protocol).hexdigest() == mse_sha, 'protocol binding changed')
    remote_mse = site.remote / 'mse-protocol.json'
    site.publish(remote_mse, protocol, 'beta', 0o444)
    plan_raw = Path(site.config['coverage_plan']['path']).read_bytes()
    plan_sha = site.config['coverage_plan']['raw_sha256']
    require(hashlib.sha256(plan_raw).hexdigest() == plan_sha, 'plan binding changed')
    remote_plan = site.remote / 'coverage-plan.json'
    site.publish(remote_plan, plan_raw, 'beta', 0o444)
    assignments = []
    for gpu, value in enumerate(plan['rounds'][site.rid - 1]['native_round']['assignments']):
        path = remote / f'training-assignment-g{gpu}.json'
        site.publish(path, canonical(value), 'beta', 0o444)
        assignments.append(dict(physical_gpu_index=gpu, path=str(path), sha256=digest(value)))
    parent = site.config['student']['initial_checkpoint_sha256']
    segment = make_continuous_segment(plan, round_id=site.rid, input_checkpoint_sha256=parent,
        mse_protocol_file_sha256=mse_sha, motion_assignments=assignments, coverage_plan_sha256=plan_sha)
    segment_path = remote / 'segment.json'
    site.publish(segment_path, canonical(segment), 'beta', 0o444)
    connection_root = getattr(site, 'connection_root', site.remote)
    connection = digest(site.read(connection_root / 'supervisor.json', 'beta'))
    commands = []
    prefix = hashlib.sha256(site.run.encode()).hexdigest()[:12]
    for gpu in (0, 1):
        name = f'continuous-{"diagnostic-" if diagnostic else ""}{prefix}-r{site.rid}-g{gpu}'
        commands.append(dict(gpu=gpu, command_id=name, operation='collect_production_segment', payload=dict(
            round_config=str(config_path.relative_to(site.jobs)), round_config_sha256=digest(site.config),
            token_file=str((site.remote / 'services.token').relative_to(site.jobs)),
            connection_status_directory=str((connection_root / 'supervisor').relative_to(site.jobs)),
            connection_config_sha256=connection, output_name=name,
            mse_protocol_file=str(remote_mse.relative_to(site.jobs)), mse_protocol_sha256=mse_sha,
            segment_file=str(segment_path.relative_to(site.jobs)), segment_sha256=digest(segment),
            coverage_plan_file=str(remote_plan.relative_to(site.jobs)), coverage_plan_sha256=plan_sha,
            motion_assignment_file=str(Path(assignments[gpu]['path']).relative_to(site.jobs)),
            motion_assignment_sha256=assignments[gpu]['sha256'])))
        if curriculum_requests is not None:
            commands[-1]['payload']['training_curriculum'] = curriculum_requests[gpu]
    intent = dict(schema='roborender.dagger-continuous-intent.v1', commands=commands,
        owners=[{k: row[k] for k in ('worker_instance_id', 'native_pid', 'gpu_uuid', 'source_sha256')}
                for row in owners], segment_path=str(segment_path), segment_sha256=digest(segment),
        input_checkpoint_sha256=parent, source_sha256=site.config['source_sha256'],
        config_sha256=digest(site.config), plan_sha256=digest(plan))
    from pipeline.dagger.rtx.code.src.dagger_rtx.population_curriculum import CONFIG_FIELD, PROTOCOL
    if site.config.get(CONFIG_FIELD) == PROTOCOL:
        population_scope=dict(schema=PROTOCOL, config_sha256=digest(site.config),
            segment_sha256=digest(segment), source_sha256=site.config['source_sha256'],
            owners=[dict(physical_gpu=g, command_id=c['command_id'], **intent['owners'][g])
                    for g,c in enumerate(commands)])
        population_path=remote/'population-scope.json'
        site.publish(population_path,canonical(population_scope),'beta',0o444)
        immutable_json(directory/'population-scope.json',population_scope)
        for c in commands:
            c['payload'].update(population_scope_file=str(population_path.relative_to(site.jobs)),
                                population_scope_sha256=digest(population_scope))
    if diagnostic:
        from pipeline.dagger.rtx.code.src.dagger_rtx.continuous_diagnostic import SCHEMA, CURRICULUM_SCHEMA, ACK, validate_scope
        scope = dict(schema=SCHEMA, purpose='diagnostic_only', run_id=site.run, round_id=site.rid,
            source_sha256=site.config['source_sha256'], config_sha256=digest(site.config),
            segment_sha256=digest(segment), coverage_plan_sha256=plan_sha,
            mse_protocol_file_sha256=mse_sha, parent_checkpoint_sha256=parent,
            optimizer_updates=4, environments_per_actor=16, full_batch_environments=32, rollout_steps=81,
            checkpoint_selection='none_diagnostic_candidate_only',
            terminal_action='finish_diagnostic_after_genuine_prefix_qualification', acknowledgement=ACK,
            owners=[dict(physical_gpu=g, command_id=c['command_id'],
                **{k: owners[g][k] for k in ('worker_instance_id', 'native_pid', 'gpu_uuid')})
                for g, c in enumerate(commands)])
        if curriculum_requests is not None:
            scope.update(schema=CURRICULUM_SCHEMA, training_curriculum=curriculum_requests)
        validate_scope(scope, config_sha256=digest(site.config), segment_sha256=digest(segment),
                       segment=segment, source_sha256=site.config['source_sha256'])
        local_scope = directory / 'diagnostic-scope.json'
        immutable_json(local_scope, scope)
        remote_scope = remote / 'diagnostic-scope.json'
        site.publish(remote_scope, canonical(scope), 'beta', 0o444)
        for c in commands:
            c['operation'] = 'probe_continuous_training'
            c['payload'].update(diagnostic_scope_file=str(remote_scope.relative_to(site.jobs)),
                                diagnostic_scope_sha256=digest(scope))
        intent.update(schema='roborender.continuous-diagnostic-intent.v1',
                      diagnostic_scope=binding(local_scope), segment=segment)
    immutable_json(directory / 'intent.json', intent)
    # If this response is lost, subsequent calls only inspect/reconcile it.
    site.pool_action('submit', commands)
    return intent


def collect(site, *, sleep=time.sleep, monotonic=time.monotonic):
    plan = load_plan(site.config['coverage_plan'])
    require(is_continuous_plan(plan), 'continuous plan required')
    directory = site.round / 'continuous'
    intent_path = directory / 'intent.json'
    intent = json.loads(intent_path.read_bytes()) if intent_path.exists() else _stage(site, plan, directory)
    require('diagnostic_scope' not in intent, 'production collector cannot adopt a bounded diagnostic')
    require(intent['config_sha256'] == digest(site.config) and intent['plan_sha256'] == digest(plan),
            'continuous intent cannot be rebound to a different experiment')
    token = site.token.read_text().strip()
    learner = ServiceClient('http://127.0.0.1:18613', token, timeout=120)
    coordinator = ServiceClient('http://127.0.0.1:18610', token, timeout=120)
    proof_path = directory / 'qualification.json'
    deadline = monotonic() + 7 * 86400  # Observer budget, never compute expiry.
    while monotonic() < deadline:
        reconcile_owners(site, intent['commands'], intent['owners'])
        response = site.pool_action('result', intent['commands'])
        results = response.get('results', [])
        require(len(results) == 2 and all(r.get('status') in {'QUEUED', 'RUNNING', 'COMPLETE'} for r in results),
                'continuous command failed/unknown; inspect saved intent, never replay')
        if all(r['status'] == 'COMPLETE' for r in results):
            break
        if not proof_path.exists():
            prefixes = [_read_prefix(site, c) for c in intent['commands']]
            if all(p is not None for p in prefixes):
                admit_prefix(site, plan, intent, prefixes, learner, directory)
        elif not (directory / 'gate-published.json').exists():
            # Reconcile a crash between proof persistence and publication; the
            # immutable gate cannot admit a different checkpoint or owner.
            publish_gate(site, intent, binding(proof_path))
            immutable_json(directory / 'gate-published.json', dict(qualification=binding(proof_path)))
        sleep(10)
    else:
        raise RuntimeError('continuous observer budget expired; allocations/owners retained, no replay')
    require(proof_path.exists(), 'terminal native work lacks actual continuous qualification')
    reconcile_owners(site, intent['commands'], intent['owners'], terminal=True)
    learner_prefix(site, learner, directory, 100)
    native_refs = []
    for command, owner, result in zip(intent['commands'], intent['owners'], results):
        receipt = result.get('receipt') or {}
        require(receipt.get('complete') is True and receipt.get('segment_sha256') == intent['segment_sha256'],
                'terminal native receipt segment differs')
        wrapped = native_wrapper(site, command, owner, intent['segment_sha256'], receipt, terminal=True)
        path = directory / f'terminal-g{command["gpu"]}.json'
        immutable_json(path, wrapped)
        native_refs.append(binding(path))
    context = coordinator.get(f'/v1/mse-rounds/{site.run}/{site.rid}')['mse_round']
    final = bound(binding(directory / 'learner-0100.json'))['output_checkpoint_sha256']
    require(context.get('status') == 'selected_pending_test' and context.get('stop_reason') == 'fixed_100_updates'
            and context['selection']['best_checkpoint_sha256'] == final,
            'continuous round has not selected exact update100')
    curriculum_enabled = all('training_curriculum' in c['payload'] for c in intent['commands'])
    output = dict(schema=('roborender.dagger-continuous-collection-result.v2' if curriculum_enabled else
                        'roborender.dagger-continuous-collection-result.v1'), run_id=site.run,
        round_id=site.rid, source_sha256=site.config['source_sha256'], coverage_plan_sha256=digest(plan),
        completed_updates=100, native_lifecycles_per_actor=[1, 1],
        model_rollout_initializations_per_actor=[100, 100], history_reset_at_window_boundary=False,
        actor_ids=actor_ids(site.config), last_candidate_checkpoint_sha256=final,
        selected_checkpoint_sha256=final, allocation_retained=True, qualification=binding(proof_path),
        native_terminal_receipts=native_refs, **site.barrier())
    from pipeline.dagger.coverage_campaign import validate_round_completion
    status = coordinator.get(f'/v1/status/{site.run}')
    validate_round_completion(site.config, output, status['iterations'], context, plan)
    immutable_json(site.round / 'production-collection-result.json', output)
    return output


def _require_diagnostic_intent(site, intent):
    from pipeline.dagger.rtx.code.src.dagger_rtx.continuous_diagnostic import validate_scope
    require(intent.get('schema') == 'roborender.continuous-diagnostic-intent.v1',
            'bounded continuous smoke requires its separate immutable intent')
    require(len(intent.get('commands', [])) == len(intent.get('owners', [])) == 2,
            'diagnostic intent needs both exact owners and commands')
    scope = bound(intent['diagnostic_scope'])
    require(digest(intent['segment']) == intent['segment_sha256'],
            'diagnostic intent segment bytes differ')
    validate_scope(scope, config_sha256=digest(site.config), segment_sha256=intent['segment_sha256'],
                   segment=intent['segment'], source_sha256=site.config['source_sha256'])
    require(intent['diagnostic_scope']['raw_sha256'] == digest(scope),
            'diagnostic scope must retain exact canonical bytes')
    for gpu, (command, owner) in enumerate(zip(intent['commands'], intent['owners'])):
        expected = scope['owners'][gpu]
        require(command['operation'] == 'probe_continuous_training'
                and command['gpu'] == gpu and owner['source_sha256'] == scope['source_sha256']
                and command['command_id'] == expected['command_id']
                and command['payload']['diagnostic_scope_sha256'] == digest(scope)
                and all(owner[k] == expected[k] for k in ('worker_instance_id', 'native_pid', 'gpu_uuid')),
                'diagnostic intent is not bound to its exact native pair')
    return scope


def require_no_fifth_update(site, learner, coordinator):
    """Empty iteration5 metadata is normal; native work, barriers/job5 are not."""
    from urllib.error import HTTPError
    from pipeline.dagger.services.client import ServiceError
    status = coordinator.get(f'/v1/status/{site.run}')
    rows = [r for r in status['iterations'] if r['round_id'] == site.rid]
    require(len(rows) == 5 and {r['iteration_id'] for r in rows} == set(range(1, 6)),
            'diagnostic must contain exactly four accepted updates and empty next-iteration metadata')
    fifth = next(r for r in rows if r['iteration_id'] == 5)
    require(fifth.get('status') == 'collecting' and fifth.get('candidate_checkpoint_sha256') is None
            and all(b['iteration_id'] <= 4 for b in status['barriers'] if b['round_id'] == site.rid),
            'diagnostic unexpectedly started the fifth update or barrier')
    context = coordinator.get(f'/v1/mse-rounds/{site.run}/{site.rid}')['mse_round']
    require(context.get('status') == 'active' and context.get('selection', {}).get('history') == []
            and context['selection'].get('best_checkpoint_sha256') == site.config['student']['initial_checkpoint_sha256'],
            'diagnostic must never select or promote a production checkpoint')
    job_id = f'{site.run}-dagger-{site.rid}-iteration-0005'
    try:
        learner.get(f'/v1/jobs/{job_id}')
    except ServiceError as exc:
        # A timeout, authentication failure, or textual error is NOT absence.
        require(isinstance(exc.__cause__, HTTPError) and exc.__cause__.code == 404,
                'learner job5 state UNKNOWN; diagnostic cannot be certified')
    else:
        raise ValueError('diagnostic unexpectedly created optimizer job5')
    return dict(next_iteration_metadata=5, next_iteration_status='collecting_empty',
                iteration5_barriers=0, optimizer_job5_absent=True, production_checkpoint_selected=False)


def collect_diagnostic(site, *, sleep=time.sleep, monotonic=time.monotonic):
    """Real first4 learner updates, then clean IDLE; never a production run.

    This is an explicit entrypoint, not a flag accepted by the production
    queue. Lost responses reattach to persisted commands without resubmission.
    A completed diagnostic cannot later resume its discarded physical state.
    """
    from pipeline.dagger.rtx.code.src.dagger_rtx.continuous_diagnostic import RESULT_SCHEMA
    plan = load_plan(site.config['coverage_plan'])
    require(is_continuous_plan(plan) and site.rid == 1,
            'separate current-source round1 continuous diagnostic required')
    directory = site.round / 'continuous-diagnostic'
    intent_path = directory / 'intent.json'
    intent = json.loads(intent_path.read_bytes()) if intent_path.exists() else _stage(site, plan, directory, diagnostic=True)
    _require_diagnostic_intent(site, intent)
    require(intent['config_sha256'] == digest(site.config) and intent['plan_sha256'] == digest(plan),
            'diagnostic intent cannot change source/config/plan')
    token = site.token.read_text().strip()
    learner = ServiceClient('http://127.0.0.1:18613', token, timeout=120)
    coordinator = ServiceClient('http://127.0.0.1:18610', token, timeout=120)
    proof_path = directory / 'qualification.json'
    deadline = monotonic() + 7*86400  # Observer bound only, never resource expiry.
    while monotonic() < deadline:
        reconcile_owners(site, intent['commands'], intent['owners'])
        results = site.pool_action('result', intent['commands']).get('results', [])
        require(len(results) == 2 and all(r.get('status') in {'QUEUED', 'RUNNING', 'COMPLETE'} for r in results),
                'diagnostic failed/UNKNOWN; preserve native state and never replay')
        if all(r['status'] == 'COMPLETE' for r in results):
            break
        if not proof_path.exists():
            prefixes = [_read_prefix(site, c) for c in intent['commands']]
            if all(p is not None for p in prefixes):
                admit_prefix(site, plan, intent, prefixes, learner, directory)
        elif not (directory / 'gate-published.json').exists():
            publish_gate(site, intent, binding(proof_path))
            immutable_json(directory / 'gate-published.json', dict(qualification=binding(proof_path)))
        sleep(10)
    else:
        raise RuntimeError('diagnostic observer budget ended; allocations and owners retained, no replay')
    require(proof_path.exists(), 'diagnostic lacks genuine full32x81 prefix qualification')
    validate_continuous_qualification(plan, bound(binding(proof_path)), intent['input_checkpoint_sha256'], round_id=1)
    prefix_by_gpu={row['physical_gpu']:row for row in
        (bound(ref) for ref in bound(binding(proof_path))['native_prefix_receipts'])}
    reconcile_owners(site, intent['commands'], intent['owners'], terminal=True)
    jobs = learner_prefix(site, learner, directory, 4)
    final = bound(jobs[-1])['output_checkpoint_sha256']
    terminal_refs = []
    for command, identity, result in zip(intent['commands'], intent['owners'], results):
        receipt = result.get('receipt') or {}
        state = receipt.get('actor_state') or {}
        require(receipt.get('complete') is True and receipt.get('diagnostic_only') is True
                and receipt.get('production_round_complete') is False
                and receipt.get('diagnostic_scope_sha256') == intent['diagnostic_scope']['raw_sha256']
                and state.get('phase') == 'DIAGNOSTIC_COMPLETE' and state.get('completed_windows') == 4
                and state.get('transition_batches') == 324 and state.get('physical_transition_rows') == 5184
                and state.get('native_environment_initializations') == 1
                and state.get('model_rollout_initializations') == 4
                and state.get('environment_reset_at_window_boundary') is False
                and state.get('selection') is None and state.get('fifth_window_started') is False
                and state.get('final_candidate_checkpoint_sha256') == final,
                'diagnostic terminal lacks exact four-update continuous scope')
        wrapped = native_wrapper(site, command, identity, intent['segment_sha256'], receipt, terminal=True,
                                 diagnostic_scope=bound(intent['diagnostic_scope']))
        if is_eight_h200_plan(plan):
            prefix=prefix_by_gpu[command['gpu']]
            require(all(wrapped[k]==prefix[k] for k in ('native_pid','worker_instance_id','command_id','segment_sha256'))
                and state.get('windows')==prefix['actor_state']['windows']
                and state.get('h200_gateway_identities')==prefix['actor_state']['h200_gateway_identities'],
                'diagnostic v2 terminal differs from its actual admitted native prefix')
        wrapped['schema'] = 'roborender.continuous-diagnostic-native-terminal.v1'
        path = directory / f'terminal-g{command["gpu"]}.json'
        immutable_json(path, wrapped); terminal_refs.append(binding(path))
    no_fifth = require_no_fifth_update(site, learner, coordinator)
    output = dict(schema=RESULT_SCHEMA, complete=True, diagnostic_only=True,
        production_round_complete=False, run_id=site.run, round_id=site.rid,
        source_sha256=site.config['source_sha256'], completed_updates=4,
        full_batch_environments=32, frames_per_update=2592,
        native_lifecycles_per_actor=[1, 1], model_rollout_initializations_per_actor=[4, 4],
        selected_checkpoint_sha256=None, last_diagnostic_candidate_checkpoint_sha256=final,
        learner_receipts=jobs, qualification=binding(proof_path), diagnostic_scope=intent['diagnostic_scope'],
        native_terminal_receipts=terminal_refs, allocation_retained=True, **no_fifth, **site.barrier())
    immutable_json(directory / 'result.json', output)
    return output
