"""Versioned native production lifetimes and full-window learner barriers.

Legacy v1 segments retain their two-window scene boundary.  A continuous-round
command owns the same physical simulator for all 100 windows; only Raven's
model-rollout buffers begin/clear each 81 steps.  A checkpoint is never a native
simulator restoration mechanism.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .contracts import canonical_json, require_identifier, require_sha256
from .mse_round_driver import MseBetaRoundDriver
from .qualification_driver import BetaRoundDriver, actor_ids

SCHEMA = 'roborender.dagger-production-segment.v1'
CONTINUOUS_SCHEMA = 'roborender.dagger-continuous-round.v1'
CONTINUOUS_PLAN_SCHEMA = 'roborender.dagger-continuous-fixed100-plan.v1'
CONTINUOUS_8H200_SCHEMA = 'roborender.dagger-continuous-round.v2'
CONTINUOUS_8H200_PLAN_SCHEMA = 'roborender.dagger-continuous-fixed100-plan.v2'
FIELDS = {'schema', 'run_id', 'round_id', 'coverage_epoch', 'cohort_index', 'cohort_id',
          'start_iteration_index', 'window_count', 'input_checkpoint_sha256',
          'mse_protocol_file_sha256', 'coverage_plan_sha256', 'episode_namespace', 'motion_assignments'}
CONTINUOUS_FIELDS = (FIELDS - {'coverage_epoch', 'cohort_index'}) | {
    'lifecycle_policy', 'qualification_after_updates'}


def is_continuous_segment(value):
    return isinstance(value, dict) and value.get('schema') in (CONTINUOUS_SCHEMA,CONTINUOUS_8H200_SCHEMA)


def _assignment_bindings(value):
    assignments = value['motion_assignments']
    if (not isinstance(assignments, list) or len(assignments) != 2
            or [r.get('physical_gpu_index') for r in assignments] != [0, 1]):
        raise ValueError('two exact native assignment bindings required')
    for row in assignments:
        if set(row) != {'physical_gpu_index', 'path', 'sha256'} or not Path(row['path']).is_absolute():
            raise ValueError('absolute native assignment binding required')
        require_sha256(row['sha256'], 'production native assignment')


def _validate_continuous_segment(value, config, plan):
    """One fresh native round; a nonzero checkpoint boundary is not a resume."""
    from .beta_round import environments_per_actor, is_continuous_fixed100, is_eight_h200
    if set(value) != CONTINUOUS_FIELDS:
        raise ValueError('exact continuous-round segment fields required')
    rid = value['round_id']
    eight = is_eight_h200(config)
    if (not is_continuous_fixed100(config)
            or plan.get('schema') != (CONTINUOUS_8H200_PLAN_SCHEMA if eight else CONTINUOUS_PLAN_SCHEMA)
            or value.get('schema') != (CONTINUOUS_8H200_SCHEMA if eight else CONTINUOUS_SCHEMA)
            or plan['topology']['h200_workers'] != (8 if eight else 4)
            or type(rid) is not int or not 1 <= rid <= 4 or rid != config['round_id']
            or value['run_id'] != config['run_id'] or plan.get('run_id') != config['run_id']
            or plan.get('source_sha256') != config['source_sha256']
            or value['coverage_plan_sha256'] != config['coverage_plan']['raw_sha256']
            or environments_per_actor(config) != 16
            or plan['topology']['environments_per_actor'] != 16
            or plan['topology']['total_environments'] != 32
            or plan.get('native_lifecycles_per_actor_per_round') != 1
            or plan.get('optimizer_updates_per_round') != 100
            or config['execution']['window_count'] != 100):
        raise ValueError('continuous round config/source/topology differs')
    if (type(value['start_iteration_index']) is not int or value['start_iteration_index'] != 0
            or type(value['window_count']) is not int or value['window_count'] != 100
            or type(value['qualification_after_updates']) is not int or value['qualification_after_updates'] != 4
            or value['lifecycle_policy'] != 'native_episode_done_only'
            or value['input_checkpoint_sha256'] != config['student']['initial_checkpoint_sha256']
            or value['cohort_id'] != f'training-r{rid}-continuous'
            or value['episode_namespace'] != f'continuous-r{rid}'):
        raise ValueError('continuous round must start fresh and preserve native episodes for 100 windows')
    for key in ('input_checkpoint_sha256', 'mse_protocol_file_sha256', 'coverage_plan_sha256'):
        require_sha256(value[key], key)
    native_round = plan['rounds'][rid - 1]['native_round']
    if (native_round.get('cohort_id') != value['cohort_id']
            or native_round.get('start_iteration') != 0 or native_round.get('window_count') != 100
            or native_round.get('qualification_after_updates') != 4
            or native_round.get('reset_at_window_boundary') is not False
            or native_round.get('lifecycle_policy') != 'native_episode_done_only'
            or native_round.get('model_rollout_lifecycle') != 'per_81_step_window'):
        raise ValueError('continuous segment is not an exact native round plan member')
    _assignment_bindings(value)
    return value


def validate_segment(value, config, coverage_plan):
    from .beta_round import environments_per_actor, is_fixed100
    if is_continuous_segment(value):
        return _validate_continuous_segment(value, config, coverage_plan)
    if not isinstance(value, dict) or set(value) != FIELDS or value['schema'] != SCHEMA:
        raise ValueError('exact production segment fields required')
    batch=environments_per_actor(config);total=2*batch;cohorts=(295+total-1)//total
    if (coverage_plan['topology']['environments_per_actor']!=batch
            or coverage_plan['topology']['total_environments']!=total
            or coverage_plan['cohorts_per_coverage_epoch']!=cohorts
            or coverage_plan['updates_per_coverage_epoch']!=2*cohorts
            or coverage_plan['train_frames_per_update']!=81*total):
        raise ValueError('native batch and coverage counts differ from the frozen plan')
    rid, epoch, index = value['round_id'], value['coverage_epoch'], value['cohort_index']
    fixed = is_fixed100(config)
    max_epochs = (coverage_plan.get('maximum_coverage_epochs', 0) if fixed else
                  coverage_plan.get('stopping', {}).get('maximum_coverage_epochs', 0))
    if fixed and (max_epochs != 5 or config['execution']['window_count'] != 100
                  or 'stopping' in coverage_plan):
        raise ValueError('fixed100 production requires five coverage epochs without stopping selection')
    if (type(rid) is not int or rid != config['round_id'] or not 1 <= rid <= 4
            or type(epoch) is not int or not 1 <= epoch <= max_epochs
            or type(index) is not int or not 0 <= index < coverage_plan['cohorts_per_coverage_epoch']
            or value['run_id'] != config['run_id']
            or type(value['window_count']) is not int or value['window_count'] != 2
            or type(value['start_iteration_index']) is not int
            or value['start_iteration_index'] != (epoch - 1) * coverage_plan['updates_per_coverage_epoch'] + index * 2
            or value['start_iteration_index'] + 2 > config['execution']['window_count']):
        raise ValueError('production segment coverage/index/budget differs')
    require_identifier(value['cohort_id'], 'production cohort')
    for key in ('input_checkpoint_sha256', 'mse_protocol_file_sha256', 'coverage_plan_sha256'):
        require_sha256(value[key], key)
    expected_namespace = f'prod-r{rid}-e{epoch:02d}-c{index:03d}'
    if (value['cohort_id'] != f'training-r{rid}-e{epoch:02d}-c{index:03d}'
            or value['episode_namespace'] != expected_namespace
            or value['coverage_plan_sha256'] != config['coverage_plan']['raw_sha256']
            or coverage_plan.get('schema') not in (('roborender.dagger-full295-fixed100-plan.v1',) if fixed else ('roborender.dagger-full295-production-plan.v1','roborender.dagger-full295-production-plan.v2'))
            or coverage_plan.get('run_id') != config['run_id']
            or coverage_plan.get('source_sha256') != config['source_sha256']):
        raise ValueError('production segment namespace/source/plan differs')
    cohort = coverage_plan['rounds'][rid - 1]['epochs'][epoch - 1]['cohorts'][index]
    if (cohort['cohort_index'] != index or cohort['start_iteration'] != value['start_iteration_index']
            or cohort['window_count'] != 2 or cohort.get('reset_at_window_boundary') is not False):
        raise ValueError('production segment is not an exact continuous plan member')
    _assignment_bindings(value)
    return value


def read_segment(path, raw_sha256, config, coverage_plan_file):
    require_sha256(raw_sha256, 'production segment file')
    raw = Path(path).read_bytes()
    plan_raw = Path(coverage_plan_file).read_bytes()
    if (len(raw) > 1 << 20 or hashlib.sha256(raw).hexdigest() != raw_sha256
            or len(plan_raw) > 32 << 20
            or hashlib.sha256(plan_raw).hexdigest() != config['coverage_plan']['raw_sha256']):
        raise ValueError('production segment/coverage bytes changed')
    return validate_segment(json.loads(raw), config, json.loads(plan_raw))


def validate_assignment_document(segment, coverage_plan, gpu, document):
    if is_continuous_segment(segment):
        cohort = coverage_plan['rounds'][segment['round_id'] - 1]['native_round']
    else:
        cohort = coverage_plan['rounds'][segment['round_id'] - 1]['epochs'][segment['coverage_epoch'] - 1]['cohorts'][segment['cohort_index']]
    if gpu not in (0, 1) or document != cohort['assignments'][gpu]:
        raise ValueError('native assignment is not the exact production cohort member')
    if (document.get('purpose') != 'training' or document.get('run_id') != segment['run_id']
            or len(document.get('lanes', [])) != coverage_plan['topology']['environments_per_actor']):
        raise ValueError('production assignment purpose/run/native B differs')
    return document


class ProductionSegmentDriver(MseBetaRoundDriver):
    state_filename = 'production-segment-state.json'

    def __init__(self, *args, segment, segment_file_sha256,
                 qualification_gate_file=None, resource_snapshot=None,
                 assignment_snapshot=None, coverage_plan_document_sha256=None, population_snapshot=None,
                 reset_inventory_binding=None,
                 diagnostic_scope=None, diagnostic_scope_sha256=None, h200_gateway_identities=None, **kwargs):
        self.segment = segment
        self.segment_sha = require_sha256(segment_file_sha256, 'production segment file')
        config = kwargs.get('config', args[3] if len(args) > 3 else {})
        from .beta_round import gateway_identities_for_actor, is_eight_h200
        eight = is_eight_h200(config)
        gpu = kwargs.get('gpu', args[5] if len(args) > 5 else None)
        if eight and h200_gateway_identities != gateway_identities_for_actor(config, gpu):
            raise ValueError('continuous v2 requires actual exclusive gateway readiness')
        self.diagnostic_scope, self.diagnostic_scope_sha256 = diagnostic_scope, diagnostic_scope_sha256
        if diagnostic_scope is not None or diagnostic_scope_sha256 is not None:
            from .continuous_diagnostic import validate_scope
            require_sha256(diagnostic_scope_sha256, 'continuous diagnostic scope')
            if hashlib.sha256(canonical_json(diagnostic_scope)).hexdigest() != diagnostic_scope_sha256:
                raise ValueError('continuous diagnostic scope must use its exact canonical bytes')
            config_sha = kwargs.get('config_sha', args[4] if len(args) > 4 else None)
            validate_scope(diagnostic_scope, config_sha256=config_sha,
                segment_sha256=self.segment_sha, segment=segment, source_sha256=config['source_sha256'])
        self.source_sha256 = require_sha256(config['source_sha256'], 'continuous source') if is_continuous_segment(segment) else None
        self.qualification_gate_file = Path(qualification_gate_file) if qualification_gate_file is not None else None
        self.resource_snapshot = resource_snapshot
        self.assignment_snapshot = assignment_snapshot
        self.population_snapshot = population_snapshot
        self.shared_population = config.get('population_curriculum_protocol') is not None
        if config.get('reset_inventory_protocol') is not None:
            from .reset_inventory import validate_binding
            validate_binding(reset_inventory_binding, lane_offset=gpu*16)
        elif reset_inventory_binding is not None:
            raise ValueError('unexpected full reset inventory proof for legacy campaign')
        if self.shared_population and not callable(population_snapshot):
            raise ValueError('new continuous protocol requires actual shared curriculum observer')
        self.coverage_plan_document_sha256 = (require_sha256(coverage_plan_document_sha256,
            'continuous canonical plan') if is_continuous_segment(segment) else None)
        super().__init__(*args, **kwargs)
        if (segment['mse_protocol_file_sha256'] != self.mse_file_sha256
                or segment['run_id'] != self.config['run_id'] or segment['round_id'] != self.round_id
                or self.runner.episode_namespace != segment['episode_namespace']):
            raise ValueError('production segment driver protocol/episode scope differs')
        if is_continuous_segment(segment) and (
                self.qualification_gate_file is None or not self.qualification_gate_file.is_absolute()
                or self.qualification_gate_file.exists() or not callable(resource_snapshot)
                or not callable(assignment_snapshot)):
            raise ValueError('fresh continuous command requires a fresh bound qualification gate and resource observer')
        self.state.update(segment=segment, segment_file_sha256=self.segment_sha)
        if reset_inventory_binding is not None:
            self.state['reset_inventory'] = dict(reset_inventory_binding)
        if eight:
            self.state['h200_gateway_identities'] = h200_gateway_identities
        if diagnostic_scope is not None:
            owner = diagnostic_scope['owners'][self.gpu]
            if owner['native_pid'] != os.getpid():
                raise ValueError('diagnostic scope does not own this native process')
            self.state.update(diagnostic_scope_sha256=diagnostic_scope_sha256,
                              diagnostic_only=True, production_round_complete=False)

    def _wait_continuous_qualification(self, checkpoint):
        """Wait inside the sole native command, with no new step or re-init."""
        resources = self.resource_snapshot()
        assignments = self.assignment_snapshot()
        if self.shared_population:
            from .population_curriculum import validate_progress
            self.state['population_curriculum']=dict(validate_progress(self.population_snapshot(),4))
        if not isinstance(assignments, list) or len(assignments) < self.config['num_envs_per_gpu']:
            raise RuntimeError('continuous qualification lacks actual native episode assignment evidence')
        self.save('WAITING_FOR_QUALIFICATION', completed_windows=4,
            transition_batches=4 * 81, physical_transition_rows=4 * 81 * self.config['num_envs_per_gpu'],
            final_candidate_checkpoint_sha256=checkpoint, resource_guard=resources,
            native_pid=os.getpid(), model_rollout_initializations=4,
            native_environment_initializations=1, environment_reset_at_window_boundary=False,
            history_reset_at_window_boundary=False, model_rollout_lifecycle='per_81_step_window',
            source_sha256=self.source_sha256, episode_assignments=assignments)
        from .durable import atomic_write_json
        prefix_path = self.path.parent / 'continuous-prefix-state.json'
        if prefix_path.exists():
            raise RuntimeError('continuous native prefix must never be replaced or replayed')
        atomic_write_json(prefix_path, self.state, mode=0o644)
        path = self.qualification_gate_file
        deadline = self.monotonic() + self.timeout
        while self.monotonic() < deadline:
            if self.runner.request_guard is not None:
                self.runner.request_guard()
            if path.exists():
                if path.is_symlink() or not path.is_file() or path.stat().st_size > 1 << 20:
                    raise RuntimeError('continuous qualification gate is not a bounded regular file')
                raw = path.read_bytes()
                gate = json.loads(raw)
                expected = dict(schema='roborender.dagger-continuous-gate.v1',
                    run_id=self.config['run_id'], round_id=self.round_id,
                    source_sha256=self.source_sha256, segment_sha256=self.segment_sha,
                    config_sha256=self.sha,
                    parent_checkpoint_sha256=self.segment['input_checkpoint_sha256'],
                    checkpoint_sha256=checkpoint, optimizer_updates=4, approved=True)
                diagnostic = getattr(self, 'diagnostic_scope', None)
                if diagnostic is not None:
                    from .continuous_diagnostic import GATE_SCHEMA
                    expected.update(schema=GATE_SCHEMA,
                        diagnostic_scope_sha256=self.diagnostic_scope_sha256,
                        decision='finish_diagnostic')
                if (set(gate) != set(expected) | {'native_pids', 'proof'}
                        or any(gate.get(k) != v for k, v in expected.items())
                        or gate['approved'] is not True
                        or set(gate['native_pids']) != {'0', '1'}
                        or any(type(pid) is not int or pid <= 0 for pid in gate['native_pids'].values())
                        or gate['native_pids'][str(self.gpu)] != os.getpid()):
                    raise RuntimeError('continuous qualification gate source/checkpoint/native identity differs')
                proof = gate['proof']
                if not isinstance(proof, dict) or set(proof) != {'path', 'raw_sha256'}:
                    raise RuntimeError('continuous qualification requires an exact proof binding')
                require_sha256(proof['raw_sha256'], 'continuous qualification proof')
                proof_path = Path(proof['path'])
                if (not proof_path.is_absolute() or proof_path.is_symlink()
                        or proof_path.parent.resolve() != path.parent.resolve()
                        or not proof_path.is_file() or proof_path.stat().st_size > 4 << 20):
                    raise RuntimeError('continuous qualification proof must be a bounded sibling file')
                proof_raw = proof_path.read_bytes()
                if hashlib.sha256(proof_raw).hexdigest() != proof['raw_sha256']:
                    raise RuntimeError('continuous qualification proof bytes changed')
                proof_value = json.loads(proof_raw)
                proof_identity = dict(schema='roborender.dagger-continuous-prefix-qualification.v1',
                    purpose='qualification', run_id=self.config['run_id'], round_id=self.round_id,
                    plan_sha256=self.coverage_plan_document_sha256,
                    parent_checkpoint_sha256=self.segment['input_checkpoint_sha256'],
                    source_sha256=self.source_sha256, optimizer_updates=4,
                    full_batch_environments=32, allocation_retained=True)
                if (any(proof_value.get(k) != v for k, v in proof_identity.items())
                        or proof_value.get('allocation_retained') is not True):
                    raise RuntimeError('continuous qualification proof source/round/parent differs')
                self.save('QUALIFIED', qualification_gate_sha256=hashlib.sha256(raw).hexdigest(),
                    qualification_proof=proof)
                return
            self.sleep(1)
        raise TimeoutError('continuous native qualification gate timed out; simulator cannot be replayed')

    def _candidate(self, index, parent):
        deadline = self.monotonic() + self.timeout
        while self.monotonic() < deadline:
            status = BetaRoundDriver.status(self)
            mse = self.mse_status()
            row = next((r for r in status['iterations'] if r['round_id'] == self.round_id
                        and r['iteration_id'] == index + 1), None)
            if row and row.get('candidate_checkpoint_sha256'):
                candidate = require_sha256(row['candidate_checkpoint_sha256'], 'candidate')
                barriers = [r for r in status['barriers'] if r['round_id'] == self.round_id
                            and r['iteration_id'] == index + 1]
                if (row['input_checkpoint_sha256'] != parent or candidate == parent
                        or len(barriers) != len(actor_ids(self.config))
                        or {r['actor_id'] for r in barriers} != set(actor_ids(self.config))
                        or any(r['checkpoint_sha256'] != parent or r['chunk_ids'] != [
                            f"d{self.round_id}-{r['actor_id']}-window-{index:06d}"] for r in barriers)):
                    raise RuntimeError('production segment learner/all-actor lineage differs')
                if row['status'] == 'awaiting_mse_validation':
                    if self.mse_protocol.get('schema') == 'roborender.dagger-fixed100-round-protocol.v1':
                        raise RuntimeError('fixed100 must never enter held-out MSE validation')
                    if mse['status'] not in {'active', 'selected_pending_test'}:
                        raise RuntimeError('production MSE validation barrier differs')
                    self.sleep(1)
                    continue
                if row['status'] == 'selected_pending_test':
                    history = mse['selection']['history']
                    selected = require_sha256(mse['selection']['best_checkpoint_sha256'], 'selected checkpoint')
                    admissible = {self.config['initial_checkpoint_sha256']}
                    admissible.update(r['candidate_checkpoint_sha256'] for r in status['iterations']
                        if r['round_id'] == self.round_id and r['iteration_id'] <= index + 1
                        and r.get('candidate_checkpoint_sha256'))
                    if (status['run']['status'] != 'selected_pending_test' or mse['status'] != 'selected_pending_test'
                            or (index + 1) % self.mse_protocol['interval_iterations'] or selected not in admissible
                            or selected not in {h['result']['checkpoint_sha256'] for h in history}
                            or history[-1]['iteration'] != index + 1
                            or mse['iteration_id'] != index + 1 or mse['checkpoint_sha256'] != candidate
                            or mse['stop_reason'] not in ({'fixed_100_updates'} if self.mse_protocol.get('schema') == 'roborender.dagger-fixed100-round-protocol.v1' else {'validation_mse_plateau', 'maximum_iterations'})
                            or (self.mse_protocol.get('schema') == 'roborender.dagger-fixed100-round-protocol.v1' and (index + 1 != 100 or selected != candidate))
                            or (mse['stop_reason'] == 'maximum_iterations' and index + 1 != self.mse_protocol['max_iterations'])
                            or any(r['round_id'] == self.round_id and r['iteration_id'] > index + 1 for r in status['iterations'])):
                        raise RuntimeError('production selection lacks a complete coverage boundary')
                    self.selected = dict(selected_checkpoint_sha256=selected, stop_reason=mse['stop_reason'])
                    return candidate
                following = next((r for r in status['iterations'] if r['round_id'] == self.round_id
                                  and r['iteration_id'] == index + 2), {})
                if (index + 1 >= self.mse_protocol['max_iterations']
                        or row['status'] != 'advanced' or mse['status'] != 'active'
                        or following.get('status') != 'collecting'
                        or following.get('input_checkpoint_sha256') != candidate
                        or status['run']['active_checkpoint_sha256'] != candidate):
                    raise RuntimeError('production candidate is not activated before the next window')
                return candidate
            self.sleep(1)
        raise TimeoutError('production segment checkpoint/MSE barrier timed out')

    def run(self):
        start = self.segment['start_iteration_index']
        parent = self.segment['input_checkpoint_sha256']
        continuous = is_continuous_segment(self.segment)
        count = self.segment['window_count'] if continuous else 2
        try:
            if continuous and (start != 0 or count != 100
                    or self.segment.get('lifecycle_policy') != 'native_episode_done_only'):
                raise RuntimeError('continuous round cannot resume or shorten native simulator history')
            initial = self.status()
            rows = [r for r in initial['iterations'] if r['round_id'] == self.round_id]
            current = next((r for r in rows if r['iteration_id'] == start + 1), {})
            if ((start == 0 and parent != self.config['initial_checkpoint_sha256'])
                    or len(rows) != start + 1 or initial['run']['status'] != 'collecting'
                    or initial['run']['active_checkpoint_sha256'] != parent
                    or current.get('status') != 'collecting' or current.get('input_checkpoint_sha256') != parent
                    or current.get('candidate_checkpoint_sha256') is not None
                    or any(r['round_id'] == self.round_id and r['iteration_id'] > start for r in initial['barriers'])):
                raise RuntimeError('production segment is not at its exact unstarted checkpoint boundary')
            if start:
                previous = next((r for r in rows if r['iteration_id'] == start), {})
                if previous.get('status') != 'advanced' or previous.get('candidate_checkpoint_sha256') != parent:
                    raise RuntimeError('production segment parent differs from preceding learner candidate')
            if not continuous:
                self.runner.lifecycle.begin()
            self.save('READY', lifecycle_initializations=0 if continuous else 1,
                **({'environment_initializations': 1, 'model_rollout_lifecycles': 0,
                    'native_environment_initializations': 1, 'model_rollout_initializations': 0,
                    'environment_reset_at_window_boundary': False,
                    'model_rollout_lifecycle': 'per_81_step_window',
                    'source_sha256': self.source_sha256, 'native_pid': os.getpid()} if continuous else {}))
            for index in range(start, start + count):
                if self.runner.phase_telemetry is not None:
                    self.runner.phase_telemetry.set_vector_step_offset(index * 81)
                self.save('COLLECTING', window_index=index, input_checkpoint_sha256=parent)
                # Raven begins/clears model rollout buffers each81; this is NOT
                # env.reset_all, scene construction, anchor resampling or RNG
                # reseeding. The same runner/cursors/native runtime survives.
                result = self.runner.run_window(index, manage_lifecycle=continuous)
                if continuous:
                    self.state['model_rollout_lifecycles'] += 1
                    self.state['model_rollout_initializations'] += 1
                    self.state['lifecycle_initializations'] += 1
                if result['transition_batches'] != 81 or result['physical_transition_rows'] != 81 * self.config['num_envs_per_gpu']:
                    raise RuntimeError('production native window is incomplete')
                continuity = None
                if self.segment['schema'] == CONTINUOUS_8H200_SCHEMA:
                    from .episode_continuity import validate_window
                    continuity = validate_window(result.get('episode_continuity'),
                        batch=self.config['num_envs_per_gpu'],index=index,
                        previous_end=(self.state['windows'][-1]['episode_continuity']['end']
                                      if self.state['windows'] else None))
                for lane in range(self.config['num_envs_per_gpu']):
                    actor = f'{self.runner.actor_prefix}-lane-{lane:03d}'
                    ack = self.request('POST', '/v1/barriers', dict(run_id=self.config['run_id'], round_id=self.round_id,
                        actor_id=actor, checkpoint_sha256=parent,
                        chunk_ids=[f'd{self.round_id}-{actor}-window-{index:06d}']))
                    if not ack.get('ok') or ack.get('iteration_id') != index + 1 or ack.get('expected') != len(actor_ids(self.config)):
                        raise RuntimeError('production actor barrier rejected')
                self.save('WAITING_FOR_CHECKPOINT', window_index=index)
                candidate = self._candidate(index, parent)
                self.state['windows'].append(dict(index=index, input_checkpoint_sha256=parent,
                                                  output_checkpoint_sha256=candidate))
                if continuity is not None:
                    self.state['windows'][-1]['episode_continuity'] = continuity
                parent = candidate
                if continuous and index + 1 == self.segment['qualification_after_updates']:
                    self._wait_continuous_qualification(candidate)
                    if getattr(self, 'diagnostic_scope', None) is not None:
                        if (len(self.state['windows']) != 4 or self.selected is not None
                                or not self.state.get('qualification_proof')):
                            raise RuntimeError('diagnostic finish lacks its genuine four-update prefix proof')
                        self.save('DIAGNOSTIC_COMPLETE', completed_windows=4,
                            transition_batches=324, physical_transition_rows=324*self.config['num_envs_per_gpu'],
                            final_candidate_checkpoint_sha256=parent, selection=None,
                            history_reset_at_window_boundary=False, diagnostic_only=True,
                            production_round_complete=False, optimizer_updates=4,
                            fifth_window_started=False, diagnostic_scope_sha256=self.diagnostic_scope_sha256)
                        return dict(self.state)
                if self.selected is not None:
                    break
            if continuous and (len(self.state['windows']) != count or self.selected is None):
                raise RuntimeError('continuous round ended without exact update100 final checkpoint selection')
            self.save('COMPLETE', completed_windows=len(self.state['windows']),
                transition_batches=81 * len(self.state['windows']),
                physical_transition_rows=81 * len(self.state['windows']) * self.config['num_envs_per_gpu'],
                final_candidate_checkpoint_sha256=parent, selection=self.selected,
                history_reset_at_window_boundary=False)
            return dict(self.state)
        except BaseException as exc:
            self.save('CRASH_UNCERTAIN', uncertain_reason=str(exc)[:512])
            raise
        finally:
            if not continuous:
                self.runner.lifecycle.finish()
