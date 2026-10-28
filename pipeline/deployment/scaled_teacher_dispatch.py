"""Durable paired teacher baseline; injected transport, no allocation APIs.

Each GPU owns one cohort at a time. Transport ambiguity stops new admissions;
the next tick reconciles the same command, never substitutes an operation.
This is a privileged-state control baseline, not a visual-student benchmark.
"""
from copy import deepcopy
import fcntl
from pathlib import Path
import re

from pipeline.deployment import scaled_success_dispatch as common
from pipeline.deployment.rtx_actor import scaled_teacher_benchmark as teacher

PLAN_SCHEMA = 'roborender.scaled-teacher-dispatch-plan.v1'
COMMIT_SCHEMA = 'roborender.scaled-teacher-cohort-commit.v1'


def build_plan(panel, owners, *, command_prefix, shadow_checkpoint_sha256):
    panel = teacher.require_panel(panel)
    owners = common._owners(owners)
    if [row['physical_gpu'] for row in owners] != [0, 1]:
        raise ValueError('teacher baseline requires both distinct RTX owners')
    if (not isinstance(command_prefix, str)
            or re.fullmatch('[a-z0-9][a-z0-9-]{0,69}', command_prefix) is None):
        raise ValueError('fresh bounded teacher campaign prefix required')

    def artifact(name, value):
        return dict(path=command_prefix+'/'+name, raw_sha256=common.digest(value), value=deepcopy(value))

    panel_file = artifact('panel.json', panel)
    commands = []
    for index in range(19):
        owner_index = index % 2
        owner, cid = owners[owner_index], f'{command_prefix}-c{index:03d}'
        spec = teacher.build_benchmark(panel, owner=owner, command_id=cid, output_name=cid,
            cohort_index=index, shadow_checkpoint_sha256=shadow_checkpoint_sha256)
        spec_file = artifact(f'benchmark-{index:03d}.json', spec)
        body = dict(schema='roborender.isaac-resident.v1', worker_instance_id=owner['worker_instance_id'],
            command_id=cid, operation=teacher.OPERATION, payload=dict(
                panel_file=panel_file['path'], panel_sha256=panel_file['raw_sha256'],
                benchmark_file=spec_file['path'], benchmark_sha256=spec_file['raw_sha256'],
                checkpoint_sha256=shadow_checkpoint_sha256, output_name=cid, **teacher.evaluation_options()))
        commands.append(dict(cohort_index=index, owner_index=owner_index, body=body,
            payload_sha256=common.digest(body), artifacts=[spec_file]))
    return dict(schema=PLAN_SCHEMA, panel=panel, owners=owners, command_prefix=command_prefix,
        shadow_not_behavior_checkpoint_sha256=shadow_checkpoint_sha256, commands=commands,
        shared_artifacts=[panel_file], observation_mode='privileged_state',
        initial_pair_integrity_barrier=True,
        compute_release_authorized=False, native_replay_allowed=False)


def validate_plan(value):
    expected = build_plan(value['panel'], value['owners'], command_prefix=value['command_prefix'],
        shadow_checkpoint_sha256=value['shadow_not_behavior_checkpoint_sha256'])
    if value != expected:
        raise ValueError('immutable teacher plan changed')
    return deepcopy(expected)


def validate_complete(plan, entry, row, resource_validator):
    validate_plan(plan)
    if entry not in plan['commands'] or not isinstance(row, dict):
        raise ValueError('teacher completion is not an admitted command')
    owner, receipt = plan['owners'][entry['owner_index']], row.get('receipt', {})
    if (row.get('status') != 'COMPLETE' or row.get('payload_sha256') != entry['payload_sha256']
            or receipt.get('complete') is not True or receipt.get('source_sha256') != owner['source_sha256']
            or receipt.get('worker_instance_id') != owner['worker_instance_id']
            or receipt.get('native_pid') != owner['native_pid']
            or receipt.get('command_id') != entry['body']['command_id']
            or receipt.get('native_initializations') != 1 or receipt.get('same_kit') is not True
            or type(receipt.get('kit_identity')) is not int or receipt['kit_identity'] <= 0
            or receipt.get('physics_pause_requested') is not True
            or receipt.get('purpose') != 'test' or receipt.get('profile') != 'explicit_evaluation'
            or type(receipt.get('optimizer_steps')) is not int or receipt['optimizer_steps'] != 0
            or any(type(receipt.get(k)) is not int or receipt[k] <= 0
                for k in ('scene_initializations', 'phase_resets'))):
        raise ValueError('actual terminal teacher native receipt required')
    result = teacher.validate_result(plan['panel'], receipt.get('result'))
    if result['benchmark'] != entry['artifacts'][0]['value']:
        raise ValueError('completed teacher descriptor differs from admitted command')
    if not callable(resource_validator):
        raise ValueError('actual native resource evidence verifier required')
    resource_validator(owner, receipt)
    return result


class Coordinator:
    """One bounded paired tick; construction never contacts a GPU or host."""
    def __init__(self, plan, root, transport, *, resource_validator):
        self.plan = validate_plan(plan)
        self.root, self.transport = Path(root), transport
        if self.root.is_symlink() or not callable(resource_validator):
            raise ValueError('physical root and actual resource verifier required')
        self.resource_validator = resource_validator
        self.root.mkdir(parents=True, exist_ok=True)
        self.plan_sha = common.digest(self.plan)
        self._verified_commits = {}
        common.immutable(self.root/'plan.json', self.plan)
        common.immutable(self.root/'controller-policy.json', dict(
            submit_attempts=1, native_replay_allowed=False, stop_peer_on_uncertainty=True,
            compute_release_authorized=False))

    def _path(self, entry):
        return self.root/f"cohort-{entry['cohort_index']:03d}"

    def _ack(self, entry):
        return dict(schema='roborender.isaac-resident.v1', status='accepted',
            command_id=entry['body']['command_id'], worker_instance_id=entry['body']['worker_instance_id'],
            payload_sha256=entry['payload_sha256'])

    def _commit(self, entry, row):
        result = validate_complete(self.plan, entry, row, self.resource_validator)
        value = dict(schema=COMMIT_SCHEMA, plan_sha256=self.plan_sha, cohort_index=entry['cohort_index'],
            owner=self.plan['owners'][entry['owner_index']], command_id=entry['body']['command_id'],
            payload_sha256=entry['payload_sha256'], native_command_receipt=deepcopy(row), result=result)
        common.immutable(self._path(entry)/'commit.json', value)
        self._verified_commits[entry['cohort_index']] = (common.digest(value), deepcopy(value))
        return value

    def _read_commit(self, entry):
        path = self._path(entry)/'commit.json'
        if not path.exists():
            return None
        value = common.read(path)
        if path.read_bytes() != common.canonical(value)+b'\n':
            raise ValueError('immutable teacher commit bytes changed')
        cached = self._verified_commits.get(entry['cohort_index'])
        if cached is not None and common.digest(value) == cached[0]:
            return deepcopy(cached[1])
        return self._commit(entry, value['native_command_receipt'])

    def _step(self, entry):
        owner, path = self.plan['owners'][entry['owner_index']], self._path(entry)
        cid = entry['body']['command_id']
        common.immutable(path/'intent.json', dict(plan_sha256=self.plan_sha, entry=entry, owner=owner))
        ready = common.validate_ready(self.transport.get_ready(deepcopy(owner)), owner)
        row = self.transport.get_command(deepcopy(owner), cid)
        if row is not None:
            if (not isinstance(row, dict) or row.get('payload_sha256') != entry['payload_sha256']
                    or row.get('status') not in {'QUEUED', 'RUNNING', 'COMPLETE', 'CRASH_UNCERTAIN'}):
                raise ValueError('teacher command identity/status differs')
            common.immutable(path/'acceptance.json', self._ack(entry))
            if row['status'] == 'COMPLETE':
                self._commit(entry, row)
                return 'COMMITTED'
            if row['status'] == 'CRASH_UNCERTAIN' or ready['phase'] in {'CRASH_UNCERTAIN', 'STOPPED'}:
                common.immutable(path/'terminal-fault.json', row)
                return 'NATIVE_TERMINAL_FAULT'
            if ready['phase'] in {'QUEUED', 'RUNNING'} and ready.get('active_command_id') != cid:
                raise ValueError('teacher owner is running unrelated work')
            return 'IN_FLIGHT'
        if (path/'acceptance.json').exists():
            raise ValueError('accepted teacher command disappeared; replay forbidden')
        if ready['phase'] in {'CRASH_UNCERTAIN', 'STOPPED'}:
            return 'NATIVE_TERMINAL_FAULT'
        if ready['phase'] != 'IDLE':
            if ready.get('active_command_id') != cid:
                raise ValueError('teacher owner busy with unrelated work')
            return 'RECONCILING_ACCEPTANCE'
        attempt = path/'submit-attempt.json'
        if attempt.exists():
            if common.read(attempt) != dict(plan_sha256=self.plan_sha, body=entry['body'], owner=owner):
                raise ValueError('teacher submit intent changed')
            return 'UNKNOWN_REQUIRES_RECONCILIATION'
        if (self.root/'STOP').exists():
            return 'NOT_STARTED_STOPPED'
        self.transport.stage_artifacts(deepcopy(owner), deepcopy(self.plan['shared_artifacts']+entry['artifacts']))
        ready = common.validate_ready(self.transport.get_ready(deepcopy(owner)), owner)
        if self.transport.get_command(deepcopy(owner), cid) is not None:
            return 'RECONCILING_ACCEPTANCE'
        if ready['phase'] != 'IDLE' or (self.root/'STOP').exists():
            return 'RECONCILING_ACCEPTANCE' if ready['phase'] != 'IDLE' else 'NOT_STARTED_STOPPED'
        common.immutable(attempt, dict(plan_sha256=self.plan_sha, body=entry['body'], owner=owner))
        ack = self.transport.submit(deepcopy(owner), deepcopy(entry['body']))
        if ack != self._ack(entry):
            raise ValueError('teacher acceptance differs from immutable intent')
        common.immutable(path/'acceptance.json', ack)
        return 'SUBMITTED'

    def tick(self):
        path = self.root/'controller.lock'
        if path.is_symlink():
            raise ValueError('teacher controller lock cannot be a symlink')
        with path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if common.read(self.root/'plan.json') != self.plan:
                raise ValueError('durable teacher plan changed')
            commits = {e['cohort_index']: self._read_commit(e) for e in self.plan['commands']}
            pending = []
            for gpu in (0, 1):
                rows = [e for e in self.plan['commands'] if e['owner_index']==gpu and commits[e['cohort_index']] is None]
                if rows:
                    pending.append(rows[0])
            # Always reconcile attempted work before new peer submissions.
            pending.sort(key=lambda e:not any((self._path(e)/name).exists()
                for name in ('submit-attempt.json','acceptance.json')))
            states = []
            for entry in pending:
                if entry['cohort_index'] >= 2 and not all(commits[i] is not None for i in (0, 1)):
                    states.append(dict(owner_index=entry['owner_index'], cohort_index=entry['cohort_index'],
                        status='WAITING_INITIAL_PAIR_INTEGRITY'))
                    continue
                try:
                    status = self._step(entry)
                except (TimeoutError, ConnectionError):
                    status = 'TRANSPORT_UNKNOWN_RECONCILE_SAME_OWNER'
                states.append(dict(owner_index=entry['owner_index'], cohort_index=entry['cohort_index'], status=status))
                if status == 'COMMITTED':
                    commits[entry['cohort_index']] = self._read_commit(entry)
                if status not in {'SUBMITTED', 'IN_FLIGHT', 'COMMITTED'}:
                    break
            values = [commits[i]['result'] for i in sorted(commits) if commits[i] is not None]
            result = None
            if len(values) == 19:
                result = teacher.aggregate_results(self.plan['panel'], values)
                common.immutable(self.root/'result.json', result, limit=common.MAX_FINAL_RESULT_LIMIT)
            successes = sum(v['cohort_result']['exported_successes'] for v in values)
            completed = sum(v['benchmark']['num_envs'] for v in values)
            return dict(schema='roborender.scaled-teacher-dispatch-progress.v1', plan_sha256=self.plan_sha,
                complete=result is not None, partial=result is None, completed=completed, expected=295,
                successes=successes, success_rate=successes/completed if completed else None,
                committed_cohorts=len(values), workers=states, observation_mode='privileged_state',
                stop_requested=(self.root/'STOP').exists(), compute_release_authorized=False)
