"""Consolidated success evaluation runtime; versioned protocols remain distinct."""
from __future__ import annotations


# ---- paired scaled evaluation ----

from copy import deepcopy as evaluation_deepcopy
import base64 as evaluation_base64
import fcntl as evaluation_fcntl
import hashlib as evaluation_hashlib
import json as evaluation_json
import math as evaluation_math
from pathlib import Path as evaluation_Path
import time as evaluation_time

from pipeline.success_rate import scaled_success_dispatch as evaluation_dispatch
from pipeline.success_rate import scaled_success_contract as evaluation_scaled
from pipeline.success_rate.rendered_student_success_contract import bind_observation_config as evaluation_bind_observation_config, validate_config as evaluation_validate_config
from pipeline.success_rate.scaled_success_qualification import validate_qualification as evaluation_validate_qualification, SCHEMA as evaluation_QUALIFICATION_SCHEMA
from pipeline.success_rate.scaled_success_resources import summarize as evaluation_summarize

evaluation_RUNTIME_SCHEMA = 'roborender.paired-scaled-runtime.v1'
evaluation_FINAL_SCHEMA = 'roborender.paired-scaled-final-result.v1'
evaluation_RUN_SCHEMA = 'roborender.paired-scaled-checkpoint-run.v1'
evaluation_TRIALS = (('b4', 4, 'paired'), ('b8', 8, 'paired'), ('b16', 16, 'paired'),
          ('b16-repeat', 16, 'paired'), ('tail7', 16, 'tail7'))
evaluation_FAULTS = {'TRANSPORT_UNKNOWN_RECONCILE_SAME_OWNER', 'NATIVE_TERMINAL_FAULT',
          'SUBMISSION_LIMIT_REQUIRES_RECONCILIATION', 'STOPPED_UNCERTAIN_ACCEPTANCE',
          'NOT_STARTED_STOPPED'}


def evaluation__raw(path, *, aggregate=False):
    path=evaluation_Path(path);limit=evaluation_dispatch.MAX_FINAL_RESULT_LIMIT if aggregate else evaluation_dispatch.LIMIT
    if (not path.is_absolute() or path.resolve(strict=True)!=path or not path.is_file()
            or path.stat().st_size>limit):
        raise ValueError('bounded physical absolute evidence required')
    before=path.stat()
    with path.open('rb') as stream:raw=stream.read(limit+1)
    after=path.stat()
    if (len(raw)>limit or (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)):
        raise ValueError('bound evidence changed while reading')
    return raw


def evaluation_ref(path, *, aggregate=False):
    return dict(path=str(evaluation_Path(path)), raw_sha256=evaluation_hashlib.sha256(evaluation__raw(path,aggregate=aggregate)).hexdigest())


def evaluation_bound(reference, *, aggregate=False):
    if (not isinstance(reference, dict) or set(reference) != {'path', 'raw_sha256'}
            or not evaluation_Path(reference['path']).is_absolute()
            or evaluation_Path(reference['path']).resolve(strict=True) != evaluation_Path(reference['path'])):
        raise ValueError('physical absolute hash-bound evidence required')
    raw=evaluation__raw(reference['path'],aggregate=aggregate)
    if evaluation_hashlib.sha256(raw).hexdigest()!=reference['raw_sha256']:
        raise ValueError('paired evidence changed')
    return evaluation_json.loads(raw)


def evaluation_final_panel(panel):
    panel = evaluation_scaled.require_panel(panel)
    if (panel['scaled_protocol']['purpose'] != 'test'
            or panel['scaled_protocol']['batch_size'] != 16 or panel['episode_count'] != 295):
        raise ValueError('only the fixed B16 full295 scaled test is production evaluation')
    return panel


def evaluation_descriptors(owners):
    return [dict(owner_id=o['worker_instance_id'], gpu_uuid=o['gpu_uuid'],
                 source_sha256=o['source_sha256']) for o in owners]


def evaluation_renderer_descriptors(config):
    return [dict(owner_id=o['server_instance'], gpu_uuid=o['gpu_uuid'],
                 source_sha256=o['source_sha256']) for o in config['renderer']['endpoints']]


def evaluation_validate_barrier(rows, owners):
    if not isinstance(rows, list) or len(rows) != 2 or len(owners) != 2:
        raise ValueError('both native owners must reach an exact IDLE barrier')
    for row, owner in zip(rows, owners):
        evaluation_dispatch.validate_ready(row, owner)
        if row['phase'] != 'IDLE':
            raise ValueError('a native owner is not quiescent')
    return rows


def evaluation_idle_barrier(transport, owners):
    return evaluation_validate_barrier([transport.get_ready(o) for o in owners], owners)


def evaluation_validate_guard_copy(owner, receipt, record):
    """Reproduce the source guard's numeric checks from its actual copied bytes."""
    from pipeline.dagger.rtx.code.src.dagger_rtx.phase_resource_guard import (
        validate_summary, classify_sample, validate_transition, policy_for,
        SUMMARY_FIELDS, SUSTAINED_SCHEMA)
    summary = receipt['resource_guard']
    if (record.get('owner') != owner or record.get('summary') != summary
            or record.get('verified') is not True):
        raise ValueError('guard was not read from this exact native owner')
    raw = evaluation_base64.b64decode(record['guard_raw_base64'], validate=True)
    if evaluation_hashlib.sha256(raw).hexdigest() != summary['raw_sha256']:
        raise ValueError('actual guard bytes do not match the native receipt')
    value = evaluation_json.loads(raw)
    validate_summary(summary); validate_summary(value)
    interval = value.get('interval_s')
    if (value.get('schema') != SUSTAINED_SCHEMA or summary.get('schema') != SUSTAINED_SCHEMA
            or type(interval) not in (int, float) or not evaluation_math.isfinite(interval) or not 0 < interval <= .5
            or value.get('policy') != policy_for(value.get('policy_version'))
            or value.get('sampling_error') is not None
            or any(value.get(k) != summary.get(k) for k in SUMMARY_FIELDS)
            or value.get('pressure_window') != summary.get('pressure_window')):
        raise ValueError('actual sustained guard policy/summary differs')
    for key in ('counter_read_attempts', 'counter_read_recoveries'):
        if key in summary and value.get(key) != summary[key]:
            raise ValueError('native counter audit changed')
    first, last = value['first_sample'], value['last_sample']
    classify_sample(first, policy_version=value['policy_version'])
    classify_sample(last, policy_version=value['policy_version']); validate_transition(first, last)
    if (first['events']['max'] != value['initial_limit_pressure_events']
            or last['events']['max'] != value['limit_pressure_events']
            or value['minimum_host_available_bytes'] > min(first['host_available_bytes'], last['host_available_bytes'])
            or type(value.get('memory_current_peak_bytes')) is not int
            or not max(first['memory_current_bytes'], last['memory_current_bytes']) <= value['memory_current_peak_bytes'] <= 21*(1 << 30)
            or value['maximum_working_set_bytes'] < max(first['working_set_bytes'], last['working_set_bytes'])
            or value['maximum_some_avg10'] < max(first['some_avg10'], last['some_avg10'])
            or value['maximum_full_avg10'] < max(first['full_avg10'], last['full_avg10'])):
        raise ValueError('actual guard extrema/endpoint counters differ')
    return value


def evaluation_validate_interval(value, plan):
    owners = evaluation_descriptors(plan['owners']); renderers = evaluation_renderer_descriptors(plan['config'])
    actual = evaluation_summarize(value['samples'], native_owners=owners, renderer_owners=renderers,
                       started_unix=value['started_unix'], ended_unix=value['ended_unix'])
    if value.get('renderer_owners') != renderers or value.get('summary') != actual:
        raise ValueError('joint resource interval does not reproduce actual owners/headroom')
    return value


def evaluation_validate_dispatch_evidence(evidence):
    plan = evaluation_dispatch.validate_plan(evaluation_bound(evidence['dispatch_plan']))
    commits = evidence['cohort_commits']
    if len(commits) != len(plan['commands']):
        raise ValueError('every exact cohort needs a committed native result')
    guard_rows=list(map(evaluation_bound,evidence['resource_guards']))
    guards = {row['summary']['raw_sha256']: row for row in guard_rows}
    if len(guards)!=len(guard_rows):
        raise ValueError('duplicate guard evidence cannot obscure exact owner attribution')
    values = []
    used_guards=set()
    for entry, reference in zip(plan['commands'], commits):
        commit = evaluation_bound(reference); owner = plan['owners'][entry['owner_index']]
        def guard(o, receipt):
            used_guards.add(receipt['resource_guard']['raw_sha256'])
            evaluation_validate_guard_copy(o, receipt, guards[receipt['resource_guard']['raw_sha256']])
        checked = evaluation_dispatch.validate_complete(plan, entry, commit['native_command_receipt'], guard)
        expected = dict(schema=evaluation_dispatch.COMMIT_SCHEMA, plan_sha256=evaluation_dispatch.digest(plan),
            cohort_index=entry['cohort_index'], owner=owner, command_id=entry['body']['command_id'],
            payload_sha256=entry['payload_sha256'], native_command_receipt=commit['native_command_receipt'], result=checked)
        if commit != expected:
            raise ValueError('cohort commit identity or scientific result changed')
        values.append(checked)
    if used_guards!=set(guards):
        raise ValueError('guard evidence includes an unrelated native command')
    result = evaluation_scaled.aggregate_results(plan['panel'], values, plan['checkpoint_sha256'],
                                      plan['config'], arm=plan['arm'])
    if result != evaluation_bound(evidence['result'],aggregate=True):
        raise ValueError('aggregate differs from exact native cohort outcomes')
    evaluation_validate_interval(evaluation_bound(evidence['resources'],aggregate=True), plan)
    all_owners = evidence['barrier_owners']
    if ([o['physical_gpu'] for o in all_owners] != [0, 1]
            or any(o not in all_owners for o in plan['owners'])):
        raise ValueError('both original native lifetimes required around every trial')
    evaluation_dispatch._owners(all_owners)
    evaluation_validate_barrier(evidence['before_idle'], all_owners)
    evaluation_validate_barrier(evidence['after_idle'], all_owners)
    return plan, result


def evaluation_validate_final_evidence(value, *, panel, checkpoint_sha256, source_sha256, arm,
                            expected_rendered_config=None):
    """The only future-production import path; raw/scalar results are insufficient."""
    panel = evaluation_final_panel(panel)
    if (value.get('schema') != evaluation_FINAL_SCHEMA or value.get('complete') is not True
            or value.get('qualification_only') is not False
            or value.get('source_sha256') != source_sha256
            or value.get('checkpoint_sha256') != checkpoint_sha256 or value.get('arm') != arm
            or value.get('allocation_release_authorized') is not False
            or value.get('historical_outcomes_reused') is not False):
        raise ValueError('actual paired final-test envelope required; qualification is not test')
    from pipeline.success_rate.paired_scaled_runtime import cpu_gate
    cpu_gate(value['cpu_validation'],source_sha256)
    plan, result = evaluation_validate_dispatch_evidence(value['final_dispatch'])
    if (plan['panel'] != panel or plan['checkpoint_sha256'] != checkpoint_sha256 or plan['arm'] != arm
            or [o['physical_gpu'] for o in plan['owners']] != [0, 1]
            or any(o['source_sha256'] != source_sha256 for o in plan['owners'])
            or (expected_rendered_config is not None and plan['config'] != expected_rendered_config)):
        raise ValueError('final test checkpoint/panel/render dependencies/native source differ')
    proof = evaluation_bound(value['qualification'])
    evaluation_validate_qualification(proof, source_sha256=source_sha256, panel=panel, config=plan['config'])
    if proof != plan['qualification_evidence'] or proof['arm'] != arm:
        raise ValueError('final dispatch did not use this actual checkpoint qualification')
    # Re-read all prefix commits and actual resource guards, not just boolean
    # qualification summaries. Ownership must be the same lifetime as final295.
    trial_refs = value['qualification_dispatches']
    if len(trial_refs) != len(evaluation_TRIALS):
        raise ValueError('all five measured qualification dispatches required')
    for reference, trial in zip(trial_refs, proof['trials'] + [proof['tail_trial']]):
        evidence = evaluation_bound(reference); qplan, _ = evaluation_validate_dispatch_evidence(evidence)
        if (qplan['panel'] != trial['panel'] or qplan['config'] != trial['config']
                or evidence['barrier_owners'] != plan['owners']
                or trial != evaluation_trial_summary(evidence)):
            raise ValueError('qualification is not bound to these original native owners/commits')
    if result.get('completed_episodes') != 295 or result.get('qualification_only') is not False:
        raise ValueError('only exact complete295 outcomes may enter the final-test ledger')
    return result


def evaluation_trial_summary(evidence):
    plan = evaluation_bound(evidence['dispatch_plan']); owners = []
    for reference in evidence['cohort_commits']:
        commit = evaluation_bound(reference); owner = commit['owner']
        owners.append(dict(physical_gpu=owner['physical_gpu'], worker_instance_id=owner['worker_instance_id'],
            gpu_uuid=owner['gpu_uuid'], native_receipt=commit['native_command_receipt']['receipt']))
    resources = evaluation_bound(evidence['resources'],aggregate=True)
    return dict(panel=plan['panel'], config=plan['config'], owners=owners,
        started_unix=resources['started_unix'], ended_unix=resources['ended_unix'],
        resources={k: resources[k] for k in ('samples', 'renderer_owners', 'summary')})


def evaluation_make_run(panel, config, owners, *, command_prefix, source_sha256, arm, cpu_validation):
    panel = evaluation_final_panel(panel); evaluation_validate_config(config, panel); owners = evaluation_dispatch._owners(owners)
    if ([o['physical_gpu'] for o in owners] != [0, 1]
            or any(o['source_sha256'] != source_sha256 for o in owners)
            or config['renderer'].get('routing') != 'cohort_parity_actor_shards_v1'
            or len(config['renderer'].get('endpoints', [])) != 8):
        raise ValueError('B16 on both RTX and eight exclusive rendered endpoints required')
    trials = []
    for label, size, variant in evaluation_TRIALS:
        qpanel = evaluation_scaled.build_panel(panel['scaled_protocol']['source_panel'], batch_size=size,
                                   qualification=True, qualification_variant=variant)
        qconfig = evaluation_bind_observation_config(config, qpanel, config['checkpoint_sha256'])
        plan = evaluation_dispatch.build_plan(qpanel, qconfig, owners[:1] if variant == 'tail7' else owners,
                                    command_prefix=command_prefix+'-'+label, arm=arm)
        trials.append(dict(label=label, plan=plan))
    return dict(schema=evaluation_RUN_SCHEMA, source_sha256=source_sha256, panel=panel, config=config,
        owners=owners, arm=arm, command_prefix=command_prefix, trials=trials,cpu_validation=evaluation_deepcopy(cpu_validation),
        qualification_selects_checkpoint=False, allocation_release_authorized=False)


class evaluation_Run:
    """Saved per-checkpoint prefix; no duplicate path or native submission retry."""
    def __init__(self, plan, root, transport, sampler, *, clock=evaluation_time.time, sleep=evaluation_time.sleep,
                 deadline=None, progress=None):
        if plan != evaluation_make_run(plan['panel'], plan['config'], plan['owners'],
                command_prefix=plan['command_prefix'], source_sha256=plan['source_sha256'], arm=plan['arm'],
                cpu_validation=plan['cpu_validation']):
            raise ValueError('paired execution plan changed')
        self.plan, self.root, self.transport, self.sampler = plan, evaluation_Path(root), transport, sampler
        self.clock, self.sleep = clock, sleep
        self.deadline,self.progress=deadline,progress
        evaluation_dispatch.immutable(self.root/'run-plan.json', plan)

    def dispatch_one(self, label, plan):
        root = self.root/label; evidence_path = root/'dispatch-evidence.json'
        if evidence_path.exists():
            value = evaluation_dispatch.read(evidence_path); evaluation_validate_dispatch_evidence(value)
            if evaluation_bound(value['dispatch_plan']) != plan:
                raise ValueError('saved trial belongs to another checkpoint/plan')
            return value
        self.sampler.healthy()
        start_path = root/'interval-start.json'
        if not start_path.exists():
            before = evaluation_idle_barrier(self.transport, self.plan['owners'])
            evaluation_dispatch.immutable(start_path, dict(started_unix=self.clock(), before_idle=before))
        started = evaluation_dispatch.read(start_path)
        # The same saved command IDs reconcile an accepted operation on resume.
        # Single-attempt coordinator never interprets a lost SSH reply as dead.
        coordinator = evaluation_dispatch.Coordinator(plan, root, self.transport,
            resource_validator=self.transport.resource_validator, max_submit_attempts=1,
            result_limit=evaluation_dispatch.MAX_FINAL_RESULT_LIMIT,stop_peer_on_uncertainty=True)
        deadline = min(self.clock()+7*86400,self.deadline) if self.deadline is not None else self.clock()+7*86400
        while True:
            if self.clock() >= deadline:
                raise TimeoutError('observer deadline; native owners, sampler and compute retained')
            self.sampler.healthy(); progress = coordinator.tick()
            if self.progress is not None:self.progress(label,evaluation_deepcopy(progress))
            if self.transport.unknown or any(w['status'] in evaluation_FAULTS for w in progress['workers']):
                raise RuntimeError('paired evaluation stopped at exact UNKNOWN/fault boundary; no replay')
            if progress['complete']:
                break
            if self.clock() >= deadline:
                raise TimeoutError('observer deadline; native owners, sampler and compute retained')
            self.sleep(5)
        after = evaluation_idle_barrier(self.transport, self.plan['owners'])
        end_path = root/'interval-end.json'
        evaluation_dispatch.immutable(end_path, dict(ended_unix=self.clock(), after_idle=after)) if not end_path.exists() else None
        ended = evaluation_dispatch.read(end_path)
        resources = self.sampler.interval(started['started_unix'], ended['ended_unix'], plan['owners'], plan['config'])
        evaluation_dispatch.immutable(root/'resources.json', resources,limit=evaluation_dispatch.MAX_FINAL_RESULT_LIMIT)
        commits = [evaluation_ref(root/f"cohort-{e['cohort_index']:03d}"/'commit.json') for e in plan['commands']]
        guards = sorted({evaluation_bound(c)['native_command_receipt']['receipt']['resource_guard']['raw_sha256'] for c in commits})
        value = dict(dispatch_plan=evaluation_ref(root/'plan.json'), cohort_commits=commits,
            resource_guards=[evaluation_ref(self.transport.guard_root/(sha+'.json')) for sha in guards],
            result=evaluation_ref(root/'result.json',aggregate=True), resources=evaluation_ref(root/'resources.json',aggregate=True),
            barrier_owners=self.plan['owners'], before_idle=started['before_idle'], after_idle=ended['after_idle'])
        evaluation_validate_dispatch_evidence(value); evaluation_dispatch.immutable(evidence_path, value)
        return value

    def execute(self):
        lock = self.root/'paired-controller.lock'
        if lock.is_symlink():
            raise ValueError('paired controller lock alias forbidden')
        with lock.open('a') as stream:
            evaluation_fcntl.flock(stream, evaluation_fcntl.LOCK_EX | evaluation_fcntl.LOCK_NB)
            from pipeline.success_rate.paired_scaled_runtime import cpu_gate
            cpu_gate(self.plan['cpu_validation'],self.plan['source_sha256'])
            summaries = []
            for trial in self.plan['trials']:
                summaries.append(evaluation_trial_summary(self.dispatch_one(trial['label'], trial['plan'])))
            proof = dict(schema=evaluation_QUALIFICATION_SCHEMA, source_sha256=self.plan['source_sha256'],
                batch_size=16, checkpoint_sha256=self.plan['config']['checkpoint_sha256'], arm=self.plan['arm'],
                compute_release_authorized=False, cpu_tests_passed=True,
                trials=summaries[:4], tail_trial=summaries[4])
            evaluation_validate_qualification(proof, source_sha256=self.plan['source_sha256'],
                                   panel=self.plan['panel'], config=self.plan['config'])
            evaluation_dispatch.immutable(self.root/'qualification.json', proof)
            plan = evaluation_dispatch.build_plan(self.plan['panel'], self.plan['config'], self.plan['owners'],
                command_prefix=self.plan['command_prefix']+'-final', arm=self.plan['arm'], qualification_evidence=proof)
            final = self.dispatch_one('final', plan)
            value = dict(schema=evaluation_FINAL_SCHEMA, complete=True, qualification_only=False,
                source_sha256=self.plan['source_sha256'], checkpoint_sha256=plan['checkpoint_sha256'], arm=plan['arm'],
                cpu_validation=self.plan['cpu_validation'],
                qualification=evaluation_ref(self.root/'qualification.json'),
                qualification_dispatches=[evaluation_ref(self.root/t['label']/'dispatch-evidence.json') for t in self.plan['trials']],
                final_dispatch=final, historical_outcomes_reused=False, allocation_release_authorized=False)
            evaluation_validate_final_evidence(value, panel=plan['panel'], checkpoint_sha256=plan['checkpoint_sha256'],
                source_sha256=self.plan['source_sha256'], arm=plan['arm'], expected_rendered_config=plan['config'])
            evaluation_dispatch.immutable(self.root/'comparison.json', value)
            return value


# ---- paired scaled runtime ----
from copy import deepcopy as deepcopy
import base64 as base64
import hashlib as hashlib
import json as json
import os as os
from pathlib import Path as Path
import shlex as shlex
import subprocess as subprocess
import time as time
import xml.etree.ElementTree as ET

from pipeline.success_rate import scaled_success_dispatch as dispatch
from pipeline.success_rate import paired_scaled_evaluation as paired
from pipeline.deployment.beta_operations import process_identity as process_identity

REJECT_OPTIMIZED_PYTHON = "import sys\nif sys.flags.optimize: raise RuntimeError('optimized Python is forbidden for native identity admission')\n"

REMOTE = REJECT_OPTIMIZED_PYTHON + r'''
import base64,hashlib,json,pathlib,sys
from urllib.request import Request,urlopen
from urllib.error import HTTPError
p=json.loads(sys.argv[1]);action=sys.argv[2];source=pathlib.Path(p['source_root'])
sys.path.insert(0,str(source));sys.path.insert(0,str(source/'pipeline/deployment/rtx_actor'))
from production_resident_pool import attest_live_owner,validate_guard
owner=p['owner'];actual=attest_live_owner(owner,p['source_sha256'],mode='production')
assert actual['host_pid']==owner['host_pid'] and actual['container_id']==owner['container_id']
def request(route,body=None):
 token=pathlib.Path(owner['token_file']);assert not token.is_symlink() and not token.stat().st_mode & 0o077
 raw=None if body is None else json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
 req=Request('http://127.0.0.1:'+str(owner['port'])+route,data=raw,headers={
  'Authorization':'Bearer '+token.read_text().strip(),'Content-Type':'application/json'})
 with urlopen(req,timeout=15) as response:raw=response.read((64<<20)+1)
 assert len(raw)<=64<<20
 return json.loads(raw)
ready=request('/ready')
for key in ('worker_instance_id','source_sha256','gpu_uuid','native_pid'):assert ready[key]==owner[key]
assert ready['native_initializations']==1
if action=='ready':result=ready
elif action=='get':
 try:result=request('/v1/commands/'+p['command_id'])
 except HTTPError as error:
  if error.code!=404:raise
  result=None
elif action=='submit':result=request('/v1/commands',p['body'])
elif action=='renderer_preflight':
 assert ready['phase']=='IDLE','renderer admission must precede native work'
 from pipeline.deployment.renderer_preflight import check_renderer_routes
 result=check_renderer_routes(p['renderer'])
elif action=='guard':
 summary=p['receipt']['resource_guard'];validate_guard(summary,owner)
 path=pathlib.Path(summary['path'])
 if path.is_relative_to('/dagger-runtime'):path=pathlib.Path(owner['runtime_root'])/path.relative_to('/dagger-runtime')
 try:raw=path.read_bytes()
 except PermissionError:
  from production_guard_access import read_owned_guard
  raw=read_owned_guard(path,owner,summary['raw_sha256'])
 assert hashlib.sha256(raw).hexdigest()==summary['raw_sha256']
 result=dict(verified=True,owner=owner,summary=summary,guard_raw_base64=base64.b64encode(raw).decode())
elif action=='stage':
 root=pathlib.Path(p['jobs_root']);relative=pathlib.Path(p['artifact']['path'])
 assert root.is_absolute() and not relative.is_absolute() and '..' not in relative.parts
 path=root/relative;raw=sys.stdin.buffer.read((64<<20)+1);assert len(raw)<=64<<20
 assert hashlib.sha256(raw).hexdigest()==p['artifact']['raw_sha256']
 assert root.resolve()==root and path.parent.resolve().is_relative_to(root)
 assert not path.is_symlink()
 path.parent.mkdir(parents=True,exist_ok=True)
 import os
 if path.exists():assert path.read_bytes()==raw
 else:
  fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
  with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
  fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 result=dict(staged=True,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
else:raise ValueError('unsupported exact-owner operation')
print(json.dumps(result,sort_keys=True,separators=(',',':')))
'''


def cpu_gate(reference, source_sha256):
    receipt = paired.bound(reference)
    if (receipt.get('schema') != 'roborender.live-pipeline-smoke-cpu-validation.v1'
            or receipt.get('source_sha256') != source_sha256 or receipt.get('passed') is not True
            or type(receipt.get('tests')) is not int or receipt['tests'] <= 0
            or any(receipt.get(k) != 0 for k in ('failures', 'errors', 'skipped'))):
        raise ValueError('actual passing current-source CPU receipt required')
    reference = receipt['junit_xml']; path = Path(reference['path'])
    if not path.is_absolute() or path.resolve(strict=True) != path or path.stat().st_size > 64 << 20:
        raise ValueError('physical bounded CPU JUnit required')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['raw_sha256']:
        raise ValueError('CPU JUnit changed')
    suites = [s for s in ET.fromstring(raw).iter('testsuite') if not list(s.iter('testsuite'))[1:]]
    if (sum(int(s.attrib.get('tests', 0)) for s in suites) != receipt['tests']
            or any(int(s.attrib.get(k, 0)) for s in suites for k in ('failures', 'errors', 'skipped'))):
        raise ValueError('CPU test counts/errors differ')
    return receipt


def validate_runtime(value, *, source_sha256, pool=None, retained=None):
    fields = {'schema', 'source_sha256', 'native', 'renderer_fleet', 'cpu_validation',
              'resource_sampler', 'allocation_release_authorized'}
    if (set(value) != fields or value.get('schema') != paired.RUNTIME_SCHEMA
            or value.get('source_sha256') != source_sha256
            or value.get('allocation_release_authorized') is not False):
        raise ValueError('explicit source-bound paired runtime required')
    native, fleet = paired.bound(value['native']), paired.bound(value['renderer_fleet'])
    from pipeline.deployment.h200.retained_renderer_attachment import validate_fleet
    validate_fleet(fleet)
    if (native.get('schema') != 'roborender.scaled-evaluation-native-bindings.v1'
            or native.get('source_sha256') != source_sha256
            or [o['physical_gpu'] for o in native.get('workers', [])] != [0, 1]):
        raise ValueError('exact paired native lifetime bindings required')
    dispatch._owners(native['workers'])
    for owner in native['workers']:
        if (owner['source_sha256'] != source_sha256 or type(owner.get('host_pid')) is not int
                or owner['host_pid'] <= 0 or not owner.get('container_id')):
            raise ValueError('actual source/host PID/container identity required')
        if pool is not None:
            worker = pool['workers'][owner['physical_gpu']]
            if any(owner.get(k) != worker[k] for k in worker):
                raise ValueError('paired owner differs from admitted production pool')
    if pool is not None and any(native.get(k) != pool[k] for k in ('source_sha256', 'jobs_root', 'assets_root')):
        raise ValueError('paired native roots differ from production pool')
    if retained is not None and (retained.get('schema') != 'roborender.full295-retained-h200.v3'
            or retained['renderer_fleet'] != value['renderer_fleet']):
        raise ValueError('eight-renderer allocation identity differs from retained provider')
    cpu_gate(value['cpu_validation'], source_sha256)
    # Admission runs before training, not only after the final checkpoint.
    # A missing/stale independently owned sampler must not defer failure until
    # the expensive final test. This observes CPU files/processes only.
    ResourceJournal(value['resource_sampler'],value)
    return native, fleet


class Transport:
    def __init__(self, site, native, root, sampler, *, run=subprocess.run):
        self.site, self.native, self.sampler, self.run = site, native, sampler, run
        self.guard_root = Path(root)/'resource-guards'; self.unknown = False

    def invoke(self, owner, action, *, input_bytes=None, **values):
        if self.unknown:
            raise ConnectionError('prior transport UNKNOWN; no further peer operation')
        self.sampler.healthy()
        parameters = dict(source_root=self.native['source_root'], source_sha256=self.native['source_sha256'],
                          jobs_root=self.native['jobs_root'], owner=owner, **values)
        argv = self.site.ssh('beta') + [shlex.join(['python3', '-B', '-c', REMOTE,
                                                   json.dumps(parameters, separators=(',', ':')), action])]
        try:
            result = self.run(argv, capture_output=True, timeout=35, check=False,input=input_bytes)
        except subprocess.TimeoutExpired as error:
            self.unknown = True
            raise TimeoutError('exact-owner SSH response unknown; never retry submission') from error
        if result.returncode == 255:
            self.unknown = True
            raise ConnectionError('exact-owner SSH connection unknown')
        if result.returncode:
            if action=='submit':
                self.unknown=True
                raise ConnectionError('native submission acceptance unknown; inspect same command before any continuation')
            raise RuntimeError('exact-owner operation failed; no native replay')
        if len(result.stdout) > 64 << 20:
            raise ValueError('bounded native response exceeded')
        return json.loads(result.stdout)

    def get_ready(self, owner): return self.invoke(owner, 'ready')
    def get_command(self, owner, command_id): return self.invoke(owner, 'get', command_id=command_id)
    def submit(self, owner, body): return self.invoke(owner, 'submit', body=body)

    def stage_artifacts(self, owner, artifacts):
        # Check every renderer (both disjoint actor shards) before the first
        # native POST. A missing tunnel is a pre-acceptance fault, not an Isaac
        # failure. Existing accepted commands are reconciled before this path.
        configs = [a['value'] for a in artifacts if isinstance(a.get('value'), dict)
                   and a['value'].get('schema') == 'roborender.rendered-student-success-config.v3']
        if len(configs) != 1:
            raise ValueError('one exact generated-only renderer config required before native dispatch')
        renderer = configs[0]['renderer']
        answer = self.invoke(owner, 'renderer_preflight', renderer=renderer)
        expected = dict(schema='roborender.renderer-route-preflight.v1',
            renderer_sha256=hashlib.sha256(json.dumps(renderer, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            verified_slots=[e['slot'] for e in renderer['endpoints']], metadata_only=True,
            native_steps=0, render_requests=0)
        if answer != expected:
            raise ValueError('actual renderer route preflight differs; no native submission')
        for artifact in artifacts:
            raw = dispatch.canonical(artifact['value'])
            if hashlib.sha256(raw).hexdigest() != artifact['raw_sha256']:
                raise ValueError('staged artifact changed')
            answer = self.invoke(owner, 'stage', artifact={k:artifact[k] for k in ('path','raw_sha256')},input_bytes=raw)
            if answer != dict(staged=True, raw_sha256=artifact['raw_sha256']):
                raise ValueError('actual staged bytes differ')

    def resource_validator(self, owner, receipt):
        value = self.invoke(owner, 'guard', receipt={'resource_guard': receipt['resource_guard']})
        paired.validate_guard_copy(owner, receipt, value)
        dispatch.immutable(self.guard_root/(receipt['resource_guard']['raw_sha256']+'.json'), value)


class ResourceJournal:
    """Observe an existing durable sampler; no launch, stop, or fake samples."""
    def __init__(self, reference, runtime, *, clock=time.time, sleep=time.sleep):
        self.reference, self.spec = reference, paired.bound(reference)
        self.clock, self.sleep = clock, sleep
        expected = dict(schema='roborender.paired-scaled-resource-journal.v1',
            source_sha256=runtime['source_sha256'], native=runtime['native'], renderer_fleet=runtime['renderer_fleet'],
            interval_seconds=5, compute_release_authorized=False, artificial_gpu_activity=False)
        if (set(self.spec) != set(expected)|{'journal', 'status_file', 'process_receipt', 'sampler_source'}
                or any(self.spec.get(k) != v for k, v in expected.items())):
            raise ValueError('independent exact-owner durable resource sampler required')
        for name in ('journal', 'status_file'):
            path = Path(self.spec[name])
            if not path.is_absolute() or path.resolve(strict=True) != path:
                raise ValueError('sampler files must be existing physical absolute paths')
        self.healthy()

    def healthy(self):
        if paired.bound(self.reference) != self.spec:
            raise ValueError('sampler specification changed')
        source = self.spec['sampler_source']; path = Path(source['path'])
        if (not path.is_absolute() or path.resolve(strict=True) != path or path.stat().st_size > 4 << 20
                or hashlib.sha256(path.read_bytes()).hexdigest() != source['raw_sha256']):
            raise ValueError('independent sampler source changed')
        process = paired.bound(self.spec['process_receipt'])
        identity = process_identity(process['pid'])
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        argv=process.get('argv')
        if (not isinstance(argv,list) or not argv or any(not isinstance(x,str) or not x for x in argv)
                or source['path'] not in argv or process.get('sampler_source')!=source
                or not identity or identity['cmdline_sha256']!=hashlib.sha256(
                    b''.join(os.fsencode(x)+b'\0' for x in argv)).hexdigest()):
            raise RuntimeError('actual sampler argv/source identity differs from its launch receipt')
        if (not identity or process.get('identity') != identity or process.get('boot_id') != boot
                or process.get('sampler_spec_sha256') != dispatch.digest(
                    {k:v for k,v in self.spec.items() if k!='process_receipt'})):
            raise RuntimeError('independent sampler process identity absent or changed')
        status = dispatch.read(self.spec['status_file'])
        latest = status.get('latest_at_unix')
        if (status.get('state') != 'RUNNING' or status.get('pid') != process['pid']
                or type(latest) not in (int, float) or not self.clock()-12 <= latest <= self.clock()+2):
            raise RuntimeError('independent resource sampler is stale or failed')
        return status

    def interval(self, started, ended, owners, config):
        deadline = self.clock()+15
        while self.healthy()['latest_at_unix'] < ended:
            if self.clock() > deadline:
                raise RuntimeError('resource sampler did not bracket the actual native interval')
            self.sleep(.5)
        path = Path(self.spec['journal'])
        if path.is_symlink() or path.stat().st_size > 512 << 20:
            raise ValueError('bounded actual sampler journal required')
        rows = []; before = None
        with path.open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):
                    break  # Concurrent writer's incomplete final record is not evidence.
                if len(line) > 4 << 20:
                    raise ValueError('oversized resource sample')
                row = json.loads(line)
                if row['at_unix'] <= started:
                    before = row
                    continue
                if not rows and before is not None:
                    rows.append(before)
                rows.append(row)
                if row['at_unix'] >= ended:
                    break
        ids = {o['worker_instance_id'] for o in owners}
        for row in rows:
            row['native'] = [o for o in row['native'] if o['owner_id'] in ids]
        renderers = paired.renderer_descriptors(config)
        value = dict(started_unix=started, ended_unix=ended, samples=rows, renderer_owners=renderers,
            summary=paired.summarize(rows, native_owners=paired.descriptors(owners), renderer_owners=renderers,
                                     started_unix=started, ended_unix=ended))
        return value


def evaluate(site, runtime):
    from pipeline.dagger.beta_validation import load_panel
    from pipeline.success_rate.rendered_student_success_contract import configuration
    native, fleet = validate_runtime(runtime, source_sha256=site.config['source_sha256'],
                                    pool=site.pool, retained=site.retained)
    if native['source_root'] != str(site.source):
        raise ValueError('native I/O source differs from frozen phase provider')
    final = site.request['final_checkpoint']; panel = paired.final_panel(load_panel(site.config))
    config = configuration(site.config['validation']['rendered_observation_config'], panel, final['sha256'])
    endpoints = config['renderer']['endpoints']
    if len(endpoints) != 8 or any(any(worker.get(k) != v for k, v in endpoint.items())
            for worker, endpoint in zip(fleet['workers'], endpoints)):
        raise ValueError('selected rendering dependencies differ from retained eight-renderer fleet')
    root = site.round/'paired-final-test'
    sampler = ResourceJournal(runtime['resource_sampler'], runtime)
    transport = Transport(site, native, root, sampler)
    site.verify_retained(); site.renderer_attachment(running=True)
    # Checkpoint copy is content-addressed/idempotent. No native command is
    # issued here; the single coordinator owns every subsequent GPU operation.
    site.copy_checkpoint(final['path'], final['sha256'])
    prefix = 'paired-'+hashlib.sha256(dispatch.canonical(dict(run=site.run, round=site.rid,
        checkpoint=final['sha256'], source=site.config['source_sha256']))).hexdigest()[:24]
    plan = paired.make_run(panel, config, native['workers'], command_prefix=prefix,
                           source_sha256=site.config['source_sha256'], arm='dagger',
                           cpu_validation=runtime['cpu_validation'])
    paired.Run(plan, root, transport, sampler).execute()
    site.renderer_attachment(running=True)
    return dict(comparison_path=str(root/'comparison.json'),
                comparison_sha256=paired.ref(root/'comparison.json')['raw_sha256'], **site.barrier())
