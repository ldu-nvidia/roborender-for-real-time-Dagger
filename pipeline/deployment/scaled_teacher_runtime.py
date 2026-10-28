"""One full295 privileged-teacher baseline, not a visual-student evaluator.

The host-specific adapter supplies an already running independent sampler and
the exact-owner Transport. This module never provisions/stops owners, renders,
releases allocations, or resumes an interrupted outer execution automatically.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from pipeline.deployment import paired_scaled_evaluation as paired
from pipeline.deployment import paired_scaled_runtime as io
from pipeline.deployment import scaled_success_dispatch as common
from pipeline.deployment import scaled_teacher_dispatch as dispatch
from pipeline.deployment.h200.retained_renderer_attachment import validate_fleet, verify_release

SCHEMA = 'roborender.scaled-teacher-runtime.v1'
AUTH_SCHEMA = 'roborender.scaled-teacher-user-authorization.v1'
CAPABILITY_SCHEMA = 'roborender.scaled-teacher-live-capability.v1'
FLAGS = ('compute_release_authorized', 'native_replay_allowed', 'automatic_restart')
REFS = ('dispatch_plan', 'native', 'renderer_fleet', 'source_manifest',
        'cpu_validation', 'sampler_binding', 'user_authorization')
FAULTS = {'TRANSPORT_UNKNOWN_RECONCILE_SAME_OWNER', 'UNKNOWN_REQUIRES_RECONCILIATION',
          'NATIVE_TERMINAL_FAULT', 'NOT_STARTED_STOPPED'}


def required_environment(owner):
    return dict(RESIDENT_SOURCE_SHA256=owner['source_sha256'],
        RESIDENT_SCALED_TEACHER=dispatch.teacher.SCHEMA,
        RESIDENT_SCALED_SUCCESS='roborender.scaled-rendered-success-protocol.v1',
        RESIDENT_PIPELINE_MODE='production', CUDA_VISIBLE_DEVICES=owner['gpu_uuid'],
        RESIDENT_PHYSICAL_GPU=str(owner['physical_gpu']), RESIDENT_PORT=str(owner['port']))


CAPABILITY_REMOTE = io.REJECT_OPTIMIZED_PYTHON + r'''
import json,pathlib,subprocess,sys
p=json.loads(sys.argv[1]);source=pathlib.Path(p['source_root'])
sys.path.insert(0,str(source));sys.path.insert(0,str(source/'pipeline/deployment/rtx_actor'))
from production_resident_pool import attest_live_owner
inspected=[]
def capture(argv,**kwargs):
 result=subprocess.run(argv,**kwargs)
 if len(result.stdout)>4<<20:raise ValueError('bounded metadata required')
 inspected.extend(json.loads(result.stdout));return result
actual=attest_live_owner(p['owner'],p['owner']['source_sha256'],mode='production',run=capture)
if len(inspected)!=1:raise ValueError('one actual container required')
environment={}
for item in inspected[0].get('Config',{}).get('Env',[]):
 key,separator,value=item.partition('=')
 if key in p['required_environment']:
  if not separator or key in environment:raise ValueError('ambiguous capability')
  environment[key]=value
if environment!=p['required_environment']:raise ValueError('teacher capability absent or changed')
print(json.dumps(dict(schema=p['schema'],owner=p['owner'],environment=environment,**actual)))
'''


def read_capability(site, native, owner, *, run=subprocess.run):
    """Read actual Docker metadata only; no model or native command call."""
    parameters = dict(source_root=native['source_root'], owner=owner,
        required_environment=required_environment(owner), schema=CAPABILITY_SCHEMA)
    argv = site.ssh('beta') + [shlex.join(['python3', '-B', '-c', CAPABILITY_REMOTE,
                                         json.dumps(parameters, separators=(',', ':'))])]
    result = run(argv, capture_output=True, timeout=35, check=False)
    if result.returncode or len(result.stdout) > 4 << 20:
        raise RuntimeError('actual teacher owner capability could not be verified')
    return validate_capability(json.loads(result.stdout), owner)


def validate_capability(value, owner):
    expected = dict(schema=CAPABILITY_SCHEMA, owner=owner, environment=required_environment(owner),
                    container_id=owner['container_id'], host_pid=owner['host_pid'])
    if value != expected:
        raise ValueError('actual exact-owner teacher launch capability required')
    return deepcopy(value)


def source_jail(root):
    root = Path(root)
    if sys.flags.optimize or not Path(__file__).resolve().is_relative_to(root):
        raise ValueError('driver must execute from its exact frozen source without optimization')
    for name, module in tuple(sys.modules.items()):
        if name == 'pipeline' or name.startswith('pipeline.'):
            paths = list(getattr(module, '__path__', []))
            if getattr(module, '__file__', None):
                paths.append(module.__file__)
            if any(not Path(p).resolve().is_relative_to(root) for p in paths):
                raise ValueError('pipeline import escaped the frozen source: '+name)


def validate_runtime(value):
    fields = {'schema', 'source_sha256', 'source_root', 'output', 'wandb', *REFS, *FLAGS}
    if (set(value) != fields or value.get('schema') != SCHEMA
            or not common._sha(value.get('source_sha256'))
            or any(value.get(k) is not False for k in FLAGS)):
        raise ValueError('exact teacher-only runtime and retention policy required')
    evidence = {key: paired.bound(value[key]) for key in REFS}
    root, output = Path(value['source_root']), Path(value['output'])
    if (not root.is_absolute() or root.resolve(strict=True) != root or not output.is_absolute()
            or output.resolve() != output or output == root or output.is_relative_to(root)):
        raise ValueError('physical frozen source and separate physical runtime output required')
    verify_release(root, value['source_manifest']['path'], value['source_sha256'])
    source_jail(root)
    io.cpu_gate(value['cpu_validation'], value['source_sha256'])
    plan = dispatch.validate_plan(evidence['dispatch_plan'])
    native, fleet = evidence['native'], validate_fleet(evidence['renderer_fleet'])
    if (native.get('schema') != 'roborender.scaled-evaluation-native-bindings.v1'
            or native.get('source_sha256') != value['source_sha256']
            or native.get('local_source_root') != str(root) or native.get('workers') != plan['owners']
            or any(o['source_sha256'] != value['source_sha256'] for o in plan['owners'])):
        raise ValueError('teacher plan must bind both fresh exact-source native lifetimes')
    sampling = evidence['sampler_binding']
    if (sampling.get('source_sha256') != value['source_sha256']
            or sampling.get('native') != value['native']
            or sampling.get('renderer_fleet') != value['renderer_fleet']
            or sampling.get('compute_release_authorized') is not False
            or sampling.get('artificial_gpu_activity') is not False):
        raise ValueError('sampler must bind these actual fresh-source owners and retained fleet')
    for key in ('source_root', 'jobs_root'):
        if not Path(native.get(key, '')).is_absolute():
            raise ValueError('exact absolute native source/jobs roots required')
    for owner in plan['owners']:
        if (type(owner.get('host_pid')) is not int or owner['host_pid'] <= 0
                or not owner.get('container_id') or not owner.get('container_name')
                or type(owner.get('port')) is not int
                or any(not Path(owner.get(k, '')).is_absolute() for k in ('token_file', 'runtime_root'))):
            raise ValueError('actual native container/process/transport identity required')
    auth = evidence['user_authorization']
    if (auth.get('schema') != AUTH_SCHEMA or auth.get('execute_scaled_teacher_full295') is not True
            or not isinstance(auth.get('user_instruction_verbatim'), str)
            or not auth['user_instruction_verbatim'].strip()
            or any(auth.get(k) is not False for k in FLAGS)):
        raise ValueError('actual explicit teacher full295 user authority required')
    wandb = value['wandb']
    if (set(wandb) != {'entity', 'project', 'id'} or wandb['entity'] != 'nvidia'
            or wandb['project'] != 'dagger'
            or re.fullmatch('[a-z0-9][a-z0-9-]{0,95}', wandb['id']) is None):
        raise ValueError('separate online teacher W&B identity required')
    return plan, native, fleet


class SamplerAdapter:
    """Use the real curriculum Sampler without relabeling its process receipts.

    interval_reader(started, ended, owners, config) reads its actual journal and
    returns the paired resource interval format. No sampler lifecycle is owned.
    """
    def __init__(self, sampler, binding, interval_reader):
        paired.bound(binding)
        self.sampler, self.binding, self.interval_reader = sampler, deepcopy(binding), interval_reader

    def healthy(self):
        paired.bound(self.binding)
        return self.sampler.healthy()

    def interval(self, started, ended, owners, config):
        self.healthy(); self.sampler.bracket(ended)
        return self.interval_reader(started, ended, owners, config)


def validate_interval(value, started, ended, owners, fleet):
    renderers = paired.renderer_descriptors(dict(renderer=dict(endpoints=fleet['workers'])))
    expected = paired.summarize(value['samples'], native_owners=paired.descriptors(owners),
        renderer_owners=renderers, started_unix=started, ended_unix=ended)
    if (value.get('started_unix') != started or value.get('ended_unix') != ended
            or value.get('renderer_owners') != renderers or value.get('summary') != expected):
        raise ValueError('whole teacher interval must reproduce actual owner-bound resource evidence')
    return value


def metrics(progress, *, complete=False):
    result = {'teacher_baseline/completed': progress['completed'],
        'teacher_baseline/expected': 295, 'teacher_baseline/successes': progress['successes'],
        'teacher_baseline/committed_cohorts': progress['committed_cohorts'],
        'teacher_baseline/full295_complete': complete, 'teacher_baseline/optimizer_steps': 0,
        'teacher_baseline/renderer_calls': 0, 'teacher_baseline/student_forward_calls': 0}
    if progress['completed']:
        result['teacher_baseline/success_rate'] = progress['success_rate']
    return result


class OnlineReporter:
    """Use approved credentials and actual online write/readback; no GPU APIs."""
    def start(self, plan, root):
        from pipeline.deployment.wandb_preflight import admit_logging
        import wandb
        self.spec = plan['wandb']
        self.api, auth = admit_logging('online', self.spec['entity'])
        self.run = wandb.init(**self.spec, name=self.spec['id'], mode='online', resume='never',
            job_type='scaled-privileged-teacher-baseline', dir=str(root), config=dict(
                source_sha256=plan['source_sha256'], dispatch_plan=plan['dispatch_plan'],
                observation_mode='privileged_state', expected_episodes=295, optimizer_steps=0,
                renderer_calls=0, student_forward_calls=0, visual_student_certified=False))
        if self.run.id != self.spec['id'] or self.run.settings.mode != 'online':
            raise ValueError('actual separate online teacher logging identity required')
        return dict(run_id=self.run.id, url=self.run.url, authenticated=auth['authenticated'])

    def progress(self, progress):
        self.run.log(metrics(progress))

    def complete(self, values):
        from pipeline.deployment.wandb_preflight import verify_remote_summary
        self.run.log(values); url = self.run.url; self.run.finish()
        verify_remote_summary(self.api, self.spec['entity'], self.spec['id'], values)
        return dict(online_readback_verified=True, run_id=self.spec['id'], url=url, metrics=values)


def publish_status(root, value):
    path = Path(root)/'status.json'
    if path.is_symlink():
        raise ValueError('runtime status cannot be a symlink')
    temporary = Path(root)/f'.status-{os.getpid()}-{time.time_ns()}.json'
    common.immutable(temporary, value)
    os.replace(temporary, path)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def execute(value, *, transport, sampler, reporter, capability_reader, clock=time.time, sleep=time.sleep):
    """Run once; an existing interrupted intent is observation-only, never replay.

    The adapter starts/attests its independent observer before calling this.
    Any fault retains accepted native work, sampler and all allocations.
    """
    plan, native, fleet = validate_runtime(value)
    if getattr(sampler, 'binding', None) != value['sampler_binding']:
        raise ValueError('actual sampler adapter must bind the admitted fresh sampler plan')
    root = Path(value['output']); root.mkdir(parents=True, exist_ok=True)
    if (root/'runtime.lock').is_symlink():
        raise ValueError('runtime lock cannot be a symlink')
    with (root/'runtime.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        common.immutable(root/'runtime-plan.json', value)
        if (root/'execution-intent.json').exists():
            # Never reconstruct/relaunch an interrupted outer workflow. The
            # dedicated read-only adapter may reconcile these same command IDs.
            return dict(schema=SCHEMA+'.status', status='UNKNOWN_REQUIRES_READ_ONLY_RECONCILIATION',
                        complete=False, partial=True, **{k:False for k in FLAGS})
        common.immutable(root/'execution-intent.json', dict(runtime_sha256=common.digest(value),
            controller_pid=os.getpid(), started_unix=clock(), **{k:False for k in FLAGS}))
        try:
            sampler.healthy()
            capabilities = [validate_capability(capability_reader(o), o) for o in plan['owners']]
            common.immutable(root/'teacher-capabilities.json', capabilities)
            common.immutable(root/'initial-idle.json', paired.idle_barrier(transport, plan['owners']))
            # Reject adoption of pre-existing native outcomes, even for the
            # same plausible command name. All19 must originate in this intent.
            for entry in plan['commands']:
                if transport.get_command(plan['owners'][entry['owner_index']], entry['body']['command_id']) is not None:
                    raise ValueError('fresh teacher command namespace already exists')
            logging = reporter.start(value, root)
            if logging.get('run_id') != value['wandb']['id'] or logging.get('authenticated') is not True:
                raise ValueError('actual authenticated teacher logging required before commands')
            common.immutable(root/'wandb.json', logging)
            started = clock()
            common.immutable(root/'interval-start.json', dict(started_unix=started))
            coordinator = dispatch.Coordinator(plan, root/'dispatch', transport,
                                               resource_validator=transport.resource_validator)
            sequence = 0
            while True:
                sampler.healthy()
                progress = coordinator.tick()
                status = dict(schema=SCHEMA+'.status', status='NATIVE_COMPLETE' if progress['complete'] else 'RUNNING',
                    progress=progress, at_unix=clock(), **{k:False for k in FLAGS})
                common.immutable(root/'progress'/f'{sequence:06d}.json', status); sequence += 1
                publish_status(root, status)
                reporter.progress(progress)
                if any(r['status'] in FAULTS for r in progress['workers']):
                    raise RuntimeError('teacher dispatch requires reconciliation; no new commands admitted')
                if progress['complete']:
                    if progress['completed'] != 295 or progress['committed_cohorts'] != 19:
                        raise ValueError('only all19 actual cohorts can complete teacher295')
                    break
                sleep(3)
            common.immutable(root/'final-idle.json', paired.idle_barrier(transport, plan['owners']))
            ended = clock()
            resources = sampler.interval(started, ended, plan['owners'],
                                         dict(renderer=dict(endpoints=fleet['workers'])))
            validate_interval(resources, started, ended, plan['owners'], fleet)
            common.immutable(root/'whole-interval-resources.json', resources, limit=common.MAX_FINAL_RESULT_LIMIT)
            values = metrics(progress, complete=True)
            values.update({'teacher_baseline/source_sha256':value['source_sha256'],
                'teacher_baseline/dispatch_plan_sha256':value['dispatch_plan']['raw_sha256'],
                'teacher_baseline/panel_sha256':plan['commands'][0]['artifacts'][0]['value']['panel_sha256'],
                'teacher_baseline/teacher_control_sha256':common.digest(
                    plan['commands'][0]['artifacts'][0]['value']['teacher_control']),
                'teacher_baseline/shadow_not_behavior_checkpoint_sha256':plan['shadow_not_behavior_checkpoint_sha256']})
            online = reporter.complete(values)
            if (online.get('online_readback_verified') is not True or online.get('run_id') != value['wandb']['id']
                    or online.get('metrics') != values or not isinstance(online.get('url'), str)):
                raise ValueError('actual exact teacher online readback required')
            common.immutable(root/'wandb-readback.json', online)
            result = dict(schema=SCHEMA+'.result', status='COMPLETE', complete=True, partial=False,
                runtime_plan=paired.ref(root/'runtime-plan.json'),
                dispatch_result=paired.ref(root/'dispatch'/'result.json', aggregate=True),
                whole_interval_resources=paired.ref(root/'whole-interval-resources.json', aggregate=True),
                initial_idle=paired.ref(root/'initial-idle.json'), final_idle=paired.ref(root/'final-idle.json'),
                teacher_capabilities=paired.ref(root/'teacher-capabilities.json'),
                wandb_readback=paired.ref(root/'wandb-readback.json'), online_readback_verified=True,
                completed=295, successes=progress['successes'], success_rate=progress['success_rate'],
                committed_cohorts=19, observation_mode='privileged_state', optimizer_steps=0,
                renderer_calls=0, student_forward_calls=0, visual_student_certified=False,
                warm_native_owners_retained=True, sampler_retained=True, **{k:False for k in FLAGS})
            common.immutable(root/'result.json', result)
            publish_status(root, result)
            return result
        except BaseException as error:
            # SDK/SSH exception text can contain secrets. Record only the class;
            # preserve actual prior progress, intents and accepted work.
            fault = dict(schema=SCHEMA+'.failure', status='FAILED_OR_UNCERTAIN',
                error_type=type(error).__name__, at_unix=clock(), accepted_work_retained=True,
                sampler_retained=True, complete=False, partial=True, **{k:False for k in FLAGS})
            common.immutable(root/'failure.json', fault)
            publish_status(root, fault)
            raise
