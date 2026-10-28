"""Explicit retry of an unsubmitted test after checkpoint transport failure.

No allocation, native state restore, training replay or automatic retry. Both
original native owners must be idle and prove the test was never accepted.
"""
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from pipeline.deployment.beta_operations import atomic, process_identity
from pipeline.dagger.beta_campaign import immutable_json
from pipeline.dagger.services.contracts import file_sha256


def validate_absent_test(request, pool, status, journals):
    command_id = f"{request['run_id']}-r{request['round_id']}-explicit-test"
    if request.get('phase') != 'evaluate' or status.get('quiescent') is not True:
        raise RuntimeError('unsubmitted evaluation and quiescent native owners required')
    if len(journals) != 2 or len(pool['workers']) != 2 or len(status.get('workers',[])) != 2:
        raise RuntimeError('both native journals required')
    source = pool['source_sha256']
    for worker, ready, journal in zip(pool['workers'], status['workers'], journals):
        for state in (ready, journal):
            if (state.get('phase') != 'IDLE' or state.get('worker_instance_id') != worker['worker_instance_id']
                    or state.get('source_sha256') != source):
                raise RuntimeError('original idle native owner/source required')
        commands = journal.get('commands', {})
        if command_id in commands or any(c.get('command', {}).get('command_id') == command_id for c in commands.values()):
            raise RuntimeError('test already accepted; never retry this recovery')
        latest = max(commands.values(), key=lambda c: c.get('accepted_at_ns', 0), default={})
        expected = f"{request['run_id']}-r{request['round_id']}-collect-g{worker['physical_gpu']}"
        if latest.get('command', {}).get('command_id') != expected or latest.get('status')!='COMPLETE':
            raise RuntimeError('latest native phase must be this completed training round')
    return command_id


def retry_unsubmitted(campaign_root, site, *, launch=False):
    root = Path(campaign_root)
    with ExitStack() as stack:
        for path in (root/'scheduler.lock', root/'evaluation-recovery.lock'):
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((root/'state.json').read_bytes())
        if state.get('phase') != 'EVALUATE':
            raise RuntimeError('campaign is not at the evaluation barrier')
        oid = state['operation_id']; directory = root/'operations'/oid
        request = json.loads((directory/'request.json').read_bytes())
        if (request['operation_id'] != oid or request['run_id'] != state['run_id']
                or request['round_id'] != state['round_id'] or request['phase'] != 'evaluate'):
            raise RuntimeError('operation identity differs')
        saved = json.loads((directory/'process.json').read_bytes())
        if saved.get('identity') and process_identity(saved['pid']) == saved['identity']:
            raise RuntimeError('prior operation is still live')
        if (directory/'receipt.json').exists():
            raise RuntimeError('completed operation must be consumed, not retried')
        trace = (directory/'console.log').read_text()
        if ('checkpoint_transfer.CheckpointTransferError: checkpoint transport retry budget exhausted' not in trace
                or 'self.copy_checkpoint(final["path"], final["sha256"])' not in trace):
            raise RuntimeError('only the proven pre-submit transfer failure is recoverable here')
        spec = json.loads((root/'operations.json').read_bytes())['commands']['evaluate']
        if file_sha256(Path(spec['entrypoint'])) != spec['entrypoint_sha256']:
            raise RuntimeError('evaluation implementation changed')
        intent = json.loads((directory/'intent.json').read_bytes())
        expected = spec['argv'] + ['--request', str(directory/'request.json'), '--receipt', str(directory/'receipt.json')]
        if intent['argv'] != expected:
            raise RuntimeError('operation command changed')
        site.verify_sources()
        status = site.pool_action('status')
        journals = [site.read(Path(w['runtime_root'])/'resident-state.json', 'beta') for w in site.pool['workers']]
        command_id = validate_absent_test(request, site.pool, status, journals)
        final = request['final_checkpoint']
        if file_sha256(Path(final['path'])) != final['sha256']:
            raise RuntimeError('selected checkpoint changed')
        evidence = dict(schema='roborender.unsubmitted-test-transfer-recovery.v1',
            operation_id=oid, command_id=command_id, checked_at=time.time(), native_status=status,
            journals=journals, checkpoint=final, compute_release_authorized=False,
            repeated_native_commands=0, repeated_optimizer_steps=0)
        if not launch:
            return evidence
        archive = directory/'attempts'/uuid.uuid4().hex
        archive.mkdir(parents=True)
        for name in ('request.json', 'intent.json', 'process.json', 'console.log'):
            shutil.copy2(directory/name, archive/name)
        immutable_json(archive/'admission.json', evidence)
        atomic(directory/'intent.json', dict(argv=expected, started_at=time.time(),
            deadline=time.time()+spec['timeout_seconds'], recovery_admission=str(archive/'admission.json')))
        with (directory/'console.log').open('wb') as log:
            child = subprocess.Popen(expected, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        atomic(directory/'process.json', dict(pid=child.pid, identity=process_identity(child.pid)))
        return dict(operation_id=oid, pid=child.pid, recovery_admission=str(archive/'admission.json'))
