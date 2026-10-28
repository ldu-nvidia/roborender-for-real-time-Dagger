"""Reconcile a host-read failure after B2 startup without restarting Isaac.

Reuse completed startup captures, or submit only the original missing zero-step
capture. Publish a distinct CPU recovery receipt; never overwrite the failed
bootstrap, initialize native owners, optimize, or release compute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import time
from urllib.error import HTTPError

from bootstrap_production_resident_pool import publish, validate_spec
from leo_binding import GPU_UUIDS
from production_guard_access import read_owned_guard
from production_resident_pool import SCHEMA, QUALIFICATION, qualification, fixed_memory_admission
from resident_isaac_client import ResidentClient


def require_idle(ready, command_id):
    event = ready.get('current_event', {})
    if (ready.get('phase') != 'IDLE' or ready.get('active_command_id') is not None
            or not (event.get('phase') == 'worker_ready'
                    or event.get('phase') == 'phase_end' and event.get('command_id') == command_id)):
        raise RuntimeError('only idle untouched startup owners may be reconciled')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    raw = args.spec.read_bytes()
    if len(raw) > 1 << 20 or hashlib.sha256(raw).hexdigest() != args.spec_sha256:
        raise RuntimeError('original bootstrap spec changed')
    spec = validate_spec(json.loads(raw))
    root = Path(spec['root'])
    failed = json.loads((root / 'bootstrap-failure.json').read_bytes())
    if (failed.get('error_type') != 'PermissionError' or failed.get('native_restart_allowed') is not False
            or failed.get('source_sha256') != spec['source_sha256']
            or (root / 'pool.json').exists() or (root / 'qualification.json').exists()):
        raise RuntimeError('only the unqualified startup metadata-read failure is recoverable')
    manifest = json.loads(Path(spec['source_manifest']).read_bytes())
    records_raw = json.dumps(manifest['files'], sort_keys=True, separators=(',', ':')).encode()
    if (manifest.get('source_sha256') != spec['source_sha256']
            or hashlib.sha256(records_raw).hexdigest() != spec['source_sha256']):
        raise RuntimeError('native source manifest differs')
    source = Path(spec['source_root']).resolve(strict=True)
    for row in manifest['files']:
        path = (source / row['path']).resolve(strict=True)
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('frozen native source changed')
    inventory = root / 'jobs/full-motion-inventory.json'
    if hashlib.sha256(inventory.read_bytes()).hexdigest() != spec['motion_inventory']['raw_sha256']:
        raise RuntimeError('native full-motion inventory differs')
    if socket.gethostname() != 'leo-rtx-6k':
        raise RuntimeError('dedicated Leo host required')
    output = args.output
    if (not output.is_absolute() or output.parent != Path('/home/ldu/roborender/runtime/production-bootstrap')
            or not output.name.startswith('recovery-') or output.exists()):
        raise RuntimeError('fresh distinct CPU recovery root required')
    workers, clients, observations = [], [], []
    for gpu in (0, 1):
        command_id = f'production-startup-g{gpu}-b2'
        runtime = root / f'gpu{gpu}'
        phases = runtime / 'phases'
        if phases.exists() and {p.name for p in phases.iterdir()} - {command_id}:
            raise RuntimeError('another native phase already exists; reconciliation required')
        client = ResidentClient(18640 + gpu, root / 'resident.token',
                                source_sha256=spec['source_sha256'], gpu_uuid=GPU_UUIDS[gpu])
        ready = client.ready()
        require_idle(ready, command_id)
        worker = dict(physical_gpu=gpu, gpu_uuid=GPU_UUIDS[gpu], port=18640 + gpu,
                      token_file=str(root / 'resident.token'), runtime_root=str(runtime),
                      container_name=spec['run_id'] + f'-g{gpu}', memory_gib=21,
                      evaluation_memory_gib=21, initial_num_envs=2,
                      worker_instance_id=ready['worker_instance_id'], native_pid=ready['native_pid'])
        try:
            result = client._request('/v1/commands/' + command_id)
            if result.get('status') != 'COMPLETE':
                raise RuntimeError('existing startup command is not complete; never replay it')
        except HTTPError as exc:
            if exc.code != 404:
                raise
            result = None
        observations.append(dict(gpu=gpu, ready=ready, result=result))
        workers.append(worker)
        clients.append(client)
    if not args.execute:
        print(json.dumps(dict(execute=False, native_restarts=0, environment_steps=0,
                              missing_capture_gpus=[r['gpu'] for r in observations if r['result'] is None],
                              native_generations=[w['worker_instance_id'] for w in workers])))
        return
    output.mkdir()
    publish(output / 'intent.json', dict(schema='roborender.production-b2-startup-recovery.v1',
        spec_path=str(args.spec), spec_sha256=args.spec_sha256, failed_bootstrap=str(root),
        native_source_sha256=spec['source_sha256'], workers=workers, observations=observations,
        native_restarts=0, permitted_operation='capture_initial_b2', environment_steps=0,
        compute_release_authorized=False, started_ns=time.time_ns()))
    results = []
    for worker, client, observation in zip(workers, clients, observations):
        name = f'production-startup-g{worker["physical_gpu"]}-b2'
        require_idle(client.ready(), name)
        result = observation['result']
        if result is None:
            acceptance = client.submit(name, 'capture_initial_b2', dict(output_name=name))
            publish(output / f'acceptance-g{worker["physical_gpu"]}.json', acceptance)
            result = client.wait(name, wall_budget=600)
        receipt = result['receipt']
        if (receipt.get('source_sha256') != spec['source_sha256']
                or receipt.get('native_pid') != worker['native_pid']
                or receipt.get('native_initializations') != 1 or receipt.get('environment_step_calls') != 0):
            raise RuntimeError('startup receipt/source/native lifetime differs')
        guard = receipt['resource_guard']
        read_owned_guard(guard['path'], worker, guard['raw_sha256'], repair=True)
        results.append(dict(gpu=worker['physical_gpu'], receipt=receipt))
        require_idle(client.ready(), name)
    q = publish(output / 'qualification.json', dict(schema=QUALIFICATION, complete=True,
        source_sha256=spec['source_sha256'], profile='initial_training_b2_capture',
        native_initializations_per_worker=1, optimizer_steps=0, results=results,
        policy_quality_certified=False, legacy_b6_b5_qualification_claimed=False))
    pool = dict(schema=SCHEMA, source_sha256=spec['source_sha256'], jobs_root=str(root / 'jobs'),
                assets_root=str(root / 'assets'), workers=workers, qualification=q)
    qualification(pool)
    memory = fixed_memory_admission(pool)
    ref = publish(output / 'pool.json', pool)
    final = dict(status='B2_NATIVE_READY', pool=ref, native_source_sha256=spec['source_sha256'],
                 native_restarts=0, environment_steps=0, optimizer_steps=0,
                 native_generations=[w['worker_instance_id'] for w in workers],
                 memory_admission=memory, workers_retained=True, compute_release_authorized=False)
    publish(output / 'result.json', final)
    print(json.dumps(final, sort_keys=True))


if __name__ == '__main__':
    main()
