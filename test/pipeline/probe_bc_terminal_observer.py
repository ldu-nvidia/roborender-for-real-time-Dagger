"""CPU-only real Docker exit/cgroup race probe; NOT Isaac qualification.

A deliberate1.2second CLI-wrapper delay reproduces the native-exit/client-lag
boundary. No GPU requested, no data mount, no daemon policy changes, no retries.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/deployment/rtx_actor'))
from pipeline.deployment.rtx_actor.bc_resource_observer import (
    PinnedCgroupObserver, ObservationUnavailable, observe_command,
)
from pipeline.deployment.rtx_actor.qualify_native_isaac6 import IMAGE
from pipeline.deployment.rtx_actor.isaac6_bc_production import publish_json
from pipeline.bc_collection.run_continuous import cleanup_exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    name = 'rr-terminal-cpu-' + uuid.uuid4().hex[:12]
    cidpath = args.output / 'container.cid'
    command = ['docker', 'run', '--rm', '--name', name,
        '--label', 'org.roborender.qualification=' + name, '--cidfile', str(cidpath),
        '--memory', '128m', '--memory-swap', '128m', '--cpus', '0.1',
        '--annotation', 'org.systemd.property.MemorySwapMax=uint64 0',
        '--network', 'none', '--no-healthcheck', '--entrypoint', '/bin/sleep', IMAGE, '3']
    # Delays only the CPU test's process wrapper, preserving Docker's exit code.
    wrapper = [sys.executable, '-c',
        'import subprocess,sys,time; c=subprocess.call(sys.argv[1:]); time.sleep(1.2); sys.exit(c)', *command]
    publish_json(args.output / 'intent.json', dict(command=wrapper, gpu_requested=False, replay_allowed=False))
    observer = None
    process = subprocess.Popen(wrapper, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    events = []
    started = time.monotonic()
    try:
        while observer is None:
            try:
                observer = PinnedCgroupObserver(cidpath, name, IMAGE, 128 << 20)
            except ObservationUnavailable:
                if process.poll() is not None or time.monotonic() - started > 10:
                    raise
                time.sleep(.05)
        observed = observer.sample()
        # Explicit CPU-only terminal fixture, never a BC manifest/qualification.
        terminal = dict(cpu_test_double=True, native_runtime_tested=False, cid=observer.cid)
        def read_live():
            try:
                return observer.sample()
            except ObservationUnavailable as exc:
                events.append(dict(time=time.monotonic(), unavailable=str(exc), client_pending=process.poll() is None))
                raise
        while True:
            sample = observe_command(process, read_live, lambda: dict(terminal), lambda: True)
            if sample.get('lifecycle') == 'completed_attested':
                break
            if time.monotonic() - started > 15:
                raise TimeoutError('CPU probe timed out')
            time.sleep(.05)
        if not events or not any(event['client_pending'] for event in events):
            raise RuntimeError('exit/client-lag boundary was not exercised')
        if sample.get('exit_code') != 0 or process.returncode != 0:
            raise RuntimeError('CPU probe exit not confirmed')
        result = dict(verdict='PASS', scope='CPU_only_exit_client_lag_reconciliation',
            native_runtime_tested=False, production_admitted=False, gpu_requested=False,
            container_id=observer.cid, initial_sample=observed, events=events,
            completion=sample, host_policy_modified=False, compute_released=False)
        publish_json(args.output / 'result.json', result)
        print(json.dumps(result))
    finally:
        # Only this probe's exact ID/image/label; never any shared allocation.
        if cidpath.exists() and cidpath.read_text().strip():
            cleanup_exact(name, cidpath, IMAGE)
        process.wait(timeout=10)


if __name__ == '__main__':
    main()
