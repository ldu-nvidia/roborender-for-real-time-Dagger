"""Explicit CPU-only Docker observer integration probe; never launches Isaac.

Uses the locally present pinned image with sleep as entrypoint, no GPU requests,
no network, no data mounts. Injection affects only one observer read, not cgroups.
This is an observer test, NOT paired BC capacity certification.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/deployment/rtx_actor'))
from pipeline.deployment.rtx_actor.bc_resource_observer import (
    PinnedCgroupObserver, ObservationUnavailable, recover_observation,
)
from pipeline.deployment.rtx_actor.qualify_native_isaac6 import IMAGE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    name = 'rr-observer-cpu-probe-' + uuid.uuid4().hex[:12]
    cap = 128 * 2**20
    command = ['docker','run','--detach','--rm','--name',name,
        '--label','org.roborender.qualification='+name,
        '--cidfile',str(args.output/'container.cid'),'--memory','128m','--memory-swap','128m',
        '--annotation','org.systemd.property.MemorySwapMax=uint64 0',
        '--cpus','0.1','--network','none','--no-healthcheck',
        '--entrypoint','/bin/sleep',IMAGE,'60']
    (args.output/'intent.json').write_text(json.dumps(dict(command=command, gpu_requested=False)))
    child = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    events, actions, readings, startup_events, startup_actions = [], [], [], [], []
    result = {'verdict':'FAIL','gpu_requested':False,'native_collection_started':False}
    try:
        observer = None
        def first_read():
            nonlocal observer
            if observer is None:
                observer = PinnedCgroupObserver(args.output/'container.cid',name,IMAGE,cap)
            return observer.sample()
        def validate(sample):
            assert sample['memory_swap.current'] == 0
            assert all(sample['events'][key] == 0 for key in ('max','oom','oom_kill'))
            assert sample['memory_current'] < cap
        readings.append(recover_observation(first_read,validate,journal=startup_events.append,
            hold=lambda:startup_actions.append('hold'),release=lambda:startup_actions.append('release'),
            host_check=lambda:None,attempts=20))
        stdout, stderr = child.communicate(timeout=20)
        assert child.returncode == 0, stderr
        cid = stdout.strip()
        assert observer.cid == cid
        count = 0
        def read():
            nonlocal count
            count += 1
            if count == 1:
                raise ObservationUnavailable('intentional one-read fault injection')
            return observer.sample()
        def journal(event):
            events.append(event)
        readings.append(recover_observation(read, validate,journal=journal,
            hold=lambda:actions.append('hold'),release=lambda:actions.append('release'),
            host_check=lambda:None))
        for _ in range(5):
            readings.append(observer.sample())
            validate(readings[-1])
        assert actions == ['hold','release']
        assert [e['status'] for e in events] == ['UNKNOWN','OBSERVED','RECOVERED']
        assert len({s['process_start_ticks'] for s in readings}) == 1
        result.update(verdict='PASS',readings=readings,events=events,actions=actions,
                      exact_container_id=cid,observer_only=True,
                      startup_events=startup_events,startup_actions=startup_actions)
    finally:
        # Stop only this explicitly created CPU probe, never any native/GPU owner.
        stdout, stderr = child.communicate(timeout=20)
        if child.returncode != 0:
            raise RuntimeError('CPU probe Docker startup failed: '+stderr)
        cid = stdout.strip()
        info = json.loads(subprocess.check_output(['docker','inspect',cid],timeout=5))[0]
        if (info['Id'] != cid or info['Name'] != '/'+name
                or info['Config']['Labels'].get('org.roborender.qualification') != name):
            raise RuntimeError('probe cleanup ownership uncertain; do not stop')
        subprocess.check_call(['docker','stop','--time','1',cid],timeout=10)
        result['probe_stopped'] = True
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('events','readings','startup_events')}))


if __name__ == '__main__':
    main()
