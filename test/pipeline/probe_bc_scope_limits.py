"""CPU-only comparison of Docker kernel limits versus persisted scope policy.

No GPU, Isaac, shared-data mounts, host-policy writes or daemon reload.
Stops only the two small sleep containers created by this probe.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/deployment/rtx_actor'))
from pipeline.deployment.rtx_actor.bc_resource_observer import PinnedCgroupObserver
from pipeline.deployment.rtx_actor.qualify_native_isaac6 import IMAGE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    reports = []
    for pinned in (False, True):
        name = 'rr-scope-cpu-' + uuid.uuid4().hex[:12]
        cidpath = args.output / (('pinned' if pinned else 'legacy') + '.cid')
        command = ['docker', 'run', '--detach', '--rm', '--name', name,
            '--label', 'org.roborender.qualification=' + name, '--cidfile', str(cidpath),
            '--memory', '128m', '--memory-swap', '128m', '--cpus', '0.1',
            '--network', 'none', '--no-healthcheck', '--entrypoint', '/bin/sleep']
        if pinned:
            command += ['--annotation', 'org.systemd.property.MemorySwapMax=uint64 0']
        command += [IMAGE, '120']
        cid = subprocess.check_output(command, text=True, timeout=20).strip()
        try:
            info = json.loads(subprocess.check_output(['docker', 'inspect', cid], timeout=5))[0]
            if (info['Id'] != cid or info['Name'] != '/' + name or info['Image'] != IMAGE
                    or info['Config']['Labels']['org.roborender.qualification'] != name):
                raise RuntimeError('probe ownership changed')
            pid = info['State']['Pid']
            group = Path(f'/proc/{pid}/cgroup').read_text().strip().removeprefix('0::')
            root = Path('/sys/fs/cgroup') / group.lstrip('/')
            properties = subprocess.check_output(['systemctl', 'show', Path(group).name,
                '--property=ControlGroup,MemoryMax,MemorySwapMax'], text=True, timeout=5)
            row = dict(pinned=pinned, cid=cid, properties=properties,
                memory_max=(root / 'memory.max').read_text().strip(),
                swap_max=(root / 'memory.swap.max').read_text().strip(), gpu_requested=False)
            if pinned:
                observer = PinnedCgroupObserver(cidpath, name, IMAGE, 128 << 20)
                row['validated_sample'] = observer.sample()
            reports.append(row)
        finally:
            info = json.loads(subprocess.check_output(['docker', 'inspect', cid], timeout=5))[0]
            if info['Id'] != cid or info['Name'] != '/' + name:
                raise RuntimeError('unknown probe identity; no stop')
            subprocess.run(['docker', 'stop', '--timeout', '1', cid], check=True, timeout=10)
    result = dict(verdict='PASS', comparisons=reports, host_policy_modified=False,
                  compute_released=False, production_admitted=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
