"""Leo-only CPU SSH endpoint with server-side stale-transport expiry.

Never changes/restarts the pod's platform sshd, GPUs, or allocations. Only the
already enrolled restricted Leo public key is accepted on loopback port2233.
The Leo cloud proxy ensures this daemon exists before each new connection.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/workspace/project/roborender/runtime/connectivity/leo-dedicated-sshd-v2')
AUTHORIZED = Path('/root/.ssh/roborender_leo_dedicated_authorized_keys')
PORT = 2233


def start_ticks(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
    except FileNotFoundError:return None


def command(root):
    return ['/usr/sbin/sshd','-D','-e','-f','/dev/null','-p',str(PORT),
        '-o','ListenAddress=127.0.0.1','-o','HostKey=/etc/ssh/ssh_host_ed25519_key',
        '-o','AuthorizedKeysFile='+str(AUTHORIZED),'-o','PidFile='+str(root/'sshd.pid'),
        '-o','AllowUsers=root','-o','PermitRootLogin=prohibit-password',
        '-o','PasswordAuthentication=no','-o','KbdInteractiveAuthentication=no',
        '-o','AuthenticationMethods=publickey','-o','UsePAM=no',
        '-o','AllowTcpForwarding=remote','-o','AllowAgentForwarding=no','-o','X11Forwarding=no',
        '-o','PermitTTY=no','-o','GatewayPorts=no','-o','PermitListen=127.0.0.1:22033',
        '-o','ClientAliveInterval=20','-o','ClientAliveCountMax=3','-o','TCPKeepAlive=yes',
        '-o','LoginGraceTime=30','-o','MaxStartups=4:30:8','-o','MaxSessions=4']


def authorized_line(path):
    rows=[r for r in path.read_text().splitlines() if r.endswith(' roborender-leo-access-20261013')]
    if (len(rows)!=1 or not rows[0].startswith('restrict,port-forwarding,permitlisten="127.0.0.1:22033",')
            or 'command="/usr/bin/python3 /workspace/project/roborender/pipeline/deployment/connectivity/leo_cpu_endpoint.py health"' not in rows[0]):
        raise RuntimeError('exact existing restricted Leo enrollment required')
    return (rows[0]+'\n').encode()


def ensure():
    if os.geteuid()!=0 or not socket.gethostname().startswith('leo-cosmos-h200-access-'):
        raise RuntimeError('existing CPU access pod root required')
    if ROOT.is_symlink():raise RuntimeError('managed root is a symlink')
    ROOT.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (ROOT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # OpenSSH StrictModes rejects the shared /workspace mount's ancestry.
        # Keep this PUBLIC authorization file in the existing private SSH dir.
        key=authorized_line(Path('/root/.ssh/authorized_keys'));path=AUTHORIZED
        if path.is_symlink() or (path.exists() and path.read_bytes()!=key):
            raise RuntimeError('dedicated key changed')
        if not path.exists():
            fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb') as stream:stream.write(key)
        argv=command(ROOT);identity=hashlib.sha256(json.dumps(argv).encode()+key).hexdigest()
        owner=ROOT/'owner.json'
        listeners=subprocess.check_output(['ss','-ltnp',f'( sport = :{PORT} )'],text=True)
        if owner.exists():
            saved=json.loads(owner.read_bytes())
            if saved.get('identity')!=identity:raise RuntimeError('dedicated daemon configuration changed')
            if start_ticks(saved['pid'])==saved['start_ticks']:
                if (Path(f'/proc/{saved["pid"]}/exe').resolve()!=Path('/usr/sbin/sshd')
                        or f'pid={saved["pid"]},' not in listeners):
                    raise RuntimeError('owned daemon is not listening correctly')
                return dict(ok=True,pid=saved['pid'],port=PORT,server_liveness_seconds=60)
        if len(listeners.strip().splitlines())!=1:raise RuntimeError('dedicated port has an unknown owner')
        subprocess.run(argv[0:1]+['-t']+argv[3:],check=True,timeout=10)
        with (ROOT/'sshd.log').open('ab') as log:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        saved=dict(pid=child.pid,start_ticks=start_ticks(child.pid),identity=identity,argv=argv)
        temporary=ROOT/'owner.json.tmp';temporary.write_text(json.dumps(saved));os.chmod(temporary,0o600);temporary.replace(owner)
        for _ in range(40):
            if child.poll() is not None:raise RuntimeError('dedicated SSH daemon exited')
            listeners=subprocess.check_output(['ss','-ltnp',f'( sport = :{PORT} )'],text=True)
            if f'pid={child.pid},' in listeners:return dict(ok=True,pid=child.pid,port=PORT,server_liveness_seconds=60)
            time.sleep(.1)
        raise TimeoutError('dedicated SSH listener not ready')


if __name__=='__main__':print(json.dumps(ensure(),sort_keys=True))
