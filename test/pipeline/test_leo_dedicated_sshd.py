from pathlib import Path
import pytest
from pipeline.deployment.connectivity.leo_dedicated_sshd import command,authorized_line


def test_dedicated_endpoint_does_not_change_platform_sshd():
    argv=command(Path('/workspace/project/roborender/runtime/connectivity/test'))
    options=[argv[i+1] for i,a in enumerate(argv[:-1]) if a=='-o']
    assert argv[:7]==['/usr/sbin/sshd','-D','-e','-f','/dev/null','-p','2233']
    for required in ('ListenAddress=127.0.0.1','ClientAliveInterval=20','ClientAliveCountMax=3',
                     'AllowTcpForwarding=remote','PermitListen=127.0.0.1:22033',
                     'PasswordAuthentication=no','AuthenticationMethods=publickey','AllowUsers=root',
                     'AuthorizedKeysFile=/root/.ssh/roborender_leo_dedicated_authorized_keys'):
        assert required in options
    assert not any('Kill' in a or 'systemctl' in a for a in argv)


def test_only_existing_restricted_leo_key_is_copied(tmp_path):
    key=tmp_path/'keys';key.write_text('ssh-ed25519 abc unrelated\n')
    with pytest.raises(RuntimeError):authorized_line(key)
    row='restrict,port-forwarding,permitlisten="127.0.0.1:22033",command="/usr/bin/python3 /workspace/project/roborender/pipeline/deployment/connectivity/leo_cpu_endpoint.py health" ssh-ed25519 abc roborender-leo-access-20261013\n'
    key.write_text('ssh-ed25519 def unrelated\n'+row)
    assert authorized_line(key)==row.encode()
    key.write_text(row+row)
    with pytest.raises(RuntimeError):authorized_line(key)
