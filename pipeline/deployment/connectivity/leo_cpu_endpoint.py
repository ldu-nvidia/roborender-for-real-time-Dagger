"""Explicit Leo binding of the existing restricted CPU enrollment endpoint.

Beta's endpoint, port, managed files and live service remain unchanged.
Only this entrypoint selects Leo's independently enrolled public keys.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpu_endpoint as endpoint


def bind():
    endpoint.CONFIG = endpoint.SSH_ROOT / "roborender_leo_access_config"
    endpoint.KNOWN = endpoint.SSH_ROOT / "roborender_leo_access_known_hosts"
    endpoint.IDENTITY = endpoint.SSH_ROOT / "roborender_leo_access_20261013"
    endpoint.ALIAS = "roborender-leo-direct"
    endpoint.PORT = 22033
    endpoint.BETA_HOST = "leo-rtx-6k"
    endpoint.BETA_PIN = "SHA256:rIKwlBBVBCMwO8TCMyQDzL/II35oOAcOkFwZKOEg5ds"
    endpoint.MARKER = "roborender-leo-access-20261013"
    def authorized_line(key):
        command = "/usr/bin/python3 /workspace/project/roborender/pipeline/deployment/connectivity/leo_cpu_endpoint.py health"
        return (f'restrict,port-forwarding,permitlisten="127.0.0.1:{endpoint.PORT}",'
                f'permitopen="127.0.0.1:1",command="{command}" '
                f'{endpoint.public_key(key)} {endpoint.MARKER}\n')
    endpoint.authorized_line = authorized_line
    return endpoint


if __name__ == "__main__":
    raise SystemExit(bind().main())
