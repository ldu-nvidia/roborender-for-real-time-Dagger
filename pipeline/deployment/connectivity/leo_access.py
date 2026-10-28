"""Leo-owned access using the tested SSH supervisor with explicit host bindings.

No Beta/Alpha/Mac routing, no copied private keys or cloud tokens. The imported
implementation runs in this process only; Beta's installed supervisor is not
modified. Installation must come from a source-hash-verified release.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import sys
import beta_access as access


def bind():
    access.BASE = Path("/home/ldu/.local/share/roborender-leo-access")
    access.TUNNEL_KEY = Path("/home/ldu/.ssh/roborender_leo_access_20261013")
    access.UNIT_NAME = "roborender-leo-access.service"
    access.BETA_HOST = "leo-rtx-6k"
    access.BETA_PIN = "SHA256:rIKwlBBVBCMwO8TCMyQDzL/II35oOAcOkFwZKOEg5ds"
    access.PORT = 22033
    access.MARKER = "# Managed by RoboRender Leo access v1"
    access.CPU_ENDPOINT = "/workspace/project/roborender/pipeline/deployment/connectivity/leo_cpu_endpoint.py"
    access.CONTROL = access.BASE / "master.sock"
    original = access.service_unit
    def service_unit(python, script, config):
        # The wrapper reapplies the exact binding on every service restart.
        return original(python, Path(script).with_name("leo_access.py"), config).replace(
            "RoboRender Beta-owned Cosmos access", "RoboRender Leo-owned Cosmos access")
    access.service_unit = service_unit
    return access


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["install", "run", "check", "upgrade-supervisor"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--expected-unit-sha256")
    a = p.parse_args()
    implementation = bind()
    config = json.loads(a.config.read_bytes())
    try:
        if a.action == "install":
            implementation.install(config)
        elif a.action == "run":
            implementation.supervise(config)
        elif a.action == "upgrade-supervisor":
            upgrade_supervisor(implementation, a.config, a.expected_unit_sha256)
        else:
            implementation.checked_config(config, installed=True)
            print(json.dumps(implementation.health(config), sort_keys=True))
    except Exception as error:
        print("RoboRender Leo access failed: " + type(error).__name__, file=sys.stderr)
        return 1
    return 0


def checked_old_unit(implementation, old, expected_sha256, config_path):
    """Only replace an explicitly inspected, uncustomized Leo-owned unit."""
    if (not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256)
            or hashlib.sha256(old.encode()).hexdigest() != expected_sha256):
        raise RuntimeError("the inspected Leo unit changed")
    match = re.search(r'ExecStart="([^"]+)" "([^"]+)" run --config "([^"]+)"', old)
    if not match:
        raise RuntimeError("unrecognized Leo service unit")
    python, script, config = map(Path, match.groups())
    runtime = Path('/home/ldu/roborender/runtime')
    if (python != implementation.SDK_PYTHON or config != config_path
            or script.name != 'leo_access.py' or not script.is_relative_to(runtime)
            or script.relative_to(runtime).parts[0] not in ('connectivity-releases', 'source-releases')):
        raise RuntimeError("service is outside the owned Leo release scope")
    expected = implementation.service_unit(python, script, config)
    if old not in (expected, expected.replace('WatchdogSec=180\nNotifyAccess=main\n', '')):
        raise RuntimeError("Leo service has unreviewed customizations")


def upgrade_supervisor(implementation, config_path, expected_sha256):
    """Idle-boundary deployment; never changes keys, routes or GPU workloads."""
    implementation.require_beta()  # bind() replaces the host with Leo.
    if config_path != implementation.BASE / 'config.json':
        raise RuntimeError('installed Leo configuration required')
    implementation.checked_config(json.loads(config_path.read_bytes()), installed=True)
    unit = Path('/home/ldu/.config/systemd/user') / implementation.UNIT_NAME
    if unit.is_symlink():
        raise RuntimeError('refusing a symlink service unit')
    old = unit.read_text()
    checked_old_unit(implementation, old, expected_sha256, config_path)
    source = Path(__file__).resolve()
    release_root = Path('/home/ldu/roborender/runtime/source-releases')
    if not source.is_relative_to(release_root) or '/frozen/' not in str(source):
        raise RuntimeError('source-verified frozen release required')
    backup = implementation.BASE / 'rollback' / expected_sha256
    backup.mkdir(parents=True, exist_ok=True, mode=0o700)
    implementation.publish_exact(backup / 'service.unit', old.encode())
    temporary = unit.with_name(unit.name + '.supervisor-upgrade')
    implementation.publish_exact(temporary, implementation.service_unit(
        implementation.SDK_PYTHON, source, config_path).encode())
    os.replace(temporary, unit)
    implementation.run(['systemctl', '--user', 'daemon-reload'])
    implementation.run(['systemctl', '--user', 'restart', '--no-block', implementation.UNIT_NAME])
    implementation.phase('leo_supervisor_upgrade_restart_requested')


if __name__ == "__main__":
    raise SystemExit(main())
