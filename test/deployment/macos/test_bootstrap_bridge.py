"""CPU-only shell lifecycle tests; SSH, sleep and uname are local fixtures."""
from test.deployment.source_paths import MACOS_ROOT
import json
import os
from pathlib import Path
import subprocess


SCRIPT = (MACOS_ROOT / "bootstrap-mac-bridge.sh")
MOCK = r'''#!/usr/bin/env python3
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
if name == 'uname':
    print('Darwin')
    sys.exit(0)
if name == 'sleep':
    sys.exit(0)
a = sys.argv[1:]
root = pathlib.Path(os.environ['BRIDGE_TEST_ROOT'])
with (root / 'calls.jsonl').open('a') as f:
    f.write(json.dumps(a) + '\n')
host = next(x for x in a if x in ('cosmos-h200', 'rtx-6k-alpha', 'rtx-6k-beta'))
if '-G' in a:
    print('hostname ' + ('10.110.20.44' if host.endswith('alpha') else '10.110.21.136'))
    print('port 22')
elif '-O' in a:
    operation = a[a.index('-O') + 1]
    if operation == 'check':
        sys.exit(1)  # End the fixture at its first monitor, after forwarding.
    if operation == 'forward' and os.environ.get('BRIDGE_TEST_FAIL_FORWARD') and ':22022:' in ' '.join(a):
        sys.exit(255)
elif a[-1].endswith('hostname'):
    if host == os.environ.get('BRIDGE_TEST_FAIL_HOST'):
        sys.exit(255)
    print({'rtx-6k-alpha': '8f68286-lcedt', 'rtx-6k-beta': '465d6b0-lcedt', 'cosmos-h200': 'leo-cosmos-h200-access-8b47858b5-hzzsf'}[host])
elif a[-1].startswith('ss '):
    if os.environ.get('BRIDGE_TEST_OCCUPIED'):
        print('LISTEN 0 128 127.0.0.1:22022 0.0.0.0:*')
elif a[-1].startswith('cat /etc/ssh/'):
    print('ssh-ed25519 public-test-fixture')
else:
    sys.stdin.read()
'''


def run_case(tmp_path, **settings):
    for name in ("ssh", "uname", "sleep"):
        target = tmp_path / name
        target.write_text(MOCK)
        target.chmod(0o700)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               BRIDGE_TEST_ROOT=str(tmp_path), ROBORENDER_BRIDGE_CONTROL_PATH="/tmp/owned-test/%C",
               **settings)
    result = subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True, timeout=5)
    records = tmp_path / "calls.jsonl"
    calls = [json.loads(line) for line in records.read_text().splitlines()] if records.exists() else []
    return result, calls


def test_connections_use_private_per_host_sockets_and_keepalives(tmp_path):
    result, calls = run_case(tmp_path)
    assert result.returncode == 1, result.stderr  # Fixture monitor closes the session.
    for args in calls:
        if "-G" not in args:
            assert "ControlPath=/tmp/owned-test/%C" in args
            assert "ServerAliveInterval=15" in args
            assert "StrictHostKeyChecking=yes" in args
            assert "ControlPersist=60" in args
        assert not any("ProxyCommand=" in arg for arg in args)
    forwards = [a[a.index("-R") + 1] for a in calls if "forward" in a]
    cancels = [a[a.index("-R") + 1] for a in calls if "cancel" in a]
    assert forwards == cancels == ["127.0.0.1:22021:10.110.20.44:22", "127.0.0.1:22022:10.110.21.136:22"]
    assert not any("exit" in a or "stop" in a for a in calls)
    assert "awaiting end-to-end health check" in result.stdout


def test_existing_listener_is_not_taken_over(tmp_path):
    result, calls = run_case(tmp_path, BRIDGE_TEST_OCCUPIED="1")
    assert result.returncode != 0 and "leaving them untouched" in result.stderr
    assert not any("-O" in a or "bash -se" in a for a in calls)


def test_partial_forward_failure_cancels_only_our_created_forward(tmp_path):
    result, calls = run_case(tmp_path, BRIDGE_TEST_FAIL_FORWARD="1")
    assert result.returncode == 255
    assert [a[a.index("-R") + 1] for a in calls if "cancel" in a] == [
        "127.0.0.1:22021:10.110.20.44:22"]
    assert "Bridge forwards installed" not in result.stdout


def test_route_failure_names_stage_before_key_authorization(tmp_path):
    result, calls = run_case(tmp_path, BRIDGE_TEST_FAIL_HOST="cosmos-h200")
    assert result.returncode == 255 and "3/7: SSH cosmos-h200" in result.stderr
    assert not any("bash -se" in a for a in calls)
