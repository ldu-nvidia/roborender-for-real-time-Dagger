"""A fresh established SSH route cannot overwrite or replay an earlier scope."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2] / 'runtime/raven-parity-overnight-20261026-t6oMpN'
spec = importlib.util.spec_from_file_location('network_resumed_test', ROOT / 'network_resumed.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_restores_only_established_loopback_ports_and_pins():
    assert m.m.PORTS == list(range(18560, 18568)) + [22036]
    assert m.m.REMOTE_ROOT.endswith('resumed-v4')
    assert m.m.UNIT.endswith('resumed-v4.service')
    assert m.m.REMOTE_ROOT in m.m.REMOTE
    assert "'--property=Restart=no'" in m.m.REMOTE
    assert "'source_known_hosts','known_hosts_sha256'" in m.m.REMOTE
    assert 'compute_release_authorized' in m.m.REMOTE


def test_existing_scope_rejected_before_remote_operation(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'NEW', tmp_path)
    monkeypatch.setattr(m.a, 'authority', lambda **kw: {})
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **kw: pytest.fail('duplicate dispatch'))
    with pytest.raises(ValueError, match='fresh network scope'):
        m.prepare()


def test_changed_plan_rejected_before_remote_operation(tmp_path, monkeypatch):
    (tmp_path / 'network-recovery-plan.json').write_text('{}')
    monkeypatch.setattr(m, 'NEW', tmp_path)
    monkeypatch.setattr(m.a, 'authority', lambda **kw: {})
    monkeypatch.setattr(m.m, 'plan', lambda: {'changed': True})
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **kw: pytest.fail('unbound route'))
    with pytest.raises(ValueError, match='plan changed'):
        m.check()
