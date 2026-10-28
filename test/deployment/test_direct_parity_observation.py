"""Transport-only recovery must not relax native/resource or source gates."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def test_direct_transport_keeps_hostkey_and_owner_pins():
    m=module('telemetry_direct_v1')
    argv=m.direct_argv(['ssh','-F','/pinned.conf','-S','/old.sock','-o','StrictHostKeyChecking=yes','retained-alias'])
    assert argv[-1]=='retained-alias' and '-F' in argv and 'StrictHostKeyChecking=yes' in argv
    assert 'HostName=10.50.176.97' in argv and 'Port=2222' in argv and 'ProxyCommand=none' in argv
    assert argv[argv.index('-S')+1]==str(m.SOCKET)
def test_fresh_route_only_replaces_proven_ended_network_unit():
    m=module('network_v3')
    assert m.m.UNIT.endswith('-v3.service') and m.OLD==ROOT/'network-v2'
    assert 'StandardError=append:' in m.m.REMOTE
    assert 'Restart=no' in m.m.REMOTE and 'native_steps=0' in (ROOT/'restore_network.py').read_text()
def test_observer_preserves_gap_and_identity_checks():
    m=module('sampler_direct_v1')
    assert m.Sampler.healthy is m.s.Sampler.healthy
    assert m.Sampler.bracket is m.s.Sampler.bracket
    assert 's.sampler_loop(' in (ROOT/'sampler_direct_v1.py').read_text()
def test_registry_recovery_uses_new_interval_without_replacing_services():
    m=module('continuous_registry_recovery_v2')
    assert m.m.RECOVERY.name=='registry-recovery-v2'
    assert m.m.v.Sampler is m.observer.Sampler
    assert 'sampler_output=str(observer.OUT)' in (ROOT/'continuous_registry_recovery_v2.py').read_text()
    assert 'services_replaced=False' in (ROOT/'continuous_registry_recovery_v2.py').read_text()

def test_all_service_readback_uses_h200_not_missing_cpu_forwards():
    import pytest
    m=module('continuous_registry_recovery_v3')
    expected={'18610':dict(role='coordinator',pid=1)}
    assert m.validate_rows(expected,expected)==expected
    for actual in ({},{'18610':dict(role='coordinator',pid=2)},{'18611':expected['18610']}):
        with pytest.raises(ValueError,match='owner'):m.validate_rows(expected,actual)
    assert 'mux_argv(transport)' in (ROOT/'continuous_registry_recovery_v3.py').read_text()
    assert 'range(18620,18628)' in m.REMOTE
