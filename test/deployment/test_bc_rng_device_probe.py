"""Guard the separate bounded device probe; never launch GPU work in unit tests."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
PATH=ROOT/'bc_rng_device_v1.py'
if not PATH.exists():pytest.skip('runtime device probe absent',allow_module_level=True)
spec=importlib.util.spec_from_file_location('bc_rng_device_probe_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture():
    files={k:dict(raw_sha256=k*64) for k in ('helper','probe','original','spec')}
    value=dict(schema='roborender.bc-collection-rng-device-differential.v1',device='cuda:0',
        resident_state_touched=False,physics_tested=False,compute_release_authorized=False,
        original_normal_method_executed=True,helper_sha256=files['helper']['raw_sha256'],
        probe_sha256=files['probe']['raw_sha256'],original_policy_sha256=files['original']['raw_sha256'],
        architecture_guard=dict(policy_spec_sha256=files['spec']['raw_sha256'],real_policy_class=True,
            declared_collector_arguments=True,resident_owner_used=False),
        comparisons=[dict(batch=b,seed=s,windows=5,rng_state_exact=True,next_reset_draws_exact=True,
            reference_optimizer_updates=5,adapter_optimizer_updates=0)
            for b in (1,2,16,32,40) for s in (31,42,901)])
    return value,files

def test_actual_device_comparison_receipt():
    value,files=fixture();assert m.validate(value,'cuda:0',files)==value

@pytest.mark.parametrize('field,value',[('resident_state_touched',True),('physics_tested',True),
    ('helper_sha256','bad'),('device','cpu'),('compute_release_authorized',True)])
def test_wrong_scope_or_source_rejected(field,value):
    record,files=fixture();record[field]=value
    with pytest.raises(ValueError):m.validate(record,'cuda:0',files)

def test_missing_comparison_rejected():
    record,files=fixture();record['comparisons'].pop()
    with pytest.raises(ValueError):m.validate(record,'cuda:0',files)

def test_no_simulator_or_allocation_operations():
    source=PATH.read_text()
    assert 'm.qualify(c,live)' in source and "'single-shot probe already admitted" in source
    assert 'post_command(' not in source and 'shutdown(' not in source
    assert "'docker','exec'" in source

def resource():
    owner=dict(worker_instance_id='owner',host_pid=10,container_id='container')
    row=dict(at_unix=100,host_available_bytes=16<<30,native=[dict(owner_id='owner',host_pid=10,
        container_id='container',vram_total_bytes=100<<30,vram_used_bytes=24<<30,
        safety_sample=dict(memory_limit_bytes=21<<30,memory_current_bytes=18<<30,
            events=dict(max=0,oom=0,oom_kill=0),swap_bytes=0))])
    return owner,row

def test_tiny_probe_requires_memory_headroom():
    owner,row=resource();assert m.resource_admit(row,owner,101)==row['native'][0]
    row['native'][0]['safety_sample']['memory_current_bytes']=20<<30
    with pytest.raises(RuntimeError):m.resource_admit(row,owner,101)

def test_stale_or_changed_owner_rejected():
    owner,row=resource()
    with pytest.raises(ValueError):m.resource_admit(row,owner,131)
    row['native'][0]['host_pid']=11
    with pytest.raises(RuntimeError):m.resource_admit(row,owner,101)
