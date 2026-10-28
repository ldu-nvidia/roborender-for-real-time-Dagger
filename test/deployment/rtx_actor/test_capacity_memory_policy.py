"""CPU-only production-memory qualification; no Isaac/GPU/systemd writes."""
from test.deployment.source_paths import ACTOR_ROOT
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ACTOR=ACTOR_ROOT
ROOT=ACTOR.parents[2]
sys.path.insert(0,str(ACTOR))

from qualify_native_isaac6 import command
from run_throughput_pair import production_capacity_options,validate_capacity_native_memory,attest_capacity_scope


def options():
    return dict(policy='disable-v1',mode='teacher-capacity',host='leo-rtx-6k',
        gpus=[0,1],steps=10,memory_gib=21,compact=True,conservative=True)


def receipt():
    return dict(native_thp_policy=dict(schema='roborender.native-thp-policy.v1',policy='disable-v1',
        disabled_after=1,native_pid=1,global_host_policy_modified=False),
        cgroup_memory={'memory.max':str(21*(1<<30)),'memory.swap.current':'0','memory.swap.max':'0'})


def test_explicit_production_policy_and_legacy_default():
    production_capacity_options(**options())
    production_capacity_options(**(options()|dict(policy='inherited',host='legacy',mode='capture-only',gpus=[1])))
    validate_capacity_native_memory(receipt())


@pytest.mark.parametrize('field,value',[('policy','bad'),('mode','single-window'),('host','beta'),
    ('gpus',[1]),('steps',3),('memory_gib',20),('compact',False),('conservative',False)])
def test_opt_in_rejects_mismatched_scope_before_launch(field,value):
    with pytest.raises(ValueError):production_capacity_options(**(options()|{field:value}))


def test_docker_explicit_native_thp_and_systemd_zero_swap(tmp_path):
    (tmp_path/'port-manifest.json').write_text('{"files":[]}')
    kwargs=dict(mode='teacher-capacity',deployment_host='leo-rtx-6k',physical_gpu=0,
        memory_gib=21,capacity_steps=10,compact_host_heap=True)
    old=command('test',tmp_path,tmp_path,ROOT/'pipeline/dagger/rtx/code',16,**kwargs)
    current=command('test',tmp_path,tmp_path,ROOT/'pipeline/dagger/rtx/code',16,
        native_thp_policy='disable-v1',**kwargs)
    assert not any(x.startswith('RESIDENT_THP_POLICY=') for x in old)
    assert '--annotation' not in old
    assert 'RESIDENT_THP_POLICY=disable-v1' in current
    assert current[current.index('--annotation')+1]=='org.systemd.property.MemorySwapMax=uint64 0'
    assert current[current.index('--memory')+1]==current[current.index('--memory-swap')+1]=='21g'
    assert current[current.index('--cpus')+1]=='12'
    assert current[current.index('--network')+1]=='bridge'
    assert not any('DAGGER_H200' in x for x in current)


@pytest.mark.parametrize('override',[dict(mode='capture-only'),dict(memory_gib=20),dict(capacity_steps=3),
    dict(compact_host_heap=False),dict(native_thp_policy='unknown')])
def test_direct_command_cannot_bypass_policy_scope(tmp_path,override):
    kwargs=dict(mode='teacher-capacity',deployment_host='leo-rtx-6k',physical_gpu=0,
        memory_gib=21,capacity_steps=10,compact_host_heap=True,native_thp_policy='disable-v1')|override
    with pytest.raises(ValueError):command('test',tmp_path,tmp_path,ROOT/'pipeline/dagger/rtx/code',16,**kwargs)


@pytest.mark.parametrize('section,key,value',[
    ('native_thp_policy','disabled_after',0),('native_thp_policy','policy','inherited'),
    ('native_thp_policy','native_pid',0),('native_thp_policy','global_host_policy_modified',True),
    ('cgroup_memory','memory.max',str(20*(1<<30))),('cgroup_memory','memory.swap.current','1'),
    ('cgroup_memory','memory.swap.max','max')])
def test_actual_native_policy_receipt_must_match(section,key,value):
    row=deepcopy(receipt());row[section][key]=value
    with pytest.raises(RuntimeError):validate_capacity_native_memory(row)


def test_systemd_scope_uses_exact_native_pid_and_shared_read_only_admission(monkeypatch):
    import run_production_resident_isaac as resident
    seen=[]
    def read(path):
        assert path==Path('/proc/123/cgroup')
        return '0::/system.slice/docker-'+('a'*64)+'.scope\n'
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(resident,'attest_systemd_scope',lambda path:seen.append(path) or {'passed':True})
    assert attest_capacity_scope({'pid':123})=={'passed':True}
    assert seen==[Path('/sys/fs/cgroup/system.slice/docker-'+('a'*64)+'.scope')]
    with pytest.raises(RuntimeError):attest_capacity_scope({'pid':0})


def test_native_configures_thp_before_scene_and_rechecks_before_receipt():
    source=(ACTOR/'isaac6_capacity_trainer.py').read_text()
    assert source.index('NATIVE_THP_CONFIGURATION = configure_from_environment()') < source.index('from dagger_rtx.native_service_trainer')
    assert 'thp = attest_native_thp(NATIVE_THP_CONFIGURATION)' in source
    assert "result['native_thp_policy'] = attest_native_thp(thp)" in source
