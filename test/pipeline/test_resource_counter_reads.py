"""Actual-file CPU fault injection, never native/production qualification."""
from test.deployment.source_paths import PIPELINE_ROOT
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(PIPELINE_ROOT/'dagger/rtx/code/src'))
from dagger_rtx.phase_resource_guard import (GIB,PhaseResourceGuard,ResourceReadError,
    sample_resources,validate_summary)


@pytest.fixture
def files(tmp_path):
    group=tmp_path/'group';group.mkdir();mem=tmp_path/'meminfo'
    mem.write_text('MemAvailable: 12582912 kB\n')
    for name,value in {'memory.events':'max 31\noom 0\noom_kill 0\n',
            'memory.pressure':'some avg10=0.00\nfull avg10=0.00\n',
            'memory.stat':'anon 1\ninactive_file '+str(GIB)+'\n',
            'memory.current':str(3*GIB),'memory.max':str(21*GIB),
            'memory.swap.current':'0','memory.swap.max':'0'}.items():
        (group/name).write_text(value)
    return group,mem


@pytest.mark.parametrize('name,bad',[
    ('meminfo','MemAvailable 12582912 kB\n'),('memory.current',''),
    ('memory.max','garbage'),('memory.swap.current',''),('memory.swap.max',''),
    ('memory.stat','inactive_file wrong\n'),
    ('memory.events','max wrong\noom 0\noom_kill 0\n'),
    ('memory.pressure','some avg10=wrong\nfull avg10=0.00\n')])
def test_one_malformed_file_recovers_complete_snapshot_with_exact_diagnostic(files,monkeypatch,name,bad):
    group,mem=files;original=Path.read_bytes;counts={}
    def read(path,*args,**kwargs):
        counts[path.name]=counts.get(path.name,0)+1
        if path.name==name and counts[name]==1:return bad.encode()
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_bytes',read)
    sample=sample_resources(cgroup_root=group,meminfo=mem)
    assert sample['read_attempts']==2 and sample['events']['max']==31
    assert sample['memory_limit_bytes']==21*GIB and sample['swap_limit_bytes']==0
    assert len(sample['recovered_read_errors'])==1
    diagnostic=sample['recovered_read_errors'][0]
    assert any(row['file'].endswith(name) for row in diagnostic['errors'])
    assert diagnostic['fatal_observations']==[]
    assert all(counts[name]==2 for name in ('meminfo','memory.events','memory.pressure',
        'memory.stat','memory.current','memory.max','memory.swap.current','memory.swap.max'))
    json.dumps(sample,allow_nan=False)


def test_malformed_second_sample_is_recovered_and_audited_not_killed(files,tmp_path,monkeypatch):
    group,mem=files;original=Path.read_bytes;reads=0
    def read(path,*args,**kwargs):
        nonlocal reads
        if path.name=='meminfo':
            reads+=1
            if reads==2:return b'malformed global meminfo row\nMemAvailable: 12582912 kB\n'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_bytes',read)
    with PhaseResourceGuard(tmp_path/'guard.json',sample=lambda:sample_resources(cgroup_root=group,meminfo=mem)) as guard:
        guard._sample();guard.check()
    value=json.loads(guard.path.read_bytes())
    assert value['passed'] is True and value['samples']==3
    assert value['counter_read_attempts']==4 and value['counter_read_recoveries']==1
    assert value['recovered_read_diagnostics'][0]['errors'][0]['field']=='meminfo'
    assert value['sampling_diagnostic'] is None
    assert validate_summary(guard.summary) is guard.summary
    with pytest.raises(RuntimeError,match='recovery audit'):
        validate_summary(guard.summary|{'counter_read_recoveries':2})


def test_persistent_malformed_never_fabricates_zero_and_stops_after_three(files,tmp_path,monkeypatch):
    group,mem=files;mem.write_text('malformed\n');counts={};original=Path.read_bytes
    def read(path,*args,**kwargs):
        counts[path.name]=counts.get(path.name,0)+1
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_bytes',read)
    with pytest.raises(ResourceReadError) as raised:sample_resources(cgroup_root=group,meminfo=mem)
    diagnostic=raised.value.diagnostic
    assert len(diagnostic['attempts'])==3 and counts['memory.events']==counts['meminfo']==3
    assert diagnostic['attempts'][0]['observed_sample']['events']['max']==31
    assert 'host_available_bytes' not in diagnostic['attempts'][0]['observed_sample']
    assert 'meminfo' in str(raised.value) and len(json.dumps(diagnostic))<16384
    with pytest.raises(RuntimeError):
        with PhaseResourceGuard(tmp_path/'guard.json',sample=lambda:sample_resources(cgroup_root=group,meminfo=mem)):pass
    failed=json.loads((tmp_path/'guard.json').read_bytes())
    assert failed['passed'] is False and failed['first_sample'] is None
    assert failed['limit_pressure_events']==31 and failed['samples']==0
    assert failed['minimum_host_available_bytes'] is None
    assert failed['cache_reclaim_events'] is None and failed['unsafe_pressure_events'] is None
    assert len(failed['sampling_diagnostic']['attempts'])==3


@pytest.mark.parametrize('name,text,field',[
    ('memory.events','bad row extra\nmax 31\noom 1\noom 0\noom_kill 0\n','events.oom'),
    ('memory.events','max 31\noom 0\noom_kill 1\n','events.oom_kill'),
    ('memory.swap.current','1','swap_bytes'),
    ('memory.pressure','some avg10=11\nfull avg10=0\n','some_avg10'),
    ('memory.pressure','some avg10=0\nfull avg10=6\n','full_avg10'),
    ('memory.max','max','memory_limit_bytes'),('memory.swap.max','max','swap_limit_bytes'),
    ('memory.current',str(21*GIB+1),'memory_current_bytes')])
def test_malformed_early_file_cannot_hide_later_unsafe_observation(files,monkeypatch,name,text,field):
    group,mem=files;mem.write_text('malformed\n');(group/name).write_text(text)
    original=Path.read_bytes;counts={}
    def read(path,*args,**kwargs):
        counts[path.name]=counts.get(path.name,0)+1
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_bytes',read)
    with pytest.raises(ResourceReadError) as raised:sample_resources(cgroup_root=group,meminfo=mem)
    assert len(raised.value.diagnostic['attempts'])==1 and counts['meminfo']==1
    attempt=raised.value.diagnostic['attempts'][0]
    assert any(row['field']==field for row in attempt['fatal_observations'])
    assert attempt['errors']
    if field=='events.oom':assert attempt['observed_sample']['events']['oom']==1


def test_counter_regression_during_read_recovery_is_terminal(files,monkeypatch):
    group,mem=files;original=Path.read_bytes;counts={}
    def read(path,*args,**kwargs):
        counts[path.name]=counts.get(path.name,0)+1
        if path.name=='meminfo' and counts[path.name]==1:return b'malformed\n'
        if path.name=='memory.events' and counts[path.name]>1:return b'max 30\noom 0\noom_kill 0\n'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_bytes',read)
    with pytest.raises(ResourceReadError,match='regressed') as raised:sample_resources(cgroup_root=group,meminfo=mem)
    assert len(raised.value.diagnostic['attempts'])==2 and counts['meminfo']==2


def test_changed_cgroup_identity_during_recovery_is_never_retried_away(files,monkeypatch):
    group,mem=files;mem.write_text('malformed\n');original=Path.stat;calls=0
    def stat(path,*args,**kwargs):
        nonlocal calls
        value=original(path,*args,**kwargs)
        if path==group:
            calls+=1
            if calls>2:return SimpleNamespace(st_dev=value.st_dev,st_ino=value.st_ino+1)
        return value
    monkeypatch.setattr(Path,'stat',stat)
    with pytest.raises(ResourceReadError,match='identity changed') as raised:sample_resources(cgroup_root=group,meminfo=mem)
    assert len(raised.value.diagnostic['attempts'])==2


def test_nonfinite_psi_is_terminal_and_json_diagnostic_is_safe(files):
    group,mem=files;(group/'memory.pressure').write_text('some avg10=nan\nfull avg10=0\n')
    with pytest.raises(ResourceReadError) as raised:sample_resources(cgroup_root=group,meminfo=mem)
    assert len(raised.value.diagnostic['attempts'])==1
    assert raised.value.diagnostic['attempts'][0]['observed_sample']['some_avg10'] is None
    assert 'nan' in json.dumps(raised.value.diagnostic,allow_nan=False)


def test_failed_native_receipt_retains_actual_oom_instead_of_old_safe_zero(files,tmp_path):
    group,mem=files
    guard=PhaseResourceGuard(tmp_path/'guard.json',sample=lambda:sample_resources(cgroup_root=group,meminfo=mem))
    guard.__enter__()
    mem.write_text('malformed\n')
    (group/'memory.events').write_text('max 32\noom 1\noom_kill 0\n')
    guard._sample()
    with pytest.raises(RuntimeError):guard.__exit__(None,None,None)
    value=json.loads(guard.path.read_bytes())
    assert value['passed'] is False and value['oom_events']==1
    assert value['limit_pressure_events']==32
    assert value['last_sample']['events']['oom']==0
    assert value['failed_sample']['events']['oom']==1


def test_invalid_utf8_recovery_preserves_exact_raw_bytes_digest(files,monkeypatch):
    group,mem=files;original=Path.read_bytes;reads=0
    def read(path,*args,**kwargs):
        nonlocal reads
        if path.name=='memory.stat':
            reads+=1
            if reads==1:return b'\xff\n'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_bytes',read)
    result=sample_resources(cgroup_root=group,meminfo=mem)
    assert result['read_attempts']==2
    row=result['recovered_read_errors'][0]['errors'][0]
    assert row['file'].endswith('memory.stat')
    assert row['raw_sha256']==hashlib.sha256(b'\xff\n').hexdigest()
    assert row['raw_excerpt']=='\\xff\n'


def test_malformed_meminfo_does_not_waive_actual_working_set_margin(files):
    group,mem=files;mem.write_text('malformed\n')
    (group/'memory.current').write_text(str(21*GIB))
    (group/'memory.stat').write_text('inactive_file 0\n')
    with pytest.raises(ResourceReadError,match='working-set') as raised:
        sample_resources(cgroup_root=group,meminfo=mem)
    assert len(raised.value.diagnostic['attempts'])==1
