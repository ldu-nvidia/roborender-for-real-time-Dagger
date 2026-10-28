"""CPU-only guard-v2 wire/file verification and bounded procfs sampling."""
from test.deployment.source_paths import REPO_ROOT
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=REPO_ROOT
sys.path.insert(0,str(ROOT/'pipeline/deployment/rtx_actor'))
sys.path.insert(0,str(ROOT/'pipeline/dagger/rtx/code/src'))
from dagger_rtx.phase_resource_guard import PhaseResourceGuard,sample_resources,GIB,SCHEMA
from production_resident_pool import validate_guard
from run_production_resident_isaac import resource_gate,enrich_memory_sample
import run_production_resident_isaac as launcher


def test_guard_metadata_import_is_stdlib_only_and_package_exports_unchanged():
    code='''
import importlib.abc,sys
class DenyNumpy(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='numpy' or fullname.startswith('numpy.'):
   raise AssertionError('resource policy imported numpy')
block=DenyNumpy();sys.meta_path.insert(0,block)
from dagger_rtx.phase_resource_guard import validate_summary
assert 'numpy' not in sys.modules
sys.meta_path.remove(block)
from dagger_rtx import CONTRACT_VERSION,RAVEN_VIEW_ORDER
assert CONTRACT_VERSION=='roborender.dagger.rtx.v1'
assert RAVEN_VIEW_ORDER==('head_color','hand_left','hand_right')
import dagger_rtx
assert dagger_rtx.__all__==['CONTRACT_VERSION','RAVEN_VIEW_ORDER']
try:dagger_rtx.unknown_export
except AttributeError:pass
else:raise AssertionError('unknown export accepted')
'''
    subprocess.run([sys.executable,'-c',
        'import sys;sys.path.insert(0,'+repr(str(ROOT/'pipeline/dagger/rtx/code/src'))+');'+code],check=True)


def actual(**changes):
    row=dict(host_available_bytes=12*GIB,memory_current_bytes=21*GIB,
        memory_limit_bytes=21*GIB,swap_bytes=0,swap_limit_bytes=0,
        inactive_file_bytes=8*GIB,working_set_bytes=13*GIB,some_avg10=0.0,full_avg10=0.0,
        process_id=12,cgroup_path='/owned',cgroup_device=1,cgroup_inode=2,
        events=dict(max=31,oom=0,oom_kill=0))
    return row|changes


def summary_fixture(tmp_path):
    with PhaseResourceGuard(tmp_path/'guard.json',sample=actual) as guard:pass
    return guard.summary,dict(runtime_root=str(tmp_path))


def republish(summary,value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    Path(summary['path']).write_bytes(raw)
    return summary|{'raw_sha256':hashlib.sha256(raw).hexdigest()}


def test_v2_actual_file_retains_31_cache_events_without_waiving_caps(tmp_path):
    summary,worker=summary_fixture(tmp_path)
    receipt=validate_guard(summary,worker)
    assert receipt['schema']==SCHEMA
    assert receipt['limit_pressure_events']==receipt['cache_reclaim_events']==31
    assert receipt['unsafe_pressure_events']==receipt['phase_limit_pressure_events']==0


@pytest.mark.parametrize('fault',['policy','raw_counter','initial_counter','identity','missing_stat',
    'raw_extrema','raw_classification','lower_peak','raw_schema','sampling_error'])
def test_hash_bound_file_cannot_lie_about_v2_proof(tmp_path,fault):
    summary,worker=summary_fixture(tmp_path)
    row=json.loads(Path(summary['path']).read_bytes())
    if fault=='policy':row['policy']['maximum_full_avg10']=100
    elif fault=='raw_counter':row['last_sample']['events']['max']=32
    elif fault=='initial_counter':row['first_sample']['events']['max']=30
    elif fault=='identity':row['last_sample']['process_id']=13
    elif fault=='missing_stat':del row['first_sample']['inactive_file_bytes']
    elif fault=='raw_extrema':row['maximum_some_avg10']=3
    elif fault=='raw_classification':row['cache_reclaim_events']=0
    elif fault=='lower_peak':row['memory_current_peak_bytes']=GIB
    elif fault=='raw_schema':row['schema']='roborender.native-phase-resource-guard.v1'
    else:row['sampling_error']='failed earlier'
    with pytest.raises((RuntimeError,KeyError)):validate_guard(republish(summary,row),worker)


def test_v2_summary_cannot_accept_legacy_file_relabel(tmp_path):
    summary,worker=summary_fixture(tmp_path)
    row=json.loads(Path(summary['path']).read_bytes())
    legacy={k:summary[k] for k in ('path','raw_sha256','passed','samples',
        'minimum_host_available_bytes','swap_peak_bytes','oom_events','limit_pressure_events')}
    legacy['limit_pressure_events']=0
    with pytest.raises(RuntimeError):validate_guard(legacy,worker)


def test_additive_read_audit_is_compared_with_hash_bound_raw_receipt(tmp_path):
    summary,worker=summary_fixture(tmp_path)
    assert validate_guard(summary,worker)['counter_read_attempts']==2
    row=json.loads(Path(summary['path']).read_bytes())
    # Each pair separately satisfies attempts=samples+recoveries. They still
    # must agree; a true checksum alone does not validate the claimed summary.
    row.update(counter_read_attempts=3,counter_read_recoveries=1)
    with pytest.raises(RuntimeError,match='counter-read audit'):
        validate_guard(republish(summary,row),worker)


def test_outer_launcher_uses_same_bounded_reclaim_classifier():
    sample=dict(pid=12,oom_killed=False,**{'memory_swap.current':0},resource_sample_v2=actual())
    assert resource_gate({'MemAvailable':12*GIB//1024},sample)==dict(
        limit_pressure_events=31,cache_reclaim_events=31,unsafe_pressure_events=0)
    for changes in ({'swap_bytes':1},{'some_avg10':10.1},{'full_avg10':5.1},
            {'working_set_bytes':21*GIB,'inactive_file_bytes':0},
            {'events':dict(max=31,oom=1,oom_kill=0)},{'process_id':13}):
        with pytest.raises(RuntimeError):resource_gate({'MemAvailable':12*GIB//1024},
            sample|{'resource_sample_v2':actual(**changes)})


def test_outer_sampler_binds_exact_inspected_native_pid(monkeypatch):
    paths=[]
    def owned(*,membership):paths.append(membership);return Path('/owned')
    monkeypatch.setattr(launcher,'own_cgroup_root',owned)
    monkeypatch.setattr(launcher,'sample_resources',lambda **kw:actual(process_id=999))
    result=enrich_memory_sample({'pid':12})
    assert paths==[Path('/proc/12/cgroup')]
    assert result['resource_sample_v2']['process_id']==12


@pytest.mark.parametrize('consistent',[True,False])
def test_only_inconsistent_cache_snapshot_gets_three_bounded_retries(tmp_path,monkeypatch,consistent):
    group=tmp_path/'owned';group.mkdir();mem=tmp_path/'meminfo'
    mem.write_text('MemAvailable: 12582912 kB\n')
    for name,value in {'memory.events':'max 31\noom 0\noom_kill 0\n',
        'memory.max':str(21*GIB),'memory.swap.current':'0','memory.swap.max':'0',
        'memory.pressure':'some avg10=0.0\nfull avg10=0.0\n',
        'memory.stat':'inactive_file '+str(8*GIB)+'\n','memory.current':str(7*GIB)}.items():
        (group/name).write_text(value)
    original=Path.read_bytes;calls={}
    def read(path,*args,**kwargs):
        calls[path.name]=calls.get(path.name,0)+1
        if path.name=='memory.current' and consistent and calls[path.name]>1:return str(9*GIB).encode()
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_bytes',read)
    if consistent:
        value=sample_resources(cgroup_root=group,meminfo=mem)
        assert value['working_set_bytes']==GIB and value['events']['max']==31
        assert calls['memory.current']==calls['memory.stat']==2
    else:
        with pytest.raises(RuntimeError,match='inconsistent'):
            sample_resources(cgroup_root=group,meminfo=mem)
        assert calls['memory.current']==calls['memory.stat']==3
    # Recovery now rereads a complete snapshot, while preserving every unsafe
    # observation from earlier attempts rather than just the cache pair.
    assert calls['memory.events']==calls['memory.pressure']==(2 if consistent else 3)
