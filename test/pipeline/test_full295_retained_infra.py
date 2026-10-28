"""CPU-only boundary tests; these do not certify native capacity or success."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
from pipeline.deployment.h200.beta_service_host import require_allocation_identity,validate_lifetime
from pipeline.deployment.retained_service_process import read_spec
from pipeline.deployment.retained_campaign_site import load_retained,RetainedResidentSite
from pipeline.deployment.h200.beta_pod import name_for
from pipeline.deployment.coverage_operations import CoverageCommandOperations

def test_old_owner_check_stays_strict(monkeypatch):
    monkeypatch.setenv('ROBORENDER_BETA_RUN_ID','dagger-leo-bc-original')
    with pytest.raises(RuntimeError):require_allocation_identity('dagger-leo-new')
    require_allocation_identity('dagger-leo-new','dagger-leo-bc-original')
    with pytest.raises(RuntimeError):require_allocation_identity('dagger-leo-new','dagger-leo-wrong')

def test_legacy_ttl_not_silently_extended():
    with pytest.raises(ValueError):validate_lifetime(7*86400,None,False)
    with pytest.raises(ValueError):validate_lifetime(24000,None,True)
    with pytest.raises(ValueError):validate_lifetime(86400,'owner',True)
    validate_lifetime(24000,'owner',True)

def test_persistent_process_cannot_launch_native_or_allocate(tmp_path):
    value=dict(schema='roborender.retained-service-process.v1',
        argv=['/usr/bin/python3','/workspace/project/roborender/pipeline/deployment/h200/beta_service_host.py',
            '--continuous-services','--retained-owner','dagger-leo-bc-original'],cwd='/workspace/project/roborender',
        env={},owner='dagger-leo-production',role='gpu-services',release_compute=False)
    path=tmp_path/'spec.json';path.write_text(json.dumps(value));assert read_spec(path)==value
    for change in (dict(release_compute=True),dict(role='native'),dict(env={'WANDB_API_KEY':'forbidden'}),
        dict(argv=['/usr/bin/python3','/workspace/project/roborender/pipeline/deployment/h200/beta_pod.py'])):
        path.write_text(json.dumps({**value,**change}))
        with pytest.raises(ValueError):read_spec(path)

def test_retained_route_and_owner_binding(tmp_path):
    route=tmp_path/'ssh.conf';known=tmp_path/'known_hosts';route.touch(mode=0o600);known.touch(mode=0o600)
    owner='dagger-leo-bc-original'
    value=dict(schema='roborender.full295-retained-h200.v1',allocation_owner_run_id=owner,
        allocation_name=name_for(owner,1),node_id='node-123',gpu_uuids=['GPU-'+str(i) for i in range(4)],
        ssh_config=str(route),known_hosts=str(known))
    path=tmp_path/'retained.json';path.write_text(json.dumps(value));sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_retained(path,sha)==value
    value['allocation_name']='unrelated';path.write_text(json.dumps(value))
    with pytest.raises(ValueError):load_retained(path,hashlib.sha256(path.read_bytes()).hexdigest())

def test_retained_provider_has_no_compute_mutations():
    site=object.__new__(RetainedResidentSite)
    for action in ('create','delete','cancel','scale','suspend'):
        with pytest.raises(ValueError):site.allocation(action)
    with pytest.raises(ValueError):site.release(confirm_release='anything')

def test_long_operations_require_new_retained_entrypoint(tmp_path):
    entry=Path('/workspace/project/roborender/pipeline/deployment/retained_campaign_site.py')
    spec=dict(argv=['/usr/bin/python3',str(entry)],entrypoint=str(entry),
        entrypoint_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),timeout_seconds=7*86400)
    commands={p:deepcopy(spec) for p in ('prepare','collect','release','evaluate','control_only','abort')}
    value=dict(schema='roborender.full295-campaign-operations.v1',commands=commands,compute_release_permitted=False)
    assert CoverageCommandOperations(value,tmp_path).binding==value
    bad=deepcopy(value);bad['compute_release_permitted']=True
    with pytest.raises(ValueError):CoverageCommandOperations(bad,tmp_path)

def test_queue_operations_bind_only_actual_runtime_refs(tmp_path):
    from pipeline.dagger.coverage_queue import operations_for
    from pipeline.bc_training.handoff_common import publish
    site=publish(tmp_path/'site.json',dict(schema='roborender.leo-site.v1',cpu_python='/usr/bin/python3'))
    retained=publish(tmp_path/'retained.json',dict(schema='roborender.full295-retained-h200.v1'))
    runtime=dict(schema='roborender.full295-runtime-bindings.v1',site=site,retained=retained,
        resident_pool=dict(path='/home/ldu/production/pool.json',sha256='a'*64),source_sha256='b'*64)
    ops=operations_for(runtime,tmp_path)
    assert ops['compute_release_permitted']is False
    assert all('retained_campaign_site.py' in s['entrypoint'] for s in ops['commands'].values())
    with pytest.raises(ValueError):operations_for({**runtime,'pretend_native_qualification':True},tmp_path)


def test_resident_collection_uses_retained_connection_root(tmp_path):
    from pipeline.deployment.resident_campaign_site import ResidentSite
    site=object.__new__(ResidentSite)
    site.jobs=tmp_path;site.remote=tmp_path/'round-1'
    site.connection_root=tmp_path/'retained-tunnels'
    site.run='dagger-leo-smoke';site.rid=1;site.gpus=[0,1]
    site.config={'run_id':site.run,'schema':'test-layout',
        'rtx_actors':[{'physical_gpu_index':g,'concurrent_environments':2} for g in site.gpus]}
    path=tmp_path/'config.json';path.write_text('{}')
    site.request={'round_config_path':str(path)}
    site.barrier=lambda:{};site.publish=lambda *args:None
    def read(path,host):
        assert path==site.connection_root/'supervisor.json' and host=='beta'
        return {'identity':'retained'}
    site.read=read
    def pool(action,commands):
        assert all(c['payload']['connection_status_directory']=='retained-tunnels/supervisor' for c in commands)
        return {'results':[{'receipt':{'actor_state':{}}} for _ in commands]}
    site.pool_action=pool
    assert len(site.collect()['actors'])==2
