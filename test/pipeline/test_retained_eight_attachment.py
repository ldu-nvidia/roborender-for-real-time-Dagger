"""CPU-only retained8 deployment checks; never native capacity certification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from pipeline.deployment.h200 import retained_renderer_attachment as a
from pipeline.deployment.h200.beta_service_host import launch_plan, gateway_environment, learner_environment, durable_intent
from pipeline.deployment.retained_campaign_site import (load_retained, RetainedResidentSite, RETAINED_EIGHT_SCHEMA,
    validate_rtx_data_route, validate_rtx_ssh_config, validate_eight_provider_root)
from pipeline.dagger.beta_campaign import bind_preparation
from test.pipeline.test_retained_native_data_route import route, config as ssh_config, ALIAS, IDENTITY
from test.pipeline.test_dagger_continuous_actor import continuous_inputs
from test.pipeline.test_dagger_fixed100_native import fixed_inputs
from test.pipeline.test_dagger_coverage_production import production


@pytest.fixture
def fleet():
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    return dict(schema='roborender.scaled-renderer-fleet.v1', allocation_owner=a.OWNER,
        allocation_name=a.ALLOCATION, hostname='owned-h200', node_id='owned-node', boot_id=boot,
        automatic_restart=False,automatic_allocation_release=False,compute_release_authorized=False,
        server_source_sha256='a'*64,workers=[dict(slot=i,port=18560+i,host='127.0.0.1',pid=9000+i,
            gpu_uuid=f'GPU-{i:08x}-1111-2222-3333-444444444444',server_instance=f'{i:032x}',
            source_sha256='a'*64,generate_source_sha256='b'*64,readiness_receipt_sha256='c'*64,
            readiness_receipt_path=f'/readiness/{i}.json',identity=dict(boot_id=boot,start_ticks=str(1000+i),
                cmdline_sha256=f'{i:064x}')) for i in range(8)])


def write_json(path,value):
    raw=a.canonical(value);path.write_bytes(raw)
    return dict(path=str(path),raw_sha256=a.digest(raw))


def test_exact8_retained_binding_rejects_legacy_relabeling(tmp_path,fleet):
    conf=tmp_path/'ssh.conf';known=tmp_path/'known_hosts';conf.touch(mode=0o600);known.touch(mode=0o600)
    ref=write_json(tmp_path/'fleet.json',fleet)
    value=dict(schema=RETAINED_EIGHT_SCHEMA,allocation_owner_run_id=a.OWNER,allocation_name=a.ALLOCATION,
        node_id=fleet['node_id'],gpu_uuids=[r['gpu_uuid'] for r in fleet['workers']],
        ssh_config=str(conf),known_hosts=str(known),rtx_data_route=dict(route(),local_port=22036),renderer_fleet=ref)
    def load(v):
        p=write_json(tmp_path/'retained.json',v);return load_retained(p['path'],p['raw_sha256'])
    assert load(value)==value
    for field,bad in [('schema','roborender.full295-retained-h200.v2'),('gpu_uuids',value['gpu_uuids'][:4]),
                      ('allocation_owner_run_id','other'),('node_id','other')]:
        with pytest.raises(ValueError):load(dict(value,**{field:bad}))


@pytest.mark.parametrize('fault',['duplicate_gpu','duplicate_pid','duplicate_instance','port','source','release'])
def test_fleet_aliases_and_identity_drift_fail(fleet,fault):
    bad=deepcopy(fleet)
    if fault.startswith('duplicate_'):
        key={'duplicate_gpu':'gpu_uuid','duplicate_pid':'pid','duplicate_instance':'server_instance'}[fault]
        bad['workers'][1][key]=bad['workers'][0][key]
    elif fault=='port':bad['workers'][4]['port']=18560
    elif fault=='source':bad['workers'][4]['source_sha256']='f'*64
    else:bad['compute_release_authorized']=True
    with pytest.raises(ValueError):a.validate_fleet(bad)


def test_attachment_plan_never_launches_a_renderer_or_uses_overlapping_devices(fleet):
    env,commands=launch_plan('/source','/state','/token','/output','gpu',retained_fleet=fleet)
    assert len(commands)==13
    assert not any('render_pool' in str(command) or 'image3f_server' in str(command) for command in commands)
    gateways=[r for r in commands if r[0].startswith('gateway-')]
    assert [r[2] for r in gateways]==[[p] for p in range(18620,18628)]
    environments=[gateway_environment(fleet,i) for i in range(8)]
    assert len({e['CUDA_VISIBLE_DEVICES'] for e in environments})==8
    assert [e['IMAGE3F_PORT'] for e in environments]==[str(p) for p in range(18560,18568)]
    assert all(e['DAGGER_STUDENT_DEVICES']=='cuda:0' for e in environments)
    _,legacy=launch_plan('/source','/state','/token','/output','gpu')
    assert legacy[-1][0]=='render_pool' and legacy[-1][2]==list(range(18620,18624))


def test_learner_is_pinned_before_spawn_without_collapsing_gateway_pool(fleet):
    assert learner_environment(fleet)=={'CUDA_VISIBLE_DEVICES':fleet['workers'][0]['gpu_uuid']}
    inherited={'CUDA_VISIBLE_DEVICES':','.join(r['gpu_uuid'] for r in fleet['workers'])}
    assert dict(inherited,**learner_environment(fleet))['CUDA_VISIBLE_DEVICES']==fleet['workers'][0]['gpu_uuid']
    assert len({dict(inherited,**gateway_environment(fleet,i))['CUDA_VISIBLE_DEVICES'] for i in range(8)})==8
    bad=deepcopy(fleet);bad['workers'][0]['gpu_uuid']=bad['workers'][1]['gpu_uuid']
    with pytest.raises(ValueError):learner_environment(bad)


def fake_live(monkeypatch,fleet):
    monkeypatch.setattr(a.socket,'gethostname',lambda:fleet['hostname'])
    monkeypatch.setenv('ROBORENDER_BETA_RUN_ID',a.OWNER);monkeypatch.setenv('ROBORENDER_BETA_ROUND_ID','1')
    monkeypatch.delenv('NODE_NAME',raising=False);monkeypatch.delenv('LEPTON_NODE_NAME',raising=False)
    bypid={r['pid']:r for r in fleet['workers']}
    monkeypatch.setattr(a,'process_identity',lambda pid:bypid[pid]['identity'])
    monkeypatch.setattr(a,'process_device_visibility',lambda pid:[bypid[pid]['gpu_uuid']])
    monkeypatch.setattr(a,'process_gpu_pids',lambda pid:{pid})
    monkeypatch.setattr(a,'owns_listener',lambda pid,port:True)
    monkeypatch.setattr(a,'read_bound',lambda path,sha:dict(transport={'server_instance':fleet['workers'][int(Path(path).stem)]['server_instance']},
        server=dict(identity=fleet['workers'][int(Path(path).stem)]['identity'],source_sha256='a'*64)))
    monkeypatch.setattr(a,'metadata',lambda port:dict(ok=True,attestation=dict(
        contract=dict(schema='roborender.dagger.student-observation.v1',mode='generated_only',renderer='Image3F',
            renderer_steps=5,renderer_cfg=1.,teacache_threshold=.3,raytraced_policy_fallback=False),
        backend_class='runtime.image3f_deploy.Image3FSession',generate_source_sha256='b'*64)))
    def output(argv,**kwargs):
        if '--query-compute-apps=gpu_uuid,pid' in argv:
            return '\n'.join(f"{r['gpu_uuid']}, {r['pid']}" for r in fleet['workers'])
        return '\n'.join(f"{r['slot']}, {r['gpu_uuid']}, NVIDIA H200, 10000, 140000" for r in fleet['workers'])
    monkeypatch.setattr(a.subprocess,'check_output',output)
    return output


def test_live_attestation_checks_all8_without_render_or_channel_open(monkeypatch,fleet):
    fake_live(monkeypatch,fleet)
    result=a.observe(fleet)
    assert len(result['renderers'])==8 and result['gpu_client_counts']==[1]*8
    assert result['render_requests']==result['renderer_restarts']==0
    assert result['compute_release_authorized'] is False
    monkeypatch.setattr(a,'process_device_visibility',lambda pid:['GPU-wrong'])
    with pytest.raises(RuntimeError):a.observe(fleet)


def test_unrelated_gpu_client_and_wrong_boot_fail_before_launch(monkeypatch,fleet):
    output=fake_live(monkeypatch,fleet)
    def extra(argv,**kwargs):
        raw=output(argv,**kwargs)
        return raw+f"\n{fleet['workers'][0]['gpu_uuid']}, 99999" if '--query-compute-apps=gpu_uuid,pid' in argv else raw
    monkeypatch.setattr(a.subprocess,'check_output',extra)
    with pytest.raises(RuntimeError,match='client population'):a.observe(fleet)
    bad=deepcopy(fleet);bad['boot_id']='different'
    for w in bad['workers']:w['identity']['boot_id']='different'
    with pytest.raises(RuntimeError,match='host/boot'):a.observe(bad)


def test_running_attachment_allows_only_exact_gateway_pids_and_listeners(monkeypatch,fleet):
    output=fake_live(monkeypatch,fleet)
    gateways=[dict(slot=i,pid=10000+i,gpu_uuid=w['gpu_uuid'],process_identity=w['identity'])
              for i,w in enumerate(fleet['workers'])]
    rows={w['pid']:dict(gpu_uuid=w['gpu_uuid'],process_identity=w['identity']) for w in fleet['workers']}
    rows.update({g['pid']:g for g in gateways})
    driver_gateways=deepcopy(gateways)
    monkeypatch.setattr(a,'process_identity',lambda pid:rows[pid]['process_identity'])
    monkeypatch.setattr(a,'process_device_visibility',lambda pid:[rows[pid]['gpu_uuid']])
    def clients(argv,**kwargs):
        raw=output(argv,**kwargs)
        if '--query-compute-apps=gpu_uuid,pid' in argv:
            raw+='\n'+'\n'.join(f"{g['gpu_uuid']}, {g['pid']}" for g in driver_gateways)
        return raw
    monkeypatch.setattr(a.subprocess,'check_output',clients)
    assert a.observe(fleet,expected_gateways=gateways)['gpu_client_counts']==[2]*8
    original=gateways[0]['pid'];gateways[0]['pid']=19999;rows[19999]=rows[original]
    with pytest.raises(RuntimeError,match='client population'):a.observe(fleet,expected_gateways=gateways)
    gateways[0]['pid']=original
    monkeypatch.setattr(a,'owns_listener',lambda pid,port:pid!=original)
    with pytest.raises(RuntimeError,match='listener'):a.observe(fleet,expected_gateways=gateways)


def test_new_and_old_native_routes_cannot_be_relabelled():
    old=route();new=dict(old,local_port=22036)
    validate_rtx_data_route(old);validate_rtx_data_route(new,eight=True)
    assert validate_rtx_ssh_config(ssh_config(new),new,ALIAS,IDENTITY,eight=True)['port']=='22036'
    with pytest.raises(ValueError):validate_rtx_data_route(old,eight=True)
    with pytest.raises(ValueError):validate_rtx_data_route(new)
    with pytest.raises(ValueError):validate_rtx_ssh_config(ssh_config(old),new,ALIAS,IDENTITY,eight=True)


def test_frozen8_provider_allows_only_registered_full_release_jails(monkeypatch):
    monkeypatch.setattr(Path,'resolve',lambda self,strict=False:self)
    verify=Mock(return_value={'verified':True});monkeypatch.setattr(a,'verify_release',verify)
    for base in ('/workspace/project/roborender/runtime/source-releases',
                 '/workspace/project/roborender-artifacts/code_releases'):
        root=Path(base)/'explicit-release'/'code';manifest=root.parent/'source-manifest.json'
        assert validate_eight_provider_root(root,manifest,'a'*64)=={'verified':True}
        verify.assert_called_with(root,manifest,'a'*64)
        with pytest.raises(ValueError):validate_eight_provider_root(root,root/'other.json','a'*64)
    for root in ('/tmp/code','/workspace/project/roborender',
                 '/workspace/project/roborender-artifacts/code_releases/release/code/nested'):
        with pytest.raises(ValueError):validate_eight_provider_root(root,Path(root).parent/'source-manifest.json','a'*64)


def test_source_and_gateway_are_verified_from_actual_bytes(tmp_path,monkeypatch,fleet):
    source=tmp_path/'code';source.mkdir();code=source/'a.py';code.write_text('value = 1\n')
    rows=[dict(path='a.py',bytes=code.stat().st_size,sha256=a.digest(code.read_bytes()),executable=False)]
    sha=a.digest(a.canonical(rows));manifest=tmp_path/'manifest.json'
    write_json(manifest,dict(schema='roborender.source-snapshot.v1',files=rows,file_count=1,source_sha256=sha))
    a.verify_release(source,manifest,sha)
    proc=tmp_path/'proc'/'100';proc.mkdir(parents=True);(proc/'cwd').symlink_to(source,target_is_directory=True)
    argv=['/python','-m','pipeline.dagger.services.render_student_server','--port','18624',
          '--devices','cuda:0','--image3f-ports','18564']
    (proc/'cmdline').write_bytes('\0'.join(argv).encode()+b'\0')
    (proc/'environ').write_bytes((f"CUDA_VISIBLE_DEVICES={fleet['workers'][4]['gpu_uuid']}\0PYTHONPATH={source}\0"
                                f"ROBORENDER_SERVICE_SOURCE_SHA256={sha}\0").encode())
    monkeypatch.setattr(a,'process_identity',lambda pid:{'live':True})
    monkeypatch.setattr(a,'owns_listener',lambda pid,port:True)
    ready=dict(ok=True,role='render_student',pid=100,gateway_instance_id='d'*64)
    row=a.gateway_identity(source,sha,fleet,4,ready,expected_pid=100,proc_root=tmp_path/'proc')
    assert row['slot']==4 and row['gpu_uuid']==fleet['workers'][4]['gpu_uuid']
    assert row['process_identity']=={'live':True}
    with pytest.raises(RuntimeError):a.gateway_identity(source,sha,fleet,0,ready,proc_root=tmp_path/'proc')
    with pytest.raises(RuntimeError):a.gateway_identity(source,sha,fleet,4,ready,expected_pid=101,proc_root=tmp_path/'proc')
    code.write_text('changed\n')
    with pytest.raises(ValueError,match='source changed'):a.verify_release(source,manifest,sha)


def test_durable_intent_disallows_even_identical_launch_retry(tmp_path):
    path=tmp_path/'intent.json';durable_intent(path,{'launch':'exact'})
    with pytest.raises(FileExistsError):durable_intent(path,{'launch':'exact'})


def test_retained8_cloud_query_is_inspect_only():
    site=object.__new__(RetainedResidentSite);site.retained=dict(allocation_owner_run_id=a.OWNER,gpu_uuids=list(range(8)))
    site.site={'broker_python':'/python'};site.json=Mock(return_value={'present':True})
    site.allocation('inspect')
    argv=site.json.call_args.args[0]
    assert argv[-2:]==['--gpus','8'] and argv[2]=='inspect'
    for action in ('create','delete','cancel','scale','suspend'):
        with pytest.raises(ValueError):site.allocation(action)


def test_prepare_copies_only_actual8_source_bound_gateway_identities(fixed_inputs,fleet):
    _,config,_,_=continuous_inputs(fixed_inputs)
    config['schema']='roborender.cross_gpu_dagger_continuous_fixed100.leo.v2'
    config['h200'].update(render_student_worker_count=8,resource_shape='custom.gpu.8xh200',gateway_pids=[],
        gateway_identities=[],actor_gateway_slots={'0':[0,1,2,3],'1':[4,5,6,7]})
    identities=[dict(slot=i,pid=100+i,gateway_instance_id=f'{i:064x}',gpu_uuid=r['gpu_uuid'],
        source_sha256=config['source_sha256']) for i,r in enumerate(fleet['workers'])]
    receipt=dict(source_sha256=config['source_sha256'],resource_shape='custom.gpu.8xh200',
        reservation='cosmos-solution-architect-xcwtffiq',allow_burst=False,genuine_backends=True,owned_control_plane=True,
        service_state_identity='a'*64,gateway_pids=list(range(100,108)),gateway_identities=identities,
        gpu_uuids=[r['gpu_uuid'] for r in identities],retained_renderer_attachment=dict(schema=a.SCHEMA,
        renderers=fleet['workers'],renderer_restarts=0,allocation_retained=True,compute_release_authorized=False))
    bound=bind_preparation(config,receipt)
    assert bound['h200']['gateway_identities']==identities and config['h200']['gateway_identities']==[]
    bad=deepcopy(receipt);bad['gateway_identities'][4]['gpu_uuid']=identities[0]['gpu_uuid']
    with pytest.raises((ValueError,RuntimeError)):bind_preparation(config,bad)
    with pytest.raises(RuntimeError):bind_preparation(config,dict(receipt,resource_shape='custom.gpu.4xh200'))
