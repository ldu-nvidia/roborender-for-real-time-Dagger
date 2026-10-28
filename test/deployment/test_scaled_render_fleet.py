"""CPU-only fleet lifecycle/identity tests; never instantiate Image3F or CUDA."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.deployment import scaled_render_fleet as fleet


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet, 'REPO', tmp_path)
    root = tmp_path / 'runtime/evaluation-b16-20261024/renderer8'
    root.mkdir(parents=True)
    value = dict(schema=fleet.SCHEMA, allocation_name=fleet.ALLOCATION, allocation_owner=fleet.OWNER,
        hostname=fleet.ALLOCATION+'-test-replica', node_id='node-test', source_sha256=fleet.SOURCE_SHA,
        replica_id=fleet.ALLOCATION+'-test-replica', boot_id='00000000-0000-0000-0000-000000000000',
        source_root='/frozen', source_manifest='/manifest', source_manifest_raw_sha256='d'*64,
        generate_source_sha256='g'.replace('g','a')*64, launcher_sha256=fleet.sha(fleet.__file__),
        root=str(root), ssh_config='/pinned-ssh', automatic_restart=False,
        automatic_allocation_release=False, compute_release_authorized=False,
        renderer_identity=dict(renderer='Image3F', steps=5, cfg=1., teacache_threshold=.3),
        workers=[dict(slot=i,gpu_uuid=f'GPU-00000000-0000-0000-0000-{i:012x}',port=18560+i) for i in range(8)])
    path = root/'deployment.json'
    fleet.immutable(path,value)
    monkeypatch.setattr(fleet, 'host_guard', Mock())
    monkeypatch.setattr(fleet, 'validate_dependencies', Mock())
    return value,path


@pytest.mark.parametrize('fault',['allocation','owner','source','restart','release','count','alias','port','slot'])
def test_exact_fleet_config_and_retention(setup,fault):
    config,_=setup
    assert fleet.validate_config(config)==config
    if fault=='allocation':config['allocation_name']='existing-one-h200'
    elif fault=='owner':config['allocation_owner']='other'
    elif fault=='source':config['source_sha256']='f'*64
    elif fault=='restart':config['automatic_restart']=True
    elif fault=='release':config['compute_release_authorized']=True
    elif fault=='count':config['workers'].pop()
    elif fault=='alias':config['workers'][1]['gpu_uuid']=config['workers'][0]['gpu_uuid']
    elif fault=='port':config['workers'][0]['port']=18550
    else:config['workers'][0]['slot']=1
    with pytest.raises(ValueError):fleet.validate_config(config)


def test_unknown_gpu_client_blocks_start_without_takeover(setup,monkeypatch):
    config,path=setup
    monkeypatch.setattr(fleet,'compute_clients',lambda:['GPU-unknown, 543'])
    popen=Mock();monkeypatch.setattr(fleet.subprocess,'Popen',popen)
    with pytest.raises(ValueError,match='no takeover'):fleet.start(config,path)
    popen.assert_not_called()
    assert not (path.parent/'start-intent.json').exists()


def test_occupied_port_blocks_launch_without_cleanup(setup,monkeypatch):
    config,path=setup
    monkeypatch.setattr(fleet,'compute_clients',lambda:[])
    sock=Mock();sock.bind.side_effect=OSError('test occupied')
    monkeypatch.setattr(fleet.socket,'socket',lambda:sock)
    popen=Mock();monkeypatch.setattr(fleet.subprocess,'Popen',popen)
    with pytest.raises(OSError):fleet.start(config,path)
    popen.assert_not_called();sock.close.assert_called_once()


def test_lost_launch_ack_never_duplicates_supervisor(setup,monkeypatch):
    config,path=setup
    fleet.immutable(path.parent/'start-intent.json',dict(prior='unresolved'))
    popen=Mock();monkeypatch.setattr(fleet.subprocess,'Popen',popen)
    with pytest.raises(ValueError,match='ambiguous'):fleet.start(config,path)
    popen.assert_not_called()


@pytest.mark.parametrize('same',[True,False])
def test_prior_owner_identity_reconciles_or_fails_without_restart(setup,monkeypatch,same):
    config,path=setup
    receipt=dict(pid=123,identity=dict(start_ticks='20'))
    fleet.immutable(path.parent/'supervisor-launch.json',receipt)
    monkeypatch.setattr(fleet,'identity',lambda pid:receipt['identity'] if same else None)
    popen=Mock();monkeypatch.setattr(fleet.subprocess,'Popen',popen)
    if same:assert fleet.start(config,path)==receipt
    else:
        with pytest.raises(ValueError,match='no automatic restart'):fleet.start(config,path)
    popen.assert_not_called()


def test_persisted_null_identity_is_not_a_live_owner(setup,monkeypatch):
    config,path=setup
    fleet.immutable(path.parent/'supervisor-launch.json',dict(pid=123,identity=None))
    monkeypatch.setattr(fleet,'identity',lambda pid:None)
    popen=Mock();monkeypatch.setattr(fleet.subprocess,'Popen',popen)
    with pytest.raises(ValueError,match='no automatic restart'):fleet.start(config,path)
    popen.assert_not_called()


@pytest.mark.parametrize('fault',[None,'hostname','replica','boot','device','unreconciled'])
def test_exact_device_receipt_pins_full_node_hostname_separately(setup,fault):
    config,path=setup
    config['hostname']='computeinstance-actual-full-node'
    proof=dict(allocation_name=fleet.ALLOCATION,run_id=fleet.OWNER,hostname=config['hostname'],
        node_id=config['node_id'],replica_id=config['replica_id'],boot_id=config['boot_id'],
        gpus=[dict(uuid=row['gpu_uuid']) for row in config['workers']],
        identity_reconciliation=dict(replica_unchanged=True,ssh_host_key_pinned=True))
    if fault=='hostname':proof['hostname']='wrong-host'
    elif fault=='replica':proof['replica_id']='wrong-replica'
    elif fault=='boot':proof['boot_id']='wrong-boot'
    elif fault=='device':proof['gpus'][0]['uuid']='wrong-gpu'
    elif fault=='unreconciled':proof['identity_reconciliation']['replica_unchanged']=False
    receipt=path.parent/'device-proof.json';fleet.immutable(receipt,proof)
    config['devices']=dict(path=str(receipt),raw_sha256=fleet.sha(receipt))
    if fault:
        with pytest.raises(ValueError,match='host/device'):fleet.validate_devices(config)
    else:
        fleet.validate_config(config)
        fleet.validate_devices(config)


def test_supervisor_owns_exactly_eight_pinned_workers_no_restart_or_release(setup,monkeypatch):
    config,path=setup
    monkeypatch.setattr(fleet,'empty_workload_guard',Mock())
    monkeypatch.setattr(fleet,'identity',lambda pid:dict(start_ticks=str(pid),boot_id='test',cmdline_sha256='e'*64))
    created=[]
    def launch(argv,**kwargs):
        created.append((argv,kwargs))
        assert (path.parent/f'worker-{len(created)-1}'/'launch-intent.json').is_file()
        return SimpleNamespace(pid=100+len(created),poll=lambda:7)
    monkeypatch.setattr(fleet.subprocess,'Popen',launch)
    result=fleet.supervise(config,path)
    assert len(created)==8 and result['phase']=='WORKER_EXITED' and not result['automatic_restart']
    assert [row[1]['env']['CUDA_VISIBLE_DEVICES'] for row in created]==[r['gpu_uuid'] for r in config['workers']]
    assert [row[1]['env']['IMAGE3F_PORT'] for row in created]==[str(18560+i) for i in range(8)]
    assert all(row[0][0]==str(fleet.PYTHON) and row[0][4]=='serve' for row in created)
    assert not result['compute_release_authorized']


def test_worker_wrong_uuid_or_unknown_gpu_client_cannot_initialize_model(setup,monkeypatch):
    config,path=setup
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','other')
    with pytest.raises(ValueError,match='CUDA UUID'):fleet.serve(config,path,0)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',config['workers'][0]['gpu_uuid'])
    (path.parent/'worker-0').mkdir()
    monkeypatch.setattr(fleet,'compute_clients',lambda:[config['workers'][0]['gpu_uuid']+', 456'])
    with pytest.raises(ValueError,match='no takeover'):fleet.serve(config,path,0)
    assert not (path.parent/'worker-0'/'owner.json').exists()


def readiness(config,slot):
    worker=config['workers'][slot]
    owner=dict(slot=slot,pid=100+slot,identity=dict(start_ticks=str(slot),boot_id='test',cmdline_sha256='a'*64),
        allocation_name=fleet.ALLOCATION,allocation_owner=fleet.OWNER,hostname=config['hostname'],
        source_sha256=config['source_sha256'],gpu_uuid=worker['gpu_uuid'],port=worker['port'],
        launcher_sha256=config['launcher_sha256'],config_sha256=fleet.sha(Path(config['root'])/'deployment.json'))
    return dict(schema='roborender.scaled-renderer-worker-readiness.v1',checked_unix=123,
        server=owner,endpoint=dict(host='127.0.0.1',port=worker['port']),renderer_identity=config['renderer_identity'],
        transport=dict(schema=fleet.CACHE_SCHEMA,server_instance=f'{slot+1:032x}',channel=f'{slot+20:032x}'),
        observation_attestation=dict(schema='roborender.image3f.observation-attestation.v1',
            backend_class='runtime.image3f_deploy.Image3FSession',generate_source_sha256=config['generate_source_sha256'],
            contract=dict(schema='roborender.dagger.student-observation.v1',mode='generated_only',renderer='Image3F',
                renderer_steps=5,renderer_cfg=1.,teacache_threshold=.3,raytraced_policy_fallback=False)),
        render_requests=0,native_steps=0,optimizer_steps=0,inference_qualified=False,
        allocation_retained=True,compute_release_authorized=False)


def test_endpoint_fleet_receipt_uses_actual_eight_identities_and_is_idempotent(setup,monkeypatch):
    config,path=setup
    values=[readiness(config,i) for i in range(8)]
    monkeypatch.setattr(fleet,'attest_worker',lambda cfg,slot:deepcopy(values[slot]))
    monkeypatch.setattr(fleet,'identity',lambda pid:values[pid-100]['server']['identity'])
    # Simulate interruption after one immutable readiness receipt was published.
    fleet.immutable(path.parent/'worker-0'/'readiness.json',values[0])
    values[0]['checked_unix']=456
    values[0]['transport']['channel']='f'*32
    result=fleet.attest(config)
    assert len(result['workers'])==8 and len(result['renderer']['endpoints'])==8
    assert not result['inference_qualified'] and not result['throughput_qualified']
    assert fleet.attest(config)==result
    assert fleet.read(path.parent/'worker-0'/'readiness.json')['checked_unix']==123
    values[0]['transport']['server_instance']='f'*32
    with pytest.raises(ValueError,match='instance changed'):fleet.attest(config)


@pytest.mark.parametrize('fault',['owner','source','native_rgb','restarted','cache'])
def test_live_metadata_handshake_checks_owner_and_generated_only(setup,monkeypatch,fault):
    config,path=setup
    value=readiness(config,0)
    owner=value['server']
    root=path.parent/'worker-0'
    fleet.immutable(root/'owner.json',owner)
    fleet.immutable(root/'launch.json',dict(pid=owner['pid'],identity=owner['identity'],config_sha256=owner['config_sha256']))
    monkeypatch.setattr(fleet,'identity',lambda pid:owner['identity'] if fault!='owner' else None)
    stream=SimpleNamespace(__enter__=lambda self:self,__exit__=lambda *args:None)
    class Stream:
        def __enter__(self):return self
        def __exit__(self,*args):return None
    monkeypatch.setattr(fleet.socket,'create_connection',lambda *args,**kwargs:Stream())
    def exchange(stream,request):
        if request['cmd']=='ping':return {'ok':True}
        if request['cmd']=='transport_open':
            return dict(ok=True,transport=dict(schema='wrong' if fault=='cache' else fleet.CACHE_SCHEMA,
                channel=request['channel'],server_instance='1'*32))
        observed=deepcopy(value['observation_attestation'])
        if fault=='source':observed['generate_source_sha256']='f'*64
        if fault=='native_rgb':observed['contract']['raytraced_policy_fallback']=True
        if fault=='restarted':monkeypatch.setattr(fleet,'identity',lambda pid:None)
        return dict(ok=True,attestation=observed)
    monkeypatch.setattr(fleet,'_exchange',exchange)
    with pytest.raises(ValueError):fleet.attest_worker(config,0)


def test_fleet_exposes_no_resource_or_workload_destruction_commands():
    text=Path(fleet.__file__).read_text()
    assert 'client.deployment' not in text and 'SIGTERM' not in text and 'SIGKILL' not in text
    assert '18550' not in text
