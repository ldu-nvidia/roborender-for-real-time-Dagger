"""Opt-in control-only strict native data routing; no SSH/GPU/compute writes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.bc_training.handoff_common import publish
from pipeline.dagger.coverage_queue import operations_for
from pipeline.deployment.retained_campaign_site import (
    RETAINED_ROUTE_SCHEMA,RetainedResidentSite,load_retained,validate_rtx_data_route,validate_rtx_ssh_config,
    validate_provider_root,
)
from pipeline.deployment.h200.beta_pod import name_for

IDENTITY='/home/ldu/.ssh/roborender_leo_h200_20261013'
OWNER='dagger-leo-bc-original'
ALIAS=name_for(OWNER,1)


def test_frozen_provider_requires_explicit_control_admission():
    validate_provider_root('/workspace/project/roborender', None, None)
    validate_provider_root('/workspace/project/roborender/runtime/source-releases/test/code', '/admission.json', 'a'*64)
    for path in ('/tmp/code', '/workspace/project/roborender/runtime/source-releases/test',
                 '/workspace/project/roborender/runtime/source-releases/test/code/nested'):
        with pytest.raises(ValueError):validate_provider_root(path, '/admission.json', 'a'*64)
    with pytest.raises(ValueError):
        validate_provider_root('/workspace/project/roborender/runtime/source-releases/test/code', None, None)


def route():
    base='/home/ldu/roborender/runtime/connectivity/retained-h200-relay-20261017'
    return dict(local_port=22035,ssh_config=dict(path=base+'/h200-relay-ssh.conf',raw_sha256='a'*64),
        known_hosts=dict(path=base+'/known_hosts',raw_sha256='b'*64))


def config(value=None):
    value=route() if value is None else value
    return (f'Host {ALIAS}\n HostName 127.0.0.1\n Port {value["local_port"]}\n User root\n'
        f' IdentityFile {IDENTITY}\n IdentitiesOnly yes\n BatchMode yes\n StrictHostKeyChecking yes\n'
        f' UserKnownHostsFile {value["known_hosts"]["path"]}\n HostKeyAlias {ALIAS}\n ConnectTimeout 12\n'
        ' ServerAliveInterval 15\n ServerAliveCountMax 3\n ControlMaster no\n ControlPath none\n RequestTTY no\n').encode()


def retained(tmp_path,*,version=2):
    conf=tmp_path/'cpu-private.conf';known=tmp_path/'known_hosts'
    conf.touch(mode=0o600);known.write_bytes(b'public-host-key-fixture\n');known.chmod(0o600)
    value=dict(schema=RETAINED_ROUTE_SCHEMA if version==2 else 'roborender.full295-retained-h200.v1',
        allocation_owner_run_id=OWNER,allocation_name=ALIAS,node_id='node-owned',
        gpu_uuids=['GPU-'+str(i) for i in range(4)],ssh_config=str(conf),known_hosts=str(known))
    if version==2:value['rtx_data_route']=route()
    return value


def test_strict_native_route_accepts_exact_single_host():
    actual=validate_rtx_ssh_config(config(),route(),ALIAS,IDENTITY)
    assert actual['hostname']=='127.0.0.1' and actual['port']=='22035'


@pytest.mark.parametrize('before,after',[
    ('HostName 127.0.0.1','HostName different'),('Port 22035','Port 22036'),
    ('StrictHostKeyChecking yes','StrictHostKeyChecking no'),('BatchMode yes','BatchMode no'),
    ('User root','User wrong'),('HostKeyAlias '+ALIAS,'HostKeyAlias wrong'),
    ('IdentityFile '+IDENTITY,'IdentityFile /other/key'),('IdentitiesOnly yes','IdentitiesOnly no'),
    ('Host '+ALIAS,'Host *'),('ConnectTimeout 12','ConnectTimeout 0'),
])
def test_wrong_target_identity_port_or_hostkey_setting_rejected(before,after):
    with pytest.raises(ValueError):validate_rtx_ssh_config(config().replace(before.encode(),after.encode()),route(),ALIAS,IDENTITY)


@pytest.mark.parametrize('extra',['ProxyCommand unsafe','ProxyJump unsafe','Include /anywhere',
    'Match exec unsafe','LocalCommand unsafe','RemoteCommand unsafe','Host another','Port 22035',
    'LocalForward 5555','IdentityAgent any'])
def test_no_proxy_hooks_includes_duplicate_hosts_or_forwarding_options(extra):
    with pytest.raises(ValueError):validate_rtx_ssh_config(config()+extra.encode()+b'\n',route(),ALIAS,IDENTITY)


def test_v2_retained_is_explicit_while_v1_remains_valid(tmp_path):
    for version in (1,2):
        value=retained(tmp_path,version=version)
        path=tmp_path/f'retained-{version}.json';path.write_text(json.dumps(value))
        assert load_retained(path,hashlib.sha256(path.read_bytes()).hexdigest())==value
    bad=retained(tmp_path);bad.pop('rtx_data_route')
    path=tmp_path/'bad.json';path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):load_retained(path,hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize('fault',['port','path','sha','extra'])
def test_route_binding_cannot_add_unapproved_paths_or_options(fault):
    value=route()
    if fault=='port':value['local_port']=22036
    elif fault=='path':value['ssh_config']['path']='/etc/ssh/ssh_config'
    elif fault=='sha':value['ssh_config']['raw_sha256']='bad'
    else:value['cloud_fallback']=True
    with pytest.raises(ValueError):validate_rtx_data_route(value)


def native_site(tmp_path):
    site=object.__new__(RetainedResidentSite);value=retained(tmp_path)
    raw=config();known=Path(value['known_hosts']).read_bytes()
    value['rtx_data_route']['ssh_config']['raw_sha256']=hashlib.sha256(raw).hexdigest()
    value['rtx_data_route']['known_hosts']['raw_sha256']=hashlib.sha256(known).hexdigest()
    blobs={value['rtx_data_route']['ssh_config']['path']:raw,value['rtx_data_route']['known_hosts']['path']:known}
    site.retained=value;site.site={'rtx_identity_file':IDENTITY};site.pod=ALIAS
    site.command=lambda argv,host:blobs[argv[-1]]
    return site,blobs


def test_actual_remote_config_and_known_host_pins_verified_before_use(tmp_path):
    site,blobs=native_site(tmp_path)
    expected=site.retained['rtx_data_route']
    assert site.native_rtx_route()==(Path(expected['ssh_config']['path']),Path(expected['known_hosts']['path']))
    blobs[expected['ssh_config']['path']]+=b'# changed bytes\n'
    with pytest.raises(RuntimeError,match='artifact changed'):site.native_rtx_route()


def test_changed_host_key_not_admitted_even_with_new_remote_hash(tmp_path):
    site,blobs=native_site(tmp_path);ref=site.retained['rtx_data_route']['known_hosts']
    blobs[ref['path']]=b'different-host-key\n';ref['raw_sha256']=hashlib.sha256(blobs[ref['path']]).hexdigest()
    with pytest.raises(RuntimeError,match='authenticated retained host key'):site.native_rtx_route()


def test_prepare_v2_uses_reviewed_route_without_cloud_proxy(tmp_path):
    site,blobs=native_site(tmp_path)
    checkpoint=tmp_path/'policy.pt';checkpoint.write_bytes(b'CPU-only-checkpoint-fixture')
    site.config={'student':{'initial_checkpoint':str(checkpoint),'initial_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}
    site.verify_sources=lambda:{'source_sha256':'c'*64};site.barrier=lambda:{};site.verify_retained=lambda:{}
    site.site.update(schema='roborender.leo-site.v1',cpu_python='/usr/bin/python3',netrc_file='/fixture/approved-netrc',
                     repository_root='/workspace/project/roborender')
    site.token=tmp_path/'token';site.token.write_text('CPU-only-placeholder');site.token.chmod(0o600)
    site.remote=Path('/owned/round');site.connection_root=Path('/owned/connections')
    site.gpu_ssh=tmp_path/'cpu-private.conf';site.retained['known_hosts']=str(tmp_path/'known_hosts')
    site.state_identity='d'*64
    ready={str(18620+i):{'pid':100+i} for i in range(4)};site.ready_bundle=lambda:ready
    old=site.command
    site.command=lambda argv,host: b'leo-rtx-6k\n' if argv==['/bin/hostname'] else old(argv,host)
    site.json=lambda *a,**k:[['python','--devices',f'cuda:{i}'] for i in range(4)]
    site.gpu_proxy=lambda *a:pytest.fail('opt-in native route must never construct cloud TTY proxy')
    published=[];supervised=[]
    site.publish=lambda path,raw,*a,**k:published.append((path,raw))
    site.supervisor=lambda *args:supervised.append(args)
    result=site.prepare()
    command=result['mse_worker_command']
    assert command[command.index('--asset-root')+1]=='/workspace/project/roborender'
    assert len(supervised)==2 and supervised[1][0]=='beta'
    assert supervised[1][2]==Path(site.retained['rtx_data_route']['ssh_config']['path'])
    assert not any('h200-ssh.conf' in str(path) or b'ProxyCommand' in raw for path,raw in published)


def test_v3_operations_require_management_but_not_unneeded_pool_overlay(tmp_path):
    site=publish(tmp_path/'site.json',{'schema':'roborender.leo-site.v1','cpu_python':'/usr/bin/python3'})
    native='c'*64
    retained_ref=publish(tmp_path/'retained.json',retained(tmp_path))
    management=publish(tmp_path/'management.json',{'native_source_sha256':native})
    runtime=dict(schema='roborender.full295-runtime-bindings.v3',site=site,retained=retained_ref,
        resident_pool={'path':'/owned/new-pool.json','sha256':'a'*64},source_sha256=native,
        control_only={'management_release':management})
    operations=operations_for(runtime,tmp_path)
    for row in operations['commands'].values():
        assert '--management-release' in row['argv'] and '--pool-control' not in row['argv']
    bad=deepcopy(runtime);bad['control_only'].clear()
    with pytest.raises(ValueError):operations_for(bad,tmp_path)
    bad=deepcopy(runtime);bad['control_only']['skip_verification']=management
    with pytest.raises(ValueError):operations_for(bad,tmp_path)
    bad=deepcopy(runtime);bad['schema']='roborender.full295-runtime-bindings.v1';bad.pop('control_only')
    full_release=operations_for(bad,tmp_path)
    assert all('--management-release' not in row['argv'] for row in full_release['commands'].values())


def test_fresh_full_release_native_route_needs_no_overlay_but_exact_source(tmp_path,monkeypatch):
    from pipeline.deployment import beta_site
    from pipeline.deployment.source_snapshot import file_record,canonical
    root=tmp_path/'full-source';root.mkdir();source=root/'controller.py';source.write_text('# CPU-only fixture\n')
    records=[file_record(root,'controller.py')]
    manifest=dict(schema='roborender.source-snapshot.v1',files=records,file_count=1,
                  source_sha256=hashlib.sha256(canonical(records)).hexdigest())
    path=tmp_path/'source-manifest.json';path.write_bytes(canonical(manifest))
    site=object.__new__(RetainedResidentSite)
    site.retained={'schema':RETAINED_ROUTE_SCHEMA};site.management_release=None
    site.site={'source_manifest':str(path)};site.config={'source_sha256':manifest['source_sha256']}
    monkeypatch.setattr(beta_site,'ROOT',root)
    monkeypatch.setattr('pipeline.deployment.source_snapshot.source_names',lambda _:['controller.py'])
    assert site.verify_sources()==manifest
    source.write_text('# changed without release\n')
    with pytest.raises(ValueError):site.verify_sources()
