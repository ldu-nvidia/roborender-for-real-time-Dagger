"""Full295 DAgger host operations on one explicitly retained H200 owner.

This opt-in provider has no allocation/create/delete path. A phase named
``release`` is only the native-quiescence barrier; compute and its authoritative
service ledger stay on the exact retained pod across all four teacher rounds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pipeline.deployment.resident_campaign_site import ResidentSite
from pipeline.deployment.beta_site import site_host
from pipeline.deployment.beta_operations import atomic
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import canonical,is_mse_production,is_corrected_smoke
from pipeline.dagger.services.contracts import file_sha256
from pipeline.deployment.h200.beta_pod import name_for

RETAINED_SCHEMA='roborender.full295-retained-h200.v1'
RETAINED_ROUTE_SCHEMA='roborender.full295-retained-h200.v2'
RETAINED_EIGHT_SCHEMA='roborender.full295-retained-h200.v3'
EIGHT_CONFIG_SCHEMA='roborender.cross_gpu_dagger_continuous_fixed100.leo.v2'


def validate_provider_root(root, management_release, management_sha256):
    """Permit frozen control code, never reinterpret its directory as assets."""
    root = Path(root).resolve()
    canonical_root = Path('/workspace/project/roborender')
    if root == canonical_root:
        return
    releases = canonical_root / 'runtime/source-releases'
    if (not root.is_relative_to(releases) or len(root.relative_to(releases).parts) != 2
            or root.name not in {'code', 'frozen'} or not management_release
            or not isinstance(management_sha256, str) or len(management_sha256) != 64
            or any(c not in '0123456789abcdef' for c in management_sha256)):
        raise ValueError('frozen provider requires an explicit verified control-only release')


def validate_eight_provider_root(root, manifest_path, source_sha256):
    """New full native source only, not a broadened control-overlay admission."""
    from pipeline.deployment.h200.retained_renderer_attachment import verify_release
    root=Path(root)
    jails=(Path('/workspace/project/roborender/runtime/source-releases'),
           Path('/workspace/project/roborender-artifacts/code_releases'))
    allowed=any(root.is_relative_to(jail) and len(root.relative_to(jail).parts)==2
                and root.name in {'code','frozen'} for jail in jails)
    if (not allowed or root.resolve(strict=True)!=root
            or Path(manifest_path)!=root.parent/'source-manifest.json'):
        raise ValueError('retained8 provider requires an exact physical frozen full-source release')
    return verify_release(root,manifest_path,source_sha256)


def validate_rtx_data_route(value, *, eight=False):
    if (not isinstance(value,dict) or set(value)!={'ssh_config','known_hosts','local_port'}
            or type(value['local_port']) is not int or value['local_port']!=(22036 if eight else 22035)):
        raise ValueError('explicit approved Leo loopback data route required')
    for name in ('ssh_config','known_hosts'):
        ref=value[name]
        if (not isinstance(ref,dict) or set(ref)!={'path','raw_sha256'}
                or not Path(ref['path']).is_absolute() or '..' in Path(ref['path']).parts
                or not Path(ref['path']).is_relative_to('/home/ldu/roborender/runtime')
                or len(ref['raw_sha256'])!=64 or any(c not in '0123456789abcdef' for c in ref['raw_sha256'])):
            raise ValueError('exact owner-scoped Leo SSH artifact binding required')
    return value


def validate_rtx_ssh_config(raw,route,allocation_name,identity_file, *, eight=False):
    """Parse a tiny single-Host native SSH route without executing ssh config.

    Includes, proxies, Match/exec, hooks, forwarding directives, wildcard hosts
    and every unreviewed option are forbidden; no cloud-TTY fallback exists.
    """
    validate_rtx_data_route(route,eight=eight)
    if not isinstance(raw,bytes) or len(raw)>65536:raise ValueError('bounded SSH config bytes required')
    expected={'host':allocation_name,'hostname':'127.0.0.1','port':str(route['local_port']),'user':'root',
        'identityfile':identity_file,'identitiesonly':'yes','batchmode':'yes','stricthostkeychecking':'yes',
        'userknownhostsfile':route['known_hosts']['path'],'hostkeyalias':allocation_name,'connecttimeout':'12'}
    optional={'serveraliveinterval':'15','serveralivecountmax':'3','controlmaster':'no',
        'controlpath':'none','requesttty':'no','exitonforwardfailure':'yes','loglevel':'ERROR',
        'passwordauthentication':'no','kbdinteractiveauthentication':'no','preferredauthentications':'publickey',
        'compression':'no','tcpkeepalive':'yes'}
    parsed={}
    for line in raw.decode('utf-8').splitlines():
        tokens=shlex.split(line,comments=True,posix=True)
        if not tokens:continue
        key=tokens[0].lower()
        if len(tokens)!=2 or key in parsed or key not in expected.keys()|optional.keys():
            raise ValueError('SSH route contains duplicate, proxy, hook or unreviewed option')
        parsed[key]=tokens[1]
    if any(parsed.get(k)!=v for k,v in expected.items()) or any(parsed[k]!=optional[k] for k in parsed.keys()-expected.keys()):
        raise ValueError('Leo SSH route target, identity or strict host-key settings differ')
    return parsed

def load_retained(path,sha256):
    path=Path(path)
    if file_sha256(path)!=sha256:raise ValueError('retained allocation binding changed')
    value=json.loads(path.read_bytes())
    fields={'schema','allocation_owner_run_id','allocation_name','node_id','gpu_uuids','ssh_config','known_hosts'}
    v3=value.get('schema')==RETAINED_EIGHT_SCHEMA
    v2=value.get('schema')==RETAINED_ROUTE_SCHEMA or v3
    count=8 if v3 else 4
    if (set(value)!=fields|({'rtx_data_route'} if v2 else set())|({'renderer_fleet'} if v3 else set())
        or value['schema'] not in (RETAINED_SCHEMA,RETAINED_ROUTE_SCHEMA,RETAINED_EIGHT_SCHEMA)
        or value['allocation_name']!=name_for(value['allocation_owner_run_id'],1)
        or len(value['gpu_uuids'])!=count or len(set(value['gpu_uuids']))!=count
        or not all(isinstance(x,str) and x.startswith('GPU-') for x in value['gpu_uuids'])
        or not isinstance(value['node_id'],str) or not value['node_id']
        or any(not Path(value[k]).is_absolute() for k in ('ssh_config','known_hosts'))):
        raise ValueError('exact versioned retained H200 owner/route binding required')
    for key in ('ssh_config','known_hosts'):
        ref=Path(value[key])
        if not ref.is_file() or ref.is_symlink() or ref.stat().st_mode&0o022:raise ValueError('unsafe retained SSH reference')
    if v2:validate_rtx_data_route(value['rtx_data_route'],eight=v3)
    if v3:
        from pipeline.deployment.h200.retained_renderer_attachment import read_bound,validate_fleet
        ref=value['renderer_fleet']
        if set(ref)!={'path','raw_sha256'}:raise ValueError('exact renderer fleet raw binding required')
        fleet=validate_fleet(read_bound(ref['path'],ref['raw_sha256']))
        if (fleet['allocation_name']!=value['allocation_name'] or fleet['allocation_owner']!=value['allocation_owner_run_id']
                or fleet['node_id']!=value['node_id'] or [r['gpu_uuid'] for r in fleet['workers']]!=value['gpu_uuids']):
            raise ValueError('retained8 fleet/owner/node/UUID mapping differs')
    return value

class RetainedResidentSite(ResidentSite):
    def __init__(self,*args,retained_file,retained_sha256,pool_control=None,pool_control_sha256=None,
                 paired_evaluation=None,paired_evaluation_sha256=None,**kwargs):
        request=args[1] if len(args)>1 else kwargs.get('request',{})
        site=args[0] if args else kwargs.get('site',{})
        if request.get('config',{}).get('schema')==EIGHT_CONFIG_SCHEMA:
            validate_eight_provider_root(ROOT,site['source_manifest'],request['config']['source_sha256'])
        else:
            validate_provider_root(ROOT, kwargs.get('management_release'), kwargs.get('management_release_sha256'))
        super().__init__(*args,**kwargs)
        if not (is_mse_production(self.config) or is_corrected_smoke(self.config)):
            raise ValueError('full295 production or corrected explicit smoke only')
        self.retained=load_retained(retained_file,retained_sha256)
        if (self.config['schema']==EIGHT_CONFIG_SCHEMA)!=(self.retained['schema']==RETAINED_EIGHT_SCHEMA):
            raise ValueError('retained8 requires explicit continuous-v2; old retained4 remains strict')
        # A fresh full release may include this opt-in route natively. Existing
        # native lifetimes instead require the existing explicit management
        # overlay; verify_sources enforces one exact source chain in both cases.
        self.pod=self.retained['allocation_name'];self.gpu_ssh=Path(self.retained['ssh_config'])
        self.bundle=self.root/'retained-services'
        self.connection_root=self.jobs/self.run/'retained-tunnels'
        self.pool_control=None
        self.paired_evaluation=None
        if (paired_evaluation is None)!=(paired_evaluation_sha256 is None):
            raise ValueError('paired evaluation requires exact path and hash')
        if paired_evaluation is not None:
            from pipeline.deployment.paired_scaled_evaluation import bound
            from pipeline.deployment.paired_scaled_runtime import validate_runtime
            self.paired_evaluation=bound(dict(path=str(paired_evaluation),raw_sha256=paired_evaluation_sha256))
            if self.config['schema']!=EIGHT_CONFIG_SCHEMA or kwargs.get('management_release') is not None:
                raise ValueError('paired v4 requires a fresh full-source eight-H200 release')
            validate_runtime(self.paired_evaluation,source_sha256=self.config['source_sha256'],
                pool=self.pool,retained=self.retained)
        if (pool_control is None)!=(pool_control_sha256 is None):
            raise ValueError('pool control requires exact path and hash')
        if pool_control is not None:
            from pipeline.bc_training.handoff_common import bound
            from pipeline.deployment.production_pool_control import validate_control
            if self.management_release is None:raise ValueError('pool helper overlay requires reviewed CPU management admission')
            self.pool_control=validate_control(bound(dict(path=str(pool_control),raw_sha256=pool_control_sha256)),
                self.config['source_sha256'])

    def verify_sources(self):
        if self.config.get('schema')==EIGHT_CONFIG_SCHEMA:
            return validate_eight_provider_root(ROOT,self.site['source_manifest'],self.config['source_sha256'])
        return super().verify_sources()

    def pool_action(self,action,commands=None):
        expected=('roborender.leo-corrected-smoke-resident-pool.v1' if is_corrected_smoke(self.config)
                  else 'roborender.leo-production-resident-pool.v1')
        if self.pool.get('schema')!=expected:
            raise ValueError('full295 execution requires actual production B2 resident startup, not legacy relabeling')
        if action not in {'status','submit','wait','result','evaluation-memory','training-memory'}:
            raise ValueError('production pool action cannot destroy native owners')
        control=getattr(self,'pool_control',None)
        if control is None:
            argv=[self.site['rtx_python'],self.source/'pipeline/deployment/rtx_actor/production_resident_pool.py']
        else:
            from pipeline.deployment.production_pool_control import REMOTE_VERIFY_AND_EXEC
            argv=[self.site['rtx_python'],'-c',REMOTE_VERIFY_AND_EXEC,canonical(control).decode()]
        argv+=['--pool',self.pool_file,'--action',action]
        if commands is not None:
            path=self.remote/(action+'-'+commands[0]['command_id']+'.json')
            self.publish(path,canonical(commands),'beta',0o444);argv+=['--commands',path]
        return self.json(argv,'beta',timeout=21900 if action=='wait' else 60)

    def allocation(self,action,keys=(),*,confirm_release=None):
        if action not in {'inspect','status'} or keys or confirm_release is not None:
            raise ValueError('retained production provider cannot mutate compute allocations')
        return self.json([self.site['broker_python'],'-',action,'--run-id',self.retained['allocation_owner_run_id'],
            '--round-id','1','--gpus',str(len(self.retained['gpu_uuids']))],'broker',data=(ROOT/'pipeline/deployment/h200/beta_pod.py').read_bytes())

    def renderer_attachment(self, *, running=False):
        """Attest the exact live retained fleet, without starting or replacing it."""
        ref=self.retained['renderer_fleet']
        argv=[self.site['cpu_python'],ROOT/'pipeline/deployment/h200/retained_renderer_attachment.py',
            '--fleet',ref['path'],'--fleet-sha256',ref['raw_sha256']]
        if running:
            ready=self.bundle/'bundle/ready.json'
            argv+=['--service-ready',ready,'--service-ready-sha256',file_sha256(ready)]
        return self.json(argv,'gpu',timeout=150)

    def verify_retained(self):
        value=self.allocation('inspect')
        if value.get('present') is not True or value.get('name')!=self.pod:raise RuntimeError('retained H200 allocation absent')
        replicas=value.get('replicas',[])
        # Replica layout varies by SDK; actual node/owner/UUIDs are also checked
        # inside the pinned SSH host, rather than guessed from a cloud hostname.
        code="""import json,os,subprocess
print(json.dumps({'owner':os.environ.get('ROBORENDER_BETA_RUN_ID'),
 'node':os.environ.get('NODE_NAME') or os.environ.get('LEPTON_NODE_NAME') or '',
 'uuids':subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()}))
"""
        host=self.json(['python3','-c',code],'gpu')
        if host['owner']!=self.retained['allocation_owner_run_id'] or host['uuids']!=self.retained['gpu_uuids']:
            raise RuntimeError('retained allocation entrypoint/device identity changed')
        nodes={r.get('node_id',r.get('node_name')) for r in replicas if isinstance(r,dict)}
        nodes.discard(None)
        if host['node']:
            if host['node']!=self.retained['node_id']:raise RuntimeError('retained node changed')
        elif self.retained['node_id'] not in nodes:
            raise RuntimeError('cloud replica did not attest the exact retained node')
        return dict(allocation_name=self.pod,allocation_owner_run_id=host['owner'],gpu_uuids=host['uuids'],
            node_id=self.retained['node_id'],allocation_retained=True,compute_release_authorized=False)

    def process(self,label,action,*,host='local',argv=None,ttl=None,env=None):
        if label not in {'gpu-services','tunnels'}:raise ValueError('production persistent process is service/tunnel only')
        directory=(self.connection_root if host=='beta' and label=='tunnels' else self.remote if host=='beta' else self.bundle)/'processes'/label
        python=self.site['rtx_python'] if host=='beta' else self.site['cpu_python']
        source=self.source if host=='beta' else ROOT
        if action=='start':
            spec=dict(schema='roborender.retained-service-process.v1',argv=list(map(str,argv)),
                cwd=str(self.source if host=='beta' else ROOT),env=env or {},owner=self.run,role=label,release_compute=False)
            self.publish(directory.with_suffix('.json'),canonical(spec),host)
        command=[python,source/'pipeline/deployment/retained_service_process.py',action,'--directory',directory]
        if action=='start':command+=['--spec',directory.with_suffix('.json')]
        return self.json(command,host)

    def ready_bundle(self,host='gpu',mode='gpu'):
        if host!='gpu' or mode!='gpu':raise ValueError('ledger must stay on retained H200 owner')
        output=self.bundle/'bundle'
        ready_path=output/'ready.json'
        eight=self.retained['schema']==RETAINED_EIGHT_SCHEMA
        if ready_path.is_file():
            value=self.read(ready_path)
            if value.get('run_id')!=self.run or value.get('state_identity')!=self.state_identity:
                raise RuntimeError('retained bundle belongs to another campaign')
            if self.process('gpu-services','status',host='gpu')['state']!='RUNNING':
                raise RuntimeError('retained bundle lost; do not silently restart its ledger')
            if eight:
                if (value.get('source_sha256')!=self.config['source_sha256']
                        or value.get('renderer_fleet_sha256')!=self.retained['renderer_fleet']['raw_sha256']
                        or value.get('renderer_start_count')!=0 or value.get('renderer_stop_authorized') is not False):
                    raise RuntimeError('retained8 attachment/source changed')
                self.renderer_attachment(running=True)
            for port,row in value['services'].items():
                live=self.json(['python3','-c',"import json,sys,urllib.request;print(urllib.request.urlopen('http://127.0.0.1:'+sys.argv[1]+'/ready',timeout=3).read().decode())",port],'gpu')
                if any(live.get(k)!=row.get(k) for k in ('pid','role','schema','gateway_instance_id') if k in row):
                    raise RuntimeError('retained service identity changed')
            return value['services']
        if eight:self.renderer_attachment()
        else:
            for _ in range(3):
                if self.command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],'gpu').strip():
                    raise RuntimeError('retained GPU clients remain; no takeover')
                time.sleep(1)
        argv=[self.site['cpu_python'],ROOT/'pipeline/deployment/h200/beta_service_host.py',
            '--source',ROOT,'--state',self.root/'services','--token',self.token,'--output',output,'--mode','gpu',
            '--run-id',self.run,'--state-identity',self.state_identity,'--netrc',self.site['netrc_file'],
            '--retained-owner',self.retained['allocation_owner_run_id'],'--continuous-services']
        if eight:
            ref=self.retained['renderer_fleet']
            argv+=['--attach-retained-renderer-fleet',ref['path'],'--renderer-fleet-sha256',ref['raw_sha256'],
                '--source-manifest',self.site['source_manifest'],'--source-sha256',self.config['source_sha256']]
        self.process('gpu-services','start',host='gpu',argv=argv)
        deadline=time.monotonic()+1320
        while time.monotonic()<deadline:
            if ready_path.is_file():return self.ready_bundle()
            if self.process('gpu-services','status',host='gpu')['state'] not in {'STARTING','RUNNING'}:
                raise RuntimeError('retained bundle failed before readiness')
            time.sleep(2)
        raise TimeoutError('retained service readiness deadline; reconcile without replay')

    def supervisor(self,host,ready,ssh_config,known):
        base=self.connection_root if host=='beta' else self.round
        ports=[18610,*range(18620,18620+len(self.retained['gpu_uuids']))] if host=='beta' else [18610,18612,18613,18614]
        cfg=dict(schema='roborender.ssh-supervisor-retained-production.v1',run_id=self.run,
            ssh_config=str(ssh_config),known_hosts=str(known),ssh_host=self.pod,lifetime_seconds=7*86400,
            forwards=[dict(local_port=p,remote_port=p,expected_ready={k:v for k,v in ready[str(p)].items()
                if k in {'ok','role','schema','pid','gateway_instance_id'}}) for p in ports])
        # One campaign owns its CPU listeners. RTX config is separately pinned
        # perround so native sidecars cannot silently retarget a gateway.
        active=self.root/'retained-services/cpu-tunnels' if host=='local' else base
        config_path=active/'supervisor.json';self.publish(config_path,canonical(cfg),host)
        if host=='local':self.publish(base/'supervisor.json',canonical(cfg))
        python,source=(self.site['rtx_python'],self.source) if host=='beta' else (self.site['cpu_python'],ROOT)
        self.process('tunnels','start',host=host,argv=[python,source/'pipeline/deployment/rtx_actor/tunnel_supervisor.py',
            '--config',config_path,'--root',active/'supervisor'])
        deadline=time.monotonic()+75
        while time.monotonic()<deadline:
            try:
                if self.read(active/'supervisor/status.json',host).get('state')=='ready':return
            except (RuntimeError,FileNotFoundError):pass
            time.sleep(1)
        raise TimeoutError('retained identity-pinned forwards not ready')

    def native_rtx_route(self):
        """Verify actual pre-provisioned Leo artifacts; never change their pins."""
        eight=self.retained['schema']==RETAINED_EIGHT_SCHEMA
        route=validate_rtx_data_route(self.retained['rtx_data_route'],eight=eight)
        code=('import pathlib,sys;p=pathlib.Path(sys.argv[1]);'
              'assert not p.is_symlink() and p.is_file() and p.stat().st_size<=65536 and not p.stat().st_mode&0o022;'
              'sys.stdout.buffer.write(p.read_bytes())')
        blobs={}
        for name in ('ssh_config','known_hosts'):
            ref=route[name];raw=self.command(['python3','-c',code,ref['path']],'beta')
            if len(raw)>65536 or hashlib.sha256(raw).hexdigest()!=ref['raw_sha256']:
                raise RuntimeError('bound Leo native SSH artifact changed')
            blobs[name]=raw
        if blobs['known_hosts']!=Path(self.retained['known_hosts']).read_bytes():
            raise RuntimeError('Leo native data route differs from authenticated retained host key')
        validate_rtx_ssh_config(blobs['ssh_config'],route,self.pod,self.site['rtx_identity_file'],eight=eight)
        return Path(route['ssh_config']['path']),Path(route['known_hosts']['path'])

    def prepare(self):
        manifest=self.verify_sources();barrier=self.barrier();retained=self.verify_retained()
        eight=self.retained['schema']==RETAINED_EIGHT_SCHEMA
        native_route=self.native_rtx_route() if self.retained['schema'] in (RETAINED_ROUTE_SCHEMA,RETAINED_EIGHT_SCHEMA) else None
        if file_sha256(Path(self.config['student']['initial_checkpoint']))!=self.config['student']['initial_checkpoint_sha256']:
            raise ValueError('MSE-selected round parent unavailable')
        if self.command(['/bin/hostname'],'beta').strip().decode()!=site_host(self.site):raise RuntimeError('wrong RTX host')
        if not self.token.is_file():raise RuntimeError('root must provision one owner-only campaign service token before prepare')
        if self.token.is_symlink() or self.token.stat().st_mode&0o077:raise ValueError('unsafe campaign token')
        ready=self.ready_bundle();pids=[ready[str(p)]['pid'] for p in range(18620,18620+len(self.retained['gpu_uuids']))]
        code="import json,pathlib,sys;print(json.dumps([pathlib.Path('/proc/'+p+'/cmdline').read_bytes().decode().split(chr(0)) for p in sys.argv[1:]]))"
        commands=self.json(['python3','-c',code,*map(str,pids)],'gpu')
        for gpu,argv in enumerate(commands):
            device='cuda:0' if eight else f'cuda:{gpu}'
            if '--fake' in argv or '--devices' not in argv or argv[argv.index('--devices')+1]!=device:
                raise RuntimeError('genuine per-GPU gateway identity differs')
        extra={}
        if eight:
            # Re-attest on the H200 owner, not by copying expected config fields.
            code="""import json,sys,pathlib
source=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(source))
from pipeline.deployment.h200.retained_renderer_attachment import read_bound,validate_fleet,verify_release,gateway_identity
verify_release(source,sys.argv[2],sys.argv[3]);fleet=validate_fleet(read_bound(sys.argv[4],sys.argv[5]))
import urllib.request
rows=[]
for slot in range(8):
 with urllib.request.urlopen('http://127.0.0.1:'+str(18620+slot)+'/ready',timeout=3) as r:ready=json.loads(r.read(16384))
 rows.append(gateway_identity(source,sys.argv[3],fleet,slot,ready))
print(json.dumps(rows))
"""
            ref=self.retained['renderer_fleet']
            process_receipts=self.json([self.site['cpu_python'],'-c',code,ROOT,self.site['source_manifest'],
                self.config['source_sha256'],ref['path'],ref['raw_sha256']],'gpu',timeout=120)
            identities=[{k:v for k,v in row.items() if k!='process_identity'} for row in process_receipts]
            if [r['pid'] for r in identities]!=pids:raise RuntimeError('gateway changed during actual retained8 preparation')
            extra=dict(gateway_identities=identities,gateway_process_receipts=process_receipts,
                renderer_fleet=ref,retained_renderer_attachment=self.renderer_attachment(running=True))
        if native_route is None:
            # Legacy behavior is unchanged unless the v2 route is explicitly
            # admitted. Opt-in native routing never enters this cloud fallback.
            remote_key=self.connection_root/'known_hosts'
            self.publish(remote_key,Path(self.retained['known_hosts']).read_bytes(),'beta')
            conf=(f'Host {self.pod}\n HostName {self.pod}\n HostKeyAlias {self.pod}\n User root\n'
                f' IdentityFile {self.site["rtx_identity_file"]}\n IdentitiesOnly yes\n BatchMode yes\n StrictHostKeyChecking yes\n'
                f' UserKnownHostsFile {remote_key}\n ConnectTimeout 12\n ProxyCommand {shlex.join(self.gpu_proxy("beta"))}\n')
            self.publish(self.connection_root/'h200-ssh.conf',conf.encode(),'beta')
            native_route=self.connection_root/'h200-ssh.conf',remote_key
        self.supervisor('local',ready,self.gpu_ssh,Path(self.retained['known_hosts']))
        self.publish(self.remote/'services.token',self.token.read_bytes(),'beta')
        self.supervisor('beta',ready,*native_route)
        return dict(source_sha256=manifest['source_sha256'],resource_shape=f"custom.gpu.{len(self.retained['gpu_uuids'])}xh200",
            reservation='cosmos-solution-architect-xcwtffiq',allow_burst=False,genuine_backends=True,
            owned_control_plane=True,gateway_pids=pids,service_state_identity=self.state_identity,token_file=str(self.token),
            mse_worker_command=[self.site['cpu_python'],str(ROOT/'pipeline/deployment/h200/mse_worker_proxy.py'),
                '--ssh-config',str(self.gpu_ssh),'--host',self.pod,'--run-id',self.retained['allocation_owner_run_id'],'--source-root',str(ROOT),
                '--asset-root',self.site['repository_root'],'--netrc-file',self.site['netrc_file']],**retained,**barrier,**extra)

    def collect(self):
        continuation=self.request.get('cohort_continuation')
        if continuation is not None:
            if not is_mse_production(self.config) or self.request.get('heldout_collection') is not None:
                raise ValueError('cohort continuation is training-only under full production')
            from pipeline.deployment.resident_production_collection import collect
            return collect(self,continuation=continuation)
        if is_corrected_smoke(self.config):
            return super().collect()
        if self.request.get('heldout_collection') is not None:
            from pipeline.deployment.resident_heldout_collection import collect
        else:
            from pipeline.deployment.resident_production_collection import collect
        return collect(self)

    def release(self,*,confirm_release=None):
        if confirm_release is not None:raise ValueError('this provider never releases compute')
        return dict(h200_allocation_retained=True,h200_allocation_released=False,same_service_ledger_owner=True,
            **self.verify_retained(),**self.barrier())

    def evaluate_explicit(self):
        from pipeline.dagger.beta_validation import load_panel
        from pipeline.deployment.rtx_actor.scaled_success_contract import is_panel
        scaled_panel=is_panel(load_panel(self.config))
        binding=getattr(self,'paired_evaluation',None)
        if binding is not None:
            if not scaled_panel:raise ValueError('paired v4 cannot evaluate a historical B2 panel')
            from pipeline.deployment.paired_scaled_runtime import evaluate
            return evaluate(self,binding)
        if scaled_panel:
            raise ValueError('scaled final test requires paired runtime v4; no GPU0/B2 fallback')
        return super().evaluate_explicit()

    def evaluate(self):return self.evaluate_explicit()
    def control_only(self):raise RuntimeError('retained production never moves a live ledger to CPU')
    def abort(self,*,confirm_release=None):
        raise RuntimeError('reconcile exact native owners manually; allocation and durable ledger retained')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('site','resident-pool','retained','request','receipt'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('site-sha256','resident-pool-sha256','retained-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--management-release',type=Path);parser.add_argument('--management-release-sha256')
    parser.add_argument('--pool-control',type=Path);parser.add_argument('--pool-control-sha256')
    parser.add_argument('--paired-evaluation',type=Path);parser.add_argument('--paired-evaluation-sha256')
    args=parser.parse_args()
    if file_sha256(args.site)!=args.site_sha256:raise ValueError('site binding changed')
    request=json.loads(args.request.read_bytes())
    site=RetainedResidentSite(json.loads(args.site.read_bytes()),request,args.resident_pool,args.resident_pool_sha256,
        retained_file=args.retained,retained_sha256=args.retained_sha256,
        pool_control=args.pool_control,pool_control_sha256=args.pool_control_sha256,
        paired_evaluation=args.paired_evaluation,paired_evaluation_sha256=args.paired_evaluation_sha256,
        management_release=args.management_release,management_release_sha256=args.management_release_sha256)
    if request['phase'] not in {'prepare','collect','release','evaluate','control_only','abort'}:raise ValueError('unknown phase')
    site.verify_sources()  # Canonical control code is pinned, even outside a frozen cwd.
    result=getattr(site,request['phase'])()
    atomic(args.receipt,dict(ok=True,operation_id=request['operation_id'],run_id=request['run_id'],
        round_id=request['round_id'],phase=request['phase'],**result))
if __name__=='__main__':main()
