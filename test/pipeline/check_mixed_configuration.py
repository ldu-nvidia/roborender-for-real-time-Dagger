"""Compose the exact Hydra launch and inspect the checkpoint, without Isaac."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    release,output,inputs,checkpoint=map(Path,sys.argv[1:])
    sys.path[:0]=[str(release),str(release/'pipeline/deployment/rtx_actor')]
    from pipeline.deployment.rtx_actor.materialize_isaac6 import materialize
    from pipeline.deployment.rtx_actor.qualify_native_isaac6 import command,IMAGE
    from pipeline.success_rate.mixed_episodes import make_spec,PAIR_SCHEMA
    from pipeline.success_rate.teacher_episodes import publish_status
    from pipeline.bc_collection.run_continuous import verify_source
    manifest=json.loads((release.parent/'source-manifest.json').read_text())
    verify_source(release,manifest)
    root=Path('/data1/workspaces/roborender')
    bc=root/'runs/bc-production10000-20261027-074500/request.json'
    sha=hashlib.sha256(bc.read_bytes()).hexdigest()
    output.mkdir(exist_ok=False)
    publish_status(output/'plan.json',dict(schema=PAIR_SCHEMA,seeds=[20261102,20261103],
        bc_request_sha256=sha,expected_episodes=128,source_sha256=manifest['source_sha256']))
    spec=make_spec(json.loads(bc.read_text()),sha,seed=20261102,
        inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest())
    publish_status(output/'teacher-episodes.json',spec)
    (output/'home').mkdir()
    mounts=dict(raven=root/'simulation_assets/rtx_BC_transfer',
        anchors=root/'simulation_assets/anchor_pool_2048',cache=root/'caches/isaac/leo-gpu0')
    code=release/'pipeline/dagger/rtx/code'
    overlay=output/'overlay'
    materialize(mounts['raven'],overlay,code.parent/'config/assets.manifest.json')
    cmd=command('rr-mixed-config-only',output,overlay,code,32,mode='mixed-episodes',
        deployment_host='leo-rtx-6k',physical_gpu=0,memory_gib=32,compact_host_heap=True,
        seed=20261102,storage_paths=mounts,teacher_episode_spec=output/'teacher-episodes.json',
        teacher_bc_request=bc,teacher_pair_dir=output,mixed_inputs=inputs,mixed_checkpoint=checkpoint)
    # Hydra exits before main/seeding/Isaac. No native operation is accepted.
    cfg=subprocess.run(cmd+['--cfg','job','--resolve'],capture_output=True,text=True,timeout=90)
    publish_status(output/'hydra-check.json',dict(returncode=cfg.returncode,stdout=cfg.stdout,stderr=cfg.stderr))
    if cfg.returncode:raise RuntimeError(cfg.stderr[-8000:])
    inspect="""import torch,json,yaml
resolved=yaml.safe_load(json.load(open('/dagger-runtime/hydra-check.json'))['stdout'])
domain=resolved['env']['config']['domain_rand']
assert domain==resolved['domain_rand'] and 'randomize_ctrl_delay' in domain, 'inherited randomization lost'
assert domain['image_augmentation']['enabled'] is False
p=torch.load('/mixed-assets/bootstrap.pt',map_location='cpu',weights_only=False)
s=p['policy_state_dict']
assert torch.equal(s['std'],torch.full((11,),.001))
print(json.dumps(dict(keys=list(p),parameter_tensors=len(s),std=s['std'].tolist(),cuda_initialized=torch.cuda.is_initialized())))
"""
    probe=subprocess.run(cmd[:cmd.index(IMAGE)+1]+['python','-c',inspect],capture_output=True,text=True,timeout=90)
    publish_status(output/'checkpoint-check.json',dict(returncode=probe.returncode,stdout=probe.stdout,stderr=probe.stderr))
    if probe.returncode:raise RuntimeError(probe.stderr[-8000:])
    print(json.dumps(dict(hydra_composition='PASS',checkpoint='PASS',native_started=False)))


if __name__=='__main__':main()
