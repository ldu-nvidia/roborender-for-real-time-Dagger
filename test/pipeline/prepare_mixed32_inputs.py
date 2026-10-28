"""CPU-only asset/metadata gate. Does not stop, launch or replay native work."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import pickle
import socket
import struct
import uuid


def metadata_rpc(port, body):
    # Same framed protocol as _Image3FLoopbackClient, metadata only. The host
    # preparation environment intentionally has no Torch/Pillow/Isaac runtime.
    raw=pickle.dumps(body,protocol=pickle.HIGHEST_PROTOCOL)
    with socket.create_connection(('127.0.0.1',port),timeout=20) as stream:
        stream.sendall(struct.pack('!Q',len(raw))+raw)
        def exact(n):
            out=b''
            while len(out)<n:
                part=stream.recv(n-len(out))
                if not part:raise RuntimeError('metadata EOF')
                out+=part
            return out
        n=struct.unpack('!Q',exact(8))[0]
        if n>8192:raise RuntimeError('unbounded metadata response')
        return pickle.loads(exact(n))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fleet',type=Path)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    from pipeline.success_rate.mixed_episodes import CHECKPOINT_SHA256
    from pipeline.success_rate.teacher_episodes import publish_status
    from pipeline.deployment.rtx_actor.rendered_endpoint_pool import (
        validate_pool_config, SHARDED_ROUTING, ACTOR_SHARDS)
    if args.output.exists():raise ValueError('immutable input publication already exists')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:
        raise ValueError('wrong bootstrap checkpoint')
    fleet=json.loads(args.fleet.read_text())
    pool=fleet['renderer']
    pool.update(routing=SHARDED_ROUTING,actor_shards=ACTOR_SHARDS)
    validate_pool_config(pool)
    proofs=[]
    for endpoint in pool['endpoints']:
        value=metadata_rpc(endpoint['port'],dict(cmd='observation_contract'))
        att=value.get('attestation',{})
        expected=dict(schema='roborender.dagger.student-observation.v1',mode='generated_only',
            renderer='Image3F',renderer_steps=5,renderer_cfg=1.,teacache_threshold=.3,
            raytraced_policy_fallback=False)
        if (value.get('ok') is not True or att.get('contract')!=expected
                or att.get('backend_class')!='runtime.image3f_deploy.Image3FSession'
                or att.get('generate_source_sha256')!=endpoint['generate_source_sha256']):
            raise RuntimeError('renderer source/contract changed')
        channel=uuid.uuid4().hex
        reply=metadata_rpc(endpoint['port'],dict(cmd='transport_open',channel=channel))
        if reply!=dict(ok=True,transport=dict(schema='roborender.image3f.cached-reply.v1',
                server_instance=endpoint['server_instance'],channel=channel)):
            raise RuntimeError('renderer process identity changed')
        proofs.append(dict(slot=endpoint['slot'],attestation=att,transport=reply['transport']))
    root=Path('/data1/workspaces/roborender/simulation_assets/anchor_pool_2048')
    anchors=[]
    for path in sorted(root.glob('*.npz')):
        if path.is_symlink():raise ValueError('anchor symlink')
        anchors.append(dict(path='/assets/anchor_pool_2048/'+path.name,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if len(anchors)!=2048:raise ValueError('incomplete original anchor inventory')
    publish_status(args.output,dict(renderer=pool,anchors=anchors,
        metadata_preflight=dict(time_ns=time.time_ns(),endpoints=proofs,
            fleet_sha256=hashlib.sha256(args.fleet.read_bytes()).hexdigest(),
            generated_images=0,live_rendering_certified=False)))
    print(json.dumps(dict(ready_metadata=True,anchors=len(anchors),endpoints=len(proofs),
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
