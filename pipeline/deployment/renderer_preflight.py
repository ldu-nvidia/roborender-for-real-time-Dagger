"""Read-only Image3F contract admission before accepting native work.

Runs on RTX, through the actual renderer route. No rendering, cached channel
creation, simulator mutation, reconnect/replay, or allocation operation.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import pickle
import socket
import struct

from pipeline.deployment.rtx_actor.rendered_endpoint_pool import validate_pool_config

SCHEMA = 'roborender.renderer-route-preflight.v1'


def _receive(sock, size):
    chunks = bytearray()
    while len(chunks) < size:
        part = sock.recv(size-len(chunks))
        if not part:
            raise ConnectionError('renderer metadata response ended early')
        chunks.extend(part)
    return bytes(chunks)


def observe(endpoint, *, connect=socket.create_connection):
    # These loopback ports are pinned authenticated SSH forwards to the owned
    # renderer. The existing wire protocol is pickle; never accept arbitrary
    # remote addresses here or deserialize unbounded responses.
    if endpoint['host'] != '127.0.0.1':
        raise ValueError('preflight requires the pinned loopback renderer route')
    wire = pickle.dumps({'cmd': 'observation_contract'}, protocol=pickle.HIGHEST_PROTOCOL)
    with connect((endpoint['host'], endpoint['port']), timeout=5) as sock:
        sock.settimeout(5)
        sock.sendall(struct.pack('!Q', len(wire))+wire)
        size = struct.unpack('!Q', _receive(sock, 8))[0]
        if not 0 < size <= 8192:
            raise ValueError('renderer metadata response exceeds bound')
        return pickle.loads(_receive(sock, size))


def check_renderer_routes(renderer, *, exchange=observe):
    validate_pool_config(renderer)
    identity = renderer['identity']
    expected = dict(schema='roborender.dagger.student-observation.v1', mode='generated_only',
        renderer='Image3F', renderer_steps=identity['steps'], renderer_cfg=identity['cfg'],
        teacache_threshold=identity['teacache_threshold'], raytraced_policy_fallback=False)

    def check(endpoint):
        answer = exchange(endpoint)
        attestation = dict(schema='roborender.image3f.observation-attestation.v1',
            contract=expected, backend_class='runtime.image3f_deploy.Image3FSession',
            generate_source_sha256=endpoint['generate_source_sha256'])
        if answer != dict(ok=True, attestation=attestation):
            raise RuntimeError('live renderer generation contract/source differs; native dispatch forbidden')
        return endpoint['slot']

    with ThreadPoolExecutor(max_workers=len(renderer['endpoints'])) as pool:
        slots = list(pool.map(check, renderer['endpoints']))
    digest = hashlib.sha256(json.dumps(renderer, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return dict(schema=SCHEMA, renderer_sha256=digest, verified_slots=slots,
        metadata_only=True, native_steps=0, render_requests=0)
