"""CPU-only deterministic endpoint routing, identity, barrier and failure tests."""
from test.deployment.source_paths import ACTOR_ROOT
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(ACTOR_ROOT))
import rendered_endpoint_pool as pool_module
import rendered_success_observations as observations
import rendered_student_success_contract as contract
from test.deployment.rtx_actor.test_rendered_success_observations import fixture as observation_fixture
from test.deployment.rtx_actor.test_rendered_student_success_contract import accepted, rendering_config


def pool_config(n=4, identity=None):
    if identity is None:
        identity = dict(renderer='Image3F', adapter_sha256=observations.ADAPTER_SHA256,
            engine_sha256='e'*64, steps=5, cfg=1., teacache_threshold=.3,
            seed=42, server_rng_isolated=True)
    return dict(identity=deepcopy(identity), routing=pool_module.ROUTING, max_inflight_per_endpoint=1,
        endpoints=[dict(slot=i, host='127.0.0.1', port=18550+i,
            gpu_uuid=f'GPU-00000000-0000-0000-0000-{i:012x}', server_instance=f'{i+1:032x}',
            source_sha256='a'*64, generate_source_sha256='b'*64,
            readiness_receipt_sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(n)])


class Client:
    def __init__(self, lane, config, *, fail=False, gate=None, tracker=None, endpoint_slot=None, channel_namespace=0):
        self.lane, self.endpoint = lane, config['endpoints'][lane % len(config['endpoints']) if endpoint_slot is None else endpoint_slot]
        self.host, self.port = self.endpoint['host'], self.endpoint['port']
        self._recovery, self._recovery_timeout_s = pool_module.CACHE_SCHEMA, 75.
        self.timeout = 120.
        self._sequence, self._server_instance = 0, None
        self._channel, self._request_lock = f'{channel_namespace+lane+100:032x}', threading.Lock()
        self.identity = config['identity']
        self.fail, self.gate, self.tracker = fail, gate, tracker
        self.calls, self.closed = 0, False

    def observation_contract(self):
        return dict(schema='roborender.image3f.observation-attestation.v1',
            backend_class='runtime.image3f_deploy.Image3FSession',
            generate_source_sha256=self.endpoint['generate_source_sha256'],
            contract=dict(schema='roborender.dagger.student-observation.v1', mode='generated_only',
                renderer='Image3F', renderer_steps=5, renderer_cfg=1., teacache_threshold=.3,
                raytraced_policy_fallback=False))

    def _exchange(self, payload, *, max_response_bytes):
        request = pickle.loads(payload)
        assert request == dict(cmd='transport_open', channel=self._channel)
        return dict(ok=True, transport=dict(schema=self._recovery, channel=self._channel,
            server_instance=self.endpoint['server_instance']))

    def render(self, **request):
        self.calls += 1
        slot = self.endpoint['slot']
        if self.tracker:
            with self.tracker['lock']:
                self.tracker['active'][slot] += 1
                self.tracker['max'][slot] = max(self.tracker['max'][slot], self.tracker['active'][slot])
                self.tracker['parallel'] = max(self.tracker['parallel'], sum(self.tracker['active']))
        try:
            if self.gate:
                self.gate.wait(timeout=5)
            if self.fail:
                raise TimeoutError('test-only renderer failure')
            self._sequence += 1
            return SimpleNamespace(runtime_ms=10., views={view: np.full((240, 416, 3), self.lane, np.uint8)
                for view in observations.VIEWS})
        finally:
            if self.tracker:
                with self.tracker['lock']:
                    self.tracker['active'][slot] -= 1

    def transport_receipt(self):
        return dict(schema=self._recovery, channel=self._channel, server_instance=self._server_instance,
            logical_requests_completed=self._sequence, transport_attempts=self.calls,
            cached_replies_recovered=0, failed=False)

    def close(self):
        self.closed = True


@pytest.mark.parametrize('count', [2, 4, 8, 16])
def test_bounded_real_endpoint_parallelism_and_lane_order(count):
    config = pool_config(min(count, 4))
    workers = len(config['endpoints'])
    tracker = dict(lock=threading.Lock(), active=[0]*workers, max=[0]*workers, parallel=0)
    barrier = threading.Barrier(workers)
    clients = [Client(lane, config, gate=barrier, tracker=tracker) for lane in range(count)]
    pool = pool_module.RenderedEndpointPool(clients, config)
    try:
        results = pool.render([{} for _ in clients], 0)
        assert [int(result.views['head_color'][0, 0, 0]) for result in results] == list(range(count))
        assert tracker['max'] == [1]*workers and tracker['parallel'] == workers
        receipt = pool.receipt()
        assert pool_module.validate_pool_receipt(receipt, count=count, steps=1, renderer=config) == receipt
        assert [row['completed'] for row in receipt['steps'][0]['endpoints']] == [count//workers]*workers
    finally:
        pool.close()


def test_partial_failure_drains_but_cannot_accept_or_replay():
    config = pool_config(2)
    barrier = threading.Barrier(2)
    clients = [Client(i, config, fail=i == 0, gate=barrier if i < 2 else None) for i in range(8)]
    pool = pool_module.RenderedEndpointPool(clients, config)
    try:
        with pytest.raises(RuntimeError, match='partial failure'):
            pool.render([{}]*8, 0)
        receipt = pool.receipt()
        assert receipt['failed'] and receipt['vector_steps_completed'] == receipt['render_requests_completed'] == 0
        assert receipt['partial_step']['accepted'] is False
        assert receipt['partial_step']['lanes'][0]['state'] == 'failed'
        assert clients[2].calls == clients[4].calls == clients[6].calls == 0
        before = [client.calls for client in clients]
        with pytest.raises(RuntimeError, match='cannot retry/replay'):
            pool.render([{}]*8, 0)
        assert [client.calls for client in clients] == before
        with pytest.raises(ValueError, match='complete exact'):
            pool_module.validate_pool_receipt(receipt, count=8, steps=1, renderer=config)
    finally:
        pool.close()


def _timed_client(lane, config, **timings):
    client = Client(lane, config)
    original = client.render
    def render(**request):
        result = original(**request)
        for key, value in timings.items():
            setattr(result, key, value)
        return result
    client.render = render
    return client


def test_optional_client_timings_preserved_per_lane_not_added_to_wall_or_aggregates():
    config = pool_config(1)
    clients = [_timed_client(0, config, worker_total_ms=12.5, loopback_total_ms=15.25),
               _timed_client(1, config, worker_total_ms=0.)]
    pool = pool_module.RenderedEndpointPool(clients, config)
    try:
        pool.render([{}, {}], 0)
        value = pool.receipt()
        row0, row1 = value['steps'][0]['lanes']
        assert row0['worker_total_ms'] == 12.5 and row0['loopback_total_ms'] == 15.25
        assert row1['worker_total_ms'] == 0. and 'loopback_total_ms' not in row1
        assert not set(pool_module.CLIENT_TIMING_FIELDS) & set(value['steps'][0]['endpoints'][0])
        assert value['client_timing_semantics'] == pool_module.CLIENT_TIMING_SEMANTICS
        assert pool_module.validate_pool_receipt(value, count=2, steps=1, renderer=config) == value
        value['steps'][0]['lanes'][0]['worker_total_ms'] += 1
        with pytest.raises(ValueError, match='trace differs'):
            pool_module.validate_pool_receipt(value, count=2, steps=1, renderer=config)
    finally:
        pool.close()


def test_legacy_client_and_receipt_without_optional_timings_still_validate():
    config = pool_config(1)
    pool = pool_module.RenderedEndpointPool([Client(0, config)], config)
    try:
        pool.render([{}], 0)
        value = pool.receipt()
        assert not set(pool_module.CLIENT_TIMING_FIELDS) & set(value['steps'][0]['lanes'][0])
        del value['client_timing_semantics']
        assert pool_module.validate_pool_receipt(value, count=1, steps=1, renderer=config) == value
    finally:
        pool.close()


@pytest.mark.parametrize('key', pool_module.CLIENT_TIMING_FIELDS)
@pytest.mark.parametrize('invalid', [-1., float('nan'), float('inf'), True, '1.0', None])
def test_invalid_optional_client_timing_fails_closed_without_replay(key, invalid):
    config = pool_config(1)
    clients = [_timed_client(0, config, **{key: invalid}), Client(1, config)]
    pool = pool_module.RenderedEndpointPool(clients, config)
    try:
        with pytest.raises(RuntimeError, match='partial failure'):
            pool.render([{}, {}], 0)
        assert clients[0].calls == 1 and clients[1].calls == 0
        assert pool.receipt()['vector_steps_completed'] == 0
        with pytest.raises(RuntimeError, match='cannot retry/replay'):
            pool.render([{}, {}], 0)
        assert clients[0].calls == 1 and clients[1].calls == 0
    finally:
        pool.close()


@pytest.mark.parametrize('fault', ['missing_semantics', 'false_semantics', 'invalid_duration'])
def test_optional_client_timing_requires_correct_semantics_and_valid_duration(fault):
    config = pool_config(1)
    pool = pool_module.RenderedEndpointPool([_timed_client(0, config, worker_total_ms=12.)], config)
    try:
        pool.render([{}], 0)
        value = pool.receipt()
        if fault == 'missing_semantics':
            del value['client_timing_semantics']
        elif fault == 'false_semantics':
            value['client_timing_semantics']['worker_total_ms'] = 'gpu_kernel_time'
        else:
            value['steps'][0]['lanes'][0]['worker_total_ms'] = -1.
        with pytest.raises(ValueError, match='optional client timing'):
            pool_module.validate_pool_receipt(value, count=1, steps=1, renderer=config)
    finally:
        pool.close()


@pytest.mark.parametrize('fault', ['instance', 'source', 'rgb', 'port', 'cache', 'deadline', 'sequence', 'duplicate_channel'])
def test_bad_live_identity_fails_before_any_generation(fault):
    config = pool_config(2)
    clients = [Client(i, config) for i in range(2)]
    c = clients[0]
    if fault == 'instance':
        c._exchange = lambda *a, **kw: dict(ok=True, transport=dict(schema=pool_module.CACHE_SCHEMA,
            channel=c._channel, server_instance='9'*32))
    elif fault in {'source', 'rgb'}:
        original = c.observation_contract
        def wrong():
            value = original()
            if fault == 'source': value['generate_source_sha256'] = '0'*64
            else: value['contract']['raytraced_policy_fallback'] = True
            return value
        c.observation_contract = wrong
    elif fault == 'port': c.port += 1
    elif fault == 'cache': c._recovery = None
    elif fault == 'deadline': c.timeout = float('inf')
    elif fault == 'sequence': c._sequence = 1
    else: clients[1]._channel = c._channel
    with pytest.raises((ValueError, RuntimeError)):
        pool_module.RenderedEndpointPool(clients, config)
    assert not any(client.calls for client in clients)


@pytest.mark.parametrize('fault', ['gpu', 'port', 'instance', 'source', 'slots', 'concurrency', 'route'])
def test_pool_configuration_cannot_mask_single_gpu_or_drift(fault):
    config = pool_config()
    if fault in {'gpu', 'port', 'instance'}:
        key = {'gpu': 'gpu_uuid', 'port': 'port', 'instance': 'server_instance'}[fault]
        config['endpoints'][1][key] = config['endpoints'][0][key]
    elif fault == 'source': config['endpoints'][1]['source_sha256'] = 'f'*64
    elif fault == 'slots': config['endpoints'][0]['slot'] = 1
    elif fault == 'concurrency': config['max_inflight_per_endpoint'] = 16
    else: config['routing'] = 'first_available_or_native_rgb'
    with pytest.raises(ValueError):
        pool_module.validate_pool_config(config)


@pytest.mark.parametrize('fault', ['endpoint', 'lane', 'sequence', 'timing', 'count', 'trace'])
def test_tampered_receipt_rejected(fault):
    config = pool_config(2)
    pool = pool_module.RenderedEndpointPool([Client(i, config) for i in range(4)], config)
    pool.render([{}]*4, 0)
    value = pool.receipt()
    pool.close()
    if fault == 'endpoint': value['endpoints'][0]['source_sha256'] = 'e'*64
    elif fault == 'lane': value['lane_bindings'][0]['endpoint_slot'] = 1
    elif fault == 'sequence': value['steps'][0]['lanes'][0]['sequence'] = 5
    elif fault == 'timing': value['steps'][0]['endpoints'][0]['elapsed_seconds'] += 1
    elif fault == 'count': value['steps'][0]['endpoints'][0]['completed'] += 1
    else: value['trace_sha256'] = '0'*64
    with pytest.raises(ValueError):
        pool_module.validate_pool_receipt(value, count=4, steps=1, renderer=config)


def test_post_handshake_restart_fails_without_rerouting():
    config = pool_config(1)
    clients = [Client(i, config) for i in range(2)]
    pool = pool_module.RenderedEndpointPool(clients, config)
    clients[0]._server_instance = 'f'*32
    with pytest.raises(RuntimeError, match='partial failure'):
        pool.render([{}]*2, 0)
    assert clients[1].calls == 0
    pool.close()


def test_one_vector_in_flight_no_unbounded_queue():
    config = pool_config(1)
    entered, release = threading.Event(), threading.Event()
    client = Client(0, config)
    original = client.render
    def block(**request):
        entered.set()
        assert release.wait(timeout=5)
        return original(**request)
    client.render = block
    pool = pool_module.RenderedEndpointPool([client], config)
    with ThreadPoolExecutor(max_workers=1) as executor:
        active = executor.submit(pool.render, [{}], 0)
        assert entered.wait(timeout=5)
        with pytest.raises(RuntimeError, match='only one bounded'):
            pool.render([{}], 0)
        release.set()
        active.result(timeout=5)
    assert client.calls == 1
    pool.close()


@pytest.mark.parametrize('failure', [False, True])
def test_adapter_all_lane_barrier_and_phase_timing(tmp_path, failure):
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    trainer, old, obs, _, calls, _ = observation_fixture(tmp_path, count=16)
    old.close()
    config = pool_config(4, old.renderer_identity)
    clients = [Client(i, config, fail=failure and i == 0) for i in range(16)]
    adapter = observations.GeneratedSuccessObservations(trainer, client_factory=lambda lane: clients[lane],
        anchor_files=old.anchor_files, anchor_seed=123, renderer_identity=old.renderer_identity,
        endpoint_pool_config=config)
    trainer._success_observation_transform = adapter
    try:
        if failure:
            with pytest.raises(RuntimeError, match='partial failure'):
                observations.apply_observation_transform(trainer, obs, torch.zeros(16), 0)
            assert calls['normalize'] == 0 and adapter.steps == 0
            assert adapter.receipt()['failed'] is True
        else:
            transformed = observations.apply_observation_transform(trainer, obs, torch.zeros(16), 0)
            assert transformed['vision_obs'] is not obs['vision_obs'] and calls['normalize'] == 1
            receipt = adapter.receipt()
            pool_module.validate_pool_receipt(receipt['renderer_endpoint_pool'], count=16, steps=1, renderer=config)
            pool_module.validate_timing_receipt(receipt['phase_timing'], steps=1)
            contract._validate_scaled_receipt(receipt, count=16, steps=1, renderer=config)
            receipt['renderer_transport_receipts'][0]['logical_requests_completed'] = 2
            with pytest.raises(ValueError, match='pinned lane/worker sequence'):
                contract._validate_scaled_receipt(receipt, count=16, steps=1, renderer=config)
            assert receipt['render_requests_completed'] == 16 and len(receipt['anchor_selections']) == 16
            assert all(row['selected_at_vector_step'] == 0 for row in receipt['anchor_selections'])
    finally:
        adapter.close()
        torch.set_num_threads(saved)
    assert all(client.closed for client in clients)


def test_v1_acceptance_unchanged_and_v2_requires_scaled_panel(accepted):
    original = rendering_config(accepted)
    assert contract.validate_config(original, accepted) == original
    value = deepcopy(original)
    value.update(schema=contract.SCALED_CONFIG_SCHEMA, renderer=pool_config(identity=original['renderer']['identity']))
    with pytest.raises(ValueError, match='explicit scaled panel'):
        contract.validate_config(value, accepted)


def test_v2_config_binding_and_missing_pool_receipt_fail_closed(accepted, monkeypatch):
    # The native agent owns real scaled-panel geometry. Isolate this contract's
    # opt-in boundary without manufacturing a production-valid scaled panel.
    original = rendering_config(accepted)
    legacy = contract.raytraced_shape(accepted)
    stub = SimpleNamespace(require_panel=lambda panel: panel, raytraced_shape=lambda panel: deepcopy(legacy))
    monkeypatch.setattr(contract, '_scaled_contract', lambda panel: stub)
    value = deepcopy(original)
    value.update(schema=contract.SCALED_CONFIG_SCHEMA, renderer=pool_config(identity=original['renderer']['identity']))
    assert contract.validate_config(value, accepted) == value
    assert contract.bind_observation_config(value, accepted, value['checkpoint_sha256']) == value
    with pytest.raises(ValueError, match='unchanged single'):
        contract.validate_config(original, accepted)
    from test.deployment.rtx_actor.test_rendered_student_success_contract import cohort_result
    row = cohort_result(accepted, config=value)
    with pytest.raises(ValueError, match='complete exact endpoint'):
        contract.validate_cohort_observations(row, count=2, config=value, panel=accepted, cohort_index=0)


@pytest.mark.parametrize('batch', [4, 8, 16])
@pytest.mark.parametrize('endpoints', [1, 4])
def test_real_scaled_panel_make_config_roundtrip(accepted, batch, endpoints):
    import scaled_success_contract as scaled
    template = rendering_config(accepted)
    panel = scaled.build_panel(accepted, batch_size=batch, qualification=True)
    config = contract.make_scaled_config(template, panel, template['checkpoint_sha256'],
        pool_config(endpoints, template['renderer']['identity'])['endpoints'])
    assert contract.validate_execution(panel, config['checkpoint_sha256'], config) == config
    assert contract.protocol_identity(panel, config)['renderer_endpoint_pool'] == config['renderer']
    assert len(config['renderer']['endpoints']) == endpoints


def test_actual_cached_client_recovers_on_same_pinned_worker_without_recompute(monkeypatch):
    import socket
    from pipeline.dagger.image3f_transport import CompleteReplyCache
    from pipeline.dagger.services.render_student import _Image3FLoopbackClient, Image3FConnectionClosed
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **kw: pytest.fail('CPU test opened network'))
    config = pool_config(1)
    cache = CompleteReplyCache()
    config['endpoints'][0]['server_instance'] = cache.instance
    client = _Image3FLoopbackClient('127.0.0.1', 18550,
        transport_recovery=pool_module.CACHE_SCHEMA, recovery_timeout_s=75)
    metadata = Client(0, config).observation_contract()
    generated, attempts, wires = [], [], []
    connection = 'a'*32
    def exchange(wire, **kwargs):
        nonlocal connection
        request = pickle.loads(wire)
        if request['cmd'] == 'observation_contract':
            return dict(ok=True, attestation=metadata)
        if request['cmd'] == 'transport_open':
            return dict(ok=True, transport=cache.open_channel(request['channel'], connection))
        attempts.append(1)
        wires.append(wire)
        def produce():
            generated.append(1)
            return dict(ok=True, runtime_ms=11., total_ms=12., views={view: client._encode(
                np.full((240, 416, 3), 7, np.uint8)) for view in observations.VIEWS})
        reply = cache.execute(request['binding'], request['payload'], connection, produce)
        if len(attempts) == 1:
            connection = 'b'*32
            raise Image3FConnectionClosed('reply lost after completed generation')
        return reply
    client._exchange = exchange
    client._recovery_sleep = lambda seconds: None
    pool = pool_module.RenderedEndpointPool([client], config)
    request = dict(prompt='CPU test only', seed=42,
        depth={v: np.zeros((240, 416), np.uint8) for v in observations.VIEWS},
        mask={v: np.zeros((240, 416, 3), np.uint8) for v in observations.VIEWS},
        previous={v: np.zeros((240, 416, 3), np.uint8) for v in observations.VIEWS})
    assert int(pool.render([request], 0)[0].views['head_color'][0, 0, 0]) == 7
    assert len(generated) == 1 and len(attempts) == 2 and wires[0] == wires[1]
    assert client.transport_receipt()['cached_replies_recovered'] == 1
    assert pool.receipt()['render_requests_completed'] == 1
    row = pool.receipt()['steps'][0]['lanes'][0]
    assert row['worker_total_ms'] == 12. and row['loopback_total_ms'] >= 0.
    assert row['renderer_runtime_ms'] == 11.
    pool.render([request], 1)
    assert len(generated) == 2 and pool.receipt()['render_requests_completed'] == 2
    pool_module.validate_pool_receipt(pool.receipt(), count=1, steps=2, renderer=config)
    pool.close()
    client._close()


def test_batched_cpu_transfer_preserves_original_per_lane_conditioning(tmp_path):
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    trainer, original, obs, _, _, _ = observation_fixture(tmp_path, count=16)
    config = pool_config(4, original.renderer_identity)
    clients = [Client(i, config) for i in range(16)]
    scaled = observations.GeneratedSuccessObservations(trainer, client_factory=lambda lane: clients[lane],
        anchor_files=original.anchor_files, anchor_seed=original.anchor_seed,
        renderer_identity=original.renderer_identity, endpoint_pool_config=config)
    try:
        original(obs, torch.zeros(16), 0)
        scaled(obs, torch.zeros(16), 0)
        old, new = original.receipt(), scaled.receipt()
        assert old['anchor_selections'] == new['anchor_selections']
        assert [row['inputs'] for row in old['bounded_first_inputs'][0]['lanes']] == [
            row['inputs'] for row in new['bounded_first_inputs'][0]['lanes']]
    finally:
        original.close()
        scaled.close()
        torch.set_num_threads(saved)


def test_per_action_binding_guard_does_not_copy_accumulated_step_receipts(tmp_path):
    trainer, adapter, _, *_ = observation_fixture(tmp_path)
    binding = dict(example='already admitted in this test')
    adapter.evaluation_binding = deepcopy(binding)
    trainer._rendered_success_evaluation_binding = binding
    adapter.receipt = lambda: pytest.fail('per-action guard copied full receipt')
    assert observations.require_observation_transform(trainer) is adapter
    adapter.close()


def test_timing_must_be_finite_and_sum_exact():
    value = dict(schema=pool_module.TIMING_SCHEMA, vector_steps_completed=1,
        preprocessing_seconds=1., render_wait_seconds=2., postprocessing_seconds=3.,
        steps=[dict(step=0, preprocessing_seconds=1., render_wait_seconds=2., postprocessing_seconds=3.)])
    assert pool_module.validate_timing_receipt(value, steps=1) == value
    value['render_wait_seconds'] += 1
    with pytest.raises(ValueError, match='sum'):
        pool_module.validate_timing_receipt(value, steps=1)
    value['render_wait_seconds'] = float('nan')
    with pytest.raises(ValueError):
        pool_module.validate_timing_receipt(value, steps=1)
