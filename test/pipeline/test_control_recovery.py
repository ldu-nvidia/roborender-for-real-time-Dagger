"""Regression of the real client's/controller's previously mismatched recovery.

CPU only. No Isaac, GPU allocation, production service or checkpoint changes.
"""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import urllib.error

import pytest

from pipeline.dagger.production_controller import get_with_transport_retry
from pipeline.dagger.services.client import IdempotentControlClient, ServiceClient, ServiceError, RetryableTransportError
from pipeline.dagger.services.recovery_policy import (
    CONTROL_RECOVERY_SECONDS, CONTROL_REQUEST_TIMEOUT_SECONDS, SSH_RECOVERY_SECONDS,
)


def fake_clock():
    value = [0.0]
    return value, lambda: value[0], lambda seconds: value.__setitem__(0, value[0] + seconds)


def deadline_client(clock, sleep, *, budget=CONTROL_RECOVERY_SECONDS, report=None):
    return IdempotentControlClient('http://127.0.0.1:18610', 'secret-token', timeout=15,
        attempts=None, recovery_seconds=budget, monotonic=clock, sleep=sleep,
        report=report if report is not None else lambda _: None)


def test_shared_deadline_outlives_tunnel_reconnect_and_request_but_not_lease():
    assert CONTROL_RECOVERY_SECONDS > SSH_RECOVERY_SECONDS + CONTROL_REQUEST_TIMEOUT_SECONDS
    assert CONTROL_RECOVERY_SECONDS < 300  # Independent scheduler-loss watchdog.
    from pipeline.deployment.rtx_actor import tunnel_supervisor
    assert tunnel_supervisor.RECOVERY_SECONDS == SSH_RECOVERY_SECONDS


@pytest.mark.parametrize('outage_seconds', [23.04, 60, 75])
def test_actual_controller_client_survives_recorded_outage(monkeypatch, outage_seconds):
    elapsed, clock, sleep = fake_clock()
    sent, events = [], []
    def request(self, method, path, body=None, content_type=None):
        sent.append((method, path, body, self.timeout))
        if elapsed[0] < outage_seconds:
            raise urllib.error.URLError(ConnectionRefusedError('transport unavailable'))
        return {'ok': True, 'run_id': 'same-owned-run'}
    monkeypatch.setattr(ServiceClient, '_request', request)
    client = deadline_client(clock, sleep, report=events.append)
    assert get_with_transport_retry(client, '/v1/status/same-owned-run')['ok']
    assert outage_seconds <= elapsed[0] < outage_seconds + 5
    assert len(sent) > 5  # Previous attempt-only client aborted after 12 seconds.
    assert all(row[:3] == sent[0][:3] for row in sent)
    assert events[-1]['action'] == 'control_transport_recovered'
    assert 'secret-token' not in json.dumps(events)


def test_permanent_outage_has_one_deadline_not_nested_retries(monkeypatch):
    elapsed, clock, sleep = fake_clock()
    sent, events = [], []
    def request(self, *args, **kwargs):
        sent.append(self.timeout)
        raise ConnectionRefusedError('private error body must not be logged')
    monkeypatch.setattr(ServiceClient, '_request', request)
    client = deadline_client(clock, sleep, report=events.append)
    with pytest.raises(RetryableTransportError, match='deadline exhausted'):
        get_with_transport_retry(client, '/v1/status/same-owned-run')
    assert elapsed[0] == CONTROL_RECOVERY_SECONDS
    assert max(sent) == 15 and min(sent) > 0
    assert events[-1]['action'] == 'control_transport_exhausted'
    assert events[-1]['native_replay_allowed'] is False
    assert 'private error body' not in json.dumps(events)


def test_last_socket_timeout_is_capped_to_remaining_budget(monkeypatch):
    elapsed, clock, sleep = fake_clock()
    timeouts = []
    def request(self, *args, **kwargs):
        timeouts.append(self.timeout)
        elapsed[0] += self.timeout
        raise TimeoutError('no response')
    monkeypatch.setattr(ServiceClient, '_request', request)
    client = deadline_client(clock, sleep, budget=20)
    with pytest.raises(RetryableTransportError):
        get_with_transport_retry(client, '/v1/status/same-owned-run')
    assert elapsed[0] == 20 and timeouts == [15, 4]
    assert client.timeout == 15  # Never mutate another request's socket timeout.


def test_wrapped_legacy_transport_error_is_not_misclassified(monkeypatch):
    class WrappedClient:
        calls = 0
        def get(self, path):
            self.calls += 1
            if self.calls == 1:
                raise RetryableTransportError('legacy request exhausted')
            return {'ok': True}
    delays = []
    monkeypatch.setattr('pipeline.dagger.production_controller.time.sleep', delays.append)
    client = WrappedClient()
    assert get_with_transport_retry(client, '/status', attempts=2)['ok']
    assert client.calls == 2 and delays == [1]


@pytest.mark.parametrize('status', [401, 403, 404, 409, 500])
def test_application_and_auth_errors_do_not_become_transport_retries(monkeypatch, status):
    elapsed, clock, sleep = fake_clock()
    calls = []
    def request(self, *args, **kwargs):
        calls.append(args)
        raise ServiceError(f'{status} semantic error')
    monkeypatch.setattr(ServiceClient, '_request', request)
    with pytest.raises(ServiceError):
        get_with_transport_retry(deadline_client(clock, sleep), '/status')
    assert len(calls) == 1 and elapsed[0] == 0


def test_raw_http_error_is_not_retried_either(monkeypatch):
    calls = []
    def request(self, *args, **kwargs):
        calls.append(args)
        raise urllib.error.HTTPError('http://127.0.0.1', 403, 'denied', {}, None)
    monkeypatch.setattr(ServiceClient, '_request', request)
    _, clock, sleep = fake_clock()
    with pytest.raises(urllib.error.HTTPError):
        get_with_transport_retry(deadline_client(clock, sleep), '/status')
    assert len(calls) == 1


@pytest.mark.parametrize('route', ['/v1/jobs', '/v1/iterations/advance', '/v2/evaluations/x/outcomes',
    '/v1/mse-rounds/register', '/v1/mse-rounds/record', '/v1/mse-rounds/finish-test'])
def test_recovered_post_preserves_exact_request_and_only_one_logical_effect(monkeypatch, route):
    elapsed, clock, sleep = fake_clock()
    payloads, effects = [], set()
    def request(self, method, path, body=None, content_type=None):
        payloads.append((method, path, body, content_type))
        effects.add(body)  # Server-side idempotent identity, including lost ack.
        if elapsed[0] < 23.04:
            raise ConnectionResetError('ambiguous acknowledgment')
        return {'ok': True}
    monkeypatch.setattr(ServiceClient, '_request', request)
    result = deadline_client(clock, sleep).post(route, {'job_id': 'same-job', 'x': 1})
    assert result['ok'] and len(effects) == 1
    assert len(payloads) > 5 and all(row == payloads[0] for row in payloads)


@pytest.mark.parametrize('route', ['/v1/env/step', '/v1/steps/prepare', '/v1/steps/commit', '/arbitrary'])
def test_no_control_retry_authority_for_simulator_or_unknown_routes(monkeypatch, route):
    calls = []
    monkeypatch.setattr(ServiceClient, '_request', lambda *args: calls.append(args))
    _, clock, sleep = fake_clock()
    with pytest.raises(ValueError, match='not permitted'):
        deadline_client(clock, sleep).post(route, {'action': [0]})
    assert not calls


@pytest.mark.parametrize('budget', [True, 0, -1, float('nan'), float('inf'), 121])
def test_unbounded_or_invalid_recovery_config_is_rejected(budget):
    with pytest.raises(ValueError):
        IdempotentControlClient('http://localhost', 'test', attempts=None, recovery_seconds=budget)


@pytest.mark.parametrize('route', ['/v1/jobs', '/v1/mse-rounds/register', '/v1/mse-rounds/record', '/v1/mse-rounds/finish-test'])
def test_cpu_real_http_disconnect_then_recovery_is_idempotent(route):
    """Actual HTTP sockets and client wrapper, not a mock that hides the error."""
    started = time.monotonic()
    seen, effects, events = [], set(), []
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            with lock:
                seen.append(body)
                effects.add(body)
            if time.monotonic() - started < .2:
                # Request accepted, but TCP reply lost. No learner/env is used.
                self.close_connection = True
                self.connection.close()
                return
            raw = b'{"ok":true}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = IdempotentControlClient(f'http://127.0.0.1:{server.server_port}', 'private-test',
            timeout=.3, attempts=None, recovery_seconds=3,
            sleep=lambda _: time.sleep(.06), report=events.append)
        assert client.post(route, {'job_id': 'same-owned-job'})['ok']
        assert len(seen) >= 2 and len(effects) == 1
        assert all(body == seen[0] for body in seen)
        assert events[-1]['action'] == 'control_transport_recovered'
        assert 'private-test' not in json.dumps(events)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_concurrent_request_deadlines_do_not_change_shared_client_timeout(monkeypatch):
    barrier = threading.Barrier(2)
    timeouts = []
    def request(self, *args, **kwargs):
        timeouts.append(self.timeout)
        barrier.wait(timeout=2)
        return {'ok': True}
    monkeypatch.setattr(ServiceClient, '_request', request)
    client = IdempotentControlClient('http://localhost', 'test', attempts=None, timeout=15,
        recovery_seconds=2, report=lambda _: None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: client.get('/status'), range(2)))
    assert all(r['ok'] for r in results) and client.timeout == 15
    assert all(0 < timeout <= 2 for timeout in timeouts)
