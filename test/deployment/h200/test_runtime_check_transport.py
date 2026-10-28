from test.deployment.source_paths import PIPELINE_ROOT
import http.client
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(PIPELINE_ROOT / "dagger/rtx/code/src"))
from dagger_rtx.contracts import ContractError
from dagger_rtx.http_transport import TransientTransportError
from pipeline.dagger.services.client import ServiceClient, ServiceError
from pipeline.deployment.h200.runtime_check import QualificationServiceClient, accepted_job


def client(guard=lambda: None):
    clock = [0.]
    return QualificationServiceClient("http://127.0.0.1:18613", "test-token", guard=guard,
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds))


def test_lost_ack_replays_identical_job_body(monkeypatch):
    calls = []
    def request(self, method, path, body, content_type):
        calls.append((method, path, body, content_type))
        if len(calls) == 1:
            raise http.client.RemoteDisconnected("lost acknowledgement")
        return {"ok": True}
    monkeypatch.setattr(ServiceClient, "_request", request)
    assert client().post("/v1/jobs", {"job_id": "immutable-job", "value": 1}) == {"ok": True}
    assert len(calls) == 2 and calls[0] == calls[1]


def test_guard_recovery_does_not_send_until_ready(monkeypatch):
    tries, calls = [], []
    def guard():
        tries.append(1)
        if len(tries) < 4:
            raise TransientTransportError("recovering")
    monkeypatch.setattr(ServiceClient, "_request", lambda *args: calls.append(args) or {"ok": True})
    assert client(guard).get("/v1/status/run") == {"ok": True}
    assert len(tries) == 4 and len(calls) == 1


@pytest.mark.parametrize("error", [ContractError("wrong identity"), ServiceError("409 body changed")])
def test_identity_and_application_errors_are_not_retried(monkeypatch, error):
    calls = []
    def request(*args):
        calls.append(args)
        raise error
    monkeypatch.setattr(ServiceClient, "_request", request)
    with pytest.raises(type(error)):
        client().get("/v1/status/run")
    assert len(calls) == 1


def test_controller_recovery_is_bounded(monkeypatch):
    calls = []
    def request(*args):
        calls.append(args)
        raise ConnectionRefusedError()
    monkeypatch.setattr(ServiceClient, "_request", request)
    with pytest.raises(RuntimeError, match="recovery expired"):
        client().get("/v1/status/run")
    assert len(calls) == 46


def test_accepted_job_visibility_retries_only_reads():
    request = {'job_id': 'test-job', 'parameters': {'iteration_id': 1}}
    clock, calls = [0.], []
    class Client:
        def get(self, path):
            calls.append(path)
            if len(calls) < 3:
                raise ServiceError('404 not_found: test-job')
            return {'job': {'job_id': 'test-job', 'request': request, 'status': 'running'}}
    job = accepted_job(Client(), 'test-job', request, monotonic=lambda: clock[0],
                       sleep=lambda s: clock.__setitem__(0, clock[0] + s))
    assert job['status'] == 'running'
    assert calls == ['/v1/jobs/test-job'] * 3


@pytest.mark.parametrize('error', ['409 body changed', '404 not_found: wrong-job'])
def test_accepted_job_does_not_retry_other_application_errors(error):
    calls = []
    class Client:
        def get(self, path):
            calls.append(path)
            raise ServiceError(error)
    with pytest.raises(ServiceError):
        accepted_job(Client(), 'test-job', {}, sleep=lambda _: pytest.fail('must not retry'))
    assert len(calls) == 1


def test_accepted_job_rejects_changed_request():
    class Client:
        def get(self, path):
            return {'job': {'job_id': 'test-job', 'request': {'different': True}}}
    with pytest.raises(RuntimeError, match='identity/request changed'):
        accepted_job(Client(), 'test-job', {})


def test_accepted_job_missing_forever_is_bounded():
    clock, calls = [0.], []
    class Client:
        def get(self, path):
            calls.append(path)
            raise ServiceError('404 not_found: test-job')
    with pytest.raises(ServiceError):
        accepted_job(Client(), 'test-job', {}, monotonic=lambda: clock[0],
                     sleep=lambda s: clock.__setitem__(0, clock[0] + s))
    assert len(calls) == 21


def test_window_receipts_are_queried_on_pinned_h200_host(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    from pipeline.deployment.h200 import runtime_check as module
    calls = []
    audit = {'window_index': 1, 'parent_checkpoint': 'a' * 64, 'committed_transitions': 648,
             'train_chunk_ids': ['example']}
    def remote(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=json.dumps(audit))
    monkeypatch.setattr(module.subprocess, 'run', remote)
    monkeypatch.setattr(module.sqlite3, 'connect', lambda *a, **kw: pytest.fail('CPU must not open H200 WAL'))
    supervisor = {'ssh_config': '/pinned/ssh_config', 'ssh_host': 'h200-owned'}
    assert module.remote_window_audit(supervisor, tmp_path, 'run', 1, 'a' * 64, 8, [1]) == audit
    assert calls[0][0][0][:6] == ['ssh', '-F', '/pinned/ssh_config', '-o', 'BatchMode=yes', 'h200-owned']
    assert calls[0][1]['timeout'] == 45
