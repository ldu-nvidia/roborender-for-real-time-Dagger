import hashlib
import json
import subprocess
from urllib.error import HTTPError, URLError

import pytest

from pipeline.deployment.observation_failure import MAX_STDERR_BYTES, classify_failure


def remote(message, code=1):
    return subprocess.CalledProcessError(code, ["hidden-command", "secret-argument"], stderr=message.encode())


@pytest.mark.parametrize("message", [
    "Error: No such container: standing-isaac-leo-prod-20261017-cache-v1-g0",
    "Error: No such object: standing-isaac-leo-prod-20261017-cache-v1-g0",
    "RuntimeError: native owner changed or stopped",
    "RuntimeError: production native owner changed",
    "RuntimeError: resident owner restarted; native state cannot be restored",
    "CRASH_UNCERTAIN",
])
def test_explicit_native_owner_unavailable_is_not_auth_or_network(message):
    result = classify_failure(remote(message))
    assert result["classification"] == "native_owner_unavailable"
    assert not result["retryable"] and result["requires_reconciliation"]


def test_missing_container_confirms_absence_but_generic_lifetime_message_does_not():
    assert classify_failure(remote("No such container: exact-owned-name"))["owner_loss_confirmed"]
    assert not classify_failure(remote("native owner changed or stopped"))["owner_loss_confirmed"]


@pytest.mark.parametrize("message,reason", [
    ("native v2 sampling failed: ValueError", "resource_sampling_unavailable"),
    ("ValueError: inconsistent actual cgroup current/inactive_file snapshot", "resource_sampling_unavailable"),
    ("RuntimeError: owned cgroup counters unavailable", "resource_sampling_unavailable"),
    ("production native cgroup resource counters unsafe or unavailable", "resource_sampling_unavailable"),
    ("owned native host-floor/swap/OOM/working-set/PSI guard failed", "resource_safety_gate_rejected"),
    ("production host headroom below8GiB", "resource_safety_gate_rejected"),
])
def test_resource_gate_errors_are_explicit_and_terminal(message, reason):
    result = classify_failure(remote(message))
    assert result["classification"] == "resource_gate_failure"
    assert result["reason_code"] == reason
    assert not result["retryable"]
    assert not result["owner_loss_confirmed"]


@pytest.mark.parametrize("message,code", [
    ("Connection closed by 10.50.145.239 port 2222", 255),
    ("Connection timed out during banner exchange", 255),
    ("kex_exchange_identification: Connection closed by remote host", 255),
    ("ssh: connect to host 127.0.0.1 port 22033: Connection refused", 255),
    ("ssh: connect to host owned-pod port 2222: Connection refused", 255),
])
def test_ssh_transport_is_retryable_but_cannot_authorize_native_actions(message, code):
    result = classify_failure(remote(message, code), channel="ssh")
    assert result["classification"] == "ssh_transport_unavailable"
    assert result["retryable"] and not result["requires_reconciliation"]
    assert not result["native_replay_allowed"] and not result["native_restart_allowed"]
    assert not result["compute_release_authorized"]


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
@pytest.mark.parametrize("port", [18640, 18641])
def test_native_loopback_refusal_requires_reconciliation_not_endless_transport_retry(host, port):
    result = classify_failure(remote(f"URLError: Connection refused: http://{host}:{port}/ready"))
    assert result["classification"] == "native_owner_unavailable"
    assert result["reason_code"] == "qualified_native_endpoint_refused"
    assert not result["retryable"] and not result["owner_loss_confirmed"]


def test_explicit_resident_context_handles_urlerror_that_omits_url():
    result = classify_failure(URLError(ConnectionRefusedError(111, "Connection refused")), resident_port=18640)
    assert result["classification"] == "native_owner_unavailable"
    assert not result["retryable"] and not result["owner_loss_confirmed"]


@pytest.mark.parametrize("message", [
    "Permission denied (publickey)", "Host key verification failed.",
    "REMOTE HOST IDENTIFICATION HAS CHANGED", "HTTP Error 401: Unauthorized", "HTTP Error 403: Forbidden",
])
def test_authentication_and_pin_failure_is_terminal_even_with_ssh255(message):
    result = classify_failure(remote(message, 255))
    assert result["classification"] == "authentication_error"
    assert not result["retryable"]


@pytest.mark.parametrize("message", [
    "native guard receipt changed", "native guard summary does not reproduce actual receipt",
    "source/GPU identity mismatch", "HTTP Error 404: Not Found", "unknown_command",
    "resident command receipt differs", "payload identity changed",
])
def test_integrity_missing_command_and_source_errors_are_not_transport(message):
    result = classify_failure(remote(message))
    assert result["classification"] == "identity_or_receipt_error"
    assert not result["retryable"]


@pytest.mark.parametrize("code,classification,retry", [
    (401, "authentication_error", False), (403, "authentication_error", False),
    (404, "identity_or_receipt_error", False), (409, "identity_or_receipt_error", False),
    (429, "resident_transport_unavailable", True), (503, "resident_transport_unavailable", True),
    (504, "resident_transport_unavailable", True), (500, "remote_control_check_failed", False),
])
def test_typed_http_reasons_do_not_need_unsafe_url_or_body(code, classification, retry):
    error = HTTPError("http://user:private-token@localhost:18640/private?token=secret", code, "secret", {}, None)
    result = classify_failure(error)
    assert result["classification"] == classification and result["retryable"] is retry
    assert "private-token" not in json.dumps(result) and "?token" not in json.dumps(result)


def test_timeout_is_bounded_observation_only():
    result = classify_failure(subprocess.TimeoutExpired(["ssh", "secret-token"], 12))
    assert result["classification"] == "ssh_observation_timeout" and result["retryable"]
    assert "secret-token" not in json.dumps(result)


def test_unknown_failure_stays_terminal_and_diagnostics_never_copy_secret_text():
    secret = "nvapi-PRIVATE-TOKEN bearer-private abcdef0123456789 " + "QWxhZGRpbjpvcGVuIHNlc2FtZQ==" * 40
    result = classify_failure(remote("arbitrary remote exception " + secret))
    assert result["classification"] == "remote_control_check_failed" and not result["retryable"]
    serialized = json.dumps(result)
    for value in ("nvapi-PRIVATE-TOKEN", "bearer-private", "QWxhZGRpbjpvcGVuIHNlc2FtZQ==", "secret-argument", "arbitrary remote exception"):
        assert value not in serialized
    assert result["stderr_sha256"] == hashlib.sha256(("arbitrary remote exception " + secret).encode()).hexdigest()
    assert not result["raw_stderr_persisted"]


def test_known_failure_keeps_only_allowlisted_public_diagnostic():
    result = classify_failure(remote("RuntimeError: native owner changed or stopped\n" + "nvapi-DO-NOT-LOG" * 500))
    assert result["classification"] == "native_owner_unavailable"
    assert "nvapi-" not in json.dumps(result)
    assert len(result["diagnostic"]) < 256


def test_oversized_stderr_is_bounded_and_digest_scope_is_truthful():
    body = b"secret-head" + b"x" * (MAX_STDERR_BYTES * 4) + b"\nRuntimeError: native owner changed or stopped"
    result = classify_failure(subprocess.CalledProcessError(1, "hidden", stderr=body))
    assert result["classification"] == "native_owner_unavailable"
    assert result["stderr_bytes"] == len(body) and result["stderr_sample_bytes"] == MAX_STDERR_BYTES
    assert result["stderr_digest_scope"] == "bounded_head_tail"
    assert "secret-head" not in json.dumps(result)


def test_exception_message_is_inspected_but_never_reproduced():
    result = classify_failure(ValueError("native v2 sampling failed: ValueError secret-password"))
    assert result["classification"] == "resource_gate_failure"
    assert result["diagnostic_source"] == "exception_message"
    assert "secret-password" not in json.dumps(result)


def test_untrusted_exception_type_name_does_not_become_a_secret_channel():
    malicious = type("NVAPI_PRIVATE_TOKEN_AS_A_CLASS_NAME", (Exception,), {})
    result = classify_failure(malicious("do-not-log"))
    assert result["error_type"] == "Exception"
    assert "PRIVATE_TOKEN" not in json.dumps(result)


@pytest.mark.parametrize("kwargs", [{"channel": "untrusted-token"}, {"resident_port": 22}, {"resident_port": True}])
def test_invalid_observation_context_is_rejected_without_echoing_inputs(kwargs):
    with pytest.raises(ValueError):
        classify_failure(RuntimeError("unknown"), **kwargs)
