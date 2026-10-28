"""Pure, stdlib-only failure classification for CPU result observers.

This module cannot submit, restart, stop, or release anything. A retryable
classification authorizes only re-observing the same accepted command after
re-attesting its owner; it never authorizes native execution or replay.

Diagnostics use a finite allowlist of public reason descriptions. Raw stderr,
exception messages, command arguments, URLs, tokens, and base64 are NEVER
copied into the returned object. Oversized stderr is sampled and its digest
scope is explicitly labelled rather than misrepresented as a full hash.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from urllib.error import HTTPError, URLError

SCHEMA = "roborender.observation-failure.v1"
MAX_STDERR_BYTES = 64 * 1024
NATIVE_PORTS = frozenset((18640, 18641))
_ERROR_TYPES = frozenset((
    "CalledProcessError", "TimeoutExpired", "TimeoutError", "ConnectionError",
    "ConnectionRefusedError", "ConnectionResetError", "BrokenPipeError",
    "URLError", "HTTPError", "OSError", "RuntimeError", "ValueError",
    "PermissionError", "FileNotFoundError", "ObservationFailure",
))
_LOCAL_NATIVE = re.compile(r"(?:127\.0\.0\.1|localhost|\[::1\])(?::|\s+port\s+)(1864[01])\b")


def _bounded(raw: bytes) -> tuple[bytes, str]:
    if len(raw) <= MAX_STDERR_BYTES:
        return raw, "full"
    half = MAX_STDERR_BYTES // 2
    return raw[:half] + raw[-half:], "bounded_head_tail"


def _stderr(exc: BaseException) -> tuple[bytes, int, str]:
    value = getattr(exc, "stderr", None)
    if isinstance(value, str):
        raw = value.encode("utf-8", errors="replace")
    elif isinstance(value, (bytes, bytearray)):
        raw = value
    else:
        raw = b""
    sample, scope = _bounded(raw)
    return sample, len(raw), scope


def _evidence(exc: BaseException, stderr: bytes) -> tuple[str, str]:
    if stderr:
        return stderr.decode("utf-8", errors="replace").lower(), "stderr"
    # Inspect, but never reproduce, a bounded exception message. This handles
    # known local RuntimeError/ValueError reasons without logging their args.
    args = getattr(exc, "args", ())
    if args and isinstance(args[0], str):
        sample, _ = _bounded(args[0].encode("utf-8", errors="replace"))
        return sample.decode("utf-8", errors="replace").lower(), "exception_message"
    reason = getattr(exc, "reason", None)
    if isinstance(reason, str):
        sample, _ = _bounded(reason.encode("utf-8", errors="replace"))
        return sample.decode("utf-8", errors="replace").lower(), "url_reason"
    return "", "typed_exception"


def classify_failure(exc: BaseException, *, channel: str = "auto", resident_port: int | None = None) -> dict:
    """Return a nonsecret observation outcome, never an action.

    ``channel`` is ``auto``, ``ssh``, or ``resident``. For a known resident
    request, pass ``resident_port`` (18640/18641) or ``channel='resident'``.
    A refusal by that qualified local endpoint requires reconciliation, unlike
    an SSH network/banner failure. Endpoint refusal alone does NOT prove the
    native process has died; ``owner_loss_confirmed`` remains false.
    """
    if channel not in ("auto", "ssh", "resident"):
        raise ValueError("known observation channel required")
    if resident_port is not None and (type(resident_port) is not int or resident_port not in NATIVE_PORTS):
        raise ValueError("known resident observation port required")
    if not isinstance(exc, BaseException):
        raise TypeError("an observation exception is required")
    raw, size, digest_scope = _stderr(exc)
    text, evidence_source = _evidence(exc, raw)
    error_type = type(exc).__name__
    code = getattr(exc, "returncode", None)
    code = code if type(code) is int and -255 <= code <= 65535 else None
    common = dict(
        schema=SCHEMA, error_type=error_type if error_type in _ERROR_TYPES else "Exception",
        returncode=code, stderr_bytes=size, stderr_sha256=hashlib.sha256(raw).hexdigest(),
        stderr_digest_scope=digest_scope, stderr_sample_bytes=len(raw),
        diagnostic_source=evidence_source, raw_stderr_persisted=False,
        native_replay_allowed=False, native_restart_allowed=False,
        compute_release_authorized=False,
    )

    def outcome(classification, reason, diagnostic, *, retry=False, confirmed=False):
        return dict(common, classification=classification, reason_code=reason,
            diagnostic=diagnostic, retryable=retry, requires_reconciliation=not retry,
            owner_loss_confirmed=confirmed)

    status = getattr(exc, "code", None) if isinstance(exc, HTTPError) else None
    if status in (401, 403) or any(marker in text for marker in (
        "permission denied", "host key verification failed", "remote host identification has changed",
        "http error 401", "http error 403", "authentication rejected",
    )):
        return outcome("authentication_error", "authentication_or_host_pin_rejected",
            "Authentication or pinned host identity was rejected; manual reconciliation is required.")

    if status in (404, 409) or any(marker in text for marker in (
        "http error 404", "http error 409", "unknown_command", "unknown command",
        "guard receipt changed", "guard summary does not reproduce", "guard summary differs",
        "guard policy/summary differs", "source binding changed", "source mismatch",
        "source/gpu identity mismatch", "identity mismatch", "payload identity changed",
        "command receipt differs", "receipt integrity", "source sha256 differs",
    )):
        return outcome("identity_or_receipt_error", "source_owner_or_receipt_binding_rejected",
            "A source, payload, accepted-command, or receipt binding was rejected; no replay is permitted.")

    resource_markers = (
        "native v2 sampling failed", "owned cgroup counters unavailable",
        "native cgroup resource counters unsafe or unavailable", "resource sampler owner mismatch",
        "invalid or unknown native v2 resource counters/psi", "owned memory cgroup is unavailable",
        "missing or ambiguous native memory psi", "inconsistent actual cgroup current/inactive_file snapshot",
        "native v2 fixed cap or working-set accounting differs",
        "owned native host-floor/swap/oom/working-set/psi guard failed",
        "production host headroom below8gib", "native resource guard failed",
        "actual successful native resource summary required",
    )
    if any(marker in text for marker in resource_markers):
        sampler = any(marker in text for marker in (
            "sampling failed", "counters unavailable", "counters unsafe or unavailable",
            "sampler owner mismatch", "cgroup is unavailable", "unknown native v2 resource counters",
            "native memory psi", "cgroup current/inactive_file snapshot",
        ))
        return outcome("resource_gate_failure",
            "resource_sampling_unavailable" if sampler else "resource_safety_gate_rejected",
            "The owned resource sampler or safety gate failed; preserve receipts and reconcile before continuation.")

    missing_container = bool(re.search(r"no such (?:container|object)(?::|\s)", text))
    explicit_owner_loss = any(marker in text for marker in (
        "native owner changed or stopped", "native owner stopped", "native owner is stopped",
        "native owner restarted", "resident owner restarted", "native owner not running",
        "native_owner_not_running", "native lifetime changed", "production native owner changed",
        "native owner exited unsuccessfully", "native container/gpu/owned runtime binding differs",
        "crash_uncertain",
    ))
    if missing_container or explicit_owner_loss:
        return outcome("native_owner_unavailable", "native_owner_missing_stopped_or_replaced",
            "The accepted native owner is missing, stopped, replaced, or uncertain; automatic native recovery is forbidden.",
            confirmed=missing_container or any(marker in text for marker in (
                "native owner stopped", "native owner is stopped", "native owner not running",
                "native_owner_not_running", "native owner exited unsuccessfully",
            )))

    reason = getattr(exc, "reason", None)
    refused = ("connection refused" in text or "connectionrefusederror" in text
               or isinstance(exc, ConnectionRefusedError) or isinstance(reason, ConnectionRefusedError))
    local_resident = resident_port in NATIVE_PORTS or channel == "resident" or bool(_LOCAL_NATIVE.search(text))
    if refused and local_resident:
        return outcome("native_owner_unavailable", "qualified_native_endpoint_refused",
            "The qualified local native endpoint refused observation; owner loss is not yet confirmed and requires reconciliation.")

    if isinstance(exc, subprocess.TimeoutExpired):
        return outcome("ssh_observation_timeout", "bounded_ssh_observation_timeout",
            "The bounded SSH observation timed out; re-observe only the same accepted command.", retry=True)
    if isinstance(exc, HTTPError):
        if status in (429, 503, 504):
            return outcome("resident_transport_unavailable", "http_observation_temporarily_unavailable",
                "The observation service is temporarily unavailable; retry only the existing read-only observation.", retry=True)
        return outcome("remote_control_check_failed", "http_application_rejected_observation",
            "The observation returned an unclassified application error; reconcile without native replay.")

    transport = any(marker in text for marker in (
        "timed out", "timeout expired", "connection reset", "connection refused", "connection closed",
        "connection unexpectedly closed", "remotedisconnected", "urlerror", "broken pipe",
        "network is unreachable", "no route to host", "banner exchange", "kex_exchange_identification",
        "connection refused by peer",
    ))
    if code == 255:
        return outcome("ssh_transport_unavailable", "ssh_network_or_banner_unavailable",
            "SSH transport or banner exchange was unavailable; retry only observation of the same owner and command.", retry=True)
    if transport or isinstance(exc, (URLError, TimeoutError, ConnectionError)):
        classification = "ssh_transport_unavailable" if channel == "ssh" else "resident_transport_unavailable"
        return outcome(classification, "observation_transport_unavailable",
            "Observation transport was unavailable; re-attest the same owner before retrying the same command read.", retry=True)
    if isinstance(exc, subprocess.CalledProcessError):
        return outcome("remote_control_check_failed", "unclassified_remote_control_failure",
            "The remote control check failed for an unclassified reason; diagnostic details were withheld and reconciliation is required.")
    return outcome("local_observer_error", "unclassified_local_observer_failure",
        "The local observer failed for an unclassified reason; diagnostic details were withheld and reconciliation is required.")
