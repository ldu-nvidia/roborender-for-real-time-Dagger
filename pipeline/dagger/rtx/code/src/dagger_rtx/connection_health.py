"""Cheap run-owned supervisor gate; never recaptures or steps the simulator."""
from __future__ import annotations
import json
from pathlib import Path
import time

from .contracts import ContractError, require_sha256
from .http_transport import TransientTransportError
from .window_runner import TransportRetryPolicy


class ConnectionHealthGate:
    def __init__(self, path: Path, *, run_id: str, config_sha256: str,
                 gateway_pids: tuple[int, ...], time_ns=time.time_ns):
        self.path, self.run_id, self.digest = Path(path), run_id, config_sha256
        self.gateway_pids, self.time_ns = tuple(gateway_pids), time_ns

    def __call__(self):
        try:
            if self.path.stat().st_size > 32768:
                raise ContractError("supervisor status exceeds its small metadata boundary")
            raw = self.path.read_bytes()
        except OSError as exc:
            raise TransientTransportError("communication supervisor status is unavailable") from exc
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise ContractError("invalid supervisor status") from exc
        if (not isinstance(value, dict) or value.get("schema") != "roborender.ssh-supervisor.v1"
                or value.get("run_id") != self.run_id or value.get("config_sha256") != self.digest):
            raise ContractError("communication supervisor identity changed")
        rows = value.get("forwards")
        if not isinstance(rows, list) or not all(isinstance(r, dict) and isinstance(r.get("expected_ready"), dict) for r in rows):
            raise ContractError("invalid supervisor forward identities")
        actual = tuple(r["expected_ready"].get("pid") for r in rows if r["expected_ready"].get("role") == "render_student")
        if not set(self.gateway_pids) <= set(actual):
            raise ContractError("communication gateway identity diverged")
        updated = value.get("updated_at_ns")
        if type(updated) is not int or updated > self.time_ns() + 1_000_000_000:
            raise ContractError("invalid supervisor heartbeat")
        if value.get("state") in {"failed", "stopped", "expired"}:
            raise ContractError("communication supervisor terminated; native restart is forbidden")
        if value.get("state") not in {"ready", "recovering"}:
            raise ContractError("unexpected supervisor state")
        if self.time_ns() - updated > 10_000_000_000:
            raise TransientTransportError("communication supervisor heartbeat is stale")
        if value["state"] != "ready" or not all(r.get("healthy") is True for r in rows):
            raise TransientTransportError("waiting for pinned SSH forwards to reconnect")


def qualification_connection_options(environ, *, run_id, gateway_pids, production=False):
    """Bind a native run to its exact run-owned tunnel supervisor.

    Transport retries reuse journaled requests; this never retries env.step.
    Keep the historical bounded qualification's 90-second budget. Production
    forwards retry independently and can take longer than that to reconnect
    (116 seconds in the October23 incident); allow a bounded five-minute
    native hold. Identity/resource failures remain immediately terminal.
    This is a fresh-source policy, never an override of an existing lifetime.
    """
    if type(production) is not bool:
        raise ContractError("production connection policy must be explicit Boolean")
    path = Path(environ.get("DAGGER_CONNECTION_STATUS", ""))
    if not path.is_absolute():
        raise ContractError("qualification requires an absolute supervisor status path")
    digest = require_sha256(environ.get("DAGGER_CONNECTION_CONFIG_SHA256", ""), "connection config")
    gate = ConnectionHealthGate(path, run_id=run_id, config_sha256=digest,
                                gateway_pids=tuple(gateway_pids))
    return {"request_guard": gate, "retry_policy": TransportRetryPolicy(
        max_attempts=180 if production else 60, initial_backoff_s=1., max_backoff_s=2.,
        max_recovery_s=300. if production else 90.)}
