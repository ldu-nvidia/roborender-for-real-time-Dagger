"""Authenticated coordinator gates for the persistent RTX production owner."""

from __future__ import annotations

import http.client
import json
from pathlib import Path
import time
from typing import Any, Mapping
from urllib.parse import urlsplit

from .contracts import canonical_json, require_sha256
from .production_driver import (
    BarrierGate,
    InitialProductionGate,
    ProductionContinuationSegment,
    ProductionDriverError,
    ProductionRunSpec,
    WindowAdvanceGate,
)


_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class CoordinatorHTTPControl:
    """Minimal polling adapter for the H200 coordinator on loopback port 18610."""

    def __init__(
        self,
        base_url: str,
        token_file: Path,
        *,
        request_timeout_s: float = 30.0,
        poll_interval_s: float = 5.0,
        wait_timeout_s: float = 7200.0,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port != 18610
            or parsed.path not in ("", "/")
        ):
            raise ValueError("production coordinator must be http://127.0.0.1:18610")
        if request_timeout_s <= 0 or poll_interval_s <= 0 or wait_timeout_s <= 0:
            raise ValueError("coordinator timeouts must be positive")
        token_file = Path(token_file)
        mode = token_file.stat().st_mode
        if mode & 0o077:
            raise PermissionError("coordinator bearer token must be owner-only")
        token = token_file.read_text(encoding="utf-8").strip()
        if not token or "\n" in token or "\r" in token:
            raise PermissionError("invalid coordinator bearer token")
        self.host = parsed.hostname
        self.port = parsed.port
        self.token = token
        self.request_timeout_s = float(request_timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self.wait_timeout_s = float(wait_timeout_s)
        self._sleep = sleep
        self._monotonic = monotonic

    def _request_once(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        if not path.startswith("/") or "?" in path or "#" in path:
            raise ProductionDriverError("invalid coordinator path")
        body = None if payload is None else canonical_json(payload)
        connection = http.client.HTTPConnection(
            self.host, self.port, timeout=self.request_timeout_s
        )
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            if body is not None:
                headers.update(
                    {
                        "Content-Type": "application/json",
                        "Content-Length": str(len(body)),
                    }
                )
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read((8 << 20) + 1)
            if len(raw) > 8 << 20:
                raise ProductionDriverError("coordinator response exceeds 8 MiB")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ProductionDriverError(
                    f"coordinator returned invalid JSON with HTTP {response.status}"
                ) from exc
            if not isinstance(value, dict):
                raise ProductionDriverError("coordinator response must be an object")
            return int(response.status), value
        finally:
            connection.close()

    def _request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        deadline = self._monotonic() + self.wait_timeout_s
        while True:
            try:
                status, value = self._request_once(method, path, payload)
            except (TimeoutError, ConnectionError, OSError, http.client.HTTPException):
                if self._monotonic() >= deadline:
                    raise ProductionDriverError("coordinator transport retry deadline expired")
                self._sleep(self.poll_interval_s)
                continue
            if 200 <= status < 300:
                return value
            if status in _TRANSIENT_STATUS and self._monotonic() < deadline:
                self._sleep(self.poll_interval_s)
                continue
            detail = str(value.get("message", value.get("error", value)))[:512]
            raise ProductionDriverError(f"coordinator HTTP {status}: {detail}")

    def _status(self, spec: ProductionRunSpec) -> dict[str, Any]:
        value = self._request("GET", f"/v1/status/{spec.run_id}")
        if value.get("ok") is not True:
            raise ProductionDriverError("coordinator status is not ok")
        run = value.get("run")
        iterations = value.get("iterations")
        barriers = value.get("barriers")
        if not isinstance(run, dict) or not isinstance(iterations, list) or not isinstance(barriers, list):
            raise ProductionDriverError("coordinator status shape mismatch")
        if (
            run.get("run_id") != spec.run_id
            or run.get("config_sha256") != spec.config_sha256
            or run.get("current_round") != spec.round_id
            or float(run.get("beta", -1)) != spec.beta
            or run.get("rollout_steps") != spec.rollout_steps
            or tuple(run.get("actor_ids", ())) != spec.actor_ids
            or run.get("initial_checkpoint_sha256") != spec.initial_checkpoint_sha256
            or run.get("validation_interval_iterations")
            != spec.validation_interval_windows
        ):
            raise ProductionDriverError("coordinator run identity diverged")
        return value

    def attest_initial(self, spec: ProductionRunSpec) -> InitialProductionGate:
        value = self._status(spec)
        run = value["run"]
        iterations = value["iterations"]
        if len(iterations) != 1 or value["barriers"]:
            raise ProductionDriverError("production run is not a fresh iteration-1 run")
        iteration = iterations[0]
        if (
            iteration.get("iteration_id") != 1
            or iteration.get("round_id") != spec.round_id
            or iteration.get("status") != "collecting"
            or iteration.get("input_checkpoint_sha256") != spec.initial_checkpoint_sha256
            or iteration.get("candidate_checkpoint_sha256") is not None
            or run.get("status") != "collecting"
            or run.get("active_checkpoint_sha256") != spec.initial_checkpoint_sha256
        ):
            raise ProductionDriverError("coordinator initial iteration identity mismatch")
        return InitialProductionGate(
            run_id=spec.run_id,
            config_sha256=spec.config_sha256,
            round_id=spec.round_id,
            active_iteration_id=1,
            active_checkpoint_sha256=spec.initial_checkpoint_sha256,
            actor_ids=spec.actor_ids,
            status="collecting",
        )

    def attest_continuation(
        self,
        spec: ProductionRunSpec,
        segment: ProductionContinuationSegment,
    ) -> InitialProductionGate:
        """Bind a fresh RTX simulator segment to an existing H200 iteration.

        This deliberately does not claim native-state restoration.  It proves
        only that all preceding learner/barrier state is durable, the current
        iteration is empty and collecting, and its immutable input checkpoint
        is exactly the one authorized by the segment manifest.
        """

        if (
            segment.run_id != spec.run_id
            or segment.config_sha256 != spec.config_sha256
        ):
            raise ProductionDriverError("continuation segment run/config mismatch")
        if segment.reset_reason in {
            "crash_uncertain",
            "operator_authorized_fresh_segment",
        } and (
            segment.h200_activity_attestation is None
            or not segment.h200_activity_attestation.is_zero
        ):
            raise ProductionDriverError(
                "unplanned continuation lacks zero-H200-activity evidence"
            )
        value = self._status(spec)
        run = value["run"]
        iterations = value["iterations"]
        barriers = value["barriers"]
        current_id = segment.start_global_window + 1
        previous_id = segment.start_global_window
        current = next(
            (row for row in iterations if row.get("iteration_id") == current_id),
            None,
        )
        previous = next(
            (row for row in iterations if row.get("iteration_id") == previous_id),
            None,
        )
        current_barriers = [
            row
            for row in barriers
            if row.get("round_id") == spec.round_id
            and row.get("iteration_id") == current_id
        ]
        previous_barriers = [
            row
            for row in barriers
            if row.get("round_id") == spec.round_id
            and row.get("iteration_id") == previous_id
        ]
        checkpoint = segment.expected_checkpoint_sha256
        if (
            run.get("status") != "collecting"
            or run.get("active_checkpoint_sha256") != checkpoint
            or not isinstance(current, dict)
            or current.get("round_id") != spec.round_id
            or current.get("status") != "collecting"
            or current.get("input_checkpoint_sha256") != checkpoint
            or current.get("candidate_checkpoint_sha256") is not None
            or current_barriers
            or not isinstance(previous, dict)
            or previous.get("round_id") != spec.round_id
            or previous.get("status") != "advanced"
            or previous.get("candidate_checkpoint_sha256") != checkpoint
            or len(previous_barriers) != spec.num_envs
        ):
            raise ProductionDriverError(
                "H200 continuation iteration/checkpoint/barrier identity mismatch"
            )
        if {row.get("actor_id") for row in previous_barriers} != set(spec.actor_ids):
            raise ProductionDriverError("H200 predecessor barrier actor set mismatch")
        return InitialProductionGate(
            run_id=spec.run_id,
            config_sha256=spec.config_sha256,
            round_id=spec.round_id,
            active_iteration_id=current_id,
            active_checkpoint_sha256=checkpoint,
            actor_ids=spec.actor_ids,
            status="collecting",
        )

    def post_window_barriers(
        self,
        spec: ProductionRunSpec,
        *,
        completed_window_index: int,
        input_checkpoint_sha256: str,
    ) -> BarrierGate:
        expected_iteration = completed_window_index + 1
        expected_chunks = tuple(
            f"d{spec.round_id}-{actor_id}-window-{completed_window_index:06d}"
            for actor_id in spec.actor_ids
        )
        last: dict[str, Any] | None = None
        for actor_id, chunk_id in zip(spec.actor_ids, expected_chunks, strict=True):
            last = self._request(
                "POST",
                "/v1/barriers",
                {
                    "run_id": spec.run_id,
                    "round_id": spec.round_id,
                    "actor_id": actor_id,
                    "checkpoint_sha256": input_checkpoint_sha256,
                    "chunk_ids": [chunk_id],
                },
            )
            if (
                last.get("ok") is not True
                or last.get("iteration_id") != expected_iteration
                or last.get("expected") != spec.num_envs
            ):
                raise ProductionDriverError("coordinator rejected an actor barrier identity")
        if last is None or last.get("arrived") != spec.num_envs or last.get("ready_for_learning") is not True:
            raise ProductionDriverError("coordinator did not reach the complete actor barrier")

        status = self._status(spec)
        rows = [
            row
            for row in status["barriers"]
            if row.get("round_id") == spec.round_id
            and row.get("iteration_id") == expected_iteration
        ]
        by_actor = {row.get("actor_id"): row for row in rows}
        if len(rows) != spec.num_envs or set(by_actor) != set(spec.actor_ids):
            raise ProductionDriverError("coordinator barrier ledger is incomplete")
        for actor_id, chunk_id in zip(spec.actor_ids, expected_chunks, strict=True):
            row = by_actor[actor_id]
            if (
                row.get("checkpoint_sha256") != input_checkpoint_sha256
                or row.get("chunk_ids") != [chunk_id]
            ):
                raise ProductionDriverError("coordinator barrier ledger diverged")
        return BarrierGate(
            run_id=spec.run_id,
            round_id=spec.round_id,
            completed_window_index=completed_window_index,
            iteration_id=expected_iteration,
            checkpoint_sha256=input_checkpoint_sha256,
            actor_ids=spec.actor_ids,
            chunk_ids=expected_chunks,
            arrived=spec.num_envs,
            expected=spec.num_envs,
            ready_for_learning=True,
        )

    def wait_for_window_advance(
        self,
        spec: ProductionRunSpec,
        *,
        completed_window_index: int,
        previous_checkpoint_sha256: str,
    ) -> WindowAdvanceGate:
        completed_iteration_id = completed_window_index + 1
        validation_boundary = (
            completed_iteration_id % spec.validation_interval_windows == 0
        )
        deadline = self._monotonic() + self.wait_timeout_s
        while True:
            value = self._status(spec)
            run = value["run"]
            iterations = value["iterations"]
            completed = next(
                (row for row in iterations if row.get("iteration_id") == completed_iteration_id),
                None,
            )
            barrier_count = sum(
                row.get("round_id") == spec.round_id
                and row.get("iteration_id") == completed_iteration_id
                for row in value["barriers"]
            )
            if run.get("status") == "learner_failed":
                raise ProductionDriverError("H200 learner failed closed")
            if isinstance(completed, dict):
                candidate = completed.get("candidate_checkpoint_sha256")
                if validation_boundary and run.get("status") in {"awaiting_validation", "validating"}:
                    if completed.get("status") not in {"awaiting_validation", "validating"}:
                        raise ProductionDriverError("final iteration validation status diverged")
                    output = require_sha256(candidate, "candidate_checkpoint_sha256")
                    return WindowAdvanceGate(
                        run_id=spec.run_id,
                        config_sha256=spec.config_sha256,
                        round_id=spec.round_id,
                        completed_window_index=completed_window_index,
                        completed_iteration_id=completed_iteration_id,
                        next_iteration_id=None,
                        output_checkpoint_sha256=output,
                        barrier_count=barrier_count,
                        actor_ids=spec.actor_ids,
                        status=str(run["status"]),
                    )
                if not validation_boundary and run.get("status") == "collecting":
                    next_id = completed_iteration_id + 1
                    following = next(
                        (row for row in iterations if row.get("iteration_id") == next_id),
                        None,
                    )
                    if completed.get("status") == "advanced" and isinstance(following, dict):
                        output = require_sha256(candidate, "candidate_checkpoint_sha256")
                        if (
                            output == previous_checkpoint_sha256
                            or following.get("status") != "collecting"
                            or following.get("input_checkpoint_sha256") != output
                            or run.get("active_checkpoint_sha256") != output
                        ):
                            raise ProductionDriverError("next checkpoint/iteration identity diverged")
                        return WindowAdvanceGate(
                            run_id=spec.run_id,
                            config_sha256=spec.config_sha256,
                            round_id=spec.round_id,
                            completed_window_index=completed_window_index,
                            completed_iteration_id=completed_iteration_id,
                            next_iteration_id=next_id,
                            output_checkpoint_sha256=output,
                            barrier_count=barrier_count,
                            actor_ids=spec.actor_ids,
                            status="collecting",
                        )
            if self._monotonic() >= deadline:
                raise ProductionDriverError("timed out waiting for H200 checkpoint advance")
            self._sleep(self.poll_interval_s)
