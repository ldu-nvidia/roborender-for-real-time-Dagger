"""In-process v4 window pipeline with one remote hot exchange per step.

The native launcher selects this path only through a hash-bound immutable v4
run configuration after all H200 gateway process generations are attested.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .batch_owner import IsaacBatchTransitionOwner
from .boundary import build_owner_commit
from .chunks import CHUNK_LENGTH
from .contracts import (
    CONTRACT_VERSION,
    ROUND_BETAS,
    ContractError,
    StepKey,
    canonical_json,
    sha256_bytes,
)
from .http_transport import TransientTransportError
from .lean_boundary_v4 import CONTRACT_VERSION_V4
from .owner import CrashUncertain, OwnerConflict
from .phase_telemetry import PhaseBoundaryTelemetry
from .window_runner import BatchWindowRunner, RolloutLifecycle, TransportRetryPolicy


@dataclass(frozen=True)
class PendingStepV4:
    keys: tuple[StepKey, ...]
    local_prepares: tuple[Mapping[str, Any], ...]
    selections: tuple[Mapping[str, Any], ...]
    local_commits: tuple[Mapping[str, Any], ...]
    prepare_idempotency_keys: tuple[str, ...]
    commit_idempotency_keys: tuple[str, ...]
    timings_ns: Mapping[str, int]
    transport_timing: Mapping[str, int | bool]


@dataclass(frozen=True)
class TransportTimingMarkerV4:
    """Opaque cursor delimiting one synchronous sharded client call."""

    sequence: int


class TransportTimingObserverV4:
    """Measure v4 HTTP time without changing the transport or wire contract.

    ``BoundedBearerBatchV4Transport`` already reports one small event after
    every HTTP request.  This sink retains only bounded scalar timing data and
    optionally forwards the event to the existing audit sink.  A caller can
    bracket a sharded client call with ``mark``/``summarize`` to distinguish
    the slowest HTTP request from client-side framing, validation, executor,
    and telemetry overhead.  No payload bytes, authorization values, or array
    contents are retained.

    The ``client_non_http_upper_bound_ns`` value is intentionally named an
    upper bound: HTTP requests overlap across shards, so subtraction cannot
    attribute all residual time uniquely without instrumenting each client.
    """

    schema = "roborender.dagger.rtx.v4-transport-timing.v1"
    _EVENTS = {"http_batch_request_v3", "http_batch_request_v4"}

    def __init__(
        self,
        delegate: Any | None = None,
        *,
        delegate_success_events: bool = True,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
        max_events: int = 4096,
    ) -> None:
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 64:
            raise ValueError("max_events must be an integer >= 64")
        self.delegate = delegate
        self.delegate_success_events = bool(delegate_success_events)
        self.monotonic_ns = monotonic_ns
        self._lock = threading.Lock()
        self._sequence = 0
        self._events: deque[dict[str, int | str]] = deque(maxlen=max_events)

    def record(self, event: str, **detail: Any) -> None:
        observed_ns = int(self.monotonic_ns())
        delegate_started_ns = observed_ns
        # Successful per-shard events are aggregated into the one vector-step
        # hot-path row below.  Production v4 disables forwarding those events
        # to the durable legacy sink, eliminating eight serialized file+dir
        # fsync pairs from every exchange.  Failures/retries remain durable so
        # fail-closed diagnosis is unchanged.
        forward_to_delegate = (
            self.delegate is not None
            and (
                self.delegate_success_events
                or detail.get("outcome") != "success"
            )
        )
        if forward_to_delegate:
            self.delegate.record(event, **detail)
        delegate_finished_ns = int(self.monotonic_ns())
        if event not in self._EVENTS:
            return
        duration_ns = detail.get("duration_ns")
        if isinstance(duration_ns, bool) or not isinstance(duration_ns, int):
            return
        row: dict[str, int | str] = {
            "duration_ns": max(0, duration_ns),
            "observed_ns": observed_ns,
            "delegate_finished_ns": delegate_finished_ns,
            "telemetry_emit_ns": max(0, delegate_finished_ns - delegate_started_ns),
            "request_body_bytes": int(detail.get("request_body_bytes", 0)),
            "response_body_bytes": int(detail.get("response_body_bytes", 0)),
            "path": str(detail.get("path", ""))[:128],
        }
        with self._lock:
            self._sequence += 1
            row["sequence"] = self._sequence
            self._events.append(row)

    def mark(self) -> TransportTimingMarkerV4:
        with self._lock:
            return TransportTimingMarkerV4(self._sequence)

    def summarize(
        self,
        marker: TransportTimingMarkerV4,
        *,
        call_started_ns: int,
        call_finished_ns: int,
    ) -> dict[str, int | bool]:
        if call_finished_ns < call_started_ns:
            raise ValueError("client timing interval is negative")
        with self._lock:
            rows = [
                dict(row)
                for row in self._events
                if int(row["sequence"]) > marker.sequence
            ]
        wall_ns = call_finished_ns - call_started_ns
        if not rows:
            return {
                "instrumented": False,
                "client_call_wall_ns": wall_ns,
                "http_request_count": 0,
            }
        durations = [int(row["duration_ns"]) for row in rows]
        observed = [int(row["observed_ns"]) for row in rows]
        finished = [int(row["delegate_finished_ns"]) for row in rows]
        request_starts = [end - duration for end, duration in zip(observed, durations)]
        critical_rtt_ns = max(durations)
        http_timeline_ns = max(observed) - min(request_starts)
        return {
            "instrumented": True,
            "client_call_wall_ns": wall_ns,
            "http_request_count": len(rows),
            "http_request_duration_sum_ns": sum(durations),
            "http_rtt_critical_path_ns": critical_rtt_ns,
            "http_timeline_span_ns": max(0, http_timeline_ns),
            "client_non_http_upper_bound_ns": max(0, wall_ns - http_timeline_ns),
            "first_http_start_lag_ns": max(0, min(request_starts) - call_started_ns),
            "last_http_observed_tail_ns": max(0, call_finished_ns - max(observed)),
            "last_telemetry_return_tail_ns": max(0, call_finished_ns - max(finished)),
            "transport_telemetry_emit_sum_ns": sum(
                int(row["telemetry_emit_ns"]) for row in rows
            ),
            "request_body_bytes": sum(int(row["request_body_bytes"]) for row in rows),
            "response_body_bytes": sum(int(row["response_body_bytes"]) for row in rows),
        }


def prewarm_readiness_v4(
    h200: Any,
    *,
    probes: int,
    expected_gateway_pids: Sequence[int] = (),
    expected_gateway_instance_ids: Sequence[str] = (),
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Establish and re-attest persistent gateway connections without state.

    This deliberately calls only ``GET /ready`` through the already-audited
    client.  It never registers anchors, creates a prepare identity, performs
    inference, or writes a production ledger row.  Consequently it is safe to
    enable before an immutable window, but it must not be described as a CUDA
    kernel warmup unless H200 later exposes a separate disposable contract.
    """

    if isinstance(probes, bool) or not isinstance(probes, int) or not 0 <= probes <= 16:
        raise ValueError("v4 readiness prewarm probes must be an integer in [0,16]")
    if probes == 0:
        return {
            "enabled": False,
            "mode": "readiness_only",
            "probes": 0,
            "ledger_mutations": 0,
            "elapsed_ns": 0,
        }
    expected_pids = tuple(int(value) for value in expected_gateway_pids)
    expected_instances = tuple(str(value) for value in expected_gateway_instance_ids)
    started_ns = int(monotonic_ns())
    identities: tuple[str, ...] | None = None
    pids: tuple[int, ...] | None = None
    for _ in range(probes):
        ready = h200.verify_ready()
        rows = ready.get("gateways")
        if not isinstance(rows, list) or not rows:
            raise ContractError("v4 readiness prewarm requires sharded gateway identities")
        current_pids = tuple(int(row.get("pid")) for row in rows)
        current_instances = tuple(str(row.get("gateway_instance_id")) for row in rows)
        if len(set(current_instances)) != len(current_instances):
            raise ContractError("v4 readiness prewarm gateway generations are not unique")
        if pids is None:
            pids = current_pids
            identities = current_instances
        elif pids != current_pids or identities != current_instances:
            raise ContractError("v4 gateway identity changed during readiness prewarm")
        if expected_pids and current_pids != expected_pids:
            raise ContractError("v4 readiness prewarm PID attestation mismatch")
        if expected_instances and current_instances != expected_instances:
            raise ContractError("v4 readiness prewarm instance attestation mismatch")
    return {
        "enabled": True,
        "mode": "readiness_only",
        "probes": probes,
        "gateway_count": len(pids or ()),
        "gateway_pids": list(pids or ()),
        "gateway_instance_ids": list(identities or ()),
        "ledger_mutations": 0,
        "elapsed_ns": int(monotonic_ns()) - started_ns,
    }


class AggregateHotPathJournalV4:
    """One compact, non-fsynced diagnostic row per executed vector step."""

    schema = "roborender.dagger.rtx.v4-hot-path.v1"

    def __init__(self, path: Path, *, clock_ns=time.time_ns) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.clock_ns = clock_ns

    def append(self, row: Mapping[str, Any]) -> None:
        body = canonical_json({"schema": self.schema, "time_ns": int(self.clock_ns()), **row})
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, body + b"\n")
        finally:
            os.close(fd)


class PipelinedBatchStepOrchestratorV4:
    """Own the v4 cross-step pipeline around the unchanged native owner."""

    wire_contract_version = CONTRACT_VERSION_V4

    def __init__(
        self,
        owner: IsaacBatchTransitionOwner,
        h200: Any,
        *,
        num_envs: int,
        journal: AggregateHotPathJournalV4 | None = None,
        phase_telemetry: PhaseBoundaryTelemetry | None = None,
        transport_timing: TransportTimingObserverV4 | None = None,
        monotonic_ns=time.perf_counter_ns,
    ) -> None:
        if num_envs != owner.num_envs or num_envs < 1:
            raise ContractError("v4 owner/orchestrator batch size mismatch")
        if getattr(h200, "protocol_version", None) != CONTRACT_VERSION_V4:
            raise ContractError("v4 orchestrator requires a v4 H200 client")
        self.owner = owner
        self.h200 = h200
        self.num_envs = num_envs
        self.remote_request_count = int(getattr(h200, "remote_request_count", 1))
        self.journal = journal
        self.phase_telemetry = phase_telemetry
        self.transport_timing = transport_timing
        self.monotonic_ns = monotonic_ns

    def _transport_marker(self) -> TransportTimingMarkerV4 | None:
        return self.transport_timing.mark() if self.transport_timing is not None else None

    def _transport_summary(
        self,
        marker: TransportTimingMarkerV4 | None,
        *,
        call_started_ns: int,
        call_finished_ns: int,
    ) -> dict[str, int | bool]:
        if marker is None or self.transport_timing is None:
            return {
                "instrumented": False,
                "client_call_wall_ns": call_finished_ns - call_started_ns,
                "http_request_count": 0,
            }
        return self.transport_timing.summarize(
            marker,
            call_started_ns=call_started_ns,
            call_finished_ns=call_finished_ns,
        )

    def _phase_start(self, keys: Sequence[StepKey], phase: str) -> bool:
        if self.phase_telemetry is None:
            return False
        return self.phase_telemetry.start(
            keys[0].rollout_step_index,
            phase,
            total=self.num_envs,
            counts={"batch_size": self.num_envs},
        )

    def _phase_end(
        self,
        keys: Sequence[StepKey],
        phase: str,
        started: bool,
        **counts: Any,
    ) -> None:
        if self.phase_telemetry is not None and started:
            self.phase_telemetry.end(
                keys[0].rollout_step_index,
                phase,
                completed=self.num_envs,
                total=self.num_envs,
                counts=counts,
            )

    def _phase_error(
        self,
        keys: Sequence[StepKey],
        phase: str,
        started: bool,
        exc: BaseException,
    ) -> None:
        if self.phase_telemetry is not None and started:
            self.phase_telemetry.error(
                keys[0].rollout_step_index,
                phase,
                exc,
                completed=0,
                total=self.num_envs,
            )

    @staticmethod
    def _idem(prefix: str, key: StepKey) -> str:
        return f"{prefix}-{sha256_bytes(canonical_json(key.as_dict()))[:48]}"

    def _validate_keys(self, keys: Sequence[StepKey], *, expected_index: int) -> list[StepKey]:
        values = list(keys)
        if len(values) != self.num_envs or any(not isinstance(key, StepKey) for key in values):
            raise ContractError("v4 vector step requires exactly B StepKey values")
        if any(key.rollout_step_index != expected_index for key in values):
            raise ContractError("v4 vector step rollout index mismatch")
        if len({key.token for key in values}) != self.num_envs:
            raise ContractError("v4 vector lane identities must be unique")
        shared = {(key.run_id, key.round_id, key.rollout_step_index) for key in values}
        if len(shared) != 1:
            raise ContractError("v4 vector lanes must share run, round, and step")
        return values

    def _local_prepare(self, keys: Sequence[StepKey]) -> tuple[list[dict[str, Any]], int]:
        phase = "capture_teacher"
        phase_started = self._phase_start(keys, phase)
        started = self.monotonic_ns()
        try:
            rows = self.owner.prepare_batch_lean_v4(
                [
                    {
                        "contract_version": CONTRACT_VERSION,
                        "request_id": self._idem("rtx-v4-prepare", key),
                        "step_key": key.as_dict(),
                        "beta": ROUND_BETAS[key.round_id],
                    }
                    for key in keys
                ]
            )
        except BaseException as exc:
            self._phase_error(keys, phase, phase_started, exc)
            raise
        elapsed = self.monotonic_ns() - started
        if len(rows) != self.num_envs:
            exc = ContractError("v4 owner returned the wrong prepare count")
            self._phase_error(keys, phase, phase_started, exc)
            raise exc
        self._phase_end(keys, phase, phase_started, prepared_lanes=len(rows))
        return rows, elapsed

    @staticmethod
    def _public_selections(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for row in rows:
            value = dict(row)
            value.pop("_selected_action_array", None)
            result.append(value)
        return result

    def _local_commit(
        self, selections: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        keys = [StepKey.from_mapping(row["step_key"]) for row in selections]
        phase = "env_step"
        phase_started = self._phase_start(keys, phase)
        started = self.monotonic_ns()
        try:
            rows = self.owner.commit_batch(
                [
                    {
                        **build_owner_commit(selection),
                        "selection_receipt": selection["selection_receipt"],
                    }
                    for selection in selections
                ]
            )
        except BaseException as exc:
            self._phase_error(keys, phase, phase_started, exc)
            raise
        elapsed = self.monotonic_ns() - started
        if len(rows) != self.num_envs:
            exc = ContractError("v4 owner returned the wrong commit count")
            self._phase_error(keys, phase, phase_started, exc)
            raise exc
        self._phase_end(
            keys,
            phase,
            phase_started,
            transition_results=len(rows),
            env_step_calls=1,
        )
        return rows, elapsed

    def bootstrap(self, keys: Sequence[StepKey]) -> PendingStepV4:
        keys = self._validate_keys(keys, expected_index=0)
        local, local_ns = self._local_prepare(keys)
        prepare_ids = tuple(self._idem("h200-v4-prepare", key) for key in keys)
        phase = "prepare_or_exchange"
        phase_started = self._phase_start(keys, phase)
        timing_marker = self._transport_marker()
        remote_started = self.monotonic_ns()
        try:
            raw = self.h200.prepare_batch(local, prepare_ids)
        except BaseException as exc:
            self._phase_error(keys, phase, phase_started, exc)
            raise
        remote_finished = self.monotonic_ns()
        remote_ns = remote_finished - remote_started
        transport_timing = self._transport_summary(
            timing_marker,
            call_started_ns=remote_started,
            call_finished_ns=remote_finished,
        )
        selections = self._public_selections(raw)
        if len(selections) != self.num_envs:
            exc = ContractError("v4 H200 returned the wrong step-0 selection count")
            self._phase_error(keys, phase, phase_started, exc)
            raise exc
        self._phase_end(
            keys,
            phase,
            phase_started,
            prepare_responses=len(selections),
            prior_commit_acknowledgements=0,
        )
        commits, commit_ns = self._local_commit(selections)
        pending = PendingStepV4(
            keys=tuple(keys),
            local_prepares=tuple(local),
            selections=tuple(selections),
            local_commits=tuple(commits),
            prepare_idempotency_keys=prepare_ids,
            commit_idempotency_keys=tuple(
                self._idem("h200-v4-commit", key) for key in keys
            ),
            timings_ns={
                "capture_teacher_and_local_prepare": local_ns,
                "h200_prepare_barrier": remote_ns,
                "native_commit_and_env_step": commit_ns,
            },
            transport_timing=transport_timing,
        )
        return pending

    def exchange(
        self, previous: PendingStepV4, current_keys: Sequence[StepKey]
    ) -> tuple[PendingStepV4, list[dict[str, Any]]]:
        previous_index = previous.keys[0].rollout_step_index
        if previous_index >= CHUNK_LENGTH - 1:
            raise ContractError("step 80 must use the final v4 flush")
        current_keys = self._validate_keys(
            current_keys, expected_index=previous_index + 1
        )
        local, local_ns = self._local_prepare(current_keys)
        prepare_ids = tuple(
            self._idem("h200-v4-prepare", key) for key in current_keys
        )
        phase = "prepare_or_exchange"
        phase_started = self._phase_start(current_keys, phase)
        timing_marker = self._transport_marker()
        remote_started = self.monotonic_ns()
        try:
            acks, raw_selections = self.h200.exchange_batch(
                previous.selections,
                previous.local_commits,
                previous.commit_idempotency_keys,
                local,
                prepare_ids,
            )
        except BaseException as exc:
            self._phase_error(current_keys, phase, phase_started, exc)
            raise
        remote_finished = self.monotonic_ns()
        remote_ns = remote_finished - remote_started
        transport_timing = self._transport_summary(
            timing_marker,
            call_started_ns=remote_started,
            call_finished_ns=remote_finished,
        )
        if len(acks) != self.num_envs or len(raw_selections) != self.num_envs:
            exc = ContractError(
                "v4 exchange did not return B acknowledgements and selections"
            )
            self._phase_error(current_keys, phase, phase_started, exc)
            raise exc
        self._phase_end(
            current_keys,
            phase,
            phase_started,
            prepare_responses=len(raw_selections),
            prior_commit_acknowledgements=len(acks),
        )
        # The exchange made the previous vector step remotely complete. Emit
        # exactly one aggregate row for that step; the exchange duration is
        # explicitly overlapping with preparation of the current step.
        self._record_completed_step(
            previous,
            commit_acknowledgements=len(acks),
            remote_commit_barrier_ns=remote_ns,
            remote_commit_overlaps_next_prepare=True,
            remote_transport_timing=transport_timing,
        )
        selections = self._public_selections(raw_selections)
        commits, commit_ns = self._local_commit(selections)
        pending = PendingStepV4(
            keys=tuple(current_keys),
            local_prepares=tuple(local),
            selections=tuple(selections),
            local_commits=tuple(commits),
            prepare_idempotency_keys=prepare_ids,
            commit_idempotency_keys=tuple(
                self._idem("h200-v4-commit", key) for key in current_keys
            ),
            timings_ns={
                "capture_teacher_and_local_prepare": local_ns,
                "h200_commit_prepare_exchange": remote_ns,
                "native_commit_and_env_step": commit_ns,
            },
            transport_timing=transport_timing,
        )
        return pending, list(acks)

    def finalize(self, pending: PendingStepV4) -> list[dict[str, Any]]:
        if pending.keys[0].rollout_step_index != CHUNK_LENGTH - 1:
            raise ContractError("v4 final flush requires rollout index 80")
        phase = "finalize"
        phase_started = self._phase_start(pending.keys, phase)
        timing_marker = self._transport_marker()
        started = self.monotonic_ns()
        try:
            acknowledgements = self.h200.finalize_batch(
                pending.selections,
                pending.local_commits,
                pending.commit_idempotency_keys,
            )
        except BaseException as exc:
            self._phase_error(pending.keys, phase, phase_started, exc)
            raise
        finished = self.monotonic_ns()
        elapsed = finished - started
        transport_timing = self._transport_summary(
            timing_marker,
            call_started_ns=started,
            call_finished_ns=finished,
        )
        if len(acknowledgements) != self.num_envs:
            exc = ContractError("v4 final flush lacks B commit acknowledgements")
            self._phase_error(pending.keys, phase, phase_started, exc)
            raise exc
        self._phase_end(
            pending.keys,
            phase,
            phase_started,
            commit_acknowledgements=len(acknowledgements),
        )
        self._record_completed_step(
            pending,
            commit_acknowledgements=len(acknowledgements),
            remote_commit_barrier_ns=elapsed,
            remote_commit_overlaps_next_prepare=False,
            remote_transport_timing=transport_timing,
        )
        return list(acknowledgements)

    def _record_completed_step(
        self,
        pending: PendingStepV4,
        *,
        commit_acknowledgements: int,
        remote_commit_barrier_ns: int,
        remote_commit_overlaps_next_prepare: bool,
        remote_transport_timing: Mapping[str, int | bool],
    ) -> None:
        if self.journal is None:
            return
        self.journal.append(
            {
                "event": "vector_step",
                "run_id": pending.keys[0].run_id,
                "round_id": pending.keys[0].round_id,
                "rollout_window_id": pending.keys[0].rollout_window_id,
                "rollout_step_index": pending.keys[0].rollout_step_index,
                "batch_size": self.num_envs,
                "prepare_mode": (
                    "initial_prepare"
                    if pending.keys[0].rollout_step_index == 0
                    else "prior_exchange"
                ),
                "commit_acknowledgements": commit_acknowledgements,
                "prepare_responses": len(pending.selections),
                "env_step_calls": 1,
                "remote_requests": self.remote_request_count,
                "timings_ns": {
                    **dict(pending.timings_ns),
                    "remote_commit_barrier": remote_commit_barrier_ns,
                },
                "selection_transport_timing": dict(pending.transport_timing),
                "ack_or_exchange_transport_timing": dict(remote_transport_timing),
                "remote_commit_overlaps_next_prepare": remote_commit_overlaps_next_prepare,
                "status": "complete",
            }
        )


class PipelinedBatchWindowRunnerV4(BatchWindowRunner):
    """Run exactly 81 native transitions with v4 commit/prepare overlap."""

    orchestrator: PipelinedBatchStepOrchestratorV4

    def __init__(
        self,
        owner: IsaacBatchTransitionOwner,
        orchestrator: PipelinedBatchStepOrchestratorV4,
        *,
        run_id: str,
        round_id: int,
        actor_prefix: str,
        lifecycle: RolloutLifecycle | None = None,
        initial_episode_serials: Sequence[int] | None = None,
        retry_policy: TransportRetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        phase_telemetry: PhaseBoundaryTelemetry | None = None,
    ) -> None:
        super().__init__(
            owner,
            orchestrator,  # type: ignore[arg-type]
            run_id=run_id,
            round_id=round_id,
            actor_prefix=actor_prefix,
            lifecycle=lifecycle,
            initial_episode_serials=initial_episode_serials,
            retry_policy=retry_policy,
            sleep=sleep,
            phase_telemetry=phase_telemetry,
        )

    def _retry(self, call, *, rollout_index: int):
        attempt = 1
        while True:
            try:
                return call()
            except (ContractError, OwnerConflict, CrashUncertain):
                raise
            except TransientTransportError:
                if attempt >= self.retry_policy.max_attempts:
                    raise
                if self.phase_telemetry is not None:
                    self.phase_telemetry.progress(
                        rollout_index,
                        "vector_step_barrier",
                        completed=0,
                        total=self.num_envs,
                        status="retrying",
                        counts={
                            "failed_attempt": attempt,
                            "next_attempt": attempt + 1,
                        },
                    )
                self._sleep(self.retry_policy.delay_after(attempt))
                attempt += 1

    def _start_vector_barrier(self, rollout_index: int) -> bool:
        if self.phase_telemetry is None:
            return False
        return self.phase_telemetry.start(
            rollout_index,
            "vector_step_barrier",
            total=self.num_envs,
            counts={"batch_size": self.num_envs},
        )

    def _end_vector_barrier(
        self,
        rollout_index: int,
        started: bool,
        *,
        commit_acknowledgements: int,
    ) -> None:
        if self.phase_telemetry is not None and started:
            self.phase_telemetry.end(
                rollout_index,
                "vector_step_barrier",
                completed=self.num_envs,
                total=self.num_envs,
                counts={
                    "commit_acknowledgements": commit_acknowledgements,
                    "durable_vector_steps": rollout_index + 1,
                    "completion_fraction": (rollout_index + 1) / CHUNK_LENGTH,
                },
            )

    def _error_vector_barrier(
        self,
        rollout_index: int,
        started: bool,
        exc: BaseException,
    ) -> None:
        if self.phase_telemetry is not None and started:
            self.phase_telemetry.error(
                rollout_index,
                "vector_step_barrier",
                exc,
                completed=0,
                total=self.num_envs,
                counts={"fail_closed": True},
            )

    def run_window(self, window_index: int, *, manage_lifecycle: bool = True) -> dict:
        active_barriers: dict[int, bool] = {}
        try:
            if manage_lifecycle:
                self.lifecycle.begin()
            self.begin_window(window_index)
            active_barriers[0] = self._start_vector_barrier(0)
            pending = self._retry(
                lambda: self.orchestrator.bootstrap(self.keys(window_index, 0)),
                rollout_index=0,
            )
            self._advance({"local_commits": list(pending.local_commits)})
            previous_acks: list[dict[str, Any]] = []
            for rollout_index in range(1, CHUNK_LENGTH):
                active_barriers[rollout_index] = self._start_vector_barrier(
                    rollout_index
                )
                current_keys = self.keys(window_index, rollout_index)
                pending, previous_acks = self._retry(
                    lambda p=pending, keys=current_keys: self.orchestrator.exchange(p, keys),
                    rollout_index=rollout_index,
                )
                if len(previous_acks) != self.num_envs:
                    raise ContractError("v4 exchange lacks the prior all-lane commit barrier")
                self._end_vector_barrier(
                    rollout_index - 1,
                    active_barriers.pop(rollout_index - 1),
                    commit_acknowledgements=len(previous_acks),
                )
                self._advance({"local_commits": list(pending.local_commits)})
            final_acks = self._retry(
                lambda: self.orchestrator.finalize(pending),
                rollout_index=CHUNK_LENGTH - 1,
            )
            commits = list(pending.local_commits)
            if sum(item.get("sealed") is True for item in commits) != self.num_envs:
                raise RuntimeError("v4 final local compact audit seal is incomplete")
            if len(final_acks) != self.num_envs:
                raise RuntimeError("v4 final remote acknowledgement barrier is incomplete")
            self._end_vector_barrier(
                CHUNK_LENGTH - 1,
                active_barriers.pop(CHUNK_LENGTH - 1),
                commit_acknowledgements=len(final_acks),
            )
            self.owner.confirm_ephemeral_window_finalized(final_acks)
            final = {
                "contract_version": CONTRACT_VERSION,
                "wire_contract_version": CONTRACT_VERSION_V4,
                "phase": "COMPLETE",
                "step_keys": [key.as_dict() for key in pending.keys],
                "local_commits": commits,
                "remote_acknowledgements": final_acks,
            }
            return {
                "contract_version": CONTRACT_VERSION,
                "batch_size": self.num_envs,
                "window_index": window_index,
                "transition_batches": CHUNK_LENGTH,
                "physical_transition_rows": CHUNK_LENGTH * self.num_envs,
                "final": final,
            }
        except BaseException as exc:
            for rollout_index, started in list(active_barriers.items()):
                self._error_vector_barrier(rollout_index, started, exc)
            raise
        finally:
            if manage_lifecycle:
                self.lifecycle.finish()
