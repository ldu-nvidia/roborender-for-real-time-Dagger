"""Finite vectorized rollout-window execution over the batch barrier."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Sequence

from .batch_orchestrator import BatchStepOrchestrator
from .batch_owner import IsaacBatchTransitionOwner
from .chunks import CHUNK_LENGTH
from .contracts import CONTRACT_VERSION, ContractError, StepKey, require_identifier
from .http_transport import TransientTransportError
from .owner import CrashUncertain, OwnerConflict
from .phase_telemetry import PhaseBoundaryTelemetry


@dataclass
class LaneCursor:
    episode_serial: int = 0
    episode_step_index: int = 0


@dataclass(frozen=True)
class RolloutLifecycle:
    """Native recurrent-state lifecycle around one 81-step window."""

    begin: Callable[[], None] = lambda: None
    finish: Callable[[], None] = lambda: None


@dataclass(frozen=True)
class TransportRetryPolicy:
    """Bounded retry policy for replay-safe H200 transport failures only."""

    max_attempts: int = 3
    initial_backoff_s: float = 0.25
    backoff_multiplier: float = 2.0
    max_backoff_s: float = 2.0
    max_recovery_s: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        for name in ("initial_backoff_s", "backoff_multiplier", "max_backoff_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative finite number")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if self.max_recovery_s is not None and (
            isinstance(self.max_recovery_s, bool) or not isinstance(self.max_recovery_s, (int, float))
            or not math.isfinite(self.max_recovery_s) or not 0 < self.max_recovery_s <= 300
        ):
            raise ValueError("max_recovery_s must be finite in (0,300]")

    def delay_after(self, failed_attempt: int) -> float:
        """Return delay after a 1-based failed attempt."""

        return min(
            self.max_backoff_s,
            self.initial_backoff_s * self.backoff_multiplier ** (failed_attempt - 1),
        )


def verify_ready_with_transient_retry(
    client,
    *,
    retry_policy: TransportRetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict:
    """Boundedly retry the read-only gateway readiness attestation.

    The supported lean clients implement ``verify_ready`` exclusively as
    authenticated ``GET /ready`` requests.  A reset or timeout can therefore
    be retried before any prepare identity, simulator step, or ledger mutation
    exists.  Contract/identity failures remain terminal and are never retried.
    """

    policy = retry_policy or TransportRetryPolicy()
    attempt = 1
    recovery_deadline = None
    while True:
        try:
            value = client.verify_ready()
            if not isinstance(value, dict):
                raise ContractError("gateway readiness attestation must be an object")
            return value
        except TransientTransportError:
            if recovery_deadline is None and policy.max_recovery_s is not None:
                recovery_deadline = monotonic() + policy.max_recovery_s
            budget_exhausted = (
                recovery_deadline is not None and monotonic() >= recovery_deadline
            )
            if attempt >= policy.max_attempts or budget_exhausted:
                raise
            delay = policy.delay_after(attempt)
            if recovery_deadline is not None:
                delay = min(delay, max(0.0, recovery_deadline - monotonic()))
            sleep(delay)
            if recovery_deadline is not None and monotonic() >= recovery_deadline:
                raise
            attempt += 1


class BatchWindowRunner:
    """Drive B scalar routes while preserving one physical batch transition.

    This object creates only identities and ordering. Capture, durability,
    action validation, and the sole vectorized ``env.step`` remain owned by
    :class:`IsaacBatchTransitionOwner`.
    """

    def __init__(
        self,
        owner: IsaacBatchTransitionOwner,
        orchestrator: BatchStepOrchestrator,
        *,
        run_id: str,
        round_id: int,
        actor_prefix: str,
        lifecycle: RolloutLifecycle | None = None,
        initial_episode_serials: Sequence[int] | None = None,
        retry_policy: TransportRetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        request_guard: Callable[[], None] | None = None,
        phase_telemetry: PhaseBoundaryTelemetry | None = None,
        episode_namespace: str | None = None,
    ) -> None:
        self.owner = owner
        self.orchestrator = orchestrator
        self.num_envs = owner.num_envs
        if orchestrator.num_envs != self.num_envs:
            raise ContractError("owner/orchestrator batch sizes differ")
        self.run_id = require_identifier(run_id, "run_id")
        if round_id not in (1, 2, 3, 4):
            raise ContractError("round_id must be an integer in [1,4]")
        self.round_id = round_id
        self.actor_prefix = require_identifier(actor_prefix, "actor_prefix")
        self.episode_namespace = (require_identifier(episode_namespace, "episode namespace")
                                  if episode_namespace is not None else None)
        serials = list(initial_episode_serials or [0] * self.num_envs)
        if len(serials) != self.num_envs or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in serials
        ):
            raise ContractError("initial episode serials must be B non-negative integers")
        self.cursors = [LaneCursor(value, 0) for value in serials]
        self.lifecycle = lifecycle or RolloutLifecycle()
        self.retry_policy = retry_policy or TransportRetryPolicy()
        self._sleep = sleep
        self._monotonic = monotonic
        self.request_guard = request_guard
        self.phase_telemetry = phase_telemetry

    def _key(self, lane: int, window_index: int, rollout_index: int) -> StepKey:
        cursor = self.cursors[lane]
        return StepKey(
            run_id=self.run_id,
            round_id=self.round_id,
            actor_id=f"{self.actor_prefix}-lane-{lane:03d}",
            rollout_window_id=f"window-{window_index:06d}",
            rollout_step_index=rollout_index,
            episode_id=((self.episode_namespace + "-") if self.episode_namespace else "")
                       + f"episode-{lane:03d}-{cursor.episode_serial:08d}",
            episode_step_index=cursor.episode_step_index,
        )

    def keys(self, window_index: int, rollout_index: int) -> list[StepKey]:
        if isinstance(window_index, bool) or not isinstance(window_index, int) or window_index < 0:
            raise ContractError("window_index must be a non-negative integer")
        if rollout_index < 0 or rollout_index >= CHUNK_LENGTH:
            raise ContractError(f"rollout_index must be in [0,{CHUNK_LENGTH - 1}]")
        return [self._key(lane, window_index, rollout_index) for lane in range(self.num_envs)]

    def begin_window(self, window_index: int) -> list[dict]:
        keys = self.keys(window_index, 0)
        return self.owner.begin_windows(
            [
                {
                    "contract_version": CONTRACT_VERSION,
                    "request_id": f"begin-{key.actor_id}-{key.rollout_window_id}",
                    "step_key": key.as_dict(),
                }
                for key in keys
            ]
        )

    def _advance(self, state: dict) -> None:
        commits = state.get("local_commits")
        if not isinstance(commits, list) or len(commits) != self.num_envs:
            raise ContractError("completed batch lacks B local commit outcomes")
        outcomes: list[tuple[bool, bool]] = []
        for commit in commits:
            result = commit.get("result") if isinstance(commit, dict) else None
            if not isinstance(result, dict):
                raise ContractError("local commit outcome is malformed")
            done = result.get("done")
            timeout = result.get("timeout")
            if not isinstance(done, bool) or not isinstance(timeout, bool):
                raise ContractError("local commit done/timeout flags must be bool")
            outcomes.append((done, timeout))
        # Validate the complete vector before changing any lane. A malformed
        # late lane must never leave earlier cursor identities partially moved.
        for cursor, (done, timeout) in zip(self.cursors, outcomes, strict=True):
            if done or timeout:
                cursor.episode_serial += 1
                cursor.episode_step_index = 0
            else:
                cursor.episode_step_index += 1

    def run_step(self, window_index: int, rollout_index: int) -> dict:
        # Recompute neither identities nor episode cursors while retrying.  The
        # orchestrator's durable phase and route-scoped idempotency keys make a
        # partial H200 prepare safe before env.step, and a partial remote commit
        # acknowledgement safe after the one irreversible env.step.
        keys = self.keys(window_index, rollout_index)
        vector_phase_started = False
        if self.phase_telemetry is not None:
            vector_phase_started = self.phase_telemetry.start(
                rollout_index,
                "vector_step_barrier",
                total=self.num_envs,
                counts={"batch_size": self.num_envs},
            )
        attempt = 1
        recovery_deadline = None
        while True:
            recovery_started = False
            if self.phase_telemetry is not None and attempt > 1:
                recovery_started = self.phase_telemetry.start(
                    rollout_index,
                    "recovery",
                    total=self.num_envs,
                    counts={"retry_attempt": attempt},
                )
            try:
                if self.request_guard is not None:
                    self.request_guard()
                state = self.orchestrator.run_batch(keys)
            except (ContractError, OwnerConflict, CrashUncertain) as exc:
                # These are semantic/fail-closed owner states.  Replaying them
                # cannot repair the request and could hide an uncertain step.
                if self.phase_telemetry is not None and recovery_started:
                    self.phase_telemetry.error(
                        rollout_index,
                        "recovery",
                        exc,
                        completed=0,
                        total=self.num_envs,
                        counts={"retry_attempt": attempt},
                    )
                if self.phase_telemetry is not None and vector_phase_started:
                    self.phase_telemetry.error(
                        rollout_index,
                        "vector_step_barrier",
                        exc,
                        completed=0,
                        total=self.num_envs,
                        counts={"attempts": attempt, "fail_closed": True},
                    )
                raise
            except TransientTransportError as exc:
                if recovery_deadline is None and self.retry_policy.max_recovery_s is not None:
                    recovery_deadline = self._monotonic() + self.retry_policy.max_recovery_s
                if self.phase_telemetry is not None and recovery_started:
                    self.phase_telemetry.error(
                        rollout_index,
                        "recovery",
                        exc,
                        completed=0,
                        total=self.num_envs,
                        counts={"retry_attempt": attempt},
                    )
                budget_exhausted = recovery_deadline is not None and self._monotonic() >= recovery_deadline
                if attempt >= self.retry_policy.max_attempts or budget_exhausted:
                    if self.phase_telemetry is not None and vector_phase_started:
                        self.phase_telemetry.error(
                            rollout_index,
                            "vector_step_barrier",
                            exc,
                            completed=0,
                            total=self.num_envs,
                            counts={"attempts": attempt, "retries_exhausted": True,
                                    "recovery_budget_exhausted": budget_exhausted},
                        )
                    raise
                if self.phase_telemetry is not None:
                    self.phase_telemetry.progress(
                        rollout_index,
                        "vector_step_barrier",
                        completed=0,
                        total=self.num_envs,
                        status="retrying",
                        counts={"failed_attempt": attempt, "next_attempt": attempt + 1},
                    )
                delay = self.retry_policy.delay_after(attempt)
                if recovery_deadline is not None:
                    delay = min(delay, max(0., recovery_deadline - self._monotonic()))
                self._sleep(delay)
                if recovery_deadline is not None and self._monotonic() >= recovery_deadline:
                    if self.phase_telemetry is not None and vector_phase_started:
                        self.phase_telemetry.error(rollout_index, "vector_step_barrier", exc,
                            completed=0, total=self.num_envs,
                            counts={"attempts": attempt, "recovery_budget_exhausted": True})
                    raise
                attempt += 1
                continue
            if self.phase_telemetry is not None and recovery_started:
                self.phase_telemetry.end(
                    rollout_index,
                    "recovery",
                    completed=self.num_envs,
                    total=self.num_envs,
                    counts={"successful_attempt": attempt},
                )
            break
        if state.get("phase") != "COMPLETE":
            exc = RuntimeError("batch orchestrator returned before COMPLETE")
            if self.phase_telemetry is not None and vector_phase_started:
                self.phase_telemetry.error(
                    rollout_index,
                    "vector_step_barrier",
                    exc,
                    completed=0,
                    total=self.num_envs,
                    counts={"attempts": attempt},
                )
            raise exc
        try:
            self._advance(state)
        except BaseException as exc:
            if self.phase_telemetry is not None and vector_phase_started:
                self.phase_telemetry.error(
                    rollout_index,
                    "vector_step_barrier",
                    exc,
                    completed=0,
                    total=self.num_envs,
                    counts={"attempts": attempt},
                )
            raise
        if self.phase_telemetry is not None and vector_phase_started:
            self.phase_telemetry.end(
                rollout_index,
                "vector_step_barrier",
                completed=self.num_envs,
                total=self.num_envs,
                counts={
                    "attempts": attempt,
                    "commit_acknowledgements": len(
                        state.get("remote_acknowledgements", ())
                    ),
                    "durable_vector_steps": rollout_index + 1,
                },
            )
        return state

    def run_window(self, window_index: int, *, manage_lifecycle: bool = True) -> dict:
        """Run and seal exactly 81 batch transitions.

        Lifecycle cleanup occurs on both success and abort. An abort after a
        native commit intent remains fail-closed in the batch owner. The
        production multi-window driver passes ``manage_lifecycle=False`` and
        owns one lifecycle around all windows; bounded callers retain the
        historical per-window default.
        """

        states: list[dict] = []
        from .episode_continuity import cursor_rows, validate_window
        initial_cursors = cursor_rows(self.keys(window_index, 0))
        resets = []
        try:
            # Keep begin inside the protected region: native rollout
            # initialization may partially mutate policy/teacher state before
            # raising, and finish must still clear both sides.
            if manage_lifecycle:
                self.lifecycle.begin()
            self.begin_window(window_index)
            for rollout_index in range(CHUNK_LENGTH):
                state = self.run_step(window_index, rollout_index)
                states.append(state)
                for lane,commit in enumerate(state['local_commits']):
                    outcome = commit['result']
                    if outcome['done'] or outcome['timeout']:
                        resets.append(dict(lane=lane,step=rollout_index,
                            done=outcome['done'],timeout=outcome['timeout']))
            final = states[-1]
            barrier_started = False
            if self.phase_telemetry is not None:
                barrier_started = self.phase_telemetry.start(
                    CHUNK_LENGTH - 1,
                    "seal_ack_barrier",
                    total=self.num_envs,
                    counts={"expected_seals": self.num_envs, "expected_acks": self.num_envs},
                )
            try:
                commits = final["local_commits"]
                sealed_count = sum(item.get("sealed") is True for item in commits)
                ack_count = len(final.get("remote_acknowledgements", ()))
                if sealed_count != self.num_envs:
                    raise RuntimeError("final batch did not seal every lane audit")
                if ack_count != self.num_envs:
                    raise RuntimeError("final batch lacks B remote acknowledgements")
            except BaseException as exc:
                if self.phase_telemetry is not None and barrier_started:
                    self.phase_telemetry.error(
                        CHUNK_LENGTH - 1,
                        "seal_ack_barrier",
                        exc,
                        completed=0,
                        total=self.num_envs,
                        counts={"sealed_lanes": locals().get("sealed_count", 0), "acknowledgements": locals().get("ack_count", 0)},
                    )
                raise
            if self.phase_telemetry is not None and barrier_started:
                self.phase_telemetry.end(
                    CHUNK_LENGTH - 1,
                    "seal_ack_barrier",
                    completed=self.num_envs,
                    total=self.num_envs,
                    counts={"sealed_lanes": sealed_count, "acknowledgements": ack_count},
                )
            return {
                "contract_version": CONTRACT_VERSION,
                "batch_size": self.num_envs,
                "window_index": window_index,
                "transition_batches": CHUNK_LENGTH,
                "physical_transition_rows": CHUNK_LENGTH * self.num_envs,
                "episode_continuity": validate_window(dict(
                    schema='roborender.native-window-cursor-continuity.v1',window_index=window_index,
                    start=initial_cursors,end=cursor_rows(self.keys(window_index+1,0)),native_resets=resets),
                    batch=self.num_envs,index=window_index),
                "final": final,
            }
        finally:
            if manage_lifecycle:
                self.lifecycle.finish()

    def run_bounded_probe(self, window_index: int, *, steps: int) -> dict:
        """Run a fresh, deliberately non-sealing capacity probe.

        Capacity probes must stop strictly before the 81-row chunk boundary.
        They exercise the real capture/selection/transition contract without
        claiming or publishing a learner-complete rollout window.
        """

        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or not 1 <= steps < CHUNK_LENGTH
        ):
            raise ContractError(
                f"bounded probe steps must be an integer in [1,{CHUNK_LENGTH - 1}]"
            )
        states: list[dict] = []
        try:
            self.lifecycle.begin()
            self.begin_window(window_index)
            for rollout_index in range(steps):
                states.append(self.run_step(window_index, rollout_index))
            final = states[-1]
            if any(item.get("sealed") is not False for item in final["local_commits"]):
                raise RuntimeError("bounded capacity probe unexpectedly sealed a lane")
            return {
                "contract_version": CONTRACT_VERSION,
                "batch_size": self.num_envs,
                "window_index": window_index,
                "transition_batches": steps,
                "physical_transition_rows": steps * self.num_envs,
                "bounded_non_sealing_probe": True,
                "final": final,
            }
        finally:
            self.lifecycle.finish()
