from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Any, Mapping, Protocol, Sequence

from .boundary import (
    H200_COMMIT_PATH,
    H200_PREPARE_PATH,
    build_h200_commit,
    build_h200_prepare,
    build_owner_commit,
    validate_h200_commit_ack,
    validate_h200_selection,
)
from .capacity_telemetry import EventSink, record_if_present
from .contracts import CONTRACT_VERSION, ROUND_BETAS, ContractError, StepKey, canonical_json, sha256_bytes
from .durable import atomic_write_json
from .http_transport import JsonTransport
from .lean_boundary_v2 import CONTRACT_VERSION_V2
from .lean_boundary_v3 import CONTRACT_VERSION_V3
from .phase_telemetry import PhaseBoundaryTelemetry


_BOUNDARY_PHASE_NAMES = {
    "local_prepare_barrier": "prepare_local_total",
    "h200_prepare_render_student_select_barrier": "prepare_encode_send_wait",
    "local_commit_vector_barrier": "commit_local_total",
    "h200_commit_barrier": "commit_encode_send_wait",
}


class BatchOwnerClient(Protocol):
    def prepare_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]: ...

    def commit_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]: ...

    def prepare_batch_lean(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]: ...


class BatchStepOrchestrator:
    """Coordinate B scalar H200 routes around one atomic RTX batch step."""

    def __init__(
        self,
        owner: BatchOwnerClient,
        h200: Any,
        runtime_root: Path,
        *,
        num_envs: int,
        max_parallel_requests: int = 1,
        telemetry: EventSink | None = None,
        phase_telemetry: PhaseBoundaryTelemetry | None = None,
        monotonic_ns=time.perf_counter_ns,
        clock_ns=time.time_ns,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.owner = owner
        self.h200 = h200
        self.wire_contract_version = getattr(h200, "protocol_version", CONTRACT_VERSION)
        self.lean_batch = self.wire_contract_version in {
            CONTRACT_VERSION_V2,
            CONTRACT_VERSION_V3,
        }
        # Retained as an internal compatibility hint for older diagnostics.
        self.lean_v2 = self.wire_contract_version == CONTRACT_VERSION_V2
        self.num_envs = num_envs
        if (
            isinstance(max_parallel_requests, bool)
            or not isinstance(max_parallel_requests, int)
            or not 1 <= max_parallel_requests <= 8
        ):
            raise ValueError("max_parallel_requests must be an integer in [1,8]")
        self.max_parallel_requests = min(max_parallel_requests, num_envs)
        if self.lean_batch:
            remote_count = getattr(h200, "remote_request_count", 1)
            if (
                isinstance(remote_count, bool)
                or not isinstance(remote_count, int)
                or not 1 <= remote_count <= 8
            ):
                raise ValueError("lean H200 remote_request_count must be in [1,8]")
            self.remote_request_count_per_phase = min(remote_count, num_envs)
        else:
            self.remote_request_count_per_phase = self.max_parallel_requests
        self.telemetry = telemetry
        self.phase_telemetry = phase_telemetry
        self.monotonic_ns = monotonic_ns
        self.clock_ns = clock_ns
        self.runtime_root = Path(runtime_root)
        self.runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    @staticmethod
    def _idem(prefix: str, key: StepKey) -> str:
        return f"{prefix}-{sha256_bytes(canonical_json(key.as_dict()))[:48]}"

    def _phase_call(self, phase: str, keys: Sequence[StepKey], call):
        boundary_phase = _BOUNDARY_PHASE_NAMES.get(phase)
        boundary_started = False
        if self.phase_telemetry is not None and boundary_phase is not None:
            boundary_started = self.phase_telemetry.start(
                keys[0].rollout_step_index,
                boundary_phase,
                total=self.num_envs,
                counts={
                    "requested_lanes": self.num_envs,
                    "parallel_requests": self.remote_request_count_per_phase,
                },
            )
        started = self.monotonic_ns()
        try:
            value = call()
        except BaseException as exc:
            duration = self.monotonic_ns() - started
            if boundary_started:
                self.phase_telemetry.error(
                    keys[0].rollout_step_index,
                    boundary_phase,
                    exc,
                    completed=0,
                    total=self.num_envs,
                    counts={"requested_lanes": self.num_envs},
                )
            record_if_present(
                self.telemetry,
                "vector_phase",
                phase=phase,
                outcome="error",
                duration_ns=duration,
                error_type=type(exc).__name__,
                step_keys=[key.as_dict() for key in keys],
                batch_size=self.num_envs,
                parallel_requests=self.remote_request_count_per_phase,
            )
            raise
        duration = self.monotonic_ns() - started
        if boundary_started:
            self.phase_telemetry.end(
                keys[0].rollout_step_index,
                boundary_phase,
                completed=self.num_envs,
                total=self.num_envs,
                counts={
                    "completed_lanes": self.num_envs,
                    "parallel_requests": self.remote_request_count_per_phase,
                },
            )
        record_if_present(
            self.telemetry,
            "vector_phase",
            phase=phase,
            outcome="success",
            duration_ns=duration,
            error_type=None,
            step_keys=[key.as_dict() for key in keys],
            batch_size=self.num_envs,
            parallel_requests=self.remote_request_count_per_phase,
        )
        return value, duration

    def _post_many(
        self,
        path: str,
        requests: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Post scalar routes concurrently while returning lane order exactly."""

        def post(request: Mapping[str, Any]) -> dict[str, Any]:
            return self.h200.post(path, request, max_response_bytes=1 << 20)

        if self.max_parallel_requests == 1:
            return [post(request) for request in requests]
        # executor.map preserves input order even when H200 workers finish out
        # of order.  The vector step remains behind completion of every future.
        with ThreadPoolExecutor(
            max_workers=self.max_parallel_requests,
            thread_name_prefix="h200-route",
        ) as pool:
            return list(pool.map(post, requests))

    def _owner_native_duration(self, section: str, field: str) -> int | None:
        status = getattr(self.owner, "status", None)
        if status is None:
            return None
        section_value = status().get(section)
        if not isinstance(section_value, Mapping):
            return None
        value = section_value.get(field)
        return value if isinstance(value, int) and value >= 0 else None

    def run_batch(self, keys: Sequence[StepKey]) -> dict[str, Any]:
        keys = list(keys)
        self._validate_keys(keys)
        identity_digest = sha256_bytes(canonical_json([key.as_dict() for key in keys]))
        first = keys[0]
        state_path = (
            self.runtime_root
            / first.run_id
            / f"D{first.round_id}"
            / "vectorized"
            / f"batch-step-{first.rollout_step_index:03d}-{identity_digest[:16]}.json"
        )
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("step_keys") != [key.as_dict() for key in keys]:
                raise RuntimeError("batch orchestrator receipt path collision")
            expected_wire = self.wire_contract_version
            if state.get("wire_contract_version", CONTRACT_VERSION) != expected_wire:
                raise RuntimeError("durable batch step was created with another wire protocol")
        else:
            state = {
                "contract_version": CONTRACT_VERSION,
                "wire_contract_version": self.wire_contract_version,
                "phase": "INIT",
                "batch_size": self.num_envs,
                "step_keys": [key.as_dict() for key in keys],
                "started_time_ns": int(self.clock_ns()),
                "started_monotonic_ns": int(self.monotonic_ns()),
                "timings_ns": {},
            }
            atomic_write_json(state_path, state)
        # Backward-compatible recovery of pre-telemetry durable receipts.
        state.setdefault("timings_ns", {})
        state.setdefault("started_time_ns", int(self.clock_ns()))
        state.setdefault("started_monotonic_ns", int(self.monotonic_ns()))
        if state["phase"] == "COMPLETE":
            return dict(state)

        local_prepares: list[dict[str, Any]] | None = None
        if state["phase"] in {"INIT", "LOCAL_PREPARED"}:
            local_prepare_requests = [
                    {
                        "contract_version": CONTRACT_VERSION,
                        "request_id": self._idem("rtx-prepare", key),
                        "step_key": key.as_dict(),
                        "beta": ROUND_BETAS[key.round_id],
                    }
                    for key in keys
                ]
            prepare_method = (
                getattr(self.owner, "prepare_batch_lean")
                if self.lean_batch
                else self.owner.prepare_batch
            )
            local_prepares, local_prepare_duration = self._phase_call(
                "local_prepare_barrier",
                keys,
                lambda: prepare_method(local_prepare_requests),
            )
            if len(local_prepares) != self.num_envs:
                raise ContractError("batch owner returned the wrong number of prepares")
        if state["phase"] == "INIT":
            assert local_prepares is not None
            state.update(
                {
                    "phase": "LOCAL_PREPARED",
                    "prepare_receipts": [item["prepare_receipt"] for item in local_prepares],
                    "capture_digests": [item["capture_digest"] for item in local_prepares],
                    "observation_sources": [
                        item["metadata"]["observation_source"] for item in local_prepares
                    ],
                }
            )
            state["timings_ns"]["local_prepare_barrier"] = local_prepare_duration
            native_capture = self._owner_native_duration(
                "last_prepare_batch", "native_capture_duration_ns"
            )
            if native_capture is not None:
                state["timings_ns"]["native_capture_batch"] = native_capture
            atomic_write_json(state_path, state)

        if state["phase"] == "LOCAL_PREPARED":
            assert local_prepares is not None
            prepare_idempotency_keys = [
                self._idem(
                    (
                        f"h200-{self.wire_contract_version.rsplit('.', 1)[-1]}-prepare"
                        if self.lean_batch
                        else "h200-prepare"
                    ),
                    key,
                )
                for key in keys
            ]
            if self.lean_batch:
                raw_selections, remote_prepare_duration = self._phase_call(
                    "h200_prepare_render_student_select_barrier",
                    keys,
                    lambda: self.h200.prepare_batch(
                        local_prepares, prepare_idempotency_keys
                    ),
                )
                selections = []
                for selection in raw_selections:
                    selection = dict(selection)
                    selection.pop("_selected_action_array", None)
                    selections.append(selection)
            else:
                # Some scalar v1 H200 prepares may succeed before a later lane
                # fails. No RTX step occurs; replay uses the same route keys.
                remote_requests = [
                    build_h200_prepare(local_prepare, idempotency_key)
                    for local_prepare, idempotency_key in zip(
                        local_prepares, prepare_idempotency_keys, strict=True
                    )
                ]
                raw_selections, remote_prepare_duration = self._phase_call(
                    "h200_prepare_render_student_select_barrier",
                    keys,
                    lambda: self._post_many(H200_PREPARE_PATH, remote_requests),
                )
                selections = []
                for raw, request, local_prepare in zip(
                    raw_selections, remote_requests, local_prepares, strict=True
                ):
                    selection = validate_h200_selection(
                        raw,
                        local_prepare,
                        expected_idempotency_key=request["idempotency_key"],
                    )
                    selection.pop("_selected_action_array", None)
                    selections.append(selection)
            state.update({"phase": "H200_SELECTED", "selections": selections})
            state["timings_ns"][
                "h200_prepare_render_student_select_barrier"
            ] = remote_prepare_duration
            atomic_write_json(state_path, state)

        selections = state["selections"]
        if state["phase"] == "H200_SELECTED":
            # This one call writes the owner's BATCH_COMMITTING intent and then
            # crosses the native vectorized env.step boundary exactly once.
            local_commits, local_commit_duration = self._phase_call(
                "local_commit_vector_barrier",
                keys,
                lambda: self.owner.commit_batch(
                    [
                        {
                            **build_owner_commit(selection),
                            # Retain the H200 selection receipt in the compact
                            # local audit.  It is intentionally not part of the
                            # pre-existing commit_digest calculation.
                            "selection_receipt": selection["selection_receipt"],
                        }
                        for selection in selections
                    ]
                ),
            )
            if len(local_commits) != self.num_envs:
                raise ContractError("batch owner returned the wrong number of commits")
            state.update({"phase": "LOCAL_COMMITTED", "local_commits": local_commits})
            state["timings_ns"]["local_commit_vector_barrier"] = local_commit_duration
            native_env_step = self._owner_native_duration(
                "last_commit_batch", "native_env_step_duration_ns"
            )
            if native_env_step is not None:
                state["timings_ns"]["native_env_step_batch"] = native_env_step
            atomic_write_json(state_path, state)

        if state["phase"] == "LOCAL_COMMITTED":
            # Partial remote acknowledgement is harmless: local env.step has
            # already happened once and retries use stable H200 idempotency keys.
            commit_idempotency_keys = [
                self._idem(
                    (
                        f"h200-{self.wire_contract_version.rsplit('.', 1)[-1]}-commit"
                        if self.lean_batch
                        else "h200-commit"
                    ),
                    key,
                )
                for key in keys
            ]
            if self.lean_batch:
                acknowledgements, remote_commit_duration = self._phase_call(
                    "h200_commit_barrier",
                    keys,
                    lambda: self.h200.commit_batch(
                        selections,
                        state["local_commits"],
                        commit_idempotency_keys,
                    ),
                )
            else:
                commit_requests = [
                    build_h200_commit(selection, local_commit, idempotency_key)
                    for selection, local_commit, idempotency_key in zip(
                        selections,
                        state["local_commits"],
                        commit_idempotency_keys,
                        strict=True,
                    )
                ]
                raw_acks, remote_commit_duration = self._phase_call(
                    "h200_commit_barrier",
                    keys,
                    lambda: self._post_many(H200_COMMIT_PATH, commit_requests),
                )
                acknowledgements = [
                    validate_h200_commit_ack(raw, request)
                    for raw, request in zip(raw_acks, commit_requests, strict=True)
                ]
            state.update(
                {
                    "phase": "COMPLETE",
                    "remote_acknowledgements": acknowledgements,
                    "completed_time_ns": int(self.clock_ns()),
                }
            )
            state["timings_ns"]["h200_commit_barrier"] = remote_commit_duration
            state["timings_ns"]["vector_step_total"] = (
                self.monotonic_ns() - state["started_monotonic_ns"]
            )
            atomic_write_json(state_path, state)
            local_commits = state["local_commits"]
            record_if_present(
                self.telemetry,
                "vector_step_complete",
                step_keys=[key.as_dict() for key in keys],
                batch_size=self.num_envs,
                parallel_requests=self.remote_request_count_per_phase,
                timings_ns=dict(state["timings_ns"]),
                observation_source_counts={
                    source: state.get("observation_sources", []).count(source)
                    for source in ("generated_controls", "raytraced")
                },
                selected_source_counts={
                    source: [item["selected_source"] for item in selections].count(source)
                    for source in ("teacher", "student")
                },
                done_count=sum(bool(item["result"]["done"]) for item in local_commits),
                timeout_count=sum(
                    bool(item["result"]["timeout"]) for item in local_commits
                ),
                unique_prepare_receipts=len(set(state["prepare_receipts"])),
                unique_capture_digests=len(set(state["capture_digests"])),
                unique_selection_receipts=len(
                    {item["selection_receipt"] for item in selections}
                ),
                unique_commit_receipts=len(
                    {item["commit_receipt"] for item in local_commits}
                ),
                unique_remote_commit_receipts=len(
                    {item["remote_commit_receipt"] for item in acknowledgements}
                ),
            )
        return dict(state)

    def _validate_keys(self, keys: Sequence[StepKey]) -> None:
        if len(keys) != self.num_envs:
            raise ContractError(
                f"run_batch requires exactly {self.num_envs} lane keys, got {len(keys)}"
            )
        if any(not isinstance(key, StepKey) for key in keys):
            raise ContractError("run_batch keys must be StepKey values")
        if len({key.token for key in keys}) != self.num_envs:
            raise ContractError("run_batch lane identities must be unique")
        if len({(key.actor_id, key.rollout_window_id) for key in keys}) != self.num_envs:
            raise ContractError("run_batch lanes must have unique actor/window routes")
        shared = {(key.run_id, key.round_id, key.rollout_step_index) for key in keys}
        if len(shared) != 1:
            raise ContractError("vectorized keys must share run, round, and rollout index")
