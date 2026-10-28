from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .backend import Capture, TransitionResult
from .batch_backend import RavenBatchBackend
from .capacity_telemetry import EventSink, record_if_present
from .chunks import CHUNK_LENGTH, CurrentRoundChunk, MemoryCurrentRoundChunk
from .contracts import (
    ACTION_DIM,
    CAPTURE_METADATA_KEYS,
    CONTRACT_VERSION,
    MODEL_VIEW_MAP,
    PINNED_RAVEN_COMMIT,
    PINNED_RAVEN_ROOT,
    RAVEN_CAPTURE_VIEWS,
    RAVEN_VIEW_ORDER,
    ArrayRule,
    ContractError,
    RavenShapeProfile,
    StepKey,
    anchor_rgb_digest,
    canonical_json,
    decode_array,
    decode_arrays,
    encode_array,
    encode_arrays,
    payload_digest,
    require_identifier,
    require_sha256,
    sha256_bytes,
    validate_action,
    validate_array,
    validate_beta,
)
from .durable import DurableJournal, atomic_write_json
from .owner import CrashUncertain, OwnerConflict
from .phase_telemetry import PhaseBoundaryTelemetry


_PROVENANCE_KEYS = {
    "raven_root",
    "raven_commit",
    "raven_videogen_trainer_sha256",
    "raven_teacher_homie_trainer_sha256",
    "isaac_capture_sha256",
    "image_dagger_config_sha256",
    "teacher_checkpoint_sha256",
    "homie_walk_sha256",
    "homie_stand_sha256",
}


class IsaacBatchTransitionOwner:
    """Sole owner of one vectorized Raven/Isaac transition boundary.

    H200 preparation and selection remain one request per environment.  This
    owner accepts or rejects those requests as a complete batch, persists one
    ``BATCH_COMMITTING`` intent, and invokes the native vectorized ``env.step``
    adapter exactly once.  It never emulates a batch with scalar environment
    steps.
    """

    def __init__(
        self,
        backend: RavenBatchBackend,
        runtime_root: Path,
        staging_root: Path,
        *,
        shape_profile: RavenShapeProfile | None = None,
        static_provenance: Mapping[str, Any] | None = None,
        clock_ns=time.time_ns,
        monotonic_ns=time.perf_counter_ns,
        persist_chunks: bool = True,
        ephemeral_hot_path: bool = False,
        telemetry: EventSink | None = None,
        phase_telemetry: PhaseBoundaryTelemetry | None = None,
    ) -> None:
        if isinstance(backend.num_envs, bool) or not isinstance(backend.num_envs, int):
            raise ContractError("batch backend num_envs must be an integer")
        if backend.num_envs < 1:
            raise ContractError("batch backend num_envs must be positive")
        self.backend = backend
        self.num_envs = backend.num_envs
        self.runtime_root = Path(runtime_root)
        self.staging_root = Path(staging_root)
        self.runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.staging_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_path = self.runtime_root / "batch_owner_state.json"
        self.journal = DurableJournal(self.runtime_root / "batch_owner_events.jsonl")
        self.profile = shape_profile or RavenShapeProfile()
        self.static_provenance = dict(static_provenance or {})
        self._validate_static_provenance()
        self.clock_ns = clock_ns
        self.monotonic_ns = monotonic_ns
        if type(persist_chunks) is not bool:
            raise ContractError("persist_chunks must be bool")
        if type(ephemeral_hot_path) is not bool:
            raise ContractError("ephemeral_hot_path must be bool")
        if ephemeral_hot_path and not persist_chunks:
            raise ContractError("ephemeral_hot_path requires final compact audit seals")
        self.persist_chunks = persist_chunks
        self.ephemeral_hot_path = ephemeral_hot_path
        self.ephemeral_marker_path = self.runtime_root / "v4_ephemeral_window_marker.json"
        self.telemetry = telemetry
        self.phase_telemetry = phase_telemetry
        self._lock = threading.RLock()
        self._captures: dict[str, Capture] = {}
        self._memory_chunks: dict[tuple[str, int, str, str], MemoryCurrentRoundChunk] = {}
        had_state = self.state_path.exists()
        if self.ephemeral_hot_path and had_state:
            raise ContractError("ephemeral_hot_path requires a fresh runtime root")
        self.state = self._load_or_new()
        if self.state.get("num_envs") != self.num_envs:
            raise ContractError("durable batch size does not match native backend num_envs")
        if bool(self.state.get("persist_chunks", True)) != self.persist_chunks:
            raise ContractError("durable chunk policy does not match requested owner mode")
        if bool(self.state.get("ephemeral_hot_path", False)) != self.ephemeral_hot_path:
            raise ContractError("durable hot-path policy does not match requested owner mode")
        unsafe = {
            "READY",
            "BATCH_PREPARING",
            "PREPARED",
            "BATCH_COMMITTING",
            "BINDING_EPISODES",
            "SEALED",
        }
        if self.ephemeral_hot_path and self.ephemeral_marker_path.exists():
            marker = json.loads(self.ephemeral_marker_path.read_text(encoding="utf-8"))
            self.state["phase"] = "CRASH_UNCERTAIN"
            self.state["uncertain_reason"] = (
                "v4 ephemeral runtime was reconstructed after its Raven/gateway generation "
                f"ended with marker status {marker.get('status', 'unknown')}; replay is forbidden"
            )
        elif had_state and self.state["phase"] in unsafe:
            prior = self.state["phase"]
            self.state["phase"] = "CRASH_UNCERTAIN"
            self.state["uncertain_reason"] = (
                f"restart observed active {prior} vectorized state without a native Raven "
                "simulator restore attestation"
            )
            self._event("CRASH_UNCERTAIN", self.state["uncertain_reason"])
            self._persist()

    def _load_or_new(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("contract_version") != CONTRACT_VERSION:
                raise ContractError("batch owner state contract version mismatch")
            return state
        return {
            "contract_version": CONTRACT_VERSION,
            "num_envs": self.num_envs,
            "persist_chunks": self.persist_chunks,
            "ephemeral_hot_path": self.ephemeral_hot_path,
            "phase": "EMPTY",
            "next_rollout_step": 0,
            "lanes": [],
            "begin_batch_digest": None,
            "begin_responses": None,
            "batch_prepare_intent": None,
            "last_prepare_batch": None,
            "last_commit_batch": None,
            "batch_commit_intent": None,
            "session": None,
        }

    def _validate_static_provenance(self) -> None:
        if set(self.static_provenance) != _PROVENANCE_KEYS:
            raise ContractError(
                f"static provenance keys must be exactly {sorted(_PROVENANCE_KEYS)}"
            )
        if self.static_provenance["raven_root"] != PINNED_RAVEN_ROOT:
            raise ContractError("static provenance raven_root is not pinned BC_render")
        if self.static_provenance["raven_commit"] != PINNED_RAVEN_COMMIT:
            raise ContractError("static provenance raven_commit mismatch")
        for name in _PROVENANCE_KEYS - {"raven_root", "raven_commit"}:
            require_sha256(self.static_provenance[name], name)

    def _persist(self) -> None:
        if self.ephemeral_hot_path:
            if self.state.get("phase") == "CRASH_UNCERTAIN":
                self._write_ephemeral_marker(
                    "crash_uncertain",
                    detail={"reason": self.state.get("uncertain_reason")},
                )
            return
        atomic_write_json(self.state_path, self.state)

    def _event(
        self,
        kind: str,
        detail: Any,
        keys: Sequence[StepKey] | None = None,
    ) -> None:
        if self.ephemeral_hot_path and kind != "CRASH_UNCERTAIN":
            return
        self.journal.append(
            {
                "contract_version": CONTRACT_VERSION,
                "time_ns": int(self.clock_ns()),
                "event": kind,
                "step_keys": [key.as_dict() for key in keys] if keys is not None else None,
                "detail": detail,
            }
        )

    def _telemetry(
        self,
        event: str,
        *,
        keys: Sequence[StepKey],
        **detail: Any,
    ) -> None:
        if self.ephemeral_hot_path:
            return
        record_if_present(
            self.telemetry,
            event,
            step_keys=[key.as_dict() for key in keys],
            batch_size=self.num_envs,
            **detail,
        )

    def _phase_start(
        self,
        keys: Sequence[StepKey],
        phase: str,
        *,
        counts: Mapping[str, Any] | None = None,
    ) -> bool:
        if self.ephemeral_hot_path or self.phase_telemetry is None:
            return False
        return self.phase_telemetry.start(
            keys[0].rollout_step_index,
            phase,
            total=self.num_envs,
            counts=counts,
        )

    def _phase_end(
        self,
        keys: Sequence[StepKey],
        phase: str,
        *,
        counts: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.ephemeral_hot_path and self.phase_telemetry is not None:
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
        exc: BaseException,
        *,
        completed: int = 0,
        counts: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.ephemeral_hot_path and self.phase_telemetry is not None:
            self.phase_telemetry.error(
                keys[0].rollout_step_index,
                phase,
                exc,
                completed=completed,
                total=self.num_envs,
                counts=counts,
            )

    def _assert_operable(self) -> None:
        if self.state["phase"] == "CRASH_UNCERTAIN":
            raise CrashUncertain(
                self.state.get("uncertain_reason", "vectorized environment transition is uncertain")
            )

    def _write_ephemeral_marker(
        self,
        status: str,
        *,
        keys: Sequence[StepKey] | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.ephemeral_hot_path:
            return
        marker = {
            "schema": "roborender.dagger.rtx.v4-ephemeral-window-marker.v1",
            "contract_version": "roborender.dagger.rtx.v4",
            "status": status,
            "time_ns": int(self.clock_ns()),
            "num_envs": self.num_envs,
            "step_keys": [key.as_dict() for key in keys] if keys is not None else None,
            "detail": dict(detail or {}),
            "restart_policy": "fatal_abandon_no_replay",
        }
        atomic_write_json(self.ephemeral_marker_path, marker, mode=0o600)

    def confirm_ephemeral_window_finalized(
        self,
        acknowledgements: Sequence[Mapping[str, Any]],
    ) -> None:
        """Durably close one v4 ephemeral window after the remote all-B flush.

        This is not a recovery checkpoint: reconstructing the owner remains
        forbidden because Raven recurrent/simulator state is not restorable.
        The completed marker only proves where the prior process stopped.
        """

        if not self.ephemeral_hot_path:
            return
        with self._lock:
            self._assert_operable()
            values = list(acknowledgements)
            if self.state.get("phase") != "SEALED" or len(values) != self.num_envs:
                raise OwnerConflict("v4 ephemeral finalize lacks local seal or all-lane ack")
            keys = [
                StepKey.from_mapping(item["step_key"])
                for item in self.state["last_commit_batch"]["responses"]
            ]
            self._write_ephemeral_marker(
                "remote_finalized",
                keys=keys,
                detail={
                    "acknowledgements": len(values),
                    "sealed_lanes": sum(
                        item.get("sealed") is True
                        for item in self.state["last_commit_batch"]["responses"]
                    ),
                },
            )
            self._memory_chunks.clear()

    def begin_windows(self, requests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Bind one per-lane rollout window without changing the scalar wire schema."""

        with self._lock:
            self._assert_operable()
            requests = self._require_batch(requests, "begin_windows")
            parsed: list[tuple[StepKey, str]] = []
            for request in requests:
                self._check_version(request)
                self._require_exact_keys(
                    request, {"contract_version", "request_id", "step_key"}, "begin_window"
                )
                key = StepKey.from_mapping(request["step_key"])
                if key.rollout_step_index != 0:
                    raise ContractError("each rollout window must begin at index zero")
                parsed.append((key, require_identifier(request["request_id"], "request_id")))
            keys = [item[0] for item in parsed]
            self._validate_batch_identity(keys, require_same_step=True)
            next_session = self._validated_session(keys)
            digest = sha256_bytes(canonical_json([dict(request) for request in requests]))
            if self.state.get("begin_batch_digest") == digest:
                return json.loads(canonical_json(self.state["begin_responses"]))
            prior_phase = self.state["phase"]
            if prior_phase not in {"EMPTY", "SEALED"}:
                raise OwnerConflict("a vectorized rollout batch is already active")

            # The v4 marker precedes episode binding because bind_episodes may
            # consume Raven RNG/native assignment state.  A process death at
            # any point after this write makes the window explicitly
            # non-replayable rather than looking like an unused runtime.
            self._write_ephemeral_marker("active", keys=keys)

            previous_lanes = self.state.get("lanes", [])
            bindings_needed: dict[int, StepKey] = {}
            continuing: dict[int, Mapping[str, Any]] = {}
            if prior_phase == "EMPTY":
                for lane, key in enumerate(keys):
                    if key.episode_step_index != 0:
                        raise ContractError("initial lane episodes must begin at episode step zero")
                    bindings_needed[lane] = key
            else:
                if len(previous_lanes) != self.num_envs:
                    raise OwnerConflict("sealed vectorized state has incomplete lane records")
                for lane, (key, prior) in enumerate(zip(keys, previous_lanes, strict=True)):
                    active = prior["active_episode"]
                    if prior.get("needs_episode_bind"):
                        if key.episode_id == active["episode_id"] or key.episode_step_index != 0:
                            raise OwnerConflict(
                                f"lane {lane} must bind a new episode after done/timeout"
                            )
                        bindings_needed[lane] = key
                    else:
                        if (
                            key.episode_id != active["episode_id"]
                            or key.episode_step_index != active["next_episode_step"]
                        ):
                            raise OwnerConflict(
                                f"lane {lane} must continue its active episode across windows"
                            )
                        continuing[lane] = dict(active["assignment"])

            bound = self._bind_episodes_durably(bindings_needed, keys)
            assignments = {**continuing, **bound}
            lanes: list[dict[str, Any]] = []
            responses: list[dict[str, Any]] = []
            for lane, ((key, request_id), assignment) in enumerate(
                zip(parsed, (assignments[i] for i in range(self.num_envs)), strict=True)
            ):
                response = {
                    "contract_version": CONTRACT_VERSION,
                    "step_key": key.as_dict(),
                    "request_id": request_id,
                    "rollout_window_id": key.rollout_window_id,
                    "episode_assignment": dict(assignment),
                }
                # Provenance is scoped to this physical 81-step window.  A
                # continuing episode is represented by a fresh index-zero
                # binding record; bindings from the preceding window must not
                # leak into this window's seal.
                current_window_bindings = [self._manifest_binding(key, assignment)]
                lanes.append(
                    {
                        "lane_index": lane,
                        "window_key": key.as_dict(),
                        "active_episode": {
                            "episode_id": key.episode_id,
                            "next_episode_step": key.episode_step_index,
                            "assignment": dict(assignment),
                        },
                        "episode_bindings": current_window_bindings,
                        "needs_episode_bind": False,
                        "sealed": None,
                    }
                )
                responses.append(response)
            self.state.update(
                {
                    "phase": "READY",
                    "next_rollout_step": 0,
                    "lanes": lanes,
                    "begin_batch_digest": digest,
                    "begin_responses": responses,
                    "batch_prepare_intent": None,
                    "last_prepare_batch": None,
                    "last_commit_batch": None,
                    "batch_commit_intent": None,
                    "session": next_session,
                }
            )
            self._event("BATCH_WINDOWS_BEGIN", responses, keys)
            self._persist()
            return json.loads(canonical_json(responses))

    def _bind_episodes_durably(
        self,
        bindings: Mapping[int, StepKey],
        all_keys: Sequence[StepKey],
    ) -> dict[int, dict[str, Any]]:
        if not bindings:
            return {}
        self.state["phase"] = "BINDING_EPISODES"
        self.state["binding_intent"] = {
            str(lane): key.as_dict() for lane, key in sorted(bindings.items())
        }
        self._persist()
        self._event("EPISODE_BIND_BATCH_INTENT", self.state["binding_intent"], all_keys)
        try:
            raw = self.backend.bind_episodes(bindings)
            if not isinstance(raw, Mapping) or set(raw) != set(bindings):
                raise ContractError("native bind_episodes returned the wrong lane set")
            result: dict[int, dict[str, Any]] = {}
            for lane in sorted(bindings):
                native = raw[lane]
                if not isinstance(native, Mapping):
                    raise ContractError(f"lane {lane} episode assignment is not an object")
                use_raytraced = native.get("use_raytraced")
                if not isinstance(use_raytraced, bool):
                    raise ContractError("native assignment use_raytraced must be bool")
                anchor_sha = require_sha256(native.get("anchor_sha256"), "anchor_sha256")
                if native.get("raven_source") != PINNED_RAVEN_ROOT:
                    raise ContractError("episode assignment is not pinned to BC_render")
                if native.get("raven_commit") != PINNED_RAVEN_COMMIT:
                    raise ContractError("episode assignment Raven commit mismatch")
                result[lane] = {
                    "use_raytraced": use_raytraced,
                    "observation_source": (
                        "raytraced" if use_raytraced else "generated_controls"
                    ),
                    "anchor_sha256": anchor_sha,
                    "assignment_source": "pinned_raven_id_sample_anchor",
                    "raven_source": PINNED_RAVEN_ROOT,
                    "raven_commit": PINNED_RAVEN_COMMIT,
                }
            self._event("EPISODE_BIND_BATCH_COMPLETE", result, all_keys)
            return result
        except BaseException as exc:
            self.state["phase"] = "CRASH_UNCERTAIN"
            self.state["uncertain_reason"] = (
                f"native bind_episodes raised {type(exc).__name__}; episode RNG consumption is uncertain"
            )
            self._event("CRASH_UNCERTAIN", self.state["uncertain_reason"], all_keys)
            self._persist()
            raise CrashUncertain(self.state["uncertain_reason"]) from exc

    def prepare_batch(self, requests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Prepare the legacy local v1 JSON/base64 representation."""

        if self.ephemeral_hot_path:
            raise ContractError("v4 ephemeral owner requires prepare_batch_lean")

        return self._prepare_batch(requests, raw_arrays=False)

    def prepare_batch_lean(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prepare raw in-process arrays for the lean v2 H200 batch boundary.

        The durable receipt contains only metadata and array descriptors. Raw
        arrays remain in the native owner's in-process capture cache until the
        exactly-once vector commit. A process restart in this phase is already
        fail-closed as CRASH_UNCERTAIN, so no unsafe array reconstruction is
        introduced.
        """

        return self._prepare_batch(requests, raw_arrays=True)

    def prepare_batch_lean_v4(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prepare the v4 lean payload without hashing RTX-only capture arrays.

        V1-v3 retain their exact historical full-capture digest.  V4 instead
        hashes its source-specific lean projection once and derives a compact
        integrity receipt over ``{contract_version, step_key, wire_digest}``.
        """

        return self._prepare_batch(
            requests,
            raw_arrays=True,
            lightweight_wire_contract="roborender.dagger.rtx.v4",
        )

    def _prepare_batch(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        raw_arrays: bool,
        lightweight_wire_contract: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._assert_operable()
            requests = self._require_batch(requests, "prepare_batch")
            parsed: list[tuple[StepKey, str, float, str]] = []
            for request in requests:
                self._check_version(request)
                self._require_exact_keys(
                    request,
                    {"contract_version", "request_id", "step_key", "beta"},
                    "prepare_step",
                )
                key = StepKey.from_mapping(request["step_key"])
                request_id = require_identifier(request["request_id"], "request_id")
                beta = validate_beta(key.round_id, request["beta"])
                digest = sha256_bytes(
                    canonical_json(
                        {"step_key": key.as_dict(), "request_id": request_id, "beta": beta}
                    )
                )
                parsed.append((key, request_id, beta, digest))
            keys = [item[0] for item in parsed]
            self._validate_batch_identity(keys, require_same_step=True)
            digests = [item[3] for item in parsed]
            prior = self.state.get("last_prepare_batch")
            if prior and prior["step_keys"] == [key.as_dict() for key in keys]:
                if prior["request_digests"] != digests:
                    raise OwnerConflict("prepare_batch retry changed a lane request")
                expected_encoding = (
                    "raw_v4_lightweight"
                    if lightweight_wire_contract is not None
                    else ("raw" if raw_arrays else "json_base64")
                )
                if prior.get("response_encoding", "json_base64") != expected_encoding:
                    raise OwnerConflict("prepare_batch retry changed local array encoding")
                if self.ephemeral_hot_path:
                    stored = json.loads(canonical_json(prior["responses"]))
                else:
                    stored = [
                        json.loads((self.runtime_root / path).read_text(encoding="utf-8"))
                        for path in prior["response_paths"]
                    ]
                if raw_arrays:
                    response_paths = prior.get("response_paths") or [None] * self.num_envs
                    for response, key, path in zip(stored, keys, response_paths, strict=True):
                        response.pop("array_descriptors", None)
                        response["arrays"] = self._capture_for(key, path).arrays
                return stored
            if self.state["phase"] != "READY":
                raise OwnerConflict(f"batch owner phase {self.state['phase']} cannot prepare")
            self._assert_lane_keys(keys)
            self._bind_resets_if_needed(keys)

            # Native capture includes teacher/Homie inference and may advance
            # recurrent/native state even though env.step has not happened.
            # Receipt publication is also a multi-file operation.  Persist a
            # fail-closed intent before crossing either boundary: after this
            # point an interrupted or failed prepare can never be replayed
            # without a certified simulator/model-state restore.
            prepare_intent = {
                "step_keys": [key.as_dict() for key in keys],
                "request_digests": digests,
                "request_ids": [item[1] for item in parsed],
            }
            self.state["phase"] = "BATCH_PREPARING"
            self.state["batch_prepare_intent"] = prepare_intent
            self._persist()
            active_phase: str | None = None
            try:
                self._event("BATCH_PREPARE_INTENT", prepare_intent, keys)
                active_phase = "capture_teacher"
                self._phase_start(
                    keys,
                    active_phase,
                    counts={"requested_lanes": self.num_envs},
                )
                capture_started_ns = self.monotonic_ns()
                captures = list(self.backend.capture_batch(keys))
                capture_duration_ns = self.monotonic_ns() - capture_started_ns
                self._telemetry(
                    "native_capture_batch",
                    keys=keys,
                    duration_ns=capture_duration_ns,
                    outcome="success",
                )
                if len(captures) != self.num_envs:
                    raise ContractError("capture_batch returned the wrong number of rows")
                self._phase_end(
                    keys,
                    active_phase,
                    counts={"captured_lanes": len(captures)},
                )
                active_phase = None

                active_phase = "prepare_encode_and_persist"
                self._phase_start(
                    keys,
                    active_phase,
                    counts={"captured_lanes": len(captures)},
                )
                responses: list[dict[str, Any]] = []
                response_paths: list[str] = []
                for lane, (capture, parsed_lane) in enumerate(
                    zip(captures, parsed, strict=True)
                ):
                    key, request_id, beta, request_digest = parsed_lane
                    assignment = self.state["lanes"][lane]["active_episode"]["assignment"]
                    self._validate_capture(capture, key, assignment)
                    metadata = dict(capture.metadata)
                    metadata.update(
                        {
                            "contract_version": CONTRACT_VERSION,
                            "step_key": key.as_dict(),
                            "beta": beta,
                            "episode_assignment": assignment,
                        }
                    )
                    wire_digest: str | None = None
                    if lightweight_wire_contract is not None:
                        if lightweight_wire_contract != "roborender.dagger.rtx.v4":
                            raise ContractError("unsupported lightweight wire contract")
                        # Import locally so the legacy owner has no v4 import
                        # cost or dependency unless this explicit method is used.
                        from .lean_boundary_v3 import project_minimal_arrays_v3
                        from .lean_boundary_v4 import lightweight_capture_digest_v4

                        lean = project_minimal_arrays_v3(
                            {"metadata": metadata, "arrays": capture.arrays},
                            shape_profile=self.profile,
                        )
                        wire_metadata = dict(metadata)
                        wire_metadata["contract_version"] = lightweight_wire_contract
                        wire_digest = payload_digest(wire_metadata, lean)
                        capture_digest = lightweight_capture_digest_v4(key, wire_digest)
                    else:
                        capture_digest = payload_digest(metadata, capture.arrays)
                    receipt = sha256_bytes(
                        canonical_json(
                            {
                                "request_digest": request_digest,
                                "capture_digest": capture_digest,
                            }
                        )
                    )
                    response: dict[str, Any] = {
                        "contract_version": CONTRACT_VERSION,
                        "step_key": key.as_dict(),
                        "request_id": request_id,
                        "prepare_receipt": receipt,
                        "capture_digest": capture_digest,
                        "metadata": metadata,
                    }
                    if wire_digest is not None:
                        response["wire_digest"] = wire_digest
                    if raw_arrays:
                        response["arrays"] = capture.arrays
                        if lightweight_wire_contract is not None:
                            # A v4 restart is unconditionally fail-closed; full
                            # descriptor hashes are neither recoverable state nor
                            # algorithm inputs, so omit that expensive reread.
                            durable_response = {
                                name: value
                                for name, value in response.items()
                                if name != "arrays"
                            }
                        else:
                            durable_response = {
                                **{
                                    name: value
                                    for name, value in response.items()
                                    if name != "arrays"
                                },
                                "array_descriptors": {
                                    name: {
                                        "dtype": value.dtype.name,
                                        "shape": list(value.shape),
                                        "sha256": sha256_bytes(
                                            np.ascontiguousarray(value).tobytes(order="C")
                                        ),
                                        "nbytes": int(value.nbytes),
                                    }
                                    for name, value in sorted(capture.arrays.items())
                                },
                            }
                    else:
                        response["arrays"] = encode_arrays(capture.arrays)
                        durable_response = response
                    rel = (
                        Path("receipts")
                        / key.run_id
                        / f"D{key.round_id}"
                        / key.actor_id
                        / key.rollout_window_id
                        / f"prepare-{key.rollout_step_index:03d}.json"
                    )
                    if not self.ephemeral_hot_path:
                        atomic_write_json(self.runtime_root / rel, durable_response)
                    self._captures[key.token] = capture
                    responses.append(response)
                    if not self.ephemeral_hot_path:
                        response_paths.append(str(rel))
                self.state["phase"] = "PREPARED"
                self.state["batch_prepare_intent"] = None
                last_prepare_batch = {
                    "step_keys": [key.as_dict() for key in keys],
                    "request_digests": digests,
                    "prepare_receipts": [item["prepare_receipt"] for item in responses],
                    "capture_digests": [item["capture_digest"] for item in responses],
                    "response_paths": response_paths,
                    "native_capture_duration_ns": capture_duration_ns,
                    "response_encoding": (
                        "raw_v4_lightweight"
                        if lightweight_wire_contract is not None
                        else ("raw" if raw_arrays else "json_base64")
                    ),
                }
                if self.ephemeral_hot_path:
                    last_prepare_batch["responses"] = [
                        {name: value for name, value in item.items() if name != "arrays"}
                        for item in responses
                    ]
                self.state["last_prepare_batch"] = last_prepare_batch
                self._persist()
                self._event(
                    "BATCH_PREPARED",
                    {
                        "prepare_receipts": [
                            item["prepare_receipt"] for item in responses
                        ],
                        "batch_size": self.num_envs,
                    },
                    keys,
                )
                self._phase_end(
                    keys,
                    active_phase,
                    counts={
                        "encoded_lanes": len(responses),
                        "durable_receipts": len(response_paths),
                    },
                )
                active_phase = None
                return responses
            except BaseException as exc:
                if active_phase is not None:
                    self._phase_error(
                        keys,
                        active_phase,
                        exc,
                        completed=(len(responses) if "responses" in locals() else 0),
                    )
                self.state["phase"] = "CRASH_UNCERTAIN"
                self.state["uncertain_reason"] = (
                    f"post-batch-prepare-intent operation raised {type(exc).__name__}; "
                    "native capture/teacher state or prepare receipt publication may be "
                    "partial, so prepare retry is forbidden"
                )
                # State is the fail-closed source of truth.  Persist it before
                # appending the diagnostic event so even a later process death
                # cannot leave a retryable BATCH_PREPARING record.
                self._persist()
                self._event("CRASH_UNCERTAIN", self.state["uncertain_reason"], keys)
                raise CrashUncertain(self.state["uncertain_reason"]) from exc

    def _bind_resets_if_needed(self, keys: Sequence[StepKey]) -> None:
        bindings: dict[int, StepKey] = {}
        for lane, (key, lane_state) in enumerate(zip(keys, self.state["lanes"], strict=True)):
            if lane_state.get("needs_episode_bind"):
                active = lane_state["active_episode"]
                if key.episode_id == active["episode_id"] or key.episode_step_index != 0:
                    raise OwnerConflict(
                        f"lane {lane} must bind a new episode at step zero after done/timeout"
                    )
                bindings[lane] = key
        assignments = self._bind_episodes_durably(bindings, keys)
        for lane, assignment in assignments.items():
            key = keys[lane]
            lane_state = self.state["lanes"][lane]
            lane_state["active_episode"] = {
                "episode_id": key.episode_id,
                "next_episode_step": 0,
                "assignment": assignment,
            }
            lane_state["episode_bindings"].append(self._manifest_binding(key, assignment))
            lane_state["needs_episode_bind"] = False
        if assignments:
            self.state["binding_intent"] = None
            self.state["phase"] = "READY"
            self._persist()

    def commit_batch(self, requests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        with self._lock:
            self._assert_operable()
            requests = self._require_batch(requests, "commit_batch")
            parsed: list[dict[str, Any]] = []
            for request in requests:
                self._check_version(request)
                commit_keys = {
                    "contract_version",
                    "step_key",
                    "prepare_receipt",
                    "selection_id",
                    "selected_source",
                    "beta",
                    "selected_action",
                }
                # New production callers retain this H200 receipt in the RTX
                # compact audit.  Accept its absence only for source/backward
                # compatibility with already-issued internal owner requests.
                if "selection_receipt" in request:
                    commit_keys.add("selection_receipt")
                self._require_exact_keys(
                    request,
                    commit_keys,
                    "commit_step",
                )
                key = StepKey.from_mapping(request["step_key"])
                action = decode_array(request["selected_action"])
                validate_action(action)
                source = request["selected_source"]
                if source not in {"teacher", "student"}:
                    raise ContractError("selected_source must be teacher or student")
                selection_id = require_identifier(request["selection_id"], "selection_id")
                selection_receipt = (
                    require_sha256(request["selection_receipt"], "selection_receipt")
                    if "selection_receipt" in request
                    else None
                )
                beta = validate_beta(key.round_id, request["beta"])
                body = {
                    "step_key": key.as_dict(),
                    "prepare_receipt": request["prepare_receipt"],
                    "selection_id": selection_id,
                    "selected_source": source,
                    "beta": beta,
                    "selected_action": encode_array(action),
                }
                parsed.append(
                    {
                        "key": key,
                        "action": action,
                        "source": source,
                        "selection_id": selection_id,
                        "selection_receipt": selection_receipt,
                        "prepare_receipt": request["prepare_receipt"],
                        "commit_digest": sha256_bytes(canonical_json(body)),
                    }
                )
            keys = [item["key"] for item in parsed]
            self._validate_batch_identity(keys, require_same_step=True)
            digests = [item["commit_digest"] for item in parsed]
            last = self.state.get("last_commit_batch")
            if last and last["step_keys"] == [key.as_dict() for key in keys]:
                if last["commit_digests"] != digests:
                    raise OwnerConflict("commit_batch retry changed a lane action or selection")
                return json.loads(canonical_json(last["responses"]))
            prepared = self.state.get("last_prepare_batch")
            if not prepared or prepared["step_keys"] != [key.as_dict() for key in keys]:
                raise OwnerConflict("commit_batch has no matching prepared batch")
            if self.state["phase"] != "PREPARED":
                raise OwnerConflict(f"batch owner phase {self.state['phase']} cannot commit")

            response_paths = prepared.get("response_paths") or [None] * self.num_envs
            captures = [
                self._capture_for(key, response_paths[lane])
                for lane, key in enumerate(keys)
            ]
            for lane, item in enumerate(parsed):
                if item["prepare_receipt"] != prepared["prepare_receipts"][lane]:
                    raise OwnerConflict(f"lane {lane} prepare receipt mismatch")
                teacher = captures[lane].arrays["teacher_action"]
                homie = captures[lane].arrays["homie_action"]
                if item["source"] == "teacher" and not np.array_equal(item["action"], teacher):
                    raise ContractError(f"lane {lane} teacher selection differs from teacher_action")
                if item["source"] == "student" and not np.array_equal(item["action"][11:], homie):
                    raise ContractError(f"lane {lane} student selection changed the Homie tail")

            selected = np.ascontiguousarray(
                np.stack([item["action"] for item in parsed], axis=0), dtype=np.float32
            )
            teachers = np.ascontiguousarray(
                np.stack([capture.arrays["teacher_action"] for capture in captures], axis=0),
                dtype=np.float32,
            )
            if selected.shape != (self.num_envs, ACTION_DIM) or teachers.shape != (
                self.num_envs,
                ACTION_DIM,
            ):
                raise ContractError("vectorized action batch has the wrong shape")

            intent = {
                "step_keys": [key.as_dict() for key in keys],
                "commit_digests": digests,
                "selection_ids": [item["selection_id"] for item in parsed],
                "selected_batch_sha256": sha256_bytes(selected.tobytes(order="C")),
                "teacher_batch_sha256": sha256_bytes(teachers.tobytes(order="C")),
            }
            self.state["phase"] = "BATCH_COMMITTING"
            self.state["batch_commit_intent"] = intent
            self._persist()
            active_phase: str | None = None
            seal_phase_active = False
            try:
                # The durable state is authoritative.  Journal publication is
                # diagnostic, but it still sits on the no-replay side of the
                # barrier: if it fails, enter CRASH_UNCERTAIN without ever
                # calling the physical vector step.
                active_phase = "commit_intent_journal"
                self._phase_start(
                    keys,
                    active_phase,
                    counts={"durable_batch_intents": 1},
                )
                self._event("BATCH_COMMIT_INTENT", intent, keys)
                self._phase_end(
                    keys,
                    active_phase,
                    counts={"durable_batch_intents": 1},
                )
                active_phase = None

                active_phase = "native_env_step"
                self._phase_start(
                    keys,
                    active_phase,
                    counts={"action_rows": self.num_envs, "action_dim": ACTION_DIM},
                )
                env_step_started_ns = self.monotonic_ns()
                results = list(self.backend.step_batch(selected, teachers))
                env_step_duration_ns = self.monotonic_ns() - env_step_started_ns
                self._telemetry(
                    "native_env_step_batch",
                    keys=keys,
                    duration_ns=env_step_duration_ns,
                    action_shape=list(selected.shape),
                    outcome="success",
                )
                if len(results) != self.num_envs:
                    raise ContractError("step_batch returned the wrong number of outcomes")
                for result in results:
                    self._validate_result(result)
                self._phase_end(
                    keys,
                    active_phase,
                    counts={"transition_results": len(results), "env_step_calls": 1},
                )
                active_phase = None

                active_phase = "commit_outcomes_and_persist"
                self._phase_start(
                    keys,
                    active_phase,
                    counts={"transition_results": len(results)},
                )
                final_step = keys[0].rollout_step_index + 1 == CHUNK_LENGTH
                if final_step:
                    self._phase_start(
                        keys,
                        "seal_local_audits",
                        counts={"audit_lanes": self.num_envs, "rows_per_lane": CHUNK_LENGTH},
                    )
                    seal_phase_active = True
                responses: list[dict[str, Any]] = []
                for lane, (item, capture, result) in enumerate(
                    zip(parsed, captures, results, strict=True)
                ):
                    key = item["key"]
                    result_dict = self._result_dict(result)
                    commit_receipt = sha256_bytes(
                        canonical_json(
                            {"commit_digest": item["commit_digest"], "result": result_dict}
                        )
                    )
                    chunk = self._chunk_for(key) if self.persist_chunks else None
                    if chunk is not None:
                        audit = {
                            "prepare_receipt": prepared["prepare_receipts"][lane],
                            "capture_digest": prepared["capture_digests"][lane],
                            "selection_id": item["selection_id"],
                            "commit_digest": item["commit_digest"],
                            "commit_receipt": commit_receipt,
                        }
                        if item["selection_receipt"] is not None:
                            audit["selection_receipt"] = item["selection_receipt"]
                        chunk.append(
                            key,
                            capture,
                            item["action"],
                            item["source"],
                            result,
                            audit=audit,
                        )
                    response: dict[str, Any] = {
                        "contract_version": CONTRACT_VERSION,
                        "step_key": key.as_dict(),
                        "commit_receipt": commit_receipt,
                        "result": result_dict,
                        "committed_steps": key.rollout_step_index + 1,
                        "sealed": False,
                    }
                    if final_step:
                        if chunk is None:
                            raise ContractError(
                                "telemetry-only owner cannot reach the 81-step sealing boundary"
                            )
                        seal = chunk.seal(
                            provenance={
                                **self.static_provenance,
                                "episode_bindings": self.state["lanes"][lane]["episode_bindings"],
                            }
                        )
                        response.update({"sealed": True, "seal": seal})
                        self.state["lanes"][lane]["sealed"] = seal
                    responses.append(response)

                if seal_phase_active:
                    self._phase_end(
                        keys,
                        "seal_local_audits",
                        counts={"sealed_lanes": len(responses), "rows_per_lane": CHUNK_LENGTH},
                    )
                    seal_phase_active = False

                ended_flags = []
                for lane, (item, result) in enumerate(zip(parsed, results, strict=True)):
                    lane_state = self.state["lanes"][lane]
                    ended = result.done or result.timeout
                    ended_flags.append(bool(ended))
                    lane_state["needs_episode_bind"] = bool(ended)
                    if not ended:
                        lane_state["active_episode"]["next_episode_step"] = (
                            item["key"].episode_step_index + 1
                        )
                next_phase = "SEALED" if keys[0].rollout_step_index + 1 == CHUNK_LENGTH else "READY"
                self.state.update(
                    {
                        "phase": next_phase,
                        "next_rollout_step": keys[0].rollout_step_index + 1,
                        "batch_commit_intent": None,
                        "last_commit_batch": {
                            "step_keys": [key.as_dict() for key in keys],
                            "commit_digests": digests,
                            "responses": responses,
                            "native_env_step_duration_ns": env_step_duration_ns,
                        },
                    }
                )
                self._event(
                    "BATCH_COMMITTED",
                    {"responses": responses, "ended_lanes": ended_flags},
                    keys,
                )
                self._persist()
                for key in keys:
                    self._captures.pop(key.token, None)
                self._phase_end(
                    keys,
                    active_phase,
                    counts={
                        "persisted_outcomes": len(responses),
                        "done_count": sum(ended_flags),
                        "sealed_lanes": (len(responses) if final_step else 0),
                    },
                )
                active_phase = None
                return responses
            except BaseException as exc:
                if seal_phase_active:
                    self._phase_error(
                        keys,
                        "seal_local_audits",
                        exc,
                        completed=(len(responses) if "responses" in locals() else 0),
                    )
                    seal_phase_active = False
                if active_phase is not None:
                    self._phase_error(
                        keys,
                        active_phase,
                        exc,
                        completed=(len(responses) if "responses" in locals() else 0),
                    )
                if isinstance(exc, CrashUncertain):
                    raise
                self.state["phase"] = "CRASH_UNCERTAIN"
                self.state["uncertain_reason"] = (
                    f"post-batch-intent operation raised {type(exc).__name__}; vectorized env.step "
                    "may have completed"
                )
                # Durable state, rather than the append-only diagnostic log,
                # is the recovery authority.  Publish CRASH_UNCERTAIN first so
                # a journal I/O failure cannot leave a retryable-looking
                # on-disk state or mask the typed fail-closed exception.
                self._persist()
                try:
                    self._event("CRASH_UNCERTAIN", self.state["uncertain_reason"], keys)
                except BaseException:
                    pass
                raise CrashUncertain(self.state["uncertain_reason"]) from exc

    def status(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(canonical_json(self.state))

    def _capture_for(self, key: StepKey, response_path: str | None) -> Capture:
        capture = self._captures.get(key.token)
        if capture is not None:
            return capture
        if response_path is None:
            raise CrashUncertain(
                "v4 ephemeral capture cache is unavailable; replay after process loss is forbidden"
            )
        response = json.loads((self.runtime_root / response_path).read_text(encoding="utf-8"))
        return Capture(
            key=key,
            metadata=dict(response["metadata"]),
            arrays=decode_arrays(response["arrays"]),
        )

    def _chunk_for(self, key: StepKey) -> CurrentRoundChunk:
        if self.ephemeral_hot_path:
            route = (key.run_id, key.round_id, key.actor_id, key.rollout_window_id)
            chunk = self._memory_chunks.get(route)
            if chunk is None:
                chunk = MemoryCurrentRoundChunk(
                    self.staging_root,
                    run_id=key.run_id,
                    round_id=key.round_id,
                    actor_id=key.actor_id,
                    rollout_window_id=key.rollout_window_id,
                )
                self._memory_chunks[route] = chunk
            return chunk
        return CurrentRoundChunk(
            self.staging_root,
            run_id=key.run_id,
            round_id=key.round_id,
            actor_id=key.actor_id,
            rollout_window_id=key.rollout_window_id,
        )

    def _validate_capture(
        self,
        capture: Capture,
        key: StepKey,
        assignment: Mapping[str, Any],
    ) -> None:
        if not isinstance(capture, Capture) or capture.key != key:
            raise ContractError("capture_batch returned a mismatched lane capture")
        rules = self.profile.capture_rules()
        missing = sorted(set(rules) - set(capture.arrays))
        if missing:
            raise ContractError(f"capture is missing required arrays: {missing}")
        ray_names = {f"raytraced_rgb__{view}" for view in RAVEN_CAPTURE_VIEWS}
        unknown = sorted(set(capture.arrays) - set(rules) - ray_names)
        if unknown:
            raise ContractError(f"capture has unknown arrays: {unknown}")
        for name, rule in rules.items():
            validate_array(name, capture.arrays[name], rule)
        if not np.array_equal(capture.arrays["teacher_action"][11:], capture.arrays["homie_action"]):
            raise ContractError("teacher_action[11:] must exactly equal homie_action")
        if anchor_rgb_digest(capture.arrays) != assignment["anchor_sha256"]:
            raise ContractError("capture anchor bytes differ from the episode assignment")
        if set(capture.metadata) != CAPTURE_METADATA_KEYS:
            raise ContractError(
                f"capture metadata keys must be exactly {sorted(CAPTURE_METADATA_KEYS)}"
            )
        if tuple(capture.metadata.get("view_order", ())) != RAVEN_VIEW_ORDER:
            raise ContractError(f"online view order must be {RAVEN_VIEW_ORDER}")
        if capture.metadata.get("model_view_map") != MODEL_VIEW_MAP:
            raise ContractError("capture model_view_map mismatch")
        if capture.metadata.get("depth_units") != "meter":
            raise ContractError("metric depth units must be meter")
        if capture.metadata.get("depth_control_encoding") != "raven_h5_v4":
            raise ContractError("depth control must come from pinned Raven h5_v4")
        if capture.metadata.get("robot_control_encoding") != "textured_robot_rgb_on_black":
            raise ContractError("robot control encoding mismatch")
        expected_source = assignment["observation_source"]
        if capture.metadata.get("observation_source") != expected_source:
            raise ContractError("capture changed the fixed episode observation source")
        if expected_source == "raytraced":
            for name in ray_names:
                values = capture.arrays.get(name)
                validate_array(
                    name,
                    values,
                    ArrayRule((self.profile.policy_h, self.profile.policy_w, 3), "float32", True),
                )
                if bool((values < 0).any()) or bool((values > 1).any()):
                    raise ContractError(f"{name} must be in [0,1]")
        elif any(name in capture.arrays for name in ray_names):
            raise ContractError("generated-controls capture carried raytraced policy RGB")
        if not isinstance(capture.metadata.get("prompt"), str) or not capture.metadata["prompt"]:
            raise ContractError("capture prompt is required")
        for name in ("env_id", "seq"):
            value = capture.metadata.get(name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ContractError(f"capture {name} must be an integer")

    def _assert_lane_keys(self, keys: Sequence[StepKey]) -> None:
        expected_step = self.state["next_rollout_step"]
        for lane, (key, lane_state) in enumerate(zip(keys, self.state["lanes"], strict=True)):
            root = StepKey.from_mapping(lane_state["window_key"])
            lhs = (key.run_id, key.round_id, key.actor_id, key.rollout_window_id)
            rhs = (root.run_id, root.round_id, root.actor_id, root.rollout_window_id)
            if lhs != rhs:
                raise OwnerConflict(f"lane {lane} step belongs to a different rollout route")
            if key.rollout_step_index != expected_step:
                raise OwnerConflict(
                    f"lane {lane} expected rollout step {expected_step}, got {key.rollout_step_index}"
                )
            active = lane_state["active_episode"]
            if lane_state.get("needs_episode_bind"):
                continue
            if key.episode_id != active["episode_id"]:
                raise OwnerConflict(f"lane {lane} episode changed without done/timeout")
            if key.episode_step_index != active["next_episode_step"]:
                raise OwnerConflict(
                    f"lane {lane} expected episode step {active['next_episode_step']}, "
                    f"got {key.episode_step_index}"
                )

    def _validated_session(self, keys: Sequence[StepKey]) -> dict[str, Any]:
        identity = {
            "run_id": keys[0].run_id,
            "round_id": keys[0].round_id,
            "lane_routes": [
                {
                    "lane_index": lane,
                    "actor_id": key.actor_id,
                    "rollout_window_id": key.rollout_window_id,
                }
                for lane, key in enumerate(keys)
            ],
        }
        prior = self.state.get("session")
        if prior is None:
            return identity
        if prior["run_id"] != identity["run_id"]:
            raise OwnerConflict("batch owner is pinned to one run_id")
        prior_round = int(prior["round_id"])
        if identity["round_id"] < prior_round or identity["round_id"] > prior_round + 1:
            raise OwnerConflict("batch rounds must be monotonic and advance by at most one")
        if [item["actor_id"] for item in prior["lane_routes"]] != [
            item["actor_id"] for item in identity["lane_routes"]
        ]:
            raise OwnerConflict("vectorized lane-to-actor assignment changed")
        return identity

    def _validate_batch_identity(
        self,
        keys: Sequence[StepKey],
        *,
        require_same_step: bool,
    ) -> None:
        if len(keys) != self.num_envs:
            raise ContractError("batch key count does not equal backend num_envs")
        if len({key.token for key in keys}) != self.num_envs:
            raise ContractError("batch lanes must have unique step identities")
        routes = {(key.actor_id, key.rollout_window_id) for key in keys}
        if len(routes) != self.num_envs:
            raise ContractError("batch lanes must have unique actor/window routes")
        shared = {(key.run_id, key.round_id, key.rollout_step_index) for key in keys}
        if require_same_step and len(shared) != 1:
            raise ContractError("all vectorized lanes must share run, round, and rollout index")

    @staticmethod
    def _validate_result(result: TransitionResult) -> None:
        if not isinstance(result, TransitionResult):
            raise ContractError("step_batch returned an invalid transition result")
        if not np.isfinite(result.reward):
            raise ContractError("transition reward is not finite")
        if not isinstance(result.done, bool) or not isinstance(result.timeout, bool):
            raise ContractError("transition done and timeout must be bool")
        if not isinstance(result.metadata, Mapping):
            raise ContractError("transition metadata must be an object")
        canonical_json(dict(result.metadata))

    @staticmethod
    def _result_dict(result: TransitionResult) -> dict[str, Any]:
        return {
            "reward": float(result.reward),
            "done": result.done,
            "timeout": result.timeout,
            "metadata": dict(result.metadata),
        }

    @staticmethod
    def _manifest_binding(key: StepKey, binding: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "episode_id": key.episode_id,
            "first_rollout_step_index": key.rollout_step_index,
            "use_raytraced": binding["use_raytraced"],
            "observation_source": binding["observation_source"],
            "anchor_sha256": binding["anchor_sha256"],
        }

    @staticmethod
    def _require_exact_keys(request: Mapping[str, Any], expected: set[str], route: str) -> None:
        if not isinstance(request, Mapping) or set(request) != expected:
            raise ContractError(f"{route} keys must be exactly {sorted(expected)}")

    @staticmethod
    def _check_version(request: Mapping[str, Any]) -> None:
        if not isinstance(request, Mapping) or request.get("contract_version") != CONTRACT_VERSION:
            raise ContractError(f"contract_version must be {CONTRACT_VERSION}")

    def _require_batch(
        self,
        requests: Sequence[Mapping[str, Any]],
        name: str,
    ) -> list[Mapping[str, Any]]:
        if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
            raise ContractError(f"{name} must be a sequence")
        result = list(requests)
        if len(result) != self.num_envs:
            raise ContractError(
                f"{name} requires exactly {self.num_envs} lane requests, got {len(result)}"
            )
        return result
