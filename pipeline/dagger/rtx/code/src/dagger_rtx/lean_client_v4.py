"""Pinned-generation, sharded RTX client for pipelined DAgger v4."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar

from .binary_transport_v4 import BatchV4Transport
from .contracts import (
    ContractError,
    RavenShapeProfile,
    canonical_json,
    require_identifier,
    sha256_bytes,
)
from .http_transport import TransientTransportError
from .lean_boundary_v4 import (
    CONTRACT_VERSION_V4,
    H200_EXCHANGE_BATCH_PATH_V4,
    H200_FINALIZE_BATCH_PATH_V4,
    H200_PREPARE_BATCH_PATH_V4,
    build_exchange_batch_v4,
    build_finalize_batch_v4,
    build_prepare_batch_v4,
    validate_exchange_batch_response_v4,
    validate_finalize_batch_response_v4,
    validate_prepare_batch_response_v4,
)


class AnchorRegistrar(Protocol):
    def register_missing_anchors(
        self, local_prepares: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]: ...


class LeanH200BatchClientV4:
    """One v4 gateway pinned to one ephemeral H200 process generation."""

    protocol_version = CONTRACT_VERSION_V4

    def __init__(
        self,
        transport: BatchV4Transport,
        *,
        anchor_registrar: AnchorRegistrar | None = None,
        shape_profile: RavenShapeProfile | None = None,
    ) -> None:
        self.transport = transport
        self.anchor_registrar = anchor_registrar
        self.shape_profile = shape_profile or RavenShapeProfile()
        self.gateway_instance_id: str | None = None
        self._scratch = bytearray()

    def verify_ready(self) -> dict[str, Any]:
        value = self.transport.get_json("/ready", max_response_bytes=1 << 20)
        if value.get("ok") is not True or value.get("role") != "render_student":
            raise ContractError("H200 v4 render/student gateway is not ready")
        if value.get("schema") != "roborender.dagger.rtx.v1":
            raise ContractError("H200 v4 render/student schema mismatch")
        versions = value.get("supported_contract_versions")
        if not isinstance(versions, list) or CONTRACT_VERSION_V4 not in versions:
            raise ContractError("H200 /ready does not advertise DAgger v4")
        required = {
            "ephemeral_window_cache",
            "pipelined_commit_prepare",
            "minimal_compressed_prepare_batch",
            "source_specific_observation_payload",
            "learner_only_persistence",
        }
        features = value.get("features")
        if not isinstance(features, Mapping) or any(
            features.get(name) is not True for name in required
        ):
            raise ContractError("H200 /ready lacks required DAgger v4 features")
        if value.get("fake") is True or value.get("fake_backend") is True:
            raise ContractError("H200 v4 gateway advertises a fake backend")
        instance = require_identifier(
            value.get("gateway_instance_id"), "gateway_instance_id", 160
        )
        if self.gateway_instance_id is None:
            self.gateway_instance_id = instance
        elif instance != self.gateway_instance_id:
            raise ContractError("H200 gateway generation changed during an active v4 window")
        return dict(value)

    def _instance(self) -> str:
        if self.gateway_instance_id is None:
            raise ContractError("v4 gateway must be /ready-attested before use")
        return self.gateway_instance_id

    def _anchors(self, local_prepares: Sequence[Mapping[str, Any]]) -> None:
        if self.anchor_registrar is None:
            # Raytraced lanes need no Image3F anchor. Generated lanes do.
            if any(
                item.get("metadata", {}).get("observation_source")
                == "generated_controls"
                for item in local_prepares
            ):
                raise ContractError("generated-controls v4 prepare lacks an anchor registrar")
            return
        self.anchor_registrar.register_missing_anchors(local_prepares)

    def _post_binary(self, path: str, body: bytes, batch_id: str) -> dict[str, Any]:
        try:
            return self.transport.post_binary(
                path, body, batch_id=batch_id, max_response_bytes=1 << 20
            )
        except TransientTransportError:
            # A retry is safe only while H200's in-memory idempotency ledger is
            # the same process generation. Re-attest before the outer retry.
            self.verify_ready()
            raise

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return self.transport.post_json(
                path, payload, max_response_bytes=1 << 20
            )
        except TransientTransportError:
            self.verify_ready()
            raise

    def prepare_batch(
        self,
        local_prepares: Sequence[Mapping[str, Any]],
        idempotency_keys: Sequence[str],
    ) -> list[dict[str, Any]]:
        self._anchors(local_prepares)
        batch_id, body, records = build_prepare_batch_v4(
            local_prepares,
            idempotency_keys,
            gateway_instance_id=self._instance(),
            shape_profile=self.shape_profile,
            scratch=self._scratch,
        )
        response = self._post_binary(H200_PREPARE_BATCH_PATH_V4, body, batch_id)
        return validate_prepare_batch_response_v4(
            response,
            batch_id=batch_id,
            gateway_instance_id=self._instance(),
            request_records=records,
            local_prepares=local_prepares,
        )

    def exchange_batch(
        self,
        previous_selections: Sequence[Mapping[str, Any]],
        previous_local_commits: Sequence[Mapping[str, Any]],
        previous_commit_idempotency_keys: Sequence[str],
        current_local_prepares: Sequence[Mapping[str, Any]],
        current_prepare_idempotency_keys: Sequence[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self._anchors(current_local_prepares)
        batch_id, body, commits, prepares = build_exchange_batch_v4(
            previous_selections,
            previous_local_commits,
            previous_commit_idempotency_keys,
            current_local_prepares,
            current_prepare_idempotency_keys,
            gateway_instance_id=self._instance(),
            shape_profile=self.shape_profile,
            scratch=self._scratch,
        )
        response = self._post_binary(H200_EXCHANGE_BATCH_PATH_V4, body, batch_id)
        return validate_exchange_batch_response_v4(
            response,
            batch_id=batch_id,
            gateway_instance_id=self._instance(),
            previous_commit_lanes=commits,
            current_prepare_records=prepares,
            current_local_prepares=current_local_prepares,
        )

    def finalize_batch(
        self,
        selections: Sequence[Mapping[str, Any]],
        local_commits: Sequence[Mapping[str, Any]],
        idempotency_keys: Sequence[str],
    ) -> list[dict[str, Any]]:
        batch_id, payload, lanes = build_finalize_batch_v4(
            selections,
            local_commits,
            idempotency_keys,
            gateway_instance_id=self._instance(),
        )
        response = self._post_json(H200_FINALIZE_BATCH_PATH_V4, payload)
        return validate_finalize_batch_response_v4(
            response,
            batch_id=batch_id,
            gateway_instance_id=self._instance(),
            request_lanes=lanes,
        )

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()


T = TypeVar("T")


class ShardedLeanH200BatchClientV4:
    """Persistent v4 gateways with a process-pinned lane assignment.

    The legacy/default assignment is deterministic ``lane_index % gateways``.
    ``cost_aware_sharding=True`` computes a static assignment exactly once from
    the first (step-0) prepare batch.  Generated-control lanes are treated as
    more expensive than raytraced lanes and greedily spread over the gateways.
    The resulting actor-to-gateway mapping is immutable through retries,
    episode resets, and every subsequent window handled by this client.  A
    later window may start only after every shard's prior final flush succeeds,
    at step zero, with the identical ordered actor set.

    Pinning is required because each H200 gateway owns an ephemeral per-window
    idempotency ledger and releases a completed actor window when that actor's
    next step-zero prepare reaches the same process.  Dynamically moving a lane
    would therefore be unsafe even after finalization or an episode reset.
    """

    protocol_version = CONTRACT_VERSION_V4

    def __init__(
        self,
        gateways: Sequence[LeanH200BatchClientV4],
        *,
        cost_aware_sharding: bool = False,
    ) -> None:
        if not 1 <= len(gateways) <= 8:
            raise ValueError("v4 requires between 1 and 8 gateway clients")
        if len({id(item) for item in gateways}) != len(gateways):
            raise ValueError("v4 gateway clients must be distinct objects")
        self.gateways = tuple(gateways)
        self.cost_aware_sharding = bool(cost_aware_sharding)
        self.remote_request_count = len(gateways)
        self._lane_to_gateway: tuple[int, ...] | None = None
        self._pinned_actor_ids: tuple[str, ...] | None = None
        self._pinned_window_id: str | None = None
        self._window_finalized = False
        self._finalize_request_digest: str | None = None
        self._shard_mapping_mode: str | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=len(gateways), thread_name_prefix="h200-v4-gateway"
        )
        self._closed = False

    @property
    def lane_to_gateway(self) -> tuple[int, ...] | None:
        """Return the immutable lane assignment after the first prepare."""

        return self._lane_to_gateway

    @property
    def shard_mapping_mode(self) -> str | None:
        """Return ``cost_aware``, ``modulo``, or ``modulo_fallback``."""

        return self._shard_mapping_mode

    @staticmethod
    def _actor_ids(
        rows: Sequence[Mapping[str, Any]], *, label: str
    ) -> tuple[str, ...]:
        actor_ids: list[str] = []
        for index, row in enumerate(rows):
            step_key = row.get("step_key")
            if not isinstance(step_key, Mapping):
                raise ContractError(f"{label}[{index}] lacks step_key")
            actor_id = step_key.get("actor_id")
            if not isinstance(actor_id, str) or not actor_id:
                raise ContractError(f"{label}[{index}] lacks actor_id")
            actor_ids.append(actor_id)
        if len(set(actor_ids)) != len(actor_ids):
            raise ContractError(f"{label} contains duplicate actor_id values")
        return tuple(actor_ids)

    @staticmethod
    def _window_id(
        rows: Sequence[Mapping[str, Any]], *, label: str
    ) -> str:
        window_ids: set[str] = set()
        for index, row in enumerate(rows):
            step_key = row.get("step_key")
            if not isinstance(step_key, Mapping):
                raise ContractError(f"{label}[{index}] lacks step_key")
            window_id = step_key.get("rollout_window_id")
            if not isinstance(window_id, str) or not window_id:
                raise ContractError(f"{label}[{index}] lacks rollout_window_id")
            window_ids.add(window_id)
        if len(window_ids) != 1:
            raise ContractError(f"{label} spans more than one rollout window")
        return next(iter(window_ids))

    @staticmethod
    def _require_step_zero(
        rows: Sequence[Mapping[str, Any]], *, label: str
    ) -> None:
        for index, row in enumerate(rows):
            step_key = row.get("step_key")
            if not isinstance(step_key, Mapping) or step_key.get("rollout_step_index") != 0:
                raise ContractError(f"{label}[{index}] is not rollout step zero")

    def _cost_balanced_assignment(
        self, local_prepares: Sequence[Mapping[str, Any]]
    ) -> tuple[tuple[int, ...], str]:
        sources: list[str] = []
        for item in local_prepares:
            metadata = item.get("metadata")
            source = (
                metadata.get("observation_source")
                if isinstance(metadata, Mapping)
                else None
            )
            if source not in {"generated_controls", "raytraced"}:
                # A malformed/older payload will subsequently be rejected by
                # the v4 contract builder.  Until then, retain deterministic
                # legacy placement rather than making a partial cost guess.
                return (
                    tuple(index % len(self.gateways) for index in range(len(local_prepares))),
                    "modulo_fallback",
                )
            sources.append(source)

        # Image3F-backed generated controls dominate a raytraced student-only
        # lane.  A conservative 4:1 relative weight balances the expensive
        # lane count first while using raytraced lanes to fill idle gateways.
        weights = {"generated_controls": 4, "raytraced": 1}
        estimated_load = [0] * len(self.gateways)
        lane_count = [0] * len(self.gateways)
        assignment = [-1] * len(local_prepares)
        ordered_indices = [
            index for index, source in enumerate(sources) if source == "generated_controls"
        ] + [index for index, source in enumerate(sources) if source == "raytraced"]
        for index in ordered_indices:
            gateway = min(
                range(len(self.gateways)),
                key=lambda item: (estimated_load[item], lane_count[item], item),
            )
            assignment[index] = gateway
            estimated_load[gateway] += weights[sources[index]]
            lane_count[gateway] += 1
        if any(item < 0 for item in assignment):
            raise AssertionError("internal v4 cost-aware shard assignment gap")
        return tuple(assignment), "cost_aware"

    def _pin_lane_assignment(
        self, local_prepares: Sequence[Mapping[str, Any]]
    ) -> None:
        actor_ids = self._actor_ids(local_prepares, label="v4 prepare lanes")
        window_id = self._window_id(local_prepares, label="v4 prepare lanes")
        if self._lane_to_gateway is not None:
            if window_id == self._pinned_window_id:
                if self._window_finalized:
                    raise ContractError("v4 rollout window was already finalized")
                if actor_ids != self._pinned_actor_ids:
                    raise ContractError(
                        "v4 lane actor order changed after gateway assignment was pinned"
                    )
                return
            if not self._window_finalized:
                raise ContractError(
                    "v4 cannot start a new window before the active window is finalized"
                )
            self._require_step_zero(local_prepares, label="v4 next-window prepare")
            if actor_ids != self._pinned_actor_ids:
                raise ContractError(
                    "v4 next-window actor order differs from the process-pinned "
                    "gateway assignment"
                )
            # H200 cache eviction is process-local: the next step-zero prepare
            # for an actor must reach the same gateway that finalized its prior
            # window.  Keep the original assignment for this client's entire
            # lifetime even if newly sampled observation sources have changed.
            self._pinned_window_id = window_id
            self._window_finalized = False
            self._finalize_request_digest = None
            return
        else:
            self._require_step_zero(local_prepares, label="v4 initial prepare")
        if self.cost_aware_sharding:
            assignment, mode = self._cost_balanced_assignment(local_prepares)
        else:
            assignment = tuple(
                index % len(self.gateways) for index in range(len(local_prepares))
            )
            mode = "modulo"
        self._lane_to_gateway = assignment
        self._pinned_actor_ids = actor_ids
        self._pinned_window_id = window_id
        self._window_finalized = False
        self._finalize_request_digest = None
        self._shard_mapping_mode = mode

    def _require_pinned_actor_order(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        label: str,
        allow_finalized: bool = False,
    ) -> None:
        if self._pinned_actor_ids is None:
            raise ContractError("v4 gateway assignment was not initialized by step-0 prepare")
        if self._window_finalized and not allow_finalized:
            raise ContractError("v4 rollout window was already finalized")
        if self._actor_ids(rows, label=label) != self._pinned_actor_ids:
            raise ContractError(
                f"{label} actor order differs from the pinned v4 gateway assignment"
            )
        if self._window_id(rows, label=label) != self._pinned_window_id:
            raise ContractError(
                f"{label} rollout window differs from the pinned v4 gateway assignment"
            )

    @staticmethod
    def _finalize_digest(
        selections: Sequence[Mapping[str, Any]],
        local_commits: Sequence[Mapping[str, Any]],
        idempotency_keys: Sequence[str],
    ) -> str:
        """Hash exactly the caller fields that form the finalize wire lanes."""

        lanes = []
        for selection, local, idempotency_key in zip(
            selections, local_commits, idempotency_keys, strict=True
        ):
            lanes.append(
                {
                    "idempotency_key": idempotency_key,
                    "step_key": selection.get("step_key"),
                    "prepare_receipt": selection.get("prepare_receipt"),
                    "selection_id": selection.get("selection_id"),
                    "selection_receipt": selection.get("selection_receipt"),
                    "commit_receipt": local.get("commit_receipt"),
                    "transition": local.get("result"),
                    "sealed": local.get("sealed"),
                    "sealed_window": local.get("seal"),
                }
            )
        return sha256_bytes(canonical_json(lanes))

    def _indices(self, lane_count: int, *, use_pinned: bool = True) -> list[list[int]]:
        if lane_count < 1:
            raise ContractError("v4 sharded batch must be non-empty")
        if use_pinned and self._lane_to_gateway is not None:
            if lane_count != len(self._lane_to_gateway):
                raise ContractError("v4 lane count changed after gateway assignment was pinned")
            return [
                [
                    lane
                    for lane, assigned_gateway in enumerate(self._lane_to_gateway)
                    if assigned_gateway == gateway
                ]
                for gateway in range(len(self.gateways))
            ]
        return [list(range(shard, lane_count, len(self.gateways))) for shard in range(len(self.gateways))]

    def _settled(
        self,
        lane_count: int,
        call: Callable[[LeanH200BatchClientV4, list[int]], Sequence[T]],
        *,
        use_pinned: bool = True,
    ) -> list[T]:
        if self._closed:
            raise RuntimeError("v4 sharded client is closed")
        futures: list[tuple[list[int], Future[Sequence[T]]]] = []
        for gateway, indices in zip(
            self.gateways,
            self._indices(lane_count, use_pinned=use_pinned),
            strict=True,
        ):
            if indices:
                futures.append((indices, self._executor.submit(call, gateway, indices)))
        result: list[T | None] = [None] * lane_count
        failures: list[BaseException] = []
        for indices, future in futures:
            try:
                rows = list(future.result())
                if len(rows) != len(indices):
                    raise ContractError("v4 gateway returned wrong shard cardinality")
                for index, row in zip(indices, rows, strict=True):
                    result[index] = row
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise failures[0]
        if any(item is None for item in result):
            raise AssertionError("internal v4 shard reassembly gap")
        return [item for item in result if item is not None]

    def verify_ready(self) -> dict[str, Any]:
        rows = self._settled(
            len(self.gateways),
            lambda gateway, _indices: [gateway.verify_ready()],
            use_pinned=False,
        )
        return {
            "contract_version": CONTRACT_VERSION_V4,
            "gateway_count": len(rows),
            "gateways": rows,
        }

    def prepare_batch(self, local_prepares, idempotency_keys):
        if not local_prepares or len(local_prepares) != len(idempotency_keys):
            raise ContractError("v4 sharded prepare cardinality mismatch")
        self._pin_lane_assignment(local_prepares)

        def call(gateway, indices):
            return gateway.prepare_batch(
                [local_prepares[i] for i in indices],
                [idempotency_keys[i] for i in indices],
            )

        return self._settled(len(local_prepares), call)

    def exchange_batch(
        self,
        previous_selections,
        previous_local_commits,
        previous_commit_idempotency_keys,
        current_local_prepares,
        current_prepare_idempotency_keys,
    ):
        lane_count = len(current_local_prepares)
        if not lane_count or any(
            len(value) != lane_count
            for value in (
                previous_selections,
                previous_local_commits,
                previous_commit_idempotency_keys,
                current_prepare_idempotency_keys,
            )
        ):
            raise ContractError("v4 sharded exchange cardinality mismatch")
        self._require_pinned_actor_order(
            previous_selections, label="v4 previous selections"
        )
        self._require_pinned_actor_order(
            previous_local_commits, label="v4 previous commits"
        )
        self._require_pinned_actor_order(
            current_local_prepares, label="v4 current prepares"
        )

        def call(gateway, indices):
            acks, selections = gateway.exchange_batch(
                [previous_selections[i] for i in indices],
                [previous_local_commits[i] for i in indices],
                [previous_commit_idempotency_keys[i] for i in indices],
                [current_local_prepares[i] for i in indices],
                [current_prepare_idempotency_keys[i] for i in indices],
            )
            return list(zip(acks, selections, strict=True))

        paired = self._settled(lane_count, call)
        return [item[0] for item in paired], [item[1] for item in paired]

    def finalize_batch(self, selections, local_commits, idempotency_keys):
        lane_count = len(selections)
        if not lane_count or len(local_commits) != lane_count or len(idempotency_keys) != lane_count:
            raise ContractError("v4 sharded finalize cardinality mismatch")
        self._require_pinned_actor_order(
            selections, label="v4 final selections", allow_finalized=True
        )
        self._require_pinned_actor_order(
            local_commits, label="v4 final commits", allow_finalized=True
        )
        request_digest = self._finalize_digest(
            selections, local_commits, idempotency_keys
        )
        if self._finalize_request_digest is None:
            # Retain the first attempt as well as a successful attempt: a
            # partial all-shard failure may already have committed remotely.
            self._finalize_request_digest = request_digest
        elif request_digest != self._finalize_request_digest:
            raise ContractError("v4 finalize retry differs from the pinned request")

        def call(gateway, indices):
            return gateway.finalize_batch(
                [selections[i] for i in indices],
                [local_commits[i] for i in indices],
                [idempotency_keys[i] for i in indices],
            )

        result = self._settled(lane_count, call)
        # Only a complete all-shard response permits the next window to start.
        # The request digest remains available so an identical finalize can be
        # replayed idempotently until that next step-zero prepare arrives.
        self._window_finalized = True
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        for gateway in self.gateways:
            gateway.close()
