"""Pinned-generation, sharded RTX client for pipelined DAgger v4."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar

from .binary_transport_v4 import BatchV4Transport
from .contracts import ContractError, RavenShapeProfile, require_identifier
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
    """Eight persistent v4 gateways with deterministic lane ``i % 8``."""

    protocol_version = CONTRACT_VERSION_V4

    def __init__(self, gateways: Sequence[LeanH200BatchClientV4]) -> None:
        if not 1 <= len(gateways) <= 8:
            raise ValueError("v4 requires between 1 and 8 gateway clients")
        if len({id(item) for item in gateways}) != len(gateways):
            raise ValueError("v4 gateway clients must be distinct objects")
        self.gateways = tuple(gateways)
        self.remote_request_count = len(gateways)
        self._executor = ThreadPoolExecutor(
            max_workers=len(gateways), thread_name_prefix="h200-v4-gateway"
        )
        self._closed = False

    def _indices(self, lane_count: int) -> list[list[int]]:
        if lane_count < 1:
            raise ContractError("v4 sharded batch must be non-empty")
        return [
            list(range(shard, lane_count, len(self.gateways)))
            for shard in range(len(self.gateways))
        ]

    def _settled(
        self,
        lane_count: int,
        call: Callable[[LeanH200BatchClientV4, list[int]], Sequence[T]],
    ) -> list[T]:
        if self._closed:
            raise RuntimeError("v4 sharded client is closed")
        futures: list[tuple[list[int], Future[Sequence[T]]]] = []
        for gateway, indices in zip(self.gateways, self._indices(lane_count), strict=True):
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
            len(self.gateways), lambda gateway, _indices: [gateway.verify_ready()]
        )
        return {
            "contract_version": CONTRACT_VERSION_V4,
            "gateway_count": len(rows),
            "gateways": rows,
        }

    def prepare_batch(self, local_prepares, idempotency_keys):
        if not local_prepares or len(local_prepares) != len(idempotency_keys):
            raise ContractError("v4 sharded prepare cardinality mismatch")

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

        def call(gateway, indices):
            return gateway.finalize_batch(
                [selections[i] for i in indices],
                [local_commits[i] for i in indices],
                [idempotency_keys[i] for i in indices],
            )

        return self._settled(lane_count, call)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        for gateway in self.gateways:
            gateway.close()
