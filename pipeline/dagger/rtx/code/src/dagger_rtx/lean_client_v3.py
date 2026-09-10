"""RTX clients for compressed, source-minimal, sharded DAgger v3."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

import numpy as np

from .binary_transport_v3 import BatchV3Transport
from .contracts import ContractError, RavenShapeProfile, require_sha256
from .durable import atomic_write_json
from .http_transport import RemoteHttpError
from .lean_boundary_v3 import (
    CONTRACT_VERSION_V3,
    H200_ANCHOR_BATCH_PATH_V3,
    H200_COMMIT_BATCH_PATH_V3,
    H200_PREPARE_BATCH_PATH_V3,
    anchor_records_from_prepares_v3,
    build_anchor_batch_v3,
    build_commit_batch_v3,
    build_prepare_batch_v3,
    validate_anchor_batch_response_v3,
    validate_commit_batch_response_v3,
    validate_prepare_batch_response_v3,
)


class LeanH200BatchClientV3:
    """One v3 gateway connection with a durable content-addressed anchor cache."""

    protocol_version = CONTRACT_VERSION_V3

    def __init__(
        self,
        transport: BatchV3Transport,
        runtime_root: Path,
        *,
        shape_profile: RavenShapeProfile | None = None,
    ) -> None:
        self.transport = transport
        self.runtime_root = Path(runtime_root)
        self.runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.cache_path = self.runtime_root / "anchor-cache.json"
        self.shape_profile = shape_profile or RavenShapeProfile()
        self._cache = self._load_cache()

    def verify_ready(self) -> dict[str, Any]:
        value = self.transport.get_json("/ready", max_response_bytes=1 << 20)
        if value.get("ok") is not True or value.get("role") != "render_student":
            raise ContractError("H200 v3 render/student gateway is not ready")
        if value.get("schema") != "roborender.dagger.rtx.v1":
            raise ContractError("H200 v3 render/student gateway schema mismatch")
        versions = value.get("supported_contract_versions")
        if not isinstance(versions, list) or CONTRACT_VERSION_V3 not in versions:
            raise ContractError("H200 /ready does not advertise DAgger v3")
        required = {
            "minimal_compressed_prepare_batch",
            "content_addressed_anchor_cache",
            "source_specific_observation_payload",
            "learner_only_persistence",
            "shared_filesystem_chunk_publish",
            "batched_commit",
        }
        features = value.get("features")
        if not isinstance(features, dict) or any(
            features.get(name) is not True for name in required
        ):
            raise ContractError("H200 /ready lacks required DAgger v3 features")
        if value.get("fake") is True or value.get("fake_backend") is True:
            raise ContractError("H200 v3 render/student gateway advertises a fake backend")
        return dict(value)

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {
                "schema": "roborender.dagger.rtx.anchor-cache.v3",
                "contract_version": CONTRACT_VERSION_V3,
                "anchors": {},
            }
        value = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema") != "roborender.dagger.rtx.anchor-cache.v3"
            or value.get("contract_version") != CONTRACT_VERSION_V3
            or not isinstance(value.get("anchors"), dict)
        ):
            raise ContractError("invalid durable v3 anchor cache")
        for digest, record in value["anchors"].items():
            require_sha256(digest, "cached anchor SHA-256")
            if (
                not isinstance(record, dict)
                or set(record) != {"bytes", "status"}
                or record.get("status") not in {"registered", "cached"}
                or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)
                or record["bytes"] <= 0
            ):
                raise ContractError("invalid durable v3 anchor cache record")
        return value

    def _persist_anchor_acks(self, acknowledgements: Sequence[Mapping[str, Any]]) -> None:
        next_cache = {
            "schema": self._cache["schema"],
            "contract_version": CONTRACT_VERSION_V3,
            "anchors": dict(self._cache["anchors"]),
        }
        for ack in acknowledgements:
            next_cache["anchors"][ack["anchor_sha256"]] = {
                "bytes": ack["bytes"],
                "status": ack["status"],
            }
        atomic_write_json(self.cache_path, next_cache)
        self._cache = next_cache

    def _register_missing_anchors(
        self, local_prepares: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        records, groups = anchor_records_from_prepares_v3(
            local_prepares, shape_profile=self.shape_profile
        )
        missing = [
            (record, group)
            for record, group in zip(records, groups, strict=True)
            if record["anchor_sha256"] not in self._cache["anchors"]
        ]
        if not missing:
            return []
        missing_records = [record for record, _group in missing]
        missing_groups = [group for _record, group in missing]
        batch_id, body = build_anchor_batch_v3(missing_records, missing_groups)
        response = self.transport.post_binary(
            H200_ANCHOR_BATCH_PATH_V3,
            body,
            batch_id=batch_id,
            max_response_bytes=1 << 20,
        )
        acknowledgements = validate_anchor_batch_response_v3(
            response, batch_id=batch_id, expected=missing_records
        )
        self._persist_anchor_acks(acknowledgements)
        return acknowledgements

    def register_missing_anchors(
        self, local_prepares: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Public cold-path anchor hook reused by later wire protocols.

        V3 behavior is unchanged; exposing the already-audited operation avoids
        duplicating content-addressed cache semantics in pipelined clients.
        """

        return self._register_missing_anchors(local_prepares)

    def _invalidate_batch_anchors(
        self, local_prepares: Sequence[Mapping[str, Any]]
    ) -> None:
        records, _groups = anchor_records_from_prepares_v3(
            local_prepares, shape_profile=self.shape_profile
        )
        next_cache = {
            "schema": self._cache["schema"],
            "contract_version": CONTRACT_VERSION_V3,
            "anchors": dict(self._cache["anchors"]),
        }
        for record in records:
            next_cache["anchors"].pop(record["anchor_sha256"], None)
        atomic_write_json(self.cache_path, next_cache)
        self._cache = next_cache

    def prepare_batch(
        self,
        local_prepares: Sequence[Mapping[str, Any]],
        idempotency_keys: Sequence[str],
    ) -> list[dict[str, Any]]:
        self._register_missing_anchors(local_prepares)
        batch_id, body, records, _groups = build_prepare_batch_v3(
            local_prepares,
            idempotency_keys,
            shape_profile=self.shape_profile,
        )
        try:
            response = self.transport.post_binary(
                H200_PREPARE_BATCH_PATH_V3,
                body,
                batch_id=batch_id,
                max_response_bytes=1 << 20,
            )
        except RemoteHttpError as exc:
            if exc.status != 412:
                raise
            self._invalidate_batch_anchors(local_prepares)
            self._register_missing_anchors(local_prepares)
            response = self.transport.post_binary(
                H200_PREPARE_BATCH_PATH_V3,
                body,
                batch_id=batch_id,
                max_response_bytes=1 << 20,
            )
        return validate_prepare_batch_response_v3(
            response,
            batch_id=batch_id,
            request_records=records,
            local_prepares=local_prepares,
        )

    def commit_batch(
        self,
        selections: Sequence[Mapping[str, Any]],
        local_commits: Sequence[Mapping[str, Any]],
        idempotency_keys: Sequence[str],
    ) -> list[dict[str, Any]]:
        batch_id, payload, lanes = build_commit_batch_v3(
            selections, local_commits, idempotency_keys
        )
        response = self.transport.post_json(
            H200_COMMIT_BATCH_PATH_V3,
            payload,
            max_response_bytes=1 << 20,
        )
        return validate_commit_batch_response_v3(
            response, batch_id=batch_id, request_lanes=lanes
        )

    @property
    def cached_anchor_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted(self._cache["anchors"]))

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()


T = TypeVar("T")


class ShardedLeanH200BatchClientV3:
    """Deterministically shard vector lanes over persistent v3 gateways.

    Lane ``i`` always uses gateway ``i % gateway_count``. All gateway calls
    settle before this client returns or raises, and successful results are
    restored to original vector order. The caller therefore retains one
    all-lane prepare barrier before crossing the native ``env.step`` boundary.
    """

    protocol_version = CONTRACT_VERSION_V3

    def __init__(self, gateways: Sequence[LeanH200BatchClientV3]) -> None:
        if not 1 <= len(gateways) <= 8:
            raise ValueError("v3 requires between 1 and 8 gateway clients")
        if len({id(gateway) for gateway in gateways}) != len(gateways):
            raise ValueError("v3 gateway clients must be distinct objects")
        self.gateways = tuple(gateways)
        self.remote_request_count = len(self.gateways)
        self._executor = ThreadPoolExecutor(
            max_workers=len(self.gateways), thread_name_prefix="h200-v3-gateway"
        )
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("v3 sharded client is closed")

    def _indices_by_gateway(self, lane_count: int) -> list[list[int]]:
        if lane_count < 1:
            raise ContractError("v3 sharded batch must contain at least one lane")
        return [
            list(range(shard, lane_count, len(self.gateways)))
            for shard in range(len(self.gateways))
        ]

    def _settled_ordered(
        self,
        lane_count: int,
        call: Callable[[LeanH200BatchClientV3, list[int]], Sequence[T]],
    ) -> list[T]:
        """Run non-empty shards concurrently, settle all, then reassemble."""

        self._require_open()
        groups = self._indices_by_gateway(lane_count)
        futures: list[tuple[list[int], Future[Sequence[T]]]] = []
        for gateway, indices in zip(self.gateways, groups, strict=True):
            if indices:
                futures.append((indices, self._executor.submit(call, gateway, indices)))
        results: list[T | None] = [None] * lane_count
        failures: list[BaseException] = []
        for indices, future in futures:
            try:
                rows = list(future.result())
                if len(rows) != len(indices):
                    raise ContractError("v3 gateway returned wrong shard cardinality")
                for index, row in zip(indices, rows, strict=True):
                    results[index] = row
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise failures[0]
        if any(row is None for row in results):
            raise AssertionError("internal v3 shard reassembly gap")
        return [row for row in results if row is not None]

    def verify_ready(self) -> dict[str, Any]:
        self._require_open()
        futures = [self._executor.submit(gateway.verify_ready) for gateway in self.gateways]
        identities: list[dict[str, Any]] = []
        failures: list[BaseException] = []
        for future in futures:
            try:
                identities.append(future.result())
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise failures[0]
        return {
            "contract_version": CONTRACT_VERSION_V3,
            "gateway_count": len(identities),
            "gateways": identities,
        }

    def prepare_batch(
        self,
        local_prepares: Sequence[Mapping[str, Any]],
        idempotency_keys: Sequence[str],
    ) -> list[dict[str, Any]]:
        if not local_prepares or len(local_prepares) != len(idempotency_keys):
            raise ContractError("v3 sharded prepare cardinality mismatch")

        def call(gateway: LeanH200BatchClientV3, indices: list[int]):
            return gateway.prepare_batch(
                [local_prepares[index] for index in indices],
                [idempotency_keys[index] for index in indices],
            )

        return self._settled_ordered(len(local_prepares), call)

    def commit_batch(
        self,
        selections: Sequence[Mapping[str, Any]],
        local_commits: Sequence[Mapping[str, Any]],
        idempotency_keys: Sequence[str],
    ) -> list[dict[str, Any]]:
        if not selections or not (
            len(selections) == len(local_commits) == len(idempotency_keys)
        ):
            raise ContractError("v3 sharded commit cardinality mismatch")

        def call(gateway: LeanH200BatchClientV3, indices: list[int]):
            return gateway.commit_batch(
                [selections[index] for index in indices],
                [local_commits[index] for index in indices],
                [idempotency_keys[index] for index in indices],
            )

        return self._settled_ordered(len(selections), call)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        for gateway in self.gateways:
            gateway.close()
