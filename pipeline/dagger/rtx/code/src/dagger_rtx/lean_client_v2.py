"""RTX-side cached-anchor and batched prepare/commit client for DAgger v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .binary_transport_v2 import BatchV2Transport
from .contracts import ContractError, RavenShapeProfile, require_sha256
from .durable import atomic_write_json
from .http_transport import RemoteHttpError
from .lean_boundary_v2 import (
    CONTRACT_VERSION_V2,
    H200_ANCHOR_BATCH_PATH_V2,
    H200_COMMIT_BATCH_PATH_V2,
    H200_PREPARE_BATCH_PATH_V2,
    anchor_records_from_prepares,
    build_anchor_batch,
    build_commit_batch,
    build_prepare_batch,
    validate_anchor_batch_response,
    validate_commit_batch_response,
    validate_prepare_batch_response,
)


class LeanH200BatchClient:
    """One remote request per prepare/commit phase with durable anchor ACKs."""

    protocol_version = CONTRACT_VERSION_V2
    required_features = frozenset(
        {
            "lean_binary_prepare_batch",
            "content_addressed_anchor_cache",
            "uncompressed_batch_persistence",
            "batched_commit",
        }
    )

    def __init__(
        self,
        transport: BatchV2Transport,
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
        if not value.get("ok") or value.get("role") != "render_student":
            raise ContractError("H200 render/student boundary is not ready")
        versions = value.get("supported_contract_versions")
        if not isinstance(versions, list) or CONTRACT_VERSION_V2 not in versions:
            raise ContractError("H200 /ready does not advertise DAgger v2")
        features = value.get("features")
        if not isinstance(features, Mapping):
            raise ContractError("H200 /ready is missing v2 feature capabilities")
        missing = sorted(
            name for name in self.required_features if features.get(name) is not True
        )
        if missing:
            raise ContractError(f"H200 /ready lacks required v2 features: {missing}")
        return dict(value)

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {
                "schema": "roborender.dagger.rtx.anchor-cache.v2",
                "contract_version": CONTRACT_VERSION_V2,
                "anchors": {},
            }
        value = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema") != "roborender.dagger.rtx.anchor-cache.v2"
            or value.get("contract_version") != CONTRACT_VERSION_V2
            or not isinstance(value.get("anchors"), dict)
        ):
            raise ContractError("invalid durable v2 anchor cache")
        for digest, record in value["anchors"].items():
            require_sha256(digest, "cached anchor SHA-256")
            if (
                not isinstance(record, dict)
                or set(record) != {"bytes", "status"}
                or record["status"] not in {"registered", "cached"}
                or isinstance(record["bytes"], bool)
                or not isinstance(record["bytes"], int)
                or record["bytes"] <= 0
            ):
                raise ContractError("invalid durable v2 anchor cache record")
        return value

    def _register_missing_anchors(
        self, local_prepares: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        records, groups = anchor_records_from_prepares(
            local_prepares, shape_profile=self.shape_profile
        )
        missing_records: list[dict[str, Any]] = []
        missing_groups: list[dict[str, np.ndarray]] = []
        for record, group in zip(records, groups, strict=True):
            digest = record["anchor_sha256"]
            if digest not in self._cache["anchors"]:
                missing_records.append(record)
                missing_groups.append(group)
        if not missing_records:
            return []
        batch_id, body = build_anchor_batch(missing_records, missing_groups)
        response = self.transport.post_binary(
            H200_ANCHOR_BATCH_PATH_V2,
            body,
            batch_id=batch_id,
            max_response_bytes=1 << 20,
        )
        acknowledgements = validate_anchor_batch_response(
            response,
            batch_id=batch_id,
            expected=missing_records,
        )
        # bytes is the H200 content-addressed NPZ artifact size (including
        # container headers), not the sum of raw ndarray payload bytes.
        # Identity is bound independently by anchor_sha256 and descriptors.
        next_cache = {
            "schema": self._cache["schema"],
            "contract_version": CONTRACT_VERSION_V2,
            "anchors": dict(self._cache["anchors"]),
        }
        for ack in acknowledgements:
            next_cache["anchors"][ack["anchor_sha256"]] = {
                "bytes": ack["bytes"],
                "status": ack["status"],
            }
        atomic_write_json(self.cache_path, next_cache)
        self._cache = next_cache
        return acknowledgements

    def prepare_batch(
        self,
        local_prepares: Sequence[Mapping[str, Any]],
        idempotency_keys: Sequence[str],
    ) -> list[dict[str, Any]]:
        self._register_missing_anchors(local_prepares)
        batch_id, body, records, _groups = build_prepare_batch(
            local_prepares,
            idempotency_keys,
            shape_profile=self.shape_profile,
        )
        try:
            response = self.transport.post_binary(
                H200_PREPARE_BATCH_PATH_V2,
                body,
                batch_id=batch_id,
                max_response_bytes=1 << 20,
            )
        except RemoteHttpError as exc:
            if exc.status != 412:
                raise
            # H200 may have intentionally replaced its content-addressed cache
            # while RTX retained durable ACKs. Invalidate only anchors required
            # by this batch, re-register them once, then replay identical
            # per-lane idempotency keys and binary prepare bytes.
            stale_anchor_records, _groups = anchor_records_from_prepares(
                local_prepares, shape_profile=self.shape_profile
            )
            next_cache = {
                "schema": self._cache["schema"],
                "contract_version": CONTRACT_VERSION_V2,
                "anchors": dict(self._cache["anchors"]),
            }
            for record in stale_anchor_records:
                next_cache["anchors"].pop(record["anchor_sha256"], None)
            atomic_write_json(self.cache_path, next_cache)
            self._cache = next_cache
            self._register_missing_anchors(local_prepares)
            response = self.transport.post_binary(
                H200_PREPARE_BATCH_PATH_V2,
                body,
                batch_id=batch_id,
                max_response_bytes=1 << 20,
            )
        return validate_prepare_batch_response(
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
        batch_id, payload, lanes = build_commit_batch(
            selections,
            local_commits,
            idempotency_keys,
        )
        response = self.transport.post_json(
            H200_COMMIT_BATCH_PATH_V2,
            payload,
            max_response_bytes=1 << 20,
        )
        return validate_commit_batch_response(
            response,
            batch_id=batch_id,
            request_lanes=lanes,
        )

    @property
    def cached_anchor_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted(self._cache["anchors"]))
