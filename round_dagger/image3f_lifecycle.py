#!/usr/bin/env python3
"""Crash-safe Image3F stream ownership and cleanup for actor-pair retries.

Every actor-pair attempt receives a distinct renderer namespace.  The attempt
publishes an immutable plan before either rank starts, each exact rank
completion publishes an immutable ownership receipt, and a sealed cohort is
not reported successful until every planned stream has an acknowledged,
idempotent ``reset_stream`` response from the endpoint selected by the same
mapping as the production client.

The module deliberately uses only the Python standard library and the
control-plane artifact validators.  It can therefore run outside Isaac and
outside the actor containers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import socket
import stat
import struct
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .actor_pair import inspect_actor_rank, require_actor_rank_completion
from .artifacts import (
    canonical_json_sha256,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
)


PLAN_SCHEMA = "fast_roborender.round_dagger.image3f_stream_plan.v2"
COMPLETION_SCHEMA = "fast_roborender.round_dagger.image3f_rank_stream_completion.v1"
RESET_ACK_SCHEMA = "fast_roborender.round_dagger.image3f_stream_reset_ack.v1"
CLEANUP_SCHEMA = "fast_roborender.round_dagger.image3f_stream_cleanup.v1"
ABANDONMENT_SCHEMA = "fast_roborender.round_dagger.image3f_rank_stream_abandonment.v1"

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_STREAM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ATTEMPT_DIR_RE = re.compile(r"attempt_[A-Za-z0-9_.-]{1,160}\Z")
_PLAN_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "cohort_index",
    "attempt_id",
    "namespace_policy",
    "attempt_namespace",
    "cohort_dir",
    "endpoint_manifest",
    "endpoint_manifest_sha256",
    "renderer_sha256",
    "local_environments",
    "actor_world_size",
    "launched_ranks",
    "streams",
}
_STREAM_KEYS = {
    "rank",
    "local_environment",
    "global_slot",
    "stream_id",
    "endpoint_index",
    "endpoint_file",
    "host",
    "port",
}
_COMPLETION_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "cohort_index",
    "rank",
    "attempt_id",
    "attempt_namespace",
    "plan_path",
    "plan_sha256",
    "stream_count",
    "stream_ids_sha256",
    "cohort_contract_sha256",
    "transition_count",
    "global_env_ids",
}
_ACK_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "cohort_index",
    "attempt_id",
    "attempt_namespace",
    "rank",
    "local_environment",
    "global_slot",
    "stream_id",
    "endpoint_index",
    "endpoint_file",
    "host",
    "port",
    "server_id",
    "renderer_identity_sha256",
    "removed",
    "acknowledged",
}
_ABANDONMENT_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "cohort_index",
    "rank",
    "attempt_id",
    "attempt_namespace",
    "plan_path",
    "plan_sha256",
    "stream_count",
    "stream_ids_sha256",
    "reason",
}
_CLEANUP_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "cohort_index",
    "contract_sha256",
    "successful_stream_count",
    "successful_stream_ids_sha256",
    "planned_stream_count",
    "acknowledged_stream_count",
    "acknowledgements",
    "status",
}
_CLEANUP_ACK_REF_KEYS = {"path", "sha256", "stream_id"}


class Image3FLifecycleError(RuntimeError):
    """An attempt plan, completion receipt, mapping, or reset is invalid."""


def _exact_mapping(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Image3FLifecycleError(f"{label} must be a mapping")
    actual = set(value)
    if actual != keys:
        raise Image3FLifecycleError(
            f"{label} has missing={sorted(keys - actual)} unknown={sorted(actual - keys)}"
        )
    return dict(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Image3FLifecycleError(f"{label} must be an integer >= {minimum}")
    return value


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None:
        raise Image3FLifecycleError(f"{label} is not a safe token")
    return value


def _stream_token(value: Any, label: str) -> str:
    if not isinstance(value, str) or _STREAM_RE.fullmatch(value) is None:
        raise Image3FLifecycleError(f"{label} is not a safe stream token")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Image3FLifecycleError(f"{label} must be an exact lowercase SHA-256")
    return value


def _absolute_regular(path: str | os.PathLike[str], label: str) -> Path:
    value = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if not value.is_absolute() or value.is_symlink() or not value.is_file():
        raise Image3FLifecycleError(f"{label} is not an absolute regular file: {value}")
    return value


def _absolute_directory(path: str | os.PathLike[str], label: str) -> Path:
    value = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if not value.is_absolute() or value.is_symlink() or not value.is_dir():
        raise Image3FLifecycleError(f"{label} is not an absolute directory: {value}")
    return value


def _read_endpoint(path: Path) -> tuple[str, int]:
    descriptor = _absolute_regular(path, "endpoint descriptor")
    try:
        fields = descriptor.read_text(encoding="utf-8").split()
    except (OSError, UnicodeError) as exc:
        raise Image3FLifecycleError(f"cannot read endpoint descriptor {descriptor}: {exc}") from exc
    if len(fields) != 2:
        raise Image3FLifecycleError(
            f"endpoint descriptor must contain exactly 'host port': {descriptor}"
        )
    host = fields[0]
    if not host or any(character.isspace() for character in host):
        raise Image3FLifecycleError(f"endpoint host is malformed: {descriptor}")
    try:
        port = int(fields[1])
    except ValueError as exc:
        raise Image3FLifecycleError(f"endpoint port is malformed: {descriptor}") from exc
    if not 1 <= port <= 65535:
        raise Image3FLifecycleError(f"endpoint port is outside 1..65535: {descriptor}")
    return host, port


def load_endpoint_manifest(
    path: str | os.PathLike[str], *, expected_count: int
) -> list[tuple[Path, str, int]]:
    manifest = _absolute_regular(path, "endpoint manifest")
    try:
        raw = [
            line.strip()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError) as exc:
        raise Image3FLifecycleError(f"cannot read endpoint manifest: {exc}") from exc
    if len(raw) != expected_count:
        raise Image3FLifecycleError(
            f"endpoint manifest has {len(raw)} entries, expected {expected_count}"
        )
    if len(set(raw)) != len(raw):
        raise Image3FLifecycleError("endpoint manifest contains duplicate descriptors")
    result: list[tuple[Path, str, int]] = []
    for item in raw:
        descriptor = _absolute_regular(item, "endpoint descriptor")
        host, port = _read_endpoint(descriptor)
        result.append((descriptor, host, port))
    if len({(host, port) for _, host, port in result}) != len(result):
        raise Image3FLifecycleError("endpoint manifest contains duplicate network endpoints")
    return result


def _manifest_entry_count(path: Path) -> int:
    try:
        count = sum(
            bool(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    except (OSError, UnicodeError) as exc:
        raise Image3FLifecycleError(f"cannot read endpoint manifest: {exc}") from exc
    if count < 1:
        raise Image3FLifecycleError("endpoint manifest is empty")
    return count


def attempt_namespace(
    *,
    campaign_id: str,
    macro_round: int,
    cohort_index: int,
    attempt_dir: str | os.PathLike[str],
) -> str:
    """Return a compact deterministic namespace unique to one attempt path."""

    campaign = _token(campaign_id, "campaign_id")
    macro_round = _integer(macro_round, "macro_round")
    cohort_index = _integer(cohort_index, "cohort_index")
    attempt = _absolute_directory(attempt_dir, "attempt directory")
    if _ATTEMPT_DIR_RE.fullmatch(attempt.name) is None:
        raise Image3FLifecycleError("attempt directory name is malformed")
    campaign_digest = hashlib.sha256(campaign.encode("utf-8")).hexdigest()[:12]
    attempt_digest = hashlib.sha256(os.fsencode(attempt)).hexdigest()[:20]
    namespace = (
        f"rda-{campaign_digest}-m{macro_round:06d}-c{cohort_index:02d}-a{attempt_digest}"
    )
    return _stream_token(namespace, "attempt namespace")


def legacy_cohort_namespace(
    *, campaign_id: str, macro_round: int, cohort_index: int
) -> str:
    """Return the exact cohort-stable namespace used before this hotfix."""

    campaign = _token(campaign_id, "campaign_id")
    macro_round = _integer(macro_round, "macro_round")
    cohort_index = _integer(cohort_index, "cohort_index")
    return _stream_token(
        f"{campaign}-m{macro_round}-c{cohort_index}", "legacy cohort namespace"
    )


def _stream_rows(
    *,
    namespace: str,
    ranks: Sequence[int],
    local_environments: int,
    endpoints: Sequence[tuple[Path, str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in ranks:
        rank = _integer(rank, "rank")
        for local_environment in range(local_environments):
            global_slot = rank * local_environments + local_environment
            endpoint_index = global_slot % len(endpoints)
            endpoint_file, host, port = endpoints[endpoint_index]
            stream_id = _stream_token(
                f"{namespace}:rank{rank}:env{global_slot}", "stream_id"
            )
            rows.append(
                {
                    "rank": rank,
                    "local_environment": local_environment,
                    "global_slot": global_slot,
                    "stream_id": stream_id,
                    "endpoint_index": endpoint_index,
                    "endpoint_file": str(endpoint_file),
                    "host": host,
                    "port": port,
                }
            )
    return rows


def create_attempt_plan(
    output: str | os.PathLike[str],
    *,
    campaign_id: str,
    macro_round: int,
    cohort_index: int,
    attempt_dir: str | os.PathLike[str],
    cohort_dir: str | os.PathLike[str],
    endpoint_manifest: str | os.PathLike[str],
    expected_renderers: int,
    renderer_sha256: str,
    local_environments: int,
    actor_world_size: int,
    launched_ranks: Sequence[int],
    namespace_policy: str = "attempt_unique",
) -> dict[str, Any]:
    attempt = _absolute_directory(attempt_dir, "attempt directory")
    output_path = Path(os.path.abspath(os.path.expanduser(os.fspath(output))))
    if output_path.parent != attempt:
        raise Image3FLifecycleError("attempt plan must be published directly in attempt directory")
    cohort = _absolute_directory(cohort_dir, "cohort directory")
    campaign = _token(campaign_id, "campaign_id")
    macro_round = _integer(macro_round, "macro_round")
    cohort_index = _integer(cohort_index, "cohort_index")
    renderer_sha = _sha256(renderer_sha256, "renderer_sha256")
    local_environments = _integer(
        local_environments, "local_environments", minimum=1
    )
    actor_world_size = _integer(actor_world_size, "actor_world_size", minimum=1)
    ranks = [_integer(rank, "launched_ranks[]") for rank in launched_ranks]
    if not ranks or ranks != sorted(set(ranks)):
        raise Image3FLifecycleError("launched_ranks must be sorted, unique, and non-empty")
    if any(rank >= actor_world_size for rank in ranks):
        raise Image3FLifecycleError("launched rank is outside actor world size")
    manifest = _absolute_regular(endpoint_manifest, "endpoint manifest")
    endpoints = load_endpoint_manifest(manifest, expected_count=expected_renderers)
    if namespace_policy == "attempt_unique":
        namespace = attempt_namespace(
            campaign_id=campaign,
            macro_round=macro_round,
            cohort_index=cohort_index,
            attempt_dir=attempt,
        )
    elif namespace_policy == "legacy_cohort":
        namespace = legacy_cohort_namespace(
            campaign_id=campaign,
            macro_round=macro_round,
            cohort_index=cohort_index,
        )
    else:
        raise Image3FLifecycleError("namespace_policy must be attempt_unique or legacy_cohort")
    document = {
        "schema": PLAN_SCHEMA,
        "campaign_id": campaign,
        "macro_round": macro_round,
        "cohort_index": cohort_index,
        "attempt_id": attempt.name,
        "namespace_policy": namespace_policy,
        "attempt_namespace": namespace,
        "cohort_dir": str(cohort),
        "endpoint_manifest": str(manifest),
        "endpoint_manifest_sha256": sha256_file(manifest),
        "renderer_sha256": renderer_sha,
        "local_environments": local_environments,
        "actor_world_size": actor_world_size,
        "launched_ranks": ranks,
        "streams": _stream_rows(
            namespace=namespace,
            ranks=ranks,
            local_environments=local_environments,
            endpoints=endpoints,
        ),
    }
    validate_attempt_plan(document, path=output_path)
    publish_immutable_json(output_path, document)
    return document


def validate_attempt_plan(
    document: Any, *, path: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    plan = _exact_mapping(document, _PLAN_KEYS, "Image3F stream plan")
    if plan["schema"] != PLAN_SCHEMA:
        raise Image3FLifecycleError("Image3F stream plan schema mismatch")
    _token(plan["campaign_id"], "campaign_id")
    _integer(plan["macro_round"], "macro_round")
    _integer(plan["cohort_index"], "cohort_index")
    attempt_id = _token(plan["attempt_id"], "attempt_id")
    if _ATTEMPT_DIR_RE.fullmatch(attempt_id) is None:
        raise Image3FLifecycleError("attempt_id is malformed")
    namespace = _stream_token(plan["attempt_namespace"], "attempt_namespace")
    namespace_policy = plan["namespace_policy"]
    if namespace_policy not in {"attempt_unique", "legacy_cohort"}:
        raise Image3FLifecycleError("stream plan namespace policy is invalid")
    cohort = _absolute_directory(plan["cohort_dir"], "cohort directory")
    manifest = _absolute_regular(plan["endpoint_manifest"], "endpoint manifest")
    if sha256_file(manifest) != _sha256(
        plan["endpoint_manifest_sha256"], "endpoint_manifest_sha256"
    ):
        raise Image3FLifecycleError("endpoint manifest bytes changed after attempt planning")
    _sha256(plan["renderer_sha256"], "renderer_sha256")
    local_environments = _integer(
        plan["local_environments"], "local_environments", minimum=1
    )
    actor_world_size = _integer(plan["actor_world_size"], "actor_world_size", minimum=1)
    ranks = plan["launched_ranks"]
    if (
        not isinstance(ranks, list)
        or not ranks
        or any(type(rank) is not int for rank in ranks)
        or ranks != sorted(set(ranks))
        or any(rank < 0 or rank >= actor_world_size for rank in ranks)
    ):
        raise Image3FLifecycleError("launched_ranks are malformed")
    endpoints = load_endpoint_manifest(
        manifest, expected_count=_manifest_entry_count(manifest)
    )
    expected_rows = _stream_rows(
        namespace=namespace,
        ranks=ranks,
        local_environments=local_environments,
        endpoints=endpoints,
    )
    raw_rows = plan["streams"]
    if not isinstance(raw_rows, list):
        raise Image3FLifecycleError("stream plan streams must be a list")
    rows = [_exact_mapping(row, _STREAM_KEYS, "stream plan row") for row in raw_rows]
    if rows != expected_rows:
        raise Image3FLifecycleError("stream plan does not match deterministic endpoint mapping")
    if path is not None:
        plan_path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
        if plan_path.parent.name != attempt_id:
            raise Image3FLifecycleError("plan path/attempt identity mismatch")
        expected_namespace = (
            attempt_namespace(
                campaign_id=plan["campaign_id"],
                macro_round=plan["macro_round"],
                cohort_index=plan["cohort_index"],
                attempt_dir=plan_path.parent,
            )
            if namespace_policy == "attempt_unique"
            else legacy_cohort_namespace(
                campaign_id=plan["campaign_id"],
                macro_round=plan["macro_round"],
                cohort_index=plan["cohort_index"],
            )
        )
        if namespace != expected_namespace:
            raise Image3FLifecycleError("attempt namespace does not match plan path")
    if not cohort.is_dir():  # pragma: no cover - retained for explanatory failure
        raise Image3FLifecycleError("cohort directory disappeared")
    return plan


def load_attempt_plan(path: str | os.PathLike[str]) -> dict[str, Any]:
    plan_path = _absolute_regular(path, "Image3F stream plan")
    return validate_attempt_plan(load_strict_json(plan_path), path=plan_path)


def _rank_streams(plan: Mapping[str, Any], rank: int) -> list[dict[str, Any]]:
    return [dict(row) for row in plan["streams"] if row["rank"] == rank]


def record_rank_completion(
    output: str | os.PathLike[str],
    *,
    plan_path: str | os.PathLike[str],
    cohort_dir: str | os.PathLike[str],
    rank: int,
) -> dict[str, Any]:
    plan_file = _absolute_regular(plan_path, "Image3F stream plan")
    plan = load_attempt_plan(plan_file)
    rank = _integer(rank, "rank")
    if rank not in plan["launched_ranks"]:
        raise Image3FLifecycleError("rank was not launched by this attempt plan")
    cohort = _absolute_directory(cohort_dir, "cohort directory")
    if str(cohort) != plan["cohort_dir"]:
        raise Image3FLifecycleError("completion cohort differs from attempt plan")
    report = require_actor_rank_completion(cohort, rank)
    streams = _rank_streams(plan, rank)
    stream_ids = [row["stream_id"] for row in streams]
    document = {
        "schema": COMPLETION_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "macro_round": plan["macro_round"],
        "cohort_index": plan["cohort_index"],
        "rank": rank,
        "attempt_id": plan["attempt_id"],
        "attempt_namespace": plan["attempt_namespace"],
        "plan_path": str(plan_file),
        "plan_sha256": sha256_file(plan_file),
        "stream_count": len(streams),
        "stream_ids_sha256": canonical_json_sha256(stream_ids),
        "cohort_contract_sha256": sha256_file(cohort / "contract.json"),
        "transition_count": report["transition_count"],
        "global_env_ids": report["global_env_ids"],
    }
    output_path = Path(os.path.abspath(os.path.expanduser(os.fspath(output))))
    if output_path.parent != plan_file.parent:
        raise Image3FLifecycleError("rank completion must be published in its attempt directory")
    abandonment = plan_file.parent / f"rank_{rank:04d}_stream_abandonment.json"
    if abandonment.exists():
        raise Image3FLifecycleError("abandoned rank attempt cannot later claim completion")
    validate_rank_completion(document)
    publish_immutable_json(output_path, document)
    return document


def validate_rank_completion(document: Any) -> dict[str, Any]:
    receipt = _exact_mapping(document, _COMPLETION_KEYS, "rank stream completion")
    if receipt["schema"] != COMPLETION_SCHEMA:
        raise Image3FLifecycleError("rank stream completion schema mismatch")
    _token(receipt["campaign_id"], "campaign_id")
    _integer(receipt["macro_round"], "macro_round")
    _integer(receipt["cohort_index"], "cohort_index")
    rank = _integer(receipt["rank"], "rank")
    _token(receipt["attempt_id"], "attempt_id")
    _stream_token(receipt["attempt_namespace"], "attempt_namespace")
    plan_path = _absolute_regular(receipt["plan_path"], "completion plan")
    if sha256_file(plan_path) != _sha256(receipt["plan_sha256"], "plan_sha256"):
        raise Image3FLifecycleError("completion plan digest mismatch")
    plan = load_attempt_plan(plan_path)
    if any(
        receipt[field] != plan[field]
        for field in (
            "campaign_id",
            "macro_round",
            "cohort_index",
            "attempt_id",
            "attempt_namespace",
        )
    ) or rank not in plan["launched_ranks"]:
        raise Image3FLifecycleError("rank completion identity differs from attempt plan")
    streams = _rank_streams(plan, rank)
    if receipt["stream_count"] != len(streams) or receipt["stream_ids_sha256"] != canonical_json_sha256(
        [row["stream_id"] for row in streams]
    ):
        raise Image3FLifecycleError("rank completion stream coverage mismatch")
    cohort = _absolute_directory(plan["cohort_dir"], "completion cohort")
    if receipt["cohort_contract_sha256"] != sha256_file(cohort / "contract.json"):
        raise Image3FLifecycleError("rank completion cohort contract digest mismatch")
    _sha256(receipt["cohort_contract_sha256"], "cohort_contract_sha256")
    _integer(receipt["transition_count"], "transition_count", minimum=1)
    envs = receipt["global_env_ids"]
    if not isinstance(envs, list) or not envs or any(type(item) is not int for item in envs):
        raise Image3FLifecycleError("rank completion global_env_ids are malformed")
    observed = require_actor_rank_completion(cohort, rank)
    if (
        receipt["transition_count"] != observed["transition_count"]
        or receipt["global_env_ids"] != observed["global_env_ids"]
    ):
        raise Image3FLifecycleError("rank completion no longer matches exact shard coverage")
    return receipt


def record_rank_abandonment(
    *, plan_path: str | os.PathLike[str], cohort_dir: str | os.PathLike[str], rank: int
) -> dict[str, Any]:
    plan_file = _absolute_regular(plan_path, "Image3F stream plan")
    plan = load_attempt_plan(plan_file)
    rank = _integer(rank, "rank")
    if rank not in plan["launched_ranks"]:
        raise Image3FLifecycleError("rank was not launched by abandonment plan")
    completion = plan_file.parent / f"rank_{rank:04d}_stream_completion.json"
    if completion.exists():
        raise Image3FLifecycleError("completed rank attempt cannot be abandoned")
    state = inspect_actor_rank(cohort_dir, rank)
    if state["state"] != "empty":
        raise Image3FLifecycleError("only an exactly empty rank attempt can be abandoned")
    streams = _rank_streams(plan, rank)
    document = {
        "schema": ABANDONMENT_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "macro_round": plan["macro_round"],
        "cohort_index": plan["cohort_index"],
        "rank": rank,
        "attempt_id": plan["attempt_id"],
        "attempt_namespace": plan["attempt_namespace"],
        "plan_path": str(plan_file),
        "plan_sha256": sha256_file(plan_file),
        "stream_count": len(streams),
        "stream_ids_sha256": canonical_json_sha256(
            [row["stream_id"] for row in streams]
        ),
        "reason": "rank_empty_after_reap",
    }
    output = plan_file.parent / f"rank_{rank:04d}_stream_abandonment.json"
    publish_immutable_json(output, document)
    return document


def _validate_abandonment(document: Any) -> dict[str, Any]:
    value = _exact_mapping(document, _ABANDONMENT_KEYS, "rank stream abandonment")
    if value["schema"] != ABANDONMENT_SCHEMA or value["reason"] != "rank_empty_after_reap":
        raise Image3FLifecycleError("rank stream abandonment schema/reason mismatch")
    plan_path = _absolute_regular(value["plan_path"], "abandonment plan")
    if sha256_file(plan_path) != value["plan_sha256"]:
        raise Image3FLifecycleError("rank stream abandonment plan digest mismatch")
    plan = load_attempt_plan(plan_path)
    rank = _integer(value["rank"], "abandoned rank")
    streams = _rank_streams(plan, rank)
    expected = {
        "campaign_id": plan["campaign_id"],
        "macro_round": plan["macro_round"],
        "cohort_index": plan["cohort_index"],
        "attempt_id": plan["attempt_id"],
        "attempt_namespace": plan["attempt_namespace"],
        "stream_count": len(streams),
        "stream_ids_sha256": canonical_json_sha256(
            [row["stream_id"] for row in streams]
        ),
    }
    if any(value.get(key) != expected_item for key, expected_item in expected.items()):
        raise Image3FLifecycleError("rank stream abandonment identity mismatch")
    return value


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise ConnectionError("Image3F server closed the connection")
        result.extend(chunk)
    return bytes(result)


def _request(host: str, port: int, payload: Mapping[str, Any], timeout_s: float) -> dict[str, Any]:
    encoded = pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL)
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall(struct.pack("!Q", len(encoded)) + encoded)
        size = struct.unpack("!Q", _recv_exact(sock, 8))[0]
        if size > 16 * 1024 * 1024:
            raise Image3FLifecycleError("Image3F reset response exceeds bounded size")
        response = pickle.loads(_recv_exact(sock, size))  # noqa: S301 - trusted cluster
    if not isinstance(response, dict):
        raise Image3FLifecycleError("Image3F reset response is not a mapping")
    return response


def _reset_with_retry(
    row: Mapping[str, Any],
    *,
    renderer_sha256: str,
    timeout_s: float,
    retries: int,
) -> dict[str, Any]:
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            response = _request(
                row["host"],
                row["port"],
                {"cmd": "reset_stream", "stream_id": row["stream_id"]},
                timeout_s,
            )
            if response.get("ok") is not True:
                raise Image3FLifecycleError(
                    f"renderer rejected stream reset: {response.get('error', 'unknown error')}"
                )
            if response.get("stream_id") not in (None, row["stream_id"]):
                raise Image3FLifecycleError("renderer reset acknowledgement stream mismatch")
            if type(response.get("removed")) is not bool:
                raise Image3FLifecycleError("renderer reset acknowledgement removed flag is invalid")
            direct_server_id = response.get("server_id")
            direct_renderer = response.get("renderer_identity_sha256")
            server_id = direct_server_id
            observed_renderer = direct_renderer
            # The deployed protocol-2 renderer acknowledges reset with
            # ``ok``/``removed`` only.  Bind that serialized response to the
            # endpoint identity with an immediate ping on the same selected
            # host/port.  Newer servers may echo identity directly; both paths
            # produce the same durable acknowledgement schema.
            if server_id is None or observed_renderer is None:
                ping = _request(
                    row["host"],
                    row["port"],
                    {"cmd": "ping", "protocol_version": 2},
                    timeout_s,
                )
                if ping.get("ok") is not True:
                    raise Image3FLifecycleError("renderer identity ping failed after reset")
                server_id = ping.get("server_id")
                observed_renderer = ping.get("renderer_identity_sha256")
                if direct_server_id is not None and direct_server_id != server_id:
                    raise Image3FLifecycleError("renderer reset/ping server identity mismatch")
                if direct_renderer is not None and direct_renderer != observed_renderer:
                    raise Image3FLifecycleError("renderer reset/ping digest mismatch")
            if not isinstance(server_id, str) or not server_id or len(server_id) > 256:
                raise Image3FLifecycleError("renderer reset acknowledgement server ID is invalid")
            if observed_renderer != renderer_sha256:
                raise Image3FLifecycleError("renderer reset acknowledgement identity mismatch")
            return {
                **response,
                "stream_id": row["stream_id"],
                "server_id": server_id,
                "renderer_identity_sha256": observed_renderer,
            }
        except (OSError, EOFError, pickle.UnpicklingError, Image3FLifecycleError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2.0, 0.1 * (2**attempt)))
    raise Image3FLifecycleError(
        f"stream reset failed after {retries + 1} attempt(s): {type(last_error).__name__}: {last_error}"
    )


def _discover_plans(
    train_dir: Path, *, allow_empty: bool = False
) -> list[tuple[Path, dict[str, Any]]]:
    attempts = train_dir / "actor_pair_attempts"
    if attempts.is_symlink() or not attempts.is_dir():
        raise Image3FLifecycleError("actor_pair_attempts directory is unsafe or missing")
    result: list[tuple[Path, dict[str, Any]]] = []
    for attempt in sorted(attempts.iterdir(), key=lambda path: path.name):
        if attempt.is_symlink() or not attempt.is_dir() or _ATTEMPT_DIR_RE.fullmatch(attempt.name) is None:
            raise Image3FLifecycleError(f"unsafe actor-pair attempt directory: {attempt}")
        plan_path = attempt / "image3f_stream_plan.json"
        if plan_path.exists():
            result.append((plan_path, load_attempt_plan(plan_path)))
    if not result and not allow_empty:
        raise Image3FLifecycleError("no Image3F actor attempt plans were found")
    return result


def _discover_completions(
    plans: Sequence[tuple[Path, Mapping[str, Any]]]
) -> dict[int, tuple[Path, dict[str, Any]]]:
    result: dict[int, tuple[Path, dict[str, Any]]] = {}
    for plan_path, plan in plans:
        for rank in plan["launched_ranks"]:
            receipt_path = plan_path.parent / f"rank_{rank:04d}_stream_completion.json"
            if not receipt_path.exists():
                continue
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise Image3FLifecycleError(f"unsafe rank completion receipt: {receipt_path}")
            receipt = validate_rank_completion(load_strict_json(receipt_path))
            previous = result.get(rank)
            if previous is not None and previous[1] != receipt:
                raise Image3FLifecycleError(f"multiple stream owners claim completed rank {rank}")
            result[rank] = (receipt_path, receipt)
    return result


def recover_completed_rank_ownership(
    train_dir: str | os.PathLike[str],
    *,
    cohort_dir: str | os.PathLike[str],
    endpoint_manifest: str | os.PathLike[str],
    expected_renderers: int,
    renderer_sha256: str,
    completed_ranks: Sequence[int],
    local_environments: int = 32,
    actor_world_size: int = 2,
) -> dict[str, Any]:
    """Backfill ownership receipts after a crash or a pre-hotfix partial run.

    A completed rank normally has an attempt-local receipt.  If exactly one
    attempt plan launched it, that immutable intent is sufficient to recover a
    receipt after the narrow shard/receipt crash window.  If no plan exists,
    the rank came from the pre-hotfix cohort-stable namespace; one deterministic
    legacy plan covers both ranks so abandoned legacy streams are also cleaned.
    Multiple unreceipted candidate plans are ambiguous and fail closed.
    """

    train = _absolute_directory(train_dir, "actor train directory")
    cohort = _absolute_directory(cohort_dir, "cohort directory")
    contract = load_strict_json(cohort / "contract.json")
    ranks = [_integer(rank, "completed_ranks[]") for rank in completed_ranks]
    if ranks != sorted(set(ranks)):
        raise Image3FLifecycleError("completed_ranks must be sorted and unique")
    for rank in ranks:
        require_actor_rank_completion(cohort, rank)
    plans = _discover_plans(train, allow_empty=True)
    completions = _discover_completions(plans)
    abandoned: set[tuple[str, int]] = set()
    for plan_path, plan in plans:
        for rank in plan["launched_ranks"]:
            key = (str(plan_path), rank)
            abandonment_path = (
                plan_path.parent / f"rank_{rank:04d}_stream_abandonment.json"
            )
            if abandonment_path.exists():
                if abandonment_path.is_symlink() or not abandonment_path.is_file():
                    raise Image3FLifecycleError("rank abandonment path is unsafe")
                _validate_abandonment(load_strict_json(abandonment_path))
                abandoned.add(key)
                continue
            if rank in completions:
                continue
            if inspect_actor_rank(cohort, rank)["state"] == "empty":
                record_rank_abandonment(
                    plan_path=plan_path, cohort_dir=cohort, rank=rank
                )
                abandoned.add(key)
    missing = [rank for rank in ranks if rank not in completions]
    legacy_missing: list[int] = []
    recovered: list[dict[str, Any]] = []
    for rank in missing:
        attempt_candidates = [
            (path, plan)
            for path, plan in plans
            if rank in plan["launched_ranks"]
            and plan["namespace_policy"] == "attempt_unique"
            and (str(path), rank) not in abandoned
        ]
        legacy_candidates = [
            (path, plan)
            for path, plan in plans
            if rank in plan["launched_ranks"]
            and plan["namespace_policy"] == "legacy_cohort"
            and (str(path), rank) not in abandoned
        ]
        if len(attempt_candidates) == 1:
            plan_path, _ = attempt_candidates[0]
            receipt_path = plan_path.parent / f"rank_{rank:04d}_stream_completion.json"
            recovered.append(
                record_rank_completion(
                    receipt_path,
                    plan_path=plan_path,
                    cohort_dir=cohort,
                    rank=rank,
                )
            )
        elif len(attempt_candidates) > 1:
            raise Image3FLifecycleError(
                f"completed rank {rank} has multiple unreceipted attempt plans"
            )
        elif len(legacy_candidates) == 1:
            plan_path, _ = legacy_candidates[0]
            recovered.append(
                record_rank_completion(
                    plan_path.parent / f"rank_{rank:04d}_stream_completion.json",
                    plan_path=plan_path,
                    cohort_dir=cohort,
                    rank=rank,
                )
            )
        elif not legacy_candidates:
            legacy_missing.append(rank)
        else:
            raise Image3FLifecycleError(
                f"completed rank {rank} has multiple legacy ownership plans"
            )
    if legacy_missing:
        legacy_dir = (
            train
            / "actor_pair_attempts"
            / f"attempt_legacy_m{contract['macro_round']:06d}_c{contract['cohort_index']:02d}"
        )
        if legacy_dir.exists() and (legacy_dir.is_symlink() or not legacy_dir.is_dir()):
            raise Image3FLifecycleError("legacy ownership directory is unsafe")
        legacy_dir.mkdir(mode=0o755, exist_ok=True)
        legacy_plan_path = legacy_dir / "image3f_stream_plan.json"
        legacy_plan = create_attempt_plan(
            legacy_plan_path,
            campaign_id=contract["campaign_id"],
            macro_round=contract["macro_round"],
            cohort_index=contract["cohort_index"],
            attempt_dir=legacy_dir,
            cohort_dir=cohort,
            endpoint_manifest=endpoint_manifest,
            expected_renderers=expected_renderers,
            renderer_sha256=renderer_sha256,
            local_environments=local_environments,
            actor_world_size=actor_world_size,
            launched_ranks=list(range(actor_world_size)),
            namespace_policy="legacy_cohort",
        )
        if legacy_plan["renderer_sha256"] != contract["renderer_sha256"]:
            raise Image3FLifecycleError("legacy ownership renderer differs from cohort")
        for rank in legacy_missing:
            recovered.append(
                record_rank_completion(
                    legacy_dir / f"rank_{rank:04d}_stream_completion.json",
                    plan_path=legacy_plan_path,
                    cohort_dir=cohort,
                    rank=rank,
                )
            )
    final_plans = _discover_plans(train, allow_empty=True)
    final = _discover_completions(final_plans)
    if any(rank not in final for rank in ranks):
        raise Image3FLifecycleError("completed rank ownership recovery is incomplete")
    return {
        "completed_ranks": ranks,
        "owned_ranks": sorted(rank for rank in final if rank in ranks),
        "recovered_receipts": len(recovered),
        "legacy_ranks": legacy_missing,
    }


def _validate_ack(document: Any, row: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    ack = _exact_mapping(document, _ACK_KEYS, "stream reset acknowledgement")
    if ack["schema"] != RESET_ACK_SCHEMA or ack["acknowledged"] is not True:
        raise Image3FLifecycleError("stream reset acknowledgement schema/status mismatch")
    expected = {
        "campaign_id": plan["campaign_id"],
        "macro_round": plan["macro_round"],
        "cohort_index": plan["cohort_index"],
        "attempt_id": plan["attempt_id"],
        "attempt_namespace": plan["attempt_namespace"],
        **dict(row),
    }
    for key, value in expected.items():
        if ack.get(key) != value:
            raise Image3FLifecycleError(f"stream reset acknowledgement {key} mismatch")
    if type(ack["removed"]) is not bool:
        raise Image3FLifecycleError("stream reset acknowledgement removed flag is invalid")
    if not isinstance(ack["server_id"], str) or not ack["server_id"]:
        raise Image3FLifecycleError("stream reset acknowledgement server ID is invalid")
    if ack["renderer_identity_sha256"] != plan["renderer_sha256"]:
        raise Image3FLifecycleError("stream reset acknowledgement renderer mismatch")
    return ack


def _reset_planned_rows(
    train: Path,
    planned: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    timeout_s: float,
    retries: int,
) -> list[dict[str, Any]]:
    cleanup_dir = train / "image3f_stream_cleanup"
    if cleanup_dir.exists() and (cleanup_dir.is_symlink() or not cleanup_dir.is_dir()):
        raise Image3FLifecycleError("Image3F cleanup path is unsafe")
    cleanup_dir.mkdir(mode=0o755, exist_ok=True)
    acknowledgements: list[dict[str, Any]] = []
    for raw_plan, raw_row in planned:
        plan, row = dict(raw_plan), dict(raw_row)
        ack_name = f"reset_{hashlib.sha256(row['stream_id'].encode('utf-8')).hexdigest()}.json"
        ack_path = cleanup_dir / ack_name
        if ack_path.exists():
            if ack_path.is_symlink() or not stat.S_ISREG(os.lstat(ack_path).st_mode):
                raise Image3FLifecycleError(f"unsafe stream reset acknowledgement: {ack_path}")
            _validate_ack(load_strict_json(ack_path), row, plan)
        else:
            response = _reset_with_retry(
                row,
                renderer_sha256=plan["renderer_sha256"],
                timeout_s=float(timeout_s),
                retries=retries,
            )
            ack = {
                "schema": RESET_ACK_SCHEMA,
                "campaign_id": plan["campaign_id"],
                "macro_round": plan["macro_round"],
                "cohort_index": plan["cohort_index"],
                "attempt_id": plan["attempt_id"],
                "attempt_namespace": plan["attempt_namespace"],
                **row,
                "server_id": response["server_id"],
                "renderer_identity_sha256": response["renderer_identity_sha256"],
                "removed": response["removed"],
                "acknowledged": True,
            }
            _validate_ack(ack, row, plan)
            publish_immutable_json(ack_path, ack)
        acknowledgements.append(
            {
                "path": str(ack_path),
                "sha256": sha256_file(ack_path),
                "stream_id": row["stream_id"],
            }
        )
    return acknowledgements


def cleanup_attempt_plan(
    train_dir: str | os.PathLike[str],
    *,
    plan_path: str | os.PathLike[str],
    timeout_s: float = 10.0,
    retries: int = 3,
) -> dict[str, Any]:
    """Reset one fully-reaped attempt, including streams from failed ranks."""

    if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0:
        raise Image3FLifecycleError("timeout_s must be finite and positive")
    retries = _integer(retries, "retries")
    train = _absolute_directory(train_dir, "actor train directory")
    plan_file = _absolute_regular(plan_path, "attempt stream plan")
    attempts = train / "actor_pair_attempts"
    if plan_file.parent.parent != attempts:
        raise Image3FLifecycleError("attempt plan is outside actor_pair_attempts")
    plan = load_attempt_plan(plan_file)
    planned = [(plan, row) for row in plan["streams"]]
    acknowledgements = _reset_planned_rows(
        train, planned, timeout_s=float(timeout_s), retries=retries
    )
    cohort = _absolute_directory(plan["cohort_dir"], "attempt cohort")
    abandoned_ranks: list[int] = []
    for rank in plan["launched_ranks"]:
        completion = plan_file.parent / f"rank_{rank:04d}_stream_completion.json"
        if not completion.exists() and inspect_actor_rank(cohort, rank)["state"] == "empty":
            record_rank_abandonment(
                plan_path=plan_file, cohort_dir=cohort, rank=rank
            )
            abandoned_ranks.append(rank)
    return {
        "attempt_id": plan["attempt_id"],
        "attempt_namespace": plan["attempt_namespace"],
        "planned_stream_count": len(planned),
        "acknowledged_stream_count": len(acknowledgements),
        "abandoned_ranks": abandoned_ranks,
    }


def cleanup_prior_attempts(
    train_dir: str | os.PathLike[str],
    *,
    timeout_s: float = 10.0,
    retries: int = 3,
) -> dict[str, Any]:
    """Reset every reaped attempt known before a retry is launched.

    The actor-pair directory lock proves no process from these plans remains
    alive.  Resetting both completed and abandoned ranks before allocating a
    new namespace releases the renderer's bounded stream capacity; durable
    per-stream ACKs make repeated reconciliation idempotent.
    """

    if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0:
        raise Image3FLifecycleError("timeout_s must be finite and positive")
    retries = _integer(retries, "retries")
    train = _absolute_directory(train_dir, "actor train directory")
    plans = _discover_plans(train, allow_empty=True)
    reports = [
        cleanup_attempt_plan(
            train,
            plan_path=plan_path,
            timeout_s=float(timeout_s),
            retries=retries,
        )
        for plan_path, _ in plans
    ]
    return {
        "attempt_count": len(reports),
        "planned_stream_count": sum(item["planned_stream_count"] for item in reports),
        "acknowledged_stream_count": sum(
            item["acknowledged_stream_count"] for item in reports
        ),
        "attempts": reports,
    }


def cleanup_sealed_cohort(
    train_dir: str | os.PathLike[str],
    *,
    cohort_dir: str | os.PathLike[str],
    timeout_s: float = 10.0,
    retries: int = 3,
) -> dict[str, Any]:
    """Reset every stream for an exact completed cohort before it is sealed.

    Successful streams are the 64 streams owned by the two immutable rank
    completion receipts.  Streams from failed attempts are reset as well so a
    long-running renderer cannot leak abandoned history across later cohorts.
    Existing valid ACKs are accepted, making the operation crash-resumable.
    """

    if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0:
        raise Image3FLifecycleError("timeout_s must be finite and positive")
    retries = _integer(retries, "retries")
    train = _absolute_directory(train_dir, "actor train directory")
    cohort = _absolute_directory(cohort_dir, "cohort directory")
    plans = _discover_plans(train)
    contract = load_strict_json(cohort / "contract.json")
    for _, plan in plans:
        for key in ("campaign_id", "macro_round", "cohort_index", "renderer_sha256"):
            if plan[key] != contract[key]:
                raise Image3FLifecycleError(f"attempt plan {key} differs from cohort contract")
        if plan["cohort_dir"] != str(cohort):
            raise Image3FLifecycleError("attempt plan cohort path differs from sealed cohort")
    completions = _discover_completions(plans)
    expected_ranks = contract["expected_ranks"]
    if sorted(completions) != expected_ranks:
        raise Image3FLifecycleError(
            f"rank stream ownership is incomplete: expected {expected_ranks}, got {sorted(completions)}"
        )
    successful_stream_ids: list[str] = []
    for rank in expected_ranks:
        receipt = completions[rank][1]
        if receipt["cohort_contract_sha256"] != sha256_file(cohort / "contract.json"):
            raise Image3FLifecycleError("rank stream ownership contract digest mismatch")
        require_actor_rank_completion(cohort, rank)
        plan = load_attempt_plan(receipt["plan_path"])
        successful_stream_ids.extend(row["stream_id"] for row in _rank_streams(plan, rank))
    if len(successful_stream_ids) != len(contract["expected_global_env_ids"]):
        raise Image3FLifecycleError("successful stream ownership does not cover all cohort environments")
    if len(set(successful_stream_ids)) != len(successful_stream_ids):
        raise Image3FLifecycleError("successful stream ownership contains duplicate IDs")

    planned: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for _, plan in plans:
        for raw_row in plan["streams"]:
            row = dict(raw_row)
            if row["stream_id"] in seen:
                raise Image3FLifecycleError("attempt plans contain a duplicate stream ID")
            seen.add(row["stream_id"])
            planned.append((plan, row))
    planned.sort(key=lambda item: (item[0]["attempt_id"], item[1]["rank"], item[1]["global_slot"]))

    acknowledgements = _reset_planned_rows(
        train, planned, timeout_s=float(timeout_s), retries=retries
    )

    report = {
        "schema": CLEANUP_SCHEMA,
        "campaign_id": contract["campaign_id"],
        "macro_round": contract["macro_round"],
        "cohort_index": contract["cohort_index"],
        "contract_sha256": sha256_file(cohort / "contract.json"),
        "successful_stream_count": len(successful_stream_ids),
        "successful_stream_ids_sha256": canonical_json_sha256(sorted(successful_stream_ids)),
        "planned_stream_count": len(planned),
        "acknowledged_stream_count": len(acknowledgements),
        "acknowledgements": acknowledgements,
        "status": "complete",
    }
    report_path = train / "image3f_stream_cleanup" / "cleanup_complete.json"
    publish_immutable_json(report_path, report)
    loaded = load_strict_json(report_path)
    if loaded != report:
        raise Image3FLifecycleError("published cleanup report differs from verified cleanup")
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def verify_cleanup_report(
    train_dir: str | os.PathLike[str], *, cohort_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    """Verify complete reset coverage without issuing any renderer request."""

    train = _absolute_directory(train_dir, "actor train directory")
    cohort = _absolute_directory(cohort_dir, "cohort directory")
    report_path = _absolute_regular(
        train / "image3f_stream_cleanup" / "cleanup_complete.json",
        "Image3F cleanup report",
    )
    report = _exact_mapping(
        load_strict_json(report_path), _CLEANUP_KEYS, "Image3F cleanup report"
    )
    if report["schema"] != CLEANUP_SCHEMA or report["status"] != "complete":
        raise Image3FLifecycleError("Image3F cleanup report schema/status mismatch")
    contract = load_strict_json(cohort / "contract.json")
    expected_identity = {
        "campaign_id": contract["campaign_id"],
        "macro_round": contract["macro_round"],
        "cohort_index": contract["cohort_index"],
        "contract_sha256": sha256_file(cohort / "contract.json"),
    }
    if any(report.get(key) != value for key, value in expected_identity.items()):
        raise Image3FLifecycleError("Image3F cleanup report cohort identity mismatch")

    plans = _discover_plans(train)
    for _, plan in plans:
        for key in ("campaign_id", "macro_round", "cohort_index", "renderer_sha256"):
            if plan[key] != contract[key]:
                raise Image3FLifecycleError(
                    f"cleanup verification plan {key} differs from cohort contract"
                )
        if plan["cohort_dir"] != str(cohort):
            raise Image3FLifecycleError(
                "cleanup verification plan cohort path differs from cohort"
            )
    completions = _discover_completions(plans)
    if sorted(completions) != contract["expected_ranks"]:
        raise Image3FLifecycleError("cleanup report rank ownership is incomplete")
    successful_ids: list[str] = []
    planned_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for rank in contract["expected_ranks"]:
        plan = load_attempt_plan(completions[rank][1]["plan_path"])
        successful_ids.extend(row["stream_id"] for row in _rank_streams(plan, rank))
    if (
        len(successful_ids) != len(contract["expected_global_env_ids"])
        or len(set(successful_ids)) != len(successful_ids)
    ):
        raise Image3FLifecycleError(
            "cleanup verification successful streams do not exactly cover cohort topology"
        )
    for _, plan in plans:
        for raw_row in plan["streams"]:
            row = dict(raw_row)
            stream_id = row["stream_id"]
            if stream_id in planned_by_id:
                raise Image3FLifecycleError("cleanup plans contain duplicate stream IDs")
            planned_by_id[stream_id] = (plan, row)
    expected_success = {
        "successful_stream_count": len(successful_ids),
        "successful_stream_ids_sha256": canonical_json_sha256(sorted(successful_ids)),
        "planned_stream_count": len(planned_by_id),
        "acknowledged_stream_count": len(planned_by_id),
    }
    if any(report.get(key) != value for key, value in expected_success.items()):
        raise Image3FLifecycleError("Image3F cleanup report coverage mismatch")
    raw_references = report["acknowledgements"]
    if not isinstance(raw_references, list) or len(raw_references) != len(planned_by_id):
        raise Image3FLifecycleError("Image3F cleanup acknowledgement list is incomplete")
    seen: set[str] = set()
    for raw_reference in raw_references:
        reference = _exact_mapping(
            raw_reference, _CLEANUP_ACK_REF_KEYS, "cleanup acknowledgement reference"
        )
        stream_id = _stream_token(reference["stream_id"], "cleanup stream ID")
        if stream_id in seen or stream_id not in planned_by_id:
            raise Image3FLifecycleError("cleanup acknowledgement coverage is invalid")
        seen.add(stream_id)
        plan, row = planned_by_id[stream_id]
        expected_path = (
            train
            / "image3f_stream_cleanup"
            / f"reset_{hashlib.sha256(stream_id.encode('utf-8')).hexdigest()}.json"
        )
        if reference["path"] != str(expected_path):
            raise Image3FLifecycleError("cleanup acknowledgement path mismatch")
        ack_path = _absolute_regular(expected_path, "stream reset acknowledgement")
        if reference["sha256"] != sha256_file(ack_path):
            raise Image3FLifecycleError("cleanup acknowledgement digest mismatch")
        _validate_ack(load_strict_json(ack_path), row, plan)
    if seen != set(planned_by_id):
        raise Image3FLifecycleError("cleanup acknowledgement set is incomplete")
    return {
        **report,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
    }


def _parse_ranks(value: str) -> list[int]:
    try:
        ranks = [int(item) for item in value.split(",") if item != ""]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ranks must be comma-separated integers") from exc
    if not ranks:
        raise argparse.ArgumentTypeError("at least one rank is required")
    return ranks


def _parse_optional_ranks(value: str) -> list[int]:
    if value == "":
        return []
    return _parse_ranks(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--campaign-id", required=True)
    plan.add_argument("--macro-round", type=int, required=True)
    plan.add_argument("--cohort-index", type=int, required=True)
    plan.add_argument("--attempt-dir", type=Path, required=True)
    plan.add_argument("--cohort-dir", type=Path, required=True)
    plan.add_argument("--endpoint-manifest", type=Path, required=True)
    plan.add_argument("--expected-renderers", type=int, required=True)
    plan.add_argument("--renderer-sha256", required=True)
    plan.add_argument("--local-environments", type=int, required=True)
    plan.add_argument("--actor-world-size", type=int, required=True)
    plan.add_argument("--launched-ranks", type=_parse_ranks, required=True)

    verify = sub.add_parser("verify-rank")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--rank", type=int, required=True)
    verify.add_argument("--expected-namespace", required=True)
    verify.add_argument("--endpoint-manifest", type=Path, required=True)
    verify.add_argument("--renderer-sha256", required=True)
    verify.add_argument("--local-environments", type=int, required=True)
    verify.add_argument("--actor-world-size", type=int, required=True)

    complete = sub.add_parser("complete-rank")
    complete.add_argument("--output", type=Path, required=True)
    complete.add_argument("--plan", type=Path, required=True)
    complete.add_argument("--cohort-dir", type=Path, required=True)
    complete.add_argument("--rank", type=int, required=True)

    recover = sub.add_parser("recover-ownership")
    recover.add_argument("--train-dir", type=Path, required=True)
    recover.add_argument("--cohort-dir", type=Path, required=True)
    recover.add_argument("--endpoint-manifest", type=Path, required=True)
    recover.add_argument("--expected-renderers", type=int, required=True)
    recover.add_argument("--renderer-sha256", required=True)
    recover.add_argument("--completed-ranks", type=_parse_optional_ranks, required=True)

    cleanup_attempt = sub.add_parser("cleanup-attempt")
    cleanup_attempt.add_argument("--train-dir", type=Path, required=True)
    cleanup_attempt.add_argument("--plan", type=Path, required=True)
    cleanup_attempt.add_argument("--timeout-s", type=float, default=10.0)
    cleanup_attempt.add_argument("--retries", type=int, default=3)

    cleanup_prior = sub.add_parser("cleanup-prior")
    cleanup_prior.add_argument("--train-dir", type=Path, required=True)
    cleanup_prior.add_argument("--timeout-s", type=float, default=10.0)
    cleanup_prior.add_argument("--retries", type=int, default=3)

    cleanup = sub.add_parser("cleanup")
    cleanup.add_argument("--train-dir", type=Path, required=True)
    cleanup.add_argument("--cohort-dir", type=Path, required=True)
    cleanup.add_argument("--timeout-s", type=float, default=10.0)
    cleanup.add_argument("--retries", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        result = create_attempt_plan(
            args.output,
            campaign_id=args.campaign_id,
            macro_round=args.macro_round,
            cohort_index=args.cohort_index,
            attempt_dir=args.attempt_dir,
            cohort_dir=args.cohort_dir,
            endpoint_manifest=args.endpoint_manifest,
            expected_renderers=args.expected_renderers,
            renderer_sha256=args.renderer_sha256,
            local_environments=args.local_environments,
            actor_world_size=args.actor_world_size,
            launched_ranks=args.launched_ranks,
        )
    elif args.command == "verify-rank":
        plan = load_attempt_plan(args.plan)
        rank = _integer(args.rank, "rank")
        expected = {
            "attempt_namespace": _stream_token(
                args.expected_namespace, "expected namespace"
            ),
            "endpoint_manifest": str(
                _absolute_regular(args.endpoint_manifest, "runtime endpoint manifest")
            ),
            "renderer_sha256": _sha256(args.renderer_sha256, "runtime renderer SHA-256"),
            "local_environments": _integer(
                args.local_environments, "runtime local environments", minimum=1
            ),
            "actor_world_size": _integer(
                args.actor_world_size, "runtime actor world size", minimum=1
            ),
        }
        mismatches = [key for key, value in expected.items() if plan[key] != value]
        if mismatches:
            raise Image3FLifecycleError(
                "runtime actor differs from stream plan: " + ", ".join(mismatches)
            )
        if rank not in plan["launched_ranks"] or len(_rank_streams(plan, rank)) != plan["local_environments"]:
            raise Image3FLifecycleError("rank is not exactly covered by attempt plan")
        result = {
            "attempt_namespace": plan["attempt_namespace"],
            "rank": rank,
            "stream_count": len(_rank_streams(plan, rank)),
        }
    elif args.command == "complete-rank":
        result = record_rank_completion(
            args.output,
            plan_path=args.plan,
            cohort_dir=args.cohort_dir,
            rank=args.rank,
        )
    elif args.command == "recover-ownership":
        result = recover_completed_rank_ownership(
            args.train_dir,
            cohort_dir=args.cohort_dir,
            endpoint_manifest=args.endpoint_manifest,
            expected_renderers=args.expected_renderers,
            renderer_sha256=args.renderer_sha256,
            completed_ranks=args.completed_ranks,
        )
    elif args.command == "cleanup-attempt":
        result = cleanup_attempt_plan(
            args.train_dir,
            plan_path=args.plan,
            timeout_s=args.timeout_s,
            retries=args.retries,
        )
    elif args.command == "cleanup-prior":
        result = cleanup_prior_attempts(
            args.train_dir,
            timeout_s=args.timeout_s,
            retries=args.retries,
        )
    else:
        if args.timeout_s <= 0:
            raise Image3FLifecycleError("timeout_s must be positive")
        result = cleanup_sealed_cohort(
            args.train_dir,
            cohort_dir=args.cohort_dir,
            timeout_s=args.timeout_s,
            retries=args.retries,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Image3FLifecycleError as error:
        raise SystemExit(f"Image3F lifecycle error: {error}") from None
