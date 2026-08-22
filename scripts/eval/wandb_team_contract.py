#!/usr/bin/env python3
"""Validate the exact W&B team authorization accepted for HSG evaluation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


ONLINE_SMOKE_SCHEMA = "fast_roborender.wandb_online_smoke.v2"
AUTHORIZED_EMAIL = "ldu@nvidia.com"
AUTHORIZED_ENTITY = "nvidia"
AUTHORIZED_SCOPE = "team"
REQUIRED_OBSERVED_KEYS = frozenset(
    {"validation/scalar", "validation/image", "validation/video"}
)
ENTITY_RE = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class TeamContractError(RuntimeError):
    """A W&B artifact or live identity violated the authorized team contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TeamContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except TeamContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TeamContractError(
            f"cannot read W&B acceptance {path}: {type(exc).__name__}"
        ) from None
    if not isinstance(payload, dict):
        raise TeamContractError("W&B acceptance must contain one JSON object")
    return payload


def validate_authorized_request(
    *, expected_email: str, entity: str, authorization_scope: str
) -> None:
    if expected_email != AUTHORIZED_EMAIL:
        raise TeamContractError("W&B account is outside the authorized evaluation scope")
    if entity != AUTHORIZED_ENTITY:
        raise TeamContractError("W&B entity is outside the authorized evaluation scope")
    if authorization_scope != AUTHORIZED_SCOPE:
        raise TeamContractError("W&B authorization scope must be team")


def team_binding() -> dict[str, Any]:
    return {
        "account_email": AUTHORIZED_EMAIL,
        "entity": AUTHORIZED_ENTITY,
        "authorization_scope": AUTHORIZED_SCOPE,
        "team_member": True,
        "entity_is_team": True,
        "personal": False,
    }


def validate_team_binding(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    expected = team_binding()
    for key, value in expected.items():
        actual = payload.get(key)
        if type(actual) is not type(value) or actual != value:
            raise TeamContractError(f"{label} {key} mismatch")
    if ENTITY_RE.fullmatch(str(payload.get("entity", ""))) is None:
        raise TeamContractError(f"{label} entity is malformed")
    return expected


def validate_online_acceptance(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema": ONLINE_SMOKE_SCHEMA,
        "status": "PASS",
        **team_binding(),
    }
    for key, value in expected.items():
        actual = payload.get(key)
        if type(actual) is not type(value) or actual != value:
            raise TeamContractError(f"W&B online acceptance {key} mismatch")
    observed = payload.get("observed_keys")
    if (
        not isinstance(observed, list)
        or any(not isinstance(key, str) for key in observed)
        or len(observed) != len(set(observed))
        or not REQUIRED_OBSERVED_KEYS.issubset(observed)
    ):
        raise TeamContractError(
            "W&B online acceptance lacks unique scalar/image/video readback keys"
        )
    return {
        **expected,
        "observed_keys": sorted(REQUIRED_OBSERVED_KEYS),
    }


def validate_live_identity(identity: object) -> dict[str, Any]:
    values = {
        "account_email": getattr(identity, "email", None),
        "entity": getattr(identity, "entity", None),
        "authorization_scope": getattr(identity, "authorization_scope", None),
        "team_member": getattr(identity, "team_member", None),
        "entity_is_team": getattr(identity, "entity_is_team", None),
        "personal": getattr(identity, "personal", None),
    }
    return validate_team_binding(values, label="live W&B identity")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--acceptance", type=Path, required=True)
    result.add_argument("--expected-email", required=True)
    result.add_argument("--entity", required=True)
    result.add_argument("--authorization-scope", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        validate_authorized_request(
            expected_email=args.expected_email,
            entity=args.entity,
            authorization_scope=args.authorization_scope,
        )
        validate_online_acceptance(load_json(args.acceptance))
    except TeamContractError as exc:
        print(f"W&B team validation failed: {exc}", file=sys.stderr)
        return 2
    print(AUTHORIZED_ENTITY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
