"""Domain-separated identity for the complete executable actor policy."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


RUNTIME_POLICY_IDENTITY_SCHEMA = (
    "fast_roborender.round_dagger.runtime_policy_identity.v1"
)
HOMIE_CONTROLLER_ROLES = ("walk", "stand")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class PolicyIdentityError(ValueError):
    """The executable-policy identity components are incomplete or malformed."""


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PolicyIdentityError(f"{label} must be a lowercase SHA-256")
    return value


def validate_homie_controller_sha256(value: Any) -> dict[str, str]:
    """Return the exact walk/stand controller byte identities."""

    if not isinstance(value, Mapping) or set(value) != set(HOMIE_CONTROLLER_ROLES):
        raise PolicyIdentityError(
            "homie_controller_sha256 must contain exactly walk and stand"
        )
    return {
        role: require_sha256(value[role], f"Homie {role} controller SHA-256")
        for role in HOMIE_CONTROLLER_ROLES
    }


def runtime_policy_identity_sha256(
    policy_state_sha256: str,
    homie_controller_sha256: Mapping[str, str],
) -> str:
    """Hash policy parameters and both executed Homie controller byte streams."""

    document = {
        "schema": RUNTIME_POLICY_IDENTITY_SCHEMA,
        "policy_state_sha256": require_sha256(
            policy_state_sha256, "policy state SHA-256"
        ),
        "homie_controller_sha256": validate_homie_controller_sha256(
            homie_controller_sha256
        ),
    }
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "HOMIE_CONTROLLER_ROLES",
    "PolicyIdentityError",
    "RUNTIME_POLICY_IDENTITY_SCHEMA",
    "require_sha256",
    "runtime_policy_identity_sha256",
    "validate_homie_controller_sha256",
]
