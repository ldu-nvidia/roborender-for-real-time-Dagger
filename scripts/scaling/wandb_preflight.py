#!/usr/bin/env python3
"""Verify W&B authentication without printing or persisting credentials.

The production DAgger launcher runs under a shared cluster account, so an
existing ``~/.netrc`` can silently select the wrong W&B user.  This preflight
performs a read-only viewer query, checks the expected account email, and
returns the personal entity slug that should be passed to ``wandb.init``.
W&B's ``viewer.entity`` is intentionally ignored for this decision because it
is the account's configured default and may name a team; only
``viewer.username`` proves the personal workspace.

No API key, netrc contents, or W&B settings are included in either successful
or failed output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Mapping


class WandbPreflightError(RuntimeError):
    """A credential, identity, or configuration check failed."""


@dataclass(frozen=True)
class WandbIdentity:
    email: str
    username: str
    entity: str
    personal: bool
    mode: str


def _field(obj: object, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _nonempty(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def validate_mode(environment: Mapping[str, str] | None = None) -> str:
    """Reject modes that would make an eight-hour run look monitored when it is not."""

    environment = os.environ if environment is None else environment
    mode = environment.get("WANDB_MODE", "online").strip().lower() or "online"
    if mode in {"offline", "dryrun", "disabled"}:
        raise WandbPreflightError(
            f"WANDB_MODE={mode!r} does not provide live production telemetry"
        )
    if mode != "online":
        raise WandbPreflightError(f"unsupported WANDB_MODE={mode!r}; expected 'online'")
    return mode


def validate_identity(
    viewer: object,
    *,
    expected_email: str,
    requested_entity: str | None = None,
    mode: str = "online",
    require_personal: bool = True,
) -> WandbIdentity:
    """Validate viewer identity and derive a personal entity slug.

    W&B entities are slugs, not email addresses.  A personal workspace uses the
    authenticated viewer's username.  ``viewer.entity`` is the configured
    default entity and can be a team, so it must never be accepted as evidence
    that an entity is personal.  Team entities are deliberately rejected unless
    ``require_personal`` is disabled by the caller.
    """

    expected_email = expected_email.strip().lower()
    if not expected_email:
        raise WandbPreflightError("an expected W&B account email is required")

    email = _nonempty(_field(viewer, "email"))
    username = _nonempty(_field(viewer, "username"))
    if email is None:
        raise WandbPreflightError("W&B viewer response did not include an email address")
    if email.lower() != expected_email:
        raise WandbPreflightError(
            "authenticated W&B account does not match the expected email"
        )
    if username is None:
        raise WandbPreflightError(
            "W&B viewer response did not include a personal username"
        )

    entity = _nonempty(requested_entity) or username
    personal = entity == username
    if require_personal and not personal:
        raise WandbPreflightError(
            "requested W&B entity is not the authenticated user's personal workspace"
        )

    return WandbIdentity(
        email=email,
        username=username,
        entity=entity,
        personal=personal,
        mode=mode,
    )


def query_identity(
    wandb_module: object,
    *,
    expected_email: str,
    requested_entity: str | None = None,
    timeout_s: float = 15.0,
    environment: Mapping[str, str] | None = None,
    require_personal: bool = True,
) -> WandbIdentity:
    """Perform the read-only W&B viewer query and validate the result."""

    mode = validate_mode(environment)
    try:
        api = wandb_module.Api(timeout=timeout_s)
        viewer = api.viewer
    except Exception as exc:  # Keep third-party errors from leaking credentials/settings.
        raise WandbPreflightError(
            f"W&B authentication/viewer query failed ({type(exc).__name__})"
        ) from None
    return validate_identity(
        viewer,
        expected_email=expected_email,
        requested_entity=requested_entity,
        mode=mode,
        require_personal=require_personal,
    )


def _load_wandb() -> object:
    try:
        import wandb
    except ImportError:
        raise WandbPreflightError("the wandb package is not installed") from None
    return wandb


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-email",
        default=os.environ.get("WANDB_EXPECTED_EMAIL"),
        help="authenticated account required for this run (or WANDB_EXPECTED_EMAIL)",
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="personal entity slug to validate; omitted means derive it from the viewer",
    )
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument(
        "--allow-team-entity",
        action="store_true",
        help="allow a non-personal entity (production RoboRender should not use this)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "shell"),
        default="text",
        help="successful output format; none includes credentials",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.expected_email:
        print(
            "W&B preflight failed: --expected-email or WANDB_EXPECTED_EMAIL is required",
            file=sys.stderr,
        )
        return 2
    try:
        identity = query_identity(
            _load_wandb(),
            expected_email=args.expected_email,
            requested_entity=args.entity,
            timeout_s=args.timeout_s,
            require_personal=not args.allow_team_entity,
        )
    except WandbPreflightError as exc:
        print(f"W&B preflight failed: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(asdict(identity), sort_keys=True))
    elif args.format == "shell":
        # W&B entity slugs are restricted to a shell-safe character set, but
        # reject anything unexpected instead of attempting clever quoting.
        if not identity.entity.replace("-", "").replace("_", "").isalnum():
            print("W&B preflight failed: entity slug is not shell-safe", file=sys.stderr)
            return 2
        print(f"WANDB_ENTITY={identity.entity}")
    else:
        print(
            f"W&B preflight OK: account={identity.email} "
            f"entity={identity.entity} personal={str(identity.personal).lower()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
