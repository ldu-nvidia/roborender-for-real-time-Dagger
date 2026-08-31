#!/usr/bin/env python3
"""Verify an explicitly authorized W&B team identity without storing secrets.

The caller supplies the authorized email and team entity at launch time.  This
module never embeds an account, team, API key, netrc value, or W&B settings in
source or output.  It performs read-only viewer and team-membership queries,
then a separate online smoke may prove write access.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Mapping


AUTHORIZED_SCOPE = "team"


class WandbPreflightError(RuntimeError):
    """A credential, identity, or authorization check failed."""


@dataclass(frozen=True)
class WandbIdentity:
    email: str
    username: str
    entity: str
    authorization_scope: str
    team_member: bool
    entity_is_team: bool
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


def validate_authorized_request(
    *, expected_email: str, requested_entity: str, authorization_scope: str
) -> None:
    """Reject any authorization request outside the user-approved team scope."""

    if not _nonempty(expected_email):
        raise WandbPreflightError("expected W&B account must be non-empty")
    if not _nonempty(requested_entity):
        raise WandbPreflightError("requested W&B entity must be non-empty")
    if authorization_scope != AUTHORIZED_SCOPE:
        raise WandbPreflightError("unsupported W&B authorization scope")


def validate_mode(environment: Mapping[str, str] | None = None) -> str:
    """Reject modes that would make a production run look monitored when it is not."""

    environment = os.environ if environment is None else environment
    mode = environment.get("WANDB_MODE", "online").strip().lower() or "online"
    if mode in {"offline", "dryrun", "disabled"}:
        raise WandbPreflightError(
            f"WANDB_MODE={mode!r} does not provide live production telemetry"
        )
    if mode != "online":
        raise WandbPreflightError(f"unsupported WANDB_MODE={mode!r}; expected 'online'")
    return mode


def _team_names(viewer: object) -> tuple[str, ...]:
    """Return explicitly listed viewer team names from supported SDK shapes."""

    teams = _field(viewer, "teams")
    if teams is None:
        raise WandbPreflightError(
            "W&B viewer response did not include team memberships"
        )

    if isinstance(teams, Mapping):
        entries = teams.get("edges")
    else:
        entries = teams
    if not isinstance(entries, (list, tuple)):
        raise WandbPreflightError("W&B viewer team memberships are malformed")

    names: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            name = _nonempty(entry)
        else:
            node = _field(entry, "node")
            if node is None:
                node = entry
            name = _nonempty(_field(node, "name"))
        if name is None:
            raise WandbPreflightError("W&B viewer team membership is malformed")
        names.append(name)
    return tuple(names)


def validate_identity(
    viewer: object,
    *,
    expected_email: str,
    requested_entity: str,
    authorization_scope: str,
    mode: str = "online",
) -> WandbIdentity:
    """Validate the pinned viewer, team membership, and team entity type."""

    validate_authorized_request(
        expected_email=expected_email,
        requested_entity=requested_entity,
        authorization_scope=authorization_scope,
    )

    email = _nonempty(_field(viewer, "email"))
    username = _nonempty(_field(viewer, "username"))
    if email is None:
        raise WandbPreflightError("W&B viewer response did not include an email address")
    if email.lower() != expected_email.lower():
        raise WandbPreflightError(
            "authenticated W&B account does not match the authorized account"
        )
    if username is None:
        raise WandbPreflightError("W&B viewer response did not include a username")

    team_names = _team_names(viewer)
    if requested_entity not in team_names:
        raise WandbPreflightError(
            "authorized W&B entity is not in the viewer's team memberships"
        )

    return WandbIdentity(
        # Emit the caller-pinned canonical identity rather than server casing
        # so the acceptance artifact has one stable byte-level account binding.
        email=expected_email.lower(),
        username=username,
        entity=requested_entity,
        authorization_scope=authorization_scope,
        team_member=True,
        entity_is_team=True,
        personal=False,
        mode=mode,
    )


def query_identity(
    wandb_module: object,
    *,
    expected_email: str,
    requested_entity: str,
    authorization_scope: str,
    timeout_s: float = 15.0,
    environment: Mapping[str, str] | None = None,
) -> WandbIdentity:
    """Perform read-only W&B viewer/team queries and validate both results."""

    validate_authorized_request(
        expected_email=expected_email,
        requested_entity=requested_entity,
        authorization_scope=authorization_scope,
    )
    mode = validate_mode(environment)
    try:
        viewer = wandb_module.Api(timeout=timeout_s).viewer
    except Exception as exc:  # Keep third-party errors from leaking credentials/settings.
        raise WandbPreflightError(
            f"W&B authentication/team query failed ({type(exc).__name__})"
        ) from None
    return validate_identity(
        viewer,
        expected_email=expected_email,
        requested_entity=requested_entity,
        authorization_scope=authorization_scope,
        mode=mode,
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
        required=True,
        help="authenticated account explicitly authorized for this run",
    )
    parser.add_argument(
        "--entity",
        required=True,
        help="team entity explicitly authorized for this run",
    )
    parser.add_argument(
        "--authorization-scope",
        required=True,
        choices=(AUTHORIZED_SCOPE,),
        help="explicit W&B authorization scope",
    )
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument(
        "--format",
        choices=("text", "json", "shell"),
        default="text",
        help="successful output format; none includes credentials",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        identity = query_identity(
            _load_wandb(),
            expected_email=args.expected_email,
            requested_entity=args.entity,
            authorization_scope=args.authorization_scope,
            timeout_s=args.timeout_s,
        )
    except WandbPreflightError as exc:
        print(f"W&B preflight failed: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(asdict(identity), sort_keys=True))
    elif args.format == "shell":
        # The entity is pinned, but retain a local shell-safety assertion.
        if not identity.entity.replace("-", "").replace("_", "").isalnum():
            print("W&B preflight failed: entity slug is not shell-safe", file=sys.stderr)
            return 2
        print(f"WANDB_ENTITY={identity.entity}")
    else:
        print(
            f"W&B preflight OK: account={identity.email} "
            f"entity={identity.entity} authorization_scope={identity.authorization_scope} "
            f"team_member={str(identity.team_member).lower()} "
            f"entity_is_team={str(identity.entity_is_team).lower()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
