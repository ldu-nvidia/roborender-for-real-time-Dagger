"""Canonical identity for one exact round-DAgger learner execution."""

from __future__ import annotations

import argparse
import hashlib
import re


IDENTITY_SCHEMA = "fast_roborender.round_dagger.learner_run_identity.v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class LearnerIdentityError(ValueError):
    """The learner semantic identity is malformed."""


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LearnerIdentityError(f"{label} must be a lowercase SHA-256")
    return value


def learner_identity_sha256(
    *,
    campaign_id: str,
    macro_round: int,
    learner_code_sha256: str,
    learner_config_sha256: str,
    replay_snapshot_file_sha256: str,
    replay_snapshot_sha256: str,
    parent_checkpoint_sha256: str,
    parent_runtime_policy_sha256: str,
    runtime_container_sha256: str,
    learner_contract_sha256: str,
) -> str:
    """Hash the complete learner semantics using a shell-reproducible encoding."""

    if not isinstance(campaign_id, str) or _TOKEN_RE.fullmatch(campaign_id) is None:
        raise LearnerIdentityError("campaign ID is malformed")
    if type(macro_round) is not int or macro_round < 0:
        raise LearnerIdentityError("macro round must be a non-negative integer")
    fields = (
        IDENTITY_SCHEMA,
        f"campaign_id={campaign_id}",
        f"macro_round={macro_round}",
        f"learner_code_sha256={_sha(learner_code_sha256, 'learner code SHA-256')}",
        f"learner_config_sha256={_sha(learner_config_sha256, 'learner config SHA-256')}",
        "replay_snapshot_file_sha256="
        f"{_sha(replay_snapshot_file_sha256, 'replay snapshot file SHA-256')}",
        "replay_snapshot_sha256="
        f"{_sha(replay_snapshot_sha256, 'replay snapshot SHA-256')}",
        "parent_checkpoint_sha256="
        f"{_sha(parent_checkpoint_sha256, 'parent checkpoint SHA-256')}",
        "parent_runtime_policy_sha256="
        f"{_sha(parent_runtime_policy_sha256, 'parent runtime policy SHA-256')}",
        "runtime_container_sha256="
        f"{_sha(runtime_container_sha256, 'runtime container SHA-256')}",
        "learner_contract_sha256="
        f"{_sha(learner_contract_sha256, 'learner contract SHA-256')}",
    )
    payload = ("\n".join(fields) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _main(argv=None) -> int:
    """Derive an identity without importing the :mod:`round_dagger` package.

    The OCI validation batch frontier deliberately executes this file by its
    absolute path.  Importing ``round_dagger.learner_identity`` first executes
    the package ``__init__`` and therefore pulls in NumPy-backed actor artifact
    code, while the compute-node host Python is intentionally only a minimal
    standard-library interpreter.  Keep this entry point standard-library-only.
    """

    parser = argparse.ArgumentParser(
        description="derive one canonical round-DAgger learner identity"
    )
    parser.add_argument("campaign_id")
    parser.add_argument("macro_round", type=int)
    parser.add_argument("learner_code_sha256")
    parser.add_argument("learner_config_sha256")
    parser.add_argument("replay_snapshot_file_sha256")
    parser.add_argument("replay_snapshot_sha256")
    parser.add_argument("parent_checkpoint_sha256")
    parser.add_argument("parent_runtime_policy_sha256")
    parser.add_argument("runtime_container_sha256")
    parser.add_argument("learner_contract_sha256")
    arguments = parser.parse_args(argv)
    identity = learner_identity_sha256(
        campaign_id=arguments.campaign_id,
        macro_round=arguments.macro_round,
        learner_code_sha256=arguments.learner_code_sha256,
        learner_config_sha256=arguments.learner_config_sha256,
        replay_snapshot_file_sha256=arguments.replay_snapshot_file_sha256,
        replay_snapshot_sha256=arguments.replay_snapshot_sha256,
        parent_checkpoint_sha256=arguments.parent_checkpoint_sha256,
        parent_runtime_policy_sha256=arguments.parent_runtime_policy_sha256,
        runtime_container_sha256=arguments.runtime_container_sha256,
        learner_contract_sha256=arguments.learner_contract_sha256,
    )
    print(identity)
    return 0


__all__ = [
    "IDENTITY_SCHEMA",
    "LearnerIdentityError",
    "learner_identity_sha256",
]


if __name__ == "__main__":
    raise SystemExit(_main())
