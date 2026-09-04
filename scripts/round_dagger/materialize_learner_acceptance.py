#!/usr/bin/env python3
"""Materialize a missing Lepton acceptance receipt without changing state.

This is a bounded audit recovery for a round that has already reached TRAINED
using older coordinator code.  It consumes the immutable bridge environment,
revalidates the complete returned transaction, and creates only the acceptance
receipt.  It never launches a learner or mutates coordinator state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger import (  # noqa: E402
    artifacts,
    learner_acceptance,
    orchestration,
    state,
)


def _materialize(
    *,
    root: Path,
    macro_round: int,
    bridge_environment: Path,
    backend_contract: Path,
) -> int:
    environment_raw = artifacts.load_strict_json(bridge_environment)
    if not isinstance(environment_raw, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in environment_raw.items()
    ):
        raise learner_acceptance.LearnerAcceptanceError(
            "bridge environment must be a string mapping"
        )
    environment = dict(environment_raw)
    if (
        environment.get("ROUND_DAGGER_CAMPAIGN_ROOT") != str(root)
        or environment.get("ROUND_DAGGER_MACRO_ROUND") != str(macro_round)
        or environment.get("ROUND_DAGGER_LEARNER_BACKEND")
        != learner_acceptance.BACKEND
    ):
        raise learner_acceptance.LearnerAcceptanceError(
            "bridge environment is not bound to the requested round"
        )
    coordinator = state.load_state(root)
    if (
        coordinator.get("last_trained_macro_round") != macro_round
        or coordinator.get("phase") not in {"TRAINED", "EVALUATED", "READY"}
    ):
        raise learner_acceptance.LearnerAcceptanceError(
            "coordinator has not durably trained the requested round"
        )

    identity = environment["ROUND_DAGGER_LEARNER_IDENTITY_SHA256"]
    handoff = (
        Path(environment["ROUND_DAGGER_REMOTE_HANDOFF_ROOT"])
        / "lepton_job_v1"
        / environment["ROUND_DAGGER_CAMPAIGN_ID"]
        / f"macro_round_{macro_round:06d}"
        / f"identity_{identity}"
    )
    output = root / "rounds" / f"macro_round_{macro_round:06d}" / "learner"
    candidate = coordinator.get("candidate_policy")
    result_path = Path(environment["ROUND_DAGGER_LEARNER_RESULT"])
    result = artifacts.load_strict_json(result_path)
    if (
        not isinstance(candidate, Mapping)
        or not isinstance(result, Mapping)
        or candidate.get("macro_round") != macro_round
        or result.get("macro_round") != macro_round
        or result.get("checkpoint_sha256") != candidate.get("checkpoint_sha256")
        or result.get("runtime_policy_sha256")
        != candidate.get("runtime_policy_sha256")
    ):
        raise learner_acceptance.LearnerAcceptanceError(
            "returned learner does not match the coordinator candidate"
        )

    receipt_path = learner_acceptance.acceptance_path(root, macro_round)
    receipt = learner_acceptance.publish_acceptance(
        receipt_path=receipt_path,
        campaign_root=root,
        macro_round=macro_round,
        backend_contract_path=Path(os.path.abspath(backend_contract.expanduser())),
        request_path=handoff / "lepton_learner_request.json",
        marker_path=handoff / "remote_learner_success.json",
        checksum_path=handoff / "remote_learner_success.json.sha256",
        return_bundle_root=handoff / "returned_bundle",
        learner_output=output,
        expected_environment=environment,
    )
    if (
        candidate.get("checkpoint_sha256")
        != receipt["candidate_checkpoint_sha256"]
        or candidate.get("runtime_policy_sha256")
        != receipt["candidate_runtime_policy_sha256"]
    ):
        raise learner_acceptance.LearnerAcceptanceError(
            "accepted learner differs from coordinator candidate"
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "receipt_path": receipt_path.relative_to(root).as_posix(),
                "receipt_sha256": artifacts.sha256_file(receipt_path),
                "content_sha256": receipt["content_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--macro-round", required=True, type=int)
    parser.add_argument("--bridge-environment", required=True, type=Path)
    parser.add_argument("--backend-contract", required=True, type=Path)
    args = parser.parse_args(argv)

    root = Path(os.path.abspath(args.campaign_root.expanduser()))
    with orchestration.campaign_mutation_lease(
        root,
        f"materialize_learner_acceptance_{args.macro_round:06d}",
    ):
        return _materialize(
            root=root,
            macro_round=args.macro_round,
            bridge_environment=args.bridge_environment,
            backend_contract=args.backend_contract,
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        learner_acceptance.LearnerAcceptanceError,
        orchestration.CampaignMutationLockError,
        OSError,
    ) as exc:
        print(f"learner acceptance recovery failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
