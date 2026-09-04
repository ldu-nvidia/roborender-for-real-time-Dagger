"""Read-only, allowlisted campaign presets for RoboRender DAgger.

This module intentionally does not import a launcher or mutate campaign state.
It validates two checked-in semantic profiles and emits a selection record that
keeps operator identity (currently the W&B run name) outside profile identity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from round_dagger.artifacts import load_strict_json, sha256_file


PROFILE_SCHEMA = "fast_roborender.round_dagger.campaign_preset.v1"
SELECTION_SCHEMA = "fast_roborender.round_dagger.campaign_selection.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRESET_DIRECTORY = REPOSITORY_ROOT / "configs" / "round_dagger" / "presets"

_PRESET_FILES = {
    "production_256x512": "production_256x512.json",
    "validation_20m": "validation_20m.json",
}
_SAFE_OPERATOR_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_EXPECTED_SEMANTICS: dict[str, dict[str, Any]] = {
    "validation_20m": {
        "mode": "validation",
        "runner": "bounded_validation",
        "topology": {
            "target_environments": 8,
            "active_environments": 8,
            "cohort_count": 1,
            "actor_ranks_per_cohort": 2,
            "environments_per_actor_rank": 4,
            "macro_round_count": 1,
            "steps_per_environment": 16,
            "transitions_per_cohort": 128,
            "transitions_per_macro_round": 128,
            "transitions_per_campaign": 128,
        },
        "learner": {
            "backend": "lepton_reserved_h200",
            "nodes": 1,
            "world_size": 8,
            "batch_size_per_rank": 1,
            "global_batch_size": 8,
            "optimizer_steps": 20,
            "save_every_steps": 10,
        },
        "evaluation": {
            "interval_steps": 10,
            "required_optimizer_steps": [0, 10, 20],
            "heldout_cases": 32,
            "paired_seed_count": 2,
            "environments_per_seed": 2,
        },
        "time_budget": {"hard_wall_time_seconds": 1200},
        "telemetry": {
            "wandb_project": "lepton-dagger-production-1",
            "wandb_run_name_binding": "operator_required",
        },
        "isolation": {
            "artifact_namespace": "validation_20m",
            "production_contract_mutation_allowed": False,
        },
        "immutable_contracts": [],
    },
    "production_256x512": {
        "mode": "production",
        "runner": "existing_production",
        "topology": {
            "target_environments": 256,
            "active_environments": 64,
            "cohort_count": 4,
            "actor_ranks_per_cohort": 2,
            "environments_per_actor_rank": 32,
            "macro_round_count": 5,
            "steps_per_environment": 512,
            "transitions_per_cohort": 32768,
            "transitions_per_macro_round": 131072,
            "transitions_per_campaign": 655360,
        },
        "learner": {
            "backend": "lepton_reserved_h200",
            "nodes": 1,
            "world_size": 8,
            "batch_size_per_rank": 8,
            "global_batch_size": 64,
            "optimizer_steps": 1200,
            "save_every_steps": 5,
        },
        "evaluation": {
            "interval_steps": 100,
            "heldout_cases": 256,
            "paired_seed_count": 2,
            "environments_per_seed": 8,
        },
        "time_budget": {"hard_wall_time_seconds": None},
        "telemetry": {
            "wandb_project": "lepton-dagger-production-1",
            "wandb_run_name_binding": "operator_required",
        },
        "isolation": {
            "artifact_namespace": "production_256x512",
            "production_contract_mutation_allowed": False,
        },
        "immutable_contracts": [
            {
                "role": "campaign",
                "path": "configs/round_dagger/production_campaign_256x512.json",
                "sha256": (
                    "a7df6f708d74575968616823221e2bfa1"
                    "ab89543dc28046fcb88f20ab02ad14a"
                ),
            },
            {
                "role": "learner",
                "path": "configs/round_dagger/production_learner_8rank_1200.json",
                "sha256": (
                    "162c86d656b5551da2434ac65b5c63159"
                    "66429fbab838962727555fcdb842695"
                ),
            },
            {
                "role": "lepton_backend",
                "path": "configs/round_dagger/production_learner_1x8h200_lepton.json",
                "sha256": (
                    "d1ca56656fcc9103c05dbf0f91c7fde4"
                    "66711f0e3d30163dc6426a1914dffb87"
                ),
            },
        ],
    },
}

_PROFILE_KEYS = {
    "schema",
    "preset_id",
    "mode",
    "description",
    "runner",
    "topology",
    "learner",
    "evaluation",
    "time_budget",
    "telemetry",
    "isolation",
    "immutable_contracts",
}


class CampaignPresetError(RuntimeError):
    """A campaign preset or operator binding is unsafe or inconsistent."""


def preset_ids() -> tuple[str, ...]:
    """Return the complete allowlist in stable display order."""

    return tuple(sorted(_PRESET_FILES))


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignPresetError(f"{label} must be a JSON object")
    return dict(value)


def _validate_contract_binding(
    binding: Any, *, repository_root: Path
) -> dict[str, str]:
    row = _mapping(binding, "immutable contract binding")
    if set(row) != {"role", "path", "sha256"}:
        raise CampaignPresetError("immutable contract binding keys are invalid")
    if row["role"] not in {"campaign", "learner", "lepton_backend"}:
        raise CampaignPresetError("immutable contract role is invalid")
    raw_path = row["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise CampaignPresetError("immutable contract path is invalid")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != raw_path
    ):
        raise CampaignPresetError(
            "immutable contract path must be canonical and relative"
        )
    digest = row["sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise CampaignPresetError("immutable contract SHA-256 is invalid")

    root = repository_root.resolve(strict=True)
    target = root.joinpath(*relative.parts)
    try:
        metadata = os.lstat(target)
        resolved = target.resolve(strict=True)
    except OSError as exc:
        raise CampaignPresetError(
            f"immutable contract is unavailable: {raw_path}"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or resolved != target
        or root not in resolved.parents
    ):
        raise CampaignPresetError(f"immutable contract path is unsafe: {raw_path}")
    if sha256_file(target) != digest:
        raise CampaignPresetError(
            f"immutable production contract changed: {raw_path}"
        )
    return {"role": row["role"], "path": raw_path, "sha256": digest}


def validate_profile(
    profile: Any,
    *,
    expected_preset_id: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Validate one allowlisted profile and every pinned production input."""

    if expected_preset_id not in _PRESET_FILES:
        raise CampaignPresetError(f"unknown campaign preset: {expected_preset_id}")
    document = _mapping(profile, "campaign preset")
    if set(document) != _PROFILE_KEYS:
        raise CampaignPresetError("campaign preset keys are invalid")
    if document["schema"] != PROFILE_SCHEMA:
        raise CampaignPresetError("campaign preset schema is invalid")
    if document["preset_id"] != expected_preset_id:
        raise CampaignPresetError("campaign preset identity does not match its file")
    if (
        not isinstance(document["description"], str)
        or not document["description"].strip()
    ):
        raise CampaignPresetError("campaign preset description is empty")

    expected = _EXPECTED_SEMANTICS[expected_preset_id]
    for key, expected_value in expected.items():
        if document[key] != expected_value:
            raise CampaignPresetError(
                f"campaign preset {expected_preset_id} has invalid {key} semantics"
            )

    topology = _mapping(document["topology"], "topology")
    environments_per_cohort = (
        topology["actor_ranks_per_cohort"]
        * topology["environments_per_actor_rank"]
    )
    transitions_per_cohort = (
        environments_per_cohort * topology["steps_per_environment"]
    )
    if topology["target_environments"] != (
        topology["cohort_count"] * environments_per_cohort
    ):
        raise CampaignPresetError("target environment count is inconsistent")
    if topology["active_environments"] != environments_per_cohort:
        raise CampaignPresetError("active environment count is inconsistent")
    if topology["transitions_per_cohort"] != transitions_per_cohort:
        raise CampaignPresetError("cohort transition count is inconsistent")
    if topology["transitions_per_macro_round"] != (
        topology["cohort_count"] * transitions_per_cohort
    ):
        raise CampaignPresetError("macro-round transition count is inconsistent")
    if topology["transitions_per_campaign"] != (
        topology["macro_round_count"]
        * topology["transitions_per_macro_round"]
    ):
        raise CampaignPresetError("campaign transition count is inconsistent")

    learner = _mapping(document["learner"], "learner")
    if learner["global_batch_size"] != (
        learner["world_size"] * learner["batch_size_per_rank"]
    ):
        raise CampaignPresetError("learner global batch size is inconsistent")
    if document["mode"] == "validation" and (
        document["time_budget"]["hard_wall_time_seconds"] != 1200
        or document["evaluation"]["heldout_cases"] != 32
        or document["evaluation"]["required_optimizer_steps"] != [0, 10, 20]
    ):
        raise CampaignPresetError("bounded validation safety limits are invalid")

    bindings = document["immutable_contracts"]
    if not isinstance(bindings, list):
        raise CampaignPresetError("immutable_contracts must be a JSON array")
    verified_bindings = [
        _validate_contract_binding(row, repository_root=repository_root)
        for row in bindings
    ]
    if document["mode"] == "production" and {
        row["role"] for row in verified_bindings
    } != {"campaign", "learner", "lepton_backend"}:
        raise CampaignPresetError("production preset contract bindings are incomplete")
    if document["mode"] == "validation" and verified_bindings:
        raise CampaignPresetError(
            "validation preset must not bind production contracts"
        )
    return document


def profile_path(preset_id: str) -> Path:
    try:
        filename = _PRESET_FILES[preset_id]
    except KeyError as exc:
        raise CampaignPresetError(f"unknown campaign preset: {preset_id}") from exc
    return PRESET_DIRECTORY / filename


def load_profile(preset_id: str) -> dict[str, Any]:
    path = profile_path(preset_id)
    try:
        profile = load_strict_json(path)
    except Exception as exc:
        raise CampaignPresetError(f"cannot load campaign preset {preset_id}") from exc
    return validate_profile(profile, expected_preset_id=preset_id)


def describe_profile(preset_id: str) -> dict[str, Any]:
    """Return validated profile bytes and repository-relative provenance."""

    path = profile_path(preset_id)
    profile = load_profile(preset_id)
    return {
        "preset_id": preset_id,
        "profile_path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "profile_sha256": sha256_file(path),
        "profile": profile,
    }


def select_profile(preset_id: str, *, wandb_run_name: str) -> dict[str, Any]:
    """Emit a deterministic, non-launching selection with operator bindings."""

    if (
        not isinstance(wandb_run_name, str)
        or _SAFE_OPERATOR_NAME.fullmatch(wandb_run_name) is None
    ):
        raise CampaignPresetError(
            "W&B run name must be a safe 1-128 character operator name"
        )
    described = describe_profile(preset_id)
    profile = described["profile"]
    return {
        "schema": SELECTION_SCHEMA,
        "selection_only": True,
        "preset_id": preset_id,
        "mode": profile["mode"],
        "profile_path": described["profile_path"],
        "profile_sha256": described["profile_sha256"],
        "artifact_namespace": profile["isolation"]["artifact_namespace"],
        "operator_bindings": {"wandb_run_name": wandb_run_name},
        "profile": profile,
    }


def _print(document: Any) -> None:
    print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect and select allowlisted DAgger campaign presets. "
            "This command never launches a job or writes campaign state."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list and validate every allowlisted preset")
    show = commands.add_parser("show", help="show one validated semantic profile")
    show.add_argument("preset_id")
    validate = commands.add_parser(
        "validate", help="validate one preset or the allowlist"
    )
    validate.add_argument("preset_id", nargs="?")
    select = commands.add_parser(
        "select", help="emit a non-launching operator selection"
    )
    select.add_argument("preset_id")
    select.add_argument("--wandb-run-name", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            _print({"presets": [describe_profile(name) for name in preset_ids()]})
        elif args.command == "show":
            _print(describe_profile(args.preset_id))
        elif args.command == "validate":
            selected = (args.preset_id,) if args.preset_id else preset_ids()
            _print(
                {
                    "status": "valid",
                    "presets": [describe_profile(name) for name in selected],
                }
            )
        elif args.command == "select":
            _print(
                select_profile(
                    args.preset_id, wandb_run_name=args.wandb_run_name
                )
            )
        else:  # pragma: no cover - argparse enforces this branch.
            raise CampaignPresetError("unsupported command")
    except CampaignPresetError as exc:
        parser.error(str(exc))
    return 0
