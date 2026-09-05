#!/usr/bin/env python3
"""Prepare or seal one immutable native-H200 BC wave cohort."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from round_dagger import artifacts  # noqa: E402
from round_dagger.conditioning_artifacts import (  # noqa: E402
    ARTIFACT_KIND,
    load_raw_conditioning_artifact,
    validate_conditioning_contract,
)
from round_dagger.gr00t_policy import REPLAY_ACTION_DIM  # noqa: E402
from round_dagger.native_actor_saturation import (  # noqa: E402
    CANDIDATES,
    GPU_COUNT,
    VIRTUAL_ACTOR_WORLD_SIZE,
    expected_shard_names,
    group_rotation,
)
from round_dagger.seed_policy import (  # noqa: E402
    ActorSeedError,
    validate_actor_seed_contract,
)


CAMPAIGN_ROOT = Path(
    "/workspace/project/roborender/data"
)
SHA = re.compile(r"^[0-9a-f]{64}$")
COHORT_SCHEMA = "fast_roborender.round_dagger.isaac_conditioning_cohort.v1"
SEAL_SCHEMA = "fast_roborender.round_dagger.isaac_conditioning_cohort_seal.v1"
SHARD_NAME = re.compile(
    r"^gen-000-bc__wave-(?P<wave>[0-9]{3})__gpu-(?P<rank>[0-9]{2})"
    r"__shard-(?P<shard>[0-9]{6})$"
)
ENVIRONMENTS_PER_SHARD = 4
ACTOR_WORLD_SIZE = VIRTUAL_ACTOR_WORLD_SIZE
ENVIRONMENTS_PER_ACTOR = 128
PRODUCTION_WAVES = 2
PHYSICAL_GPU_COUNT = GPU_COUNT


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing required environment: {name}")
    return value


def integer(name: str, *, minimum: int = 0) -> int:
    raw = required(name)
    if re.fullmatch(r"0|[1-9][0-9]*", raw) is None or int(raw) < minimum:
        raise SystemExit(f"{name} must be a canonical integer >= {minimum}")
    return int(raw)


def digest(name: str) -> str:
    value = required(name)
    if SHA.fullmatch(value) is None:
        raise SystemExit(f"{name} must be an exact lowercase SHA-256")
    return value


def context() -> dict:
    if CANDIDATES != (ENVIRONMENTS_PER_ACTOR,):
        raise SystemExit("actor candidate contract must be exactly (128,)")
    if ACTOR_WORLD_SIZE != 1 or PHYSICAL_GPU_COUNT != 8:
        raise SystemExit("actor/physical GPU topology contract differs")
    if Path(required("ROUND_DAGGER_CAMPAIGN_ROOT")) != CAMPAIGN_ROOT:
        raise SystemExit("campaign root differs from canonical data root")
    if required("ROUND_DAGGER_GENERATION_ID") != "gen-000-bc":
        raise SystemExit("generation ID must be gen-000-bc")
    generation = CAMPAIGN_ROOT / "BC/collection"
    if Path(required("ROUND_DAGGER_GENERATION_ROOT")) != generation:
        raise SystemExit("generation root differs")
    wave = integer("ROUND_DAGGER_SATURATION_WAVE_INDEX")
    if wave not in range(PRODUCTION_WAVES):
        raise SystemExit("BC wave index must be exactly 0 or 1")
    if integer("ROUND_DAGGER_COHORT_INDEX") != wave or integer("ROUND_DAGGER_ROLLOUT_ID") != wave:
        raise SystemExit("cohort/rollout identity differs from wave index")
    cohort = generation / "conditioning" / "waves" / f"wave-{wave:03d}"
    if Path(required("ROUND_DAGGER_COHORT_DIR")) != cohort:
        raise SystemExit("cohort path differs from the exact wave namespace")
    envs = integer("ROUND_DAGGER_ENVS_PER_GPU", minimum=1)
    if envs != ENVIRONMENTS_PER_ACTOR:
        raise SystemExit("BC wave must contain exactly 128 environments per actor")
    if integer("ROUND_DAGGER_STEPS_PER_ENVIRONMENT", minimum=1) != 512:
        raise SystemExit("BC wave must contain exactly 512 steps per environment")
    if integer("ROUND_DAGGER_ACTOR_WORLD_SIZE", minimum=1) != ACTOR_WORLD_SIZE:
        raise SystemExit("BC wave must contain exactly one actor rank")
    if (
        required("ROUND_DAGGER_ACTOR_ROLE") != "bc_collect"
        or required("ROUND_DAGGER_PURPOSE") != "train"
        or required("ROUND_DAGGER_BETA") != "1.0"
    ):
        raise SystemExit("BC wave role/purpose/beta differs")
    offset = integer("ROUND_DAGGER_GLOBAL_ENV_OFFSET")
    if offset != wave * ACTOR_WORLD_SIZE * envs:
        raise SystemExit("global environment offset differs from wave identity")
    environments_per_shard = integer(
        "ROUND_DAGGER_ENVS_PER_SHARD", minimum=1
    )
    if environments_per_shard != ENVIRONMENTS_PER_SHARD:
        raise SystemExit("BC wave shards must contain exactly four environments")
    if envs % environments_per_shard:
        raise SystemExit("BC wave environment count must divide exactly into shards")
    names = json.loads(required("ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON"))
    exact_names = expected_shard_names(
        wave_index=wave,
        envs_per_gpu=envs,
        environments_per_shard=environments_per_shard,
    )
    if names != exact_names:
        raise SystemExit("expected shard inventory differs from exact wave naming")
    try:
        seed_contract = json.loads(required("ROUND_DAGGER_ACTOR_SEED_CONTRACT_JSON"))
    except json.JSONDecodeError as exc:
        raise SystemExit("actor seed contract is not valid JSON") from exc
    if (
        digest("ROUND_DAGGER_ACTOR_SEED_CONTRACT_SHA256")
        != artifacts.canonical_json_sha256(seed_contract)
    ):
        raise SystemExit("actor seed contract SHA-256 mismatch")
    expected_env_ids = list(range(offset, offset + ACTOR_WORLD_SIZE * envs))
    try:
        seed_contract = validate_actor_seed_contract(
            seed_contract,
            campaign_id=required("ROUND_DAGGER_CAMPAIGN_ID"),
            macro_round=integer("ROUND_DAGGER_MACRO_ROUND"),
            cohort_index=wave,
            expected_ranks=list(range(ACTOR_WORLD_SIZE)),
            expected_global_env_ids=expected_env_ids,
        )
    except ActorSeedError as exc:
        raise SystemExit(f"actor seed contract is invalid: {exc}") from exc
    if integer("ROUND_DAGGER_SIMULATOR_BASE_SEED", minimum=1) != seed_contract[
        "simulator_base_seed"
    ]:
        raise SystemExit("simulator base seed differs from actor seed contract")
    try:
        train_group_ids = json.loads(required("ROUND_DAGGER_TRAIN_GROUP_IDS_JSON"))
        supplied_rotation = json.loads(required("ROUND_DAGGER_GROUP_ROTATION_JSON"))
    except json.JSONDecodeError as exc:
        raise SystemExit("train group IDs/rotation are not valid JSON") from exc
    if (
        not isinstance(train_group_ids, list)
        or not train_group_ids
        or any(type(group_id) is not str or not group_id for group_id in train_group_ids)
        or len(set(train_group_ids)) != len(train_group_ids)
    ):
        raise SystemExit("train group IDs must be nonempty unique strings")
    exact_rotation = group_rotation(train_group_ids, envs, wave)
    if supplied_rotation != exact_rotation:
        raise SystemExit("train group rotation differs from the exact wave allocation")
    return {
        "wave": wave,
        "cohort": cohort,
        "envs": envs,
        "environments_per_shard": environments_per_shard,
        "offset": offset,
        "expected_names": names,
        "actor_seed_contract": seed_contract,
        "split_manifest_sha256": digest(
            "ROUND_DAGGER_TRAIN_PARTITION_SHA256"
        ),
        "split_sha256": digest("ROUND_DAGGER_SPLIT_SHA256"),
        "train_group_ids": train_group_ids,
        "group_rotation": exact_rotation,
    }


def provenance() -> dict:
    gpu_count = integer("ROUND_DAGGER_GPU_COUNT", minimum=1)
    if gpu_count != PHYSICAL_GPU_COUNT:
        raise SystemExit("physical GPU count must attest exactly eight H200s")
    try:
        physical_gpu_inventory = json.loads(
            required("ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON")
        )
    except json.JSONDecodeError as exc:
        raise SystemExit("physical GPU inventory is not valid JSON") from exc
    physical_gpu_inventory = artifacts.validate_physical_gpu_inventory(
        physical_gpu_inventory
    )
    inventory_sha256 = digest("ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256")
    if inventory_sha256 != artifacts.canonical_json_sha256(physical_gpu_inventory):
        raise SystemExit("physical GPU inventory SHA-256 mismatch")
    values = {
        "observation_source": required("ROUND_DAGGER_OBSERVATION_SOURCE"),
        "observation_runtime_sha256": digest(
            "ROUND_DAGGER_OBSERVATION_RUNTIME_SHA256"
        ),
        "isaac_runtime_sha256": digest("ROUND_DAGGER_ISAAC_RUNTIME_SHA256"),
        "actor_python_freeze_sha256": digest(
            "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256"
        ),
        "actor_source_inventory_sha256": digest(
            "ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256"
        ),
        "gr00t_revision": required("ROUND_DAGGER_GR00T_REVISION"),
        "gr00t_source_tree_sha256": digest(
            "ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256"
        ),
        "gpu_name": required("ROUND_DAGGER_GPU_NAME"),
        "compute_capability": required("ROUND_DAGGER_COMPUTE_CAPABILITY"),
        "gpu_count": gpu_count,
        "physical_gpu_inventory": physical_gpu_inventory,
        "physical_gpu_inventory_sha256": inventory_sha256,
    }
    return values


def prepare(values: dict) -> dict:
    count = ACTOR_WORLD_SIZE * values["envs"]
    cohort = values["cohort"]
    if cohort.is_symlink():
        raise SystemExit("conditioning cohort cannot be a symbolic link")
    cohort.mkdir(parents=True, exist_ok=True)
    shards = cohort / "shards"
    if shards.is_symlink():
        raise SystemExit("conditioning shard directory cannot be a symbolic link")
    shards.mkdir(exist_ok=True)
    document = {
        "schema": COHORT_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "status": "prepared",
        "learner_replay_ready": False,
        "campaign_id": required("ROUND_DAGGER_CAMPAIGN_ID"),
        "generation_id": "gen-000-bc",
        "macro_round": integer("ROUND_DAGGER_MACRO_ROUND"),
        "cohort_index": integer("ROUND_DAGGER_COHORT_INDEX"),
        "purpose": "train",
        "expected_ranks": list(range(ACTOR_WORLD_SIZE)),
        "expected_global_env_ids": list(
            range(values["offset"], values["offset"] + count)
        ),
        "expected_transitions": count * 512,
        "expected_shard_names": values["expected_names"],
        "environments_per_shard": values["environments_per_shard"],
        "control_contract": validate_conditioning_contract(),
        "action_contract": {"dtype": "float32", "shape": [REPLAY_ACTION_DIM]},
        "split_manifest_sha256": values["split_manifest_sha256"],
        "split_sha256": values["split_sha256"],
        "train_group_ids": values["train_group_ids"],
        "group_rotation": values["group_rotation"],
        "policy_sha256": digest("ROUND_DAGGER_POLICY_SHA256"),
        "runtime_policy_sha256": digest("ROUND_DAGGER_RUNTIME_POLICY_SHA256"),
        "teacher_sha256": digest("ROUND_DAGGER_TEACHER_SHA256"),
        "beta_sha256": digest("ROUND_DAGGER_BETA_SHA256"),
        "config_sha256": digest("ROUND_DAGGER_CONFIG_SHA256"),
        "actor_seed_contract": values["actor_seed_contract"],
        "provenance": provenance(),
    }
    contract_path = cohort / "contract.json"
    contract_sha256 = artifacts.publish_immutable_json(contract_path, document)
    return {
        **document,
        "path": str(contract_path),
        "sha256": contract_sha256,
        "transition_count": document["expected_transitions"],
    }


def seal(values: dict) -> dict:
    cohort = values["cohort"]
    contract_path = cohort / "contract.json"
    contract = artifacts.load_strict_json(contract_path)
    prepared = prepare(values)
    if contract != {key: prepared[key] for key in contract}:
        raise SystemExit("conditioning cohort contract differs from preparation")
    contract_sha256 = artifacts.sha256_file(contract_path)
    shard_dir = cohort / "shards"
    if not shard_dir.is_dir() or shard_dir.is_symlink():
        raise SystemExit("conditioning shard directory is missing or unsafe")
    actual_names = sorted(path.name for path in shard_dir.iterdir())
    if actual_names != sorted(values["expected_names"]):
        raise SystemExit("conditioning shard inventory differs from its contract")

    expected_count = ACTOR_WORLD_SIZE * values["envs"] * 512
    expected_ids = list(
        range(
            values["offset"],
            values["offset"] + ACTOR_WORLD_SIZE * values["envs"],
        )
    )
    expected_stems = sorted(
        name[:-5] for name in values["expected_names"] if name.endswith(".json")
    )
    environments_per_shard = values["environments_per_shard"]
    parts_per_rank = values["envs"] // environments_per_shard
    observed_ids: list[int] = []
    transition_count = 0
    episode_count = 0
    image_count = 0
    shard_reports = []
    expected_provenance = provenance()
    for stem in expected_stems:
        match = SHARD_NAME.fullmatch(stem)
        if match is None or int(match.group("wave")) != values["wave"]:
            raise SystemExit(f"conditioning shard identity is malformed: {stem}")
        rank = int(match.group("rank"))
        shard_index = int(match.group("shard"))
        wave_first_shard_index = (
            values["wave"] * ACTOR_WORLD_SIZE * parts_per_rank
        )
        wave_shard_ordinal = shard_index - wave_first_shard_index
        ordinal_rank, part_index = divmod(wave_shard_ordinal, parts_per_rank)
        if (
            rank not in range(ACTOR_WORLD_SIZE)
            or wave_shard_ordinal < 0
            or ordinal_rank != rank
            or part_index not in range(parts_per_rank)
        ):
            raise SystemExit(f"conditioning shard rank/part ordinal differs: {stem}")
        rank_env_start = values["offset"] + rank * values["envs"]
        expected_shard_ids = list(
            range(
                rank_env_start + part_index * environments_per_shard,
                rank_env_start + (part_index + 1) * environments_per_shard,
            )
        )
        motion_offset = values["group_rotation"]["rank_group_offsets"][rank]
        expected_motion_ids = [
            values["train_group_ids"][
                (motion_offset + part_index * environments_per_shard + local_index)
                % len(values["train_group_ids"])
            ]
            for local_index in range(environments_per_shard)
        ]
        report, arrays = load_raw_conditioning_artifact(shard_dir / f"{stem}.json")
        metadata = report["render"]["metadata"]
        exact_metadata = {
            "campaign_id": required("ROUND_DAGGER_CAMPAIGN_ID"),
            "macro_round": integer("ROUND_DAGGER_MACRO_ROUND"),
            "cohort_index": values["wave"],
            "rank": rank,
            "rollout_id": values["wave"],
            "part_index": part_index,
            "shard_id": stem,
            "policy_sha256": digest("ROUND_DAGGER_POLICY_SHA256"),
            "runtime_policy_sha256": digest("ROUND_DAGGER_RUNTIME_POLICY_SHA256"),
            "teacher_sha256": digest("ROUND_DAGGER_TEACHER_SHA256"),
            "beta_sha256": digest("ROUND_DAGGER_BETA_SHA256"),
            "config_sha256": digest("ROUND_DAGGER_CONFIG_SHA256"),
            "split_manifest_sha256": values["split_manifest_sha256"],
            "split_sha256": values["split_sha256"],
            "group_ids": values["train_group_ids"],
            "motion_ids": expected_motion_ids,
            "panel_sha256": None,
            **expected_provenance,
        }
        if any(metadata.get(key) != value for key, value in exact_metadata.items()):
            raise SystemExit(f"conditioning shard provenance differs: {stem}")
        if (
            report["purpose"] != "train"
            or report["learner_replay_ready"] is not False
            or report["num_steps"] != 512
            or report["num_envs"] != environments_per_shard
            or report["action_dim"] != REPLAY_ACTION_DIM
            or report["control_contract"] != contract["control_contract"]
        ):
            raise SystemExit(f"conditioning shard contract differs: {stem}")
        ids = arrays["global_env_ids"]
        if not np.array_equal(ids, np.broadcast_to(ids[0:1], ids.shape)):
            raise SystemExit(f"conditioning shard environment IDs changed: {stem}")
        shard_ids = [int(value) for value in ids[0].tolist()]
        if shard_ids != expected_shard_ids:
            raise SystemExit(
                f"conditioning shard environment ownership differs: {stem}"
            )
        observed_ids.extend(shard_ids)
        transition_count += report["transition_count"]
        episode_count += report["episode_count"]
        image_count += report["image_count"]
        shard_reports.append({
            "shard_id": stem,
            "rank": rank,
            "part_index": part_index,
            "global_env_ids": shard_ids,
            "motion_ids": expected_motion_ids,
            "transition_count": report["transition_count"],
            "episode_count": report["episode_count"],
            "image_count": report["image_count"],
            "manifest": {
                "path": f"shards/{stem}.json",
                "sha256": report["manifest_sha256"],
                "size_bytes": (shard_dir / f"{stem}.json").stat().st_size,
            },
            "payload": {
                "path": f"shards/{stem}.npz",
                "sha256": report["payload"]["sha256"],
                "size_bytes": report["payload"]["size_bytes"],
            },
        })
    if transition_count != expected_count or sorted(observed_ids) != expected_ids:
        raise SystemExit("sealed BC wave differs from its exact coverage contract")
    if len(observed_ids) != len(set(observed_ids)):
        raise SystemExit("sealed BC wave repeats global environment identities")
    seal_document = {
        "schema": SEAL_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "sealed": True,
        "learner_replay_ready": False,
        "campaign_id": required("ROUND_DAGGER_CAMPAIGN_ID"),
        "generation_id": "gen-000-bc",
        "macro_round": integer("ROUND_DAGGER_MACRO_ROUND"),
        "cohort_index": values["wave"],
        "purpose": "train",
        "contract_sha256": contract_sha256,
        "split_manifest_sha256": contract["split_manifest_sha256"],
        "split_sha256": contract["split_sha256"],
        "train_group_ids": contract["train_group_ids"],
        "group_rotation": contract["group_rotation"],
        "transition_count": transition_count,
        "episode_count": episode_count,
        "image_count": image_count,
        "global_env_ids": sorted(observed_ids),
        "shard_count": len(shard_reports),
        "shards": shard_reports,
    }
    seal_path = cohort / "seal.json"
    seal_sha256 = artifacts.publish_immutable_json(seal_path, seal_document)
    return {
        **seal_document,
        "path": str(seal_path),
        "sha256": seal_sha256,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in {"prepare", "seal"}:
        raise SystemExit("usage: native_h200_bc_wave_artifact.py prepare|seal")
    values = context()
    report = prepare(values) if argv[1] == "prepare" else seal(values)
    print(json.dumps({
        "operation": argv[1],
        "cohort": str(values["cohort"]),
        "transition_count": report.get("transition_count", report.get("expected_transitions")),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
