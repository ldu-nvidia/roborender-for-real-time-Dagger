#!/usr/bin/env python3
"""Run the one-time 512-episode final success test on eight local H200s."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger.artifacts import (  # noqa: E402
    load_strict_json,
    publish_immutable_json,
)
from round_dagger.gr00t_policy import load_policy_spec  # noqa: E402
from round_dagger.native_h200_experiment import (  # noqa: E402
    Controller,
    FINAL_TEST_RESULT_SCHEMA,
)
from round_dagger.native_h200_fleet import (  # noqa: E402
    CANONICAL_CAMPAIGN_ROOT,
    NativeH200FleetError,
    query_h200_inventory,
    require_environment,
    safe_directory,
    safe_python_launcher,
    safe_regular,
    sanitized_environment,
    sha256_file,
    strict_document,
)


TEST_ROOT = CANONICAL_CAMPAIGN_ROOT / "test/heldout"
TEST_MOTION_ROOT = TEST_ROOT / "input/motions"
TEACHER_CHECKPOINT_ROOT = Path("/workspace/project/roborender/checkpoints/teachers/robot_policy")
ACTOR_ENTRYPOINT = ROOT / "scripts/round_dagger/actor_training_entrypoint.py"
TRAINER = ROOT / "GR00T-VisualSim2Real/gr00t/rl/train_agent_trl.py"
NATIVE_CONFIG = (
    ROOT
    / "GR00T-VisualSim2Real/gr00t/rl/config/exp/loco_manip/"
    "wsdpt_round_actor_native_3view.yaml"
)
EPISODES_PER_GPU = 64


def _runtime(root: Path) -> tuple[Path, dict[str, str]]:
    root = safe_directory(root, "qualified Isaac runtime")
    isaac = safe_directory(root / "runtime/isaacsim-5.1", "Isaac runtime", root=root)
    python = safe_python_launcher(
        isaac / "bin/python", "Isaac Python", target_root=root
    )
    isaaclab = safe_directory(root / "runtime/IsaacLab/source", "IsaacLab source", root=root)
    dependencies = safe_directory(
        root / "runtime/actor-python-site-packages", "actor dependencies", root=root
    )
    system = safe_directory(
        root / "packages/root/usr/lib/x86_64-linux-gnu", "system libraries", root=root
    )
    cudnn = safe_directory(
        isaac / "lib/python3.11/site-packages/nvidia/cudnn/lib",
        "cuDNN libraries",
        root=root,
    )
    experience = safe_regular(
        root / "runtime/IsaacLab/apps/phc.isaaclab.python.headless.rendering.kit",
        "headless Isaac experience",
        root=root,
        read_only=False,
    )
    python_paths = [
        isaaclab / name
        for name in (
            "isaaclab", "isaaclab_assets", "isaaclab_contrib", "isaaclab_mimic",
            "isaaclab_rl", "isaaclab_tasks",
        )
    ] + [
        dependencies,
        dependencies / "cmeel.prefix/lib/python3.11/site-packages",
        ROOT,
        ROOT / "GR00T-VisualSim2Real",
    ]
    for path in python_paths:
        safe_directory(path, "qualified Python import root")
    environment = {
        "PYTHONPATH": ":".join(str(path) for path in python_paths),
        "LD_LIBRARY_PATH": f"{cudnn}:{system}:/usr/lib/x86_64-linux-gnu",
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "OMNI_KIT_ALLOW_ROOT": "1",
        "HYDRA_FULL_ERROR": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONUNBUFFERED": "1",
        "ROBORENDER_HEADLESS_EXPERIENCE": str(experience),
    }
    return python, environment


def _teacher() -> tuple[Path, str]:
    root = TEACHER_CHECKPOINT_ROOT
    candidates = sorted(root.glob("*.pt")) if root.is_dir() and not root.is_symlink() else []
    if len(candidates) != 1:
        raise NativeH200FleetError("final test requires one canonical teacher checkpoint")
    path = safe_regular(candidates[0], "teacher checkpoint", root=TEACHER_CHECKPOINT_ROOT)
    return path, sha256_file(path)


def _joint_stratum(strata: dict[str, str]) -> str:
    aliases = {
        "table_height": "h",
        "initial_planar_distance_bin": "d",
        "initial_world_bearing_bin": "b",
        "trajectory_length_bin": "l",
    }
    return ".".join(f"{alias}_{strata[field]}" for field, alias in aliases.items())


def _test_assignments(cases: dict, partition: dict) -> list[dict]:
    raw_cases = cases.get("cases")
    seeds = cases.get("evaluation_seeds")
    if (
        cases.get("schema") != "fast_roborender.round_dagger.sealed_test_cases.v1"
        or cases.get("case_count") != 512
        or not isinstance(raw_cases, list)
        or not isinstance(seeds, list)
        or len(seeds) != 8
        or len(set(seeds)) != 8
    ):
        raise NativeH200FleetError("private test case registry is malformed")
    groups = {row["group_id"]: row for row in partition.get("groups", [])}
    if len(groups) != 5 or partition.get("purpose") != "test":
        raise NativeH200FleetError("test motion partition is malformed")
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in raw_cases:
        if not isinstance(row, dict) or set(row) != {
            "case_id", "evaluation_seed", "group_id", "reset_index"
        }:
            raise NativeH200FleetError("private test case row is malformed")
        if row["group_id"] not in groups:
            raise NativeH200FleetError("private test case escaped the test split")
        by_seed[row["evaluation_seed"]].append(row)
    assignments: list[dict] = []
    for physical_gpu, seed in enumerate(seeds):
        seed_cases = by_seed.get(seed, [])
        counts = Counter(row["group_id"] for row in seed_cases)
        if len(seed_cases) != EPISODES_PER_GPU or sorted(counts.values()) != [12, 13, 13, 13, 13]:
            raise NativeH200FleetError("one test seed is not a balanced 64-case panel")
        ordered_groups = sorted(counts, key=lambda group: (-counts[group], group))
        lookup = {(row["group_id"], row["reset_index"]): row for row in seed_cases}
        seen = Counter()
        ordered_cases = []
        for env_index in range(EPISODES_PER_GPU):
            group = ordered_groups[env_index % len(ordered_groups)]
            key = (group, seen[group])
            try:
                row = lookup[key]
            except KeyError as exc:
                raise NativeH200FleetError("test reset assignment is not contiguous") from exc
            seen[group] += 1
            ordered_cases.append({**row, "env_index": env_index})
        derived_partition = {
            **partition,
            "groups": [groups[group] for group in ordered_groups],
        }
        assignments.append(
            {
                "physical_gpu": physical_gpu,
                "seed": seed,
                "group_ids": ordered_groups,
                "cases": ordered_cases,
                "partition": derived_partition,
            }
        )
    if {row["case_id"] for item in assignments for row in item["cases"]} != {
        row["case_id"] for row in raw_cases
    }:
        raise NativeH200FleetError("test assignments do not cover every sealed case once")
    return assignments


def _command(
    *,
    python: Path,
    checkpoint: Path,
    teacher: Path,
    partition: Path,
    partition_sha256: str,
    split_sha256: str,
    registry_sha256: str,
    group_ids: list[str],
    seed: int,
    run_dir: Path,
) -> list[str]:
    groups_json = json.dumps(group_ids, separators=(",", ":"))
    return [
        str(python), str(ACTOR_ENTRYPOINT), str(TRAINER),
        "+exp=loco_manip/wsdpt_round_actor_native_3view",
        f"seed={seed}",
        f"num_envs={EPISODES_PER_GPU}",
        "algo.config.num_steps_per_env=512",
        "algo.config.num_learning_epochs=0",
        "algo.config.rollout_with_teacher_num_steps=0",
        "algo.config.enforce_teacher_rollout=false",
        "algo.config.native_dagger.role=test",
        "algo.config.native_dagger.purpose=test",
        "algo.config.native_dagger.observation_source=native",
        "algo.config.native_dagger.rollout_action_source=student",
        "algo.config.native_dagger.beta_schedule=constant",
        "algo.config.native_dagger.beta_start=0.0",
        "algo.config.native_dagger.beta_end=0.0",
        "algo.config.native_dagger.beta_decay_iters=1",
        f"algo.config.native_dagger.split_manifest_path={partition}",
        f"algo.config.native_dagger.split_manifest_sha256={partition_sha256}",
        f"algo.config.native_dagger.split_sha256={split_sha256}",
        f"algo.config.native_dagger.group_ids={groups_json}",
        f"algo.config.native_dagger.panel_sha256={registry_sha256}",
        "algo.config.eval.eval_num_envs_episodes=true",
        "algo.config.eval.save_videos=false",
        "algo.config.actor.backbone.vision_module.module_config_dict.layer_config.pretrained=false",
        "env.config.reset_from_dataset.use_motion_file_dir=true",
        f"env.config.reset_from_dataset.motion_file_dir={TEST_MOTION_ROOT}",
        f"env.config.reset_from_dataset.motion_split_manifest={partition}",
        f"env.config.reset_from_dataset.motion_split_manifest_sha256={partition_sha256}",
        "env.config.reset_from_dataset.motion_split_role=test",
        f"env.config.reset_from_dataset.motion_split_group_ids={groups_json}",
        "env.config.reset_from_dataset.motion_split_verify_content=true",
        "env.config.reset_from_dataset.motion_split_offset=0",
        "env.config.reset_from_dataset.motion_split_limit=null",
        "env.config.reset_from_dataset.motion_assignment_offset=0",
        f"checkpoint={checkpoint}",
        "resume=false",
        f"teacher_actor_path={teacher}",
        "simulator.config.cameras.policy_views=[head_color,hand_left,hand_right]",
        "policy_view_count=3",
        "env.config.native_policy_view_count=3",
        "simulator.config.render_results=false",
        "headless=true",
        "use_wandb=false",
        f"experiment_name=final-test-gpu-{run_dir.name}",
        f"experiment_dir={run_dir}",
        f"hydra.run.dir={run_dir / 'hydra'}",
        f"output_dir={run_dir / 'output'}",
    ]


def _aggregate(
    *,
    assignments: list[dict],
    run_root: Path,
    registry: dict,
    cases_sha256: str,
    checkpoint_sha256: str,
    generation_id: str,
    protocol_id: str,
    split_sha256: str,
    group_strata: dict[str, str],
) -> dict:
    outcomes = []
    for assignment in assignments:
        path = run_root / f"gpu-{assignment['physical_gpu']:02d}/output/metrics_eval.json"
        metrics = strict_document(path, "test rank metrics", root=run_root)
        summary = metrics.get("_native_closed_loop")
        records = metrics.get("_roborender_fixed_seed_episodes")
        if (
            not isinstance(summary, dict)
            or summary.get("schema") != "roborender.native_closed_loop.v2"
            or summary.get("role") != "test"
            or summary.get("purpose") != "test"
            or summary.get("split_sha256") != split_sha256
            or summary.get("panel_sha256") != registry["test_registry_sha256"]
            or summary.get("episode_count") != EPISODES_PER_GPU
            or summary.get("safety_observed") is not True
            or not isinstance(records, list)
            or len(records) != EPISODES_PER_GPU
        ):
            raise NativeH200FleetError("test rank metrics contract differs")
        by_env = {row.get("env_index"): row for row in records if isinstance(row, dict)}
        if set(by_env) != set(range(EPISODES_PER_GPU)):
            raise NativeH200FleetError("test rank does not contain one outcome per environment")
        for case in assignment["cases"]:
            record = by_env[case["env_index"]]
            if record.get("seed") != assignment["seed"] or type(record.get("success")) is not bool:
                raise NativeH200FleetError("test episode seed/success outcome differs")
            safety = record.get("safety_violation")
            if type(safety) is not bool:
                raise NativeH200FleetError("test episode lacks explicit safety evidence")
            outcomes.append(
                {
                    "case_id": case["case_id"],
                    "group_id": case["group_id"],
                    "success": record["success"],
                    "safety_violation": safety,
                }
            )
    if len(outcomes) != 512 or len({row["case_id"] for row in outcomes}) != 512:
        raise NativeH200FleetError("final test outcomes are incomplete or duplicated")
    per_stratum: dict[str, dict[str, int]] = {}
    for name in sorted(registry["stratum_case_counts"]):
        rows = [row for row in outcomes if group_strata[row["group_id"]] == name]
        per_stratum[name] = {
            "episode_count": len(rows),
            "success_count": sum(row["success"] for row in rows),
            "safety_violations": sum(row["safety_violation"] for row in rows),
        }
    if {name: row["episode_count"] for name, row in per_stratum.items()} != registry[
        "stratum_case_counts"
    ]:
        raise NativeH200FleetError("test outcome strata differ from the sealed registry")
    return {
        "schema": FINAL_TEST_RESULT_SCHEMA,
        "campaign_id": "roborender-h200-v001",
        "protocol_id": protocol_id,
        "generation_id": generation_id,
        "checkpoint_sha256": checkpoint_sha256,
        "test_registry_sha256": registry["test_registry_sha256"],
        "test_cases_sha256": cases_sha256,
        "split_sha256": split_sha256,
        "episode_count": len(outcomes),
        "success_count": sum(row["success"] for row in outcomes),
        "safety_violations": sum(row["safety_violation"] for row in outcomes),
        "per_stratum": per_stratum,
    }


def main() -> int:
    values = require_environment("test")
    if (
        values.get("ROUND_DAGGER_OBSERVATION_SOURCE") != "native"
        or values.get("ROUND_DAGGER_EVALUATION_PURPOSE") != "test_success"
        or values.get("ROUND_DAGGER_EVALUATOR_WORLD_SIZE") != "8"
    ):
        raise NativeH200FleetError("final test must be native success-only on eight H200s")
    required = (
        "ROUND_DAGGER_POLICY_CHECKPOINT", "ROUND_DAGGER_POLICY_SHA256",
        "ROUND_DAGGER_TEST_REGISTRY", "ROUND_DAGGER_TEST_REGISTRY_SHA256",
        "ROUND_DAGGER_TEST_CASES", "ROUND_DAGGER_TEST_CASES_SHA256",
        "ROUND_DAGGER_TEST_PARTITION", "ROUND_DAGGER_TEST_PARTITION_SHA256",
        "ROUND_DAGGER_TEST_RESULT_DIR", "ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT",
    )
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise NativeH200FleetError("test binding is incomplete: " + ", ".join(missing))
    state = Controller(CANONICAL_CAMPAIGN_ROOT).status()
    if state["phase"] != "final_selected":
        raise NativeH200FleetError("final checkpoint must be frozen before test access")
    checkpoint = safe_regular(
        Path(values["ROUND_DAGGER_POLICY_CHECKPOINT"]),
        "final checkpoint",
        root=CANONICAL_CAMPAIGN_ROOT,
        expected_sha256=values["ROUND_DAGGER_POLICY_SHA256"],
    )
    spec_path = safe_regular(
        Path(f"{checkpoint}.policy_spec.json"),
        "final policy specification",
        root=CANONICAL_CAMPAIGN_ROOT,
    )
    spec = load_policy_spec(spec_path)
    if spec.get("checkpoint_sha256") != values["ROUND_DAGGER_POLICY_SHA256"]:
        raise NativeH200FleetError("final policy specification names another checkpoint")
    registry = strict_document(
        Path(values["ROUND_DAGGER_TEST_REGISTRY"]), "test registry",
        root=TEST_ROOT, expected_sha256=values["ROUND_DAGGER_TEST_REGISTRY_SHA256"],
    )
    cases = strict_document(
        Path(values["ROUND_DAGGER_TEST_CASES"]), "private test cases",
        root=TEST_ROOT, expected_sha256=values["ROUND_DAGGER_TEST_CASES_SHA256"],
    )
    partition = strict_document(
        Path(values["ROUND_DAGGER_TEST_PARTITION"]), "test partition",
        root=TEST_ROOT, expected_sha256=values["ROUND_DAGGER_TEST_PARTITION_SHA256"],
    )
    if (
        registry.get("schema") != "fast_roborender.round_dagger.sealed_test_registry.v1"
        or registry.get("cases_sha256") != values["ROUND_DAGGER_TEST_CASES_SHA256"]
        or partition.get("split_sha256") != values["ROUND_DAGGER_SPLIT_SHA256"]
    ):
        raise NativeH200FleetError("sealed test contracts disagree")
    registry = {**registry, "test_registry_sha256": values["ROUND_DAGGER_TEST_REGISTRY_SHA256"]}
    assignments = _test_assignments(cases, partition)
    result_dir = safe_directory(
        Path(values["ROUND_DAGGER_TEST_RESULT_DIR"]),
        "final test result directory", root=CANONICAL_CAMPAIGN_ROOT,
    )
    if result_dir != CANONICAL_CAMPAIGN_ROOT / "test/heldout/results":
        raise NativeH200FleetError("final test result escaped its one-time namespace")
    result_path = result_dir / "result.json"
    if result_path.exists():
        existing = strict_document(result_path, "final test result", root=result_dir)
        if (
            existing.get("schema") != FINAL_TEST_RESULT_SCHEMA
            or existing.get("checkpoint_sha256") != values["ROUND_DAGGER_POLICY_SHA256"]
        ):
            raise NativeH200FleetError("existing final test result differs")
        print(f"NATIVE_H200_TEST_RESULT_READY path={result_path} sha256={sha256_file(result_path)}")
        return 0

    inventory = query_h200_inventory()
    python, runtime_environment = _runtime(Path(values["ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT"]))
    teacher, teacher_sha = _teacher()
    safe_regular(ACTOR_ENTRYPOINT, "actor entrypoint", root=ROOT, read_only=False)
    safe_regular(TRAINER, "GR00T trainer", root=ROOT, read_only=False)
    safe_regular(NATIVE_CONFIG, "native actor config", root=ROOT, read_only=False)
    safe_directory(TEST_MOTION_ROOT, "sealed test motion root", root=TEST_ROOT)
    run_root = result_dir / "runtime"
    run_root.mkdir(mode=0o700)
    split = strict_document(
        CANONICAL_CAMPAIGN_ROOT / "BC/contracts/group_split.v1.json",
        "campaign group split", root=CANONICAL_CAMPAIGN_ROOT,
    )
    group_strata = {
        row["group_id"]: _joint_stratum(row["strata"])
        for row in split.get("groups", [])
        if row.get("group_id") in set(partition["groups"][index]["group_id"] for index in range(5))
    }
    if set(group_strata) != set(registry["groups"]):
        raise NativeH200FleetError("test strata do not match the sealed group split")

    processes: list[tuple[subprocess.Popen, object, Path]] = []
    for assignment in assignments:
        gpu = assignment["physical_gpu"]
        run_dir = run_root / f"gpu-{gpu:02d}"
        run_dir.mkdir(mode=0o700)
        (run_dir / "hydra").mkdir(mode=0o700)
        (run_dir / "output").mkdir(mode=0o700)
        process_state = run_dir / "process-state"
        process_state.mkdir(mode=0o700)
        process_directories = {
            name: process_state / name
            for name in ("cache", "config", "data", "runtime", "tmp")
        }
        for path in process_directories.values():
            path.mkdir(mode=0o700)
        partition_path = run_dir / "motion_partition.v1.json"
        partition_sha = publish_immutable_json(partition_path, assignment["partition"])
        log_path = run_dir / "test.log"
        log_handle = log_path.open("xb")
        command = _command(
            python=python,
            checkpoint=checkpoint,
            teacher=teacher,
            partition=partition_path,
            partition_sha256=partition_sha,
            split_sha256=values["ROUND_DAGGER_SPLIT_SHA256"],
            registry_sha256=values["ROUND_DAGGER_TEST_REGISTRY_SHA256"],
            group_ids=assignment["group_ids"],
            seed=assignment["seed"],
            run_dir=run_dir,
        )
        environment = sanitized_environment(
            {
                **runtime_environment,
                "CUDA_VISIBLE_DEVICES": inventory[gpu]["uuid"],
                "ROUND_DAGGER_POLICY_CHECKPOINT": str(checkpoint),
                "ROUND_DAGGER_POLICY_SHA256": values["ROUND_DAGGER_POLICY_SHA256"],
                "ROUND_DAGGER_RUNTIME_POLICY_SHA256": spec["runtime_policy_sha256"],
                "ROUND_DAGGER_TEACHER_CHECKPOINT": str(teacher),
                "ROUND_DAGGER_TEACHER_SHA256": teacher_sha,
                "ROUND_DAGGER_OBSERVATION_SOURCE": "native",
                "ROUND_DAGGER_ACTOR_ROLE": "test",
                "ROUND_DAGGER_PURPOSE": "test",
                "ROUND_DAGGER_EXPECT_POLICY_VIEW_COUNT": "3",
                "EXPECTED_TRAIN_CHECKPOINT_SHA256": values["ROUND_DAGGER_POLICY_SHA256"],
                "TRAIN_RUN_DIR": str(run_dir),
                "XDG_CACHE_HOME": str(process_directories["cache"]),
                "XDG_CONFIG_HOME": str(process_directories["config"]),
                "XDG_DATA_HOME": str(process_directories["data"]),
                "XDG_RUNTIME_DIR": str(process_directories["runtime"]),
                "TMPDIR": str(process_directories["tmp"]),
            },
            cuda_devices=inventory[gpu]["uuid"],
        )
        process = subprocess.Popen(
            command,
            cwd=ROOT / "GR00T-VisualSim2Real",
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log_handle, log_path))
    failures = []
    for process, log_handle, log_path in processes:
        returncode = process.wait()
        log_handle.close()
        if returncode:
            failures.append(f"{log_path.name}:{returncode}")
    if failures:
        raise NativeH200FleetError("final test rank failure: " + ", ".join(failures))

    document = _aggregate(
        assignments=assignments,
        run_root=run_root,
        registry=registry,
        cases_sha256=values["ROUND_DAGGER_TEST_CASES_SHA256"],
        checkpoint_sha256=values["ROUND_DAGGER_POLICY_SHA256"],
        generation_id=values["ROUND_DAGGER_GENERATION_ID"],
        protocol_id=values["ROUND_DAGGER_PROTOCOL_ID"],
        split_sha256=values["ROUND_DAGGER_SPLIT_SHA256"],
        group_strata=group_strata,
    )
    digest = publish_immutable_json(result_path, document)
    print(f"NATIVE_H200_TEST_FLEET_OK path={result_path} sha256={digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError, NativeH200FleetError) as error:
        print(f"native H200 final test fleet failed: {error}", file=sys.stderr)
        raise SystemExit(2)
