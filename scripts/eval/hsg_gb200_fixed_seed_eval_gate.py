#!/usr/bin/env python3
"""Content-addressed release gates for HSG GB200 fixed-seed policy evaluation.

The scientific scoring implementation lives in :mod:`fixed_seed_gate`.  This
module adds the HSG deployment boundary around it: immutable source revisions,
source-tree digests, the sealed SM100 trainer image, exact checkpoint/config
bytes, and a successful upload/readback marker for the authorized W&B team.

The evaluator intentionally uses synchronized *simulator RGB* observations.
It does not set ``IMAGE_DAGGER_EVAL_RELABEL=1`` and therefore must not be cited
as an Image3F-in-the-loop evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fixed_seed_gate  # noqa: E402
import wandb_team_contract as team_contract  # noqa: E402


SCHEMA = "fast_roborender.hsg_fixed_seed_preflight.v2"
ACCEPTANCE_SCHEMA = "fast_roborender.hsg_fixed_seed_acceptance.v2"
WANDB_SCHEMA = "fast_roborender.hsg_fixed_seed_wandb_upload.v2"
ROLE_ORDER = ("baseline", "early", "mid", "final")
EXPECTED_PROTOCOL = "one_episode_per_environment_fixed_seed_blocks"
EXPECTED_OBSERVATION_MODE = "raytraced"
REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ENTITY_RE = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class GateError(RuntimeError):
    """An HSG evaluation artifact violated the release contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except GateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GateError(f"{path} must contain exactly one JSON object")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GateError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise GateError(f"{label} must be a lowercase 64-hex SHA-256")
    return value


def _require_revision(value: Any, label: str) -> str:
    if not isinstance(value, str) or REVISION_RE.fullmatch(value) is None:
        raise GateError(f"{label} must be a lowercase 40-hex Git revision")
    return value


def _require_file(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"cannot resolve {label} {path}: {exc}") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise GateError(f"{label} is not a non-empty regular file: {resolved}")
    return resolved


def _require_dir(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"cannot resolve {label} {path}: {exc}") from exc
    if not resolved.is_dir():
        raise GateError(f"{label} is not a directory: {resolved}")
    return resolved


def _require_hash(path: Path, expected: str, label: str) -> str:
    expected = _require_sha256(expected, f"expected {label}")
    actual = sha256_file(_require_file(path, label))
    if actual != expected:
        raise GateError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def source_tree_sha256(root: Path, *, excluded_top_level: Sequence[str] = ()) -> str:
    """Hash one Git-less staged tree, excluding volatile bytecode and metadata."""

    root = _require_dir(root, "source tree")
    excluded = set(excluded_top_level)
    candidates: list[tuple[Path, Path]] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in excluded:
            continue
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.is_dir() or path.suffix in {".pyc", ".pyo"}:
            continue
        candidates.append((relative, path))
    digest = hashlib.sha256()
    for relative, path in sorted(candidates, key=lambda item: item[0].as_posix()):
        name = relative.as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update((path.lstat().st_mode & 0o777).to_bytes(2, "big"))
        if path.is_symlink():
            payload = os.readlink(path).encode("utf-8")
            digest.update(b"L" + len(payload).to_bytes(8, "big") + payload)
        elif path.is_file():
            digest.update(b"F")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
        else:
            raise GateError(f"unsupported source-tree entry: {path}")
    return digest.hexdigest()


def _revision_marker(repo: Path, expected: str, label: str) -> str:
    expected = _require_revision(expected, f"expected {label} revision")
    marker = _require_file(repo / ".source_revision", f"{label} revision marker")
    actual = marker.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise GateError(f"{label} revision mismatch: {actual!r} != {expected!r}")
    return actual


def _parse_expected_seeds(raw: str) -> list[int]:
    try:
        seeds = [int(piece) for piece in raw.split(",") if piece]
    except ValueError as exc:
        raise GateError("expected seeds must be comma-separated integers") from exc
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise GateError("expected seeds must contain at least two unique nonnegative integers")
    return seeds


def validate_fixed_seed_manifest(
    manifest_path: Path,
    *,
    train_run_dir: Path,
    expected_seeds: Sequence[int],
    expected_num_envs: int,
) -> dict[str, Any]:
    manifest_path = _require_file(manifest_path, "fixed-seed manifest")
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != fixed_seed_gate.SCHEMA_VERSION:
        raise GateError("fixed-seed manifest schema mismatch")
    if manifest.get("protocol") != EXPECTED_PROTOCOL:
        raise GateError("fixed-seed protocol mismatch")
    if manifest.get("observation_mode") != EXPECTED_OBSERVATION_MODE:
        raise GateError("HSG fixed-seed gate requires simulator-RGB/raytraced evaluation")
    if manifest.get("seeds") != list(expected_seeds):
        raise GateError("fixed-seed list changed")
    if manifest.get("num_envs_per_seed") != expected_num_envs:
        raise GateError("fixed-seed environment count changed")
    if manifest.get("expected_episodes_per_seed") != expected_num_envs:
        raise GateError("one-episode-per-environment contract changed")

    run_dir = _require_dir(train_run_dir, "training run")
    declared_run = _require_dir(Path(str(manifest.get("train_run_dir", ""))), "declared training run")
    if declared_run != run_dir:
        raise GateError("fixed-seed manifest points at a different training run")
    reference = _require_file(Path(str(manifest.get("reference_config", ""))), "reference config")
    if reference != (run_dir / "config.yaml").resolve():
        raise GateError("reference config must be the training run's exact config.yaml")
    reference_hash = _require_sha256(
        manifest.get("reference_config_sha256"), "reference_config_sha256"
    )
    if sha256_file(reference) != reference_hash:
        raise GateError("reference config changed after fixed-seed manifest creation")
    if manifest.get("launch_time_reference_config_sha256") != reference_hash:
        raise GateError("reference config lacks a matching launch-time hash")

    roles = manifest.get("checkpoint_roles")
    if not isinstance(roles, list) or [role.get("label") for role in roles if isinstance(role, dict)] != list(ROLE_ORDER):
        raise GateError(f"checkpoint roles must be {ROLE_ORDER}")
    normalized_roles: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    steps: list[int] = []
    for index, role in enumerate(roles):
        if not isinstance(role, dict):
            raise GateError(f"checkpoint role {index} is malformed")
        path = _require_file(Path(str(role.get("checkpoint", ""))), f"{role['label']} checkpoint")
        if path in seen_paths:
            raise GateError("checkpoint roles must use four distinct files")
        seen_paths.add(path)
        expected_hash = _require_sha256(role.get("checkpoint_sha256"), f"{role['label']} checkpoint hash")
        if sha256_file(path) != expected_hash:
            raise GateError(f"{role['label']} checkpoint changed after manifest creation")
        if index == 0:
            if path != (run_dir / "initial_policy.pt").resolve():
                raise GateError("baseline must be the run's iteration-zero initial_policy.pt")
            if role.get("checkpoint_step") is not None:
                raise GateError("baseline checkpoint_step must be null")
            if manifest.get("launch_time_baseline_sha256") != expected_hash:
                raise GateError("baseline lacks a matching launch-time hash")
        else:
            if path.parent != run_dir:
                raise GateError(f"{role['label']} checkpoint must be inside the training run")
            step = role.get("checkpoint_step")
            if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
                raise GateError(f"{role['label']} checkpoint_step must be positive")
            if path.name != f"model_step_{step:06d}.pt":
                raise GateError(f"{role['label']} checkpoint filename/step mismatch")
            steps.append(step)
        normalized_roles.append(
            {"label": role["label"], "path": str(path), "sha256": expected_hash, "step": role.get("checkpoint_step")}
        )
    if steps != sorted(steps) or len(set(steps)) != 3:
        raise GateError("early/mid/final checkpoint steps must be strictly increasing")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(ROLE_ORDER):
        raise GateError("fixed-seed manifest must contain exactly four tasks")
    for index, (task, role) in enumerate(zip(tasks, roles, strict=True)):
        if not isinstance(task, dict) or task.get("task_index") != index:
            raise GateError(f"fixed-seed task {index} identity mismatch")
        for key in ("label", "checkpoint", "checkpoint_step", "checkpoint_sha256"):
            if task.get(key) != role.get(key):
                raise GateError(f"fixed-seed task {index} {key} mismatch")
        if task.get("seeds") != list(expected_seeds):
            raise GateError(f"fixed-seed task {index} seed mismatch")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "train_run_dir": str(run_dir),
        "reference_config": str(reference),
        "reference_config_sha256": reference_hash,
        "seeds": list(expected_seeds),
        "num_envs_per_seed": expected_num_envs,
        "roles": normalized_roles,
    }


def validate_wandb_acceptance(path: Path, expected_sha256: str) -> dict[str, Any]:
    acceptance_path = _require_file(path, "authorized-team W&B acceptance")
    acceptance_hash = _require_hash(
        acceptance_path,
        expected_sha256,
        "authorized-team W&B acceptance",
    )
    try:
        normalized = team_contract.validate_online_acceptance(
            team_contract.load_json(acceptance_path)
        )
    except team_contract.TeamContractError as exc:
        raise GateError(str(exc)) from None
    return {
        "path": str(acceptance_path),
        "sha256": acceptance_hash,
        **normalized,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = _require_dir(args.root, "fast_roborender source")
    gr00t = _require_dir(args.gr00t_root, "GR00T source")
    fast_revision = _revision_marker(root, args.fast_revision, "fast_roborender")
    gr00t_revision = _revision_marker(gr00t, args.gr00t_revision, "GR00T")
    container_hash = _require_hash(
        args.trainer_container, args.trainer_container_sha256, "SM100 trainer container"
    )
    expected_seeds = _parse_expected_seeds(args.expected_seeds)
    fixed = validate_fixed_seed_manifest(
        args.fixed_seed_manifest,
        train_run_dir=args.train_run_dir,
        expected_seeds=expected_seeds,
        expected_num_envs=args.expected_num_envs,
    )
    expected_manifest_hash = _require_sha256(
        args.fixed_seed_manifest_sha256, "expected fixed-seed manifest SHA-256"
    )
    if fixed["manifest_sha256"] != expected_manifest_hash:
        raise GateError("fixed-seed manifest hash changed before HSG preflight")
    wandb_acceptance = validate_wandb_acceptance(
        args.wandb_acceptance,
        args.wandb_acceptance_sha256,
    )
    return {
        "schema": SCHEMA,
        "status": "validated",
        "observation_mode": "simulator_rgb",
        "image3f_in_the_loop": False,
        "production_observation_distribution_validated": False,
        "fast_roborender_revision": fast_revision,
        "gr00t_revision": gr00t_revision,
        "fast_roborender_source_tree_sha256": source_tree_sha256(
            root, excluded_top_level=("GR00T-VisualSim2Real",)
        ),
        "gr00t_source_tree_sha256": source_tree_sha256(gr00t),
        "trainer_container": str(_require_file(args.trainer_container, "SM100 trainer container")),
        "trainer_container_sha256": container_hash,
        "fixed_seed": fixed,
        "wandb_online_acceptance": wandb_acceptance,
    }


def _without_created_at(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "created_at"}


def postrun(args: argparse.Namespace) -> dict[str, Any]:
    preflight_path = _require_file(args.preflight, "HSG fixed-seed preflight")
    if sha256_file(preflight_path) != _require_sha256(args.preflight_sha256, "preflight SHA-256"):
        raise GateError("HSG fixed-seed preflight changed")
    preflight_payload = load_json(preflight_path)
    if preflight_payload.get("schema") != SCHEMA or preflight_payload.get("status") != "validated":
        raise GateError("HSG fixed-seed preflight is not validated")
    if preflight_payload.get("image3f_in_the_loop") is not False:
        raise GateError("HSG fixed-seed observation-mode attestation changed")
    if preflight_payload.get("production_observation_distribution_validated") is not False:
        raise GateError("simulator-RGB gate cannot attest the production observation distribution")
    preflight_wandb = preflight_payload.get("wandb_online_acceptance")
    if not isinstance(preflight_wandb, dict):
        raise GateError("HSG preflight lacks its authorized-team W&B binding")
    try:
        team_contract.validate_online_acceptance(preflight_wandb)
        team_contract.validate_authorized_request(
            expected_email=args.expected_wandb_email,
            entity=args.expected_wandb_entity,
            authorization_scope=team_contract.AUTHORIZED_SCOPE,
        )
    except team_contract.TeamContractError as exc:
        raise GateError(str(exc)) from None
    preflight_wandb_hash = _require_sha256(
        preflight_wandb.get("sha256"),
        "preflight W&B online acceptance SHA-256",
    )

    manifest_path = _require_file(args.fixed_seed_manifest, "fixed-seed manifest")
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != _require_sha256(args.fixed_seed_manifest_sha256, "manifest SHA-256"):
        raise GateError("fixed-seed manifest changed before postrun")
    preflight_fixed = preflight_payload.get("fixed_seed")
    if not isinstance(preflight_fixed, dict):
        raise GateError("HSG fixed-seed preflight lacks its fixed-seed artifact binding")
    if preflight_fixed.get("manifest_sha256") != manifest_hash:
        raise GateError("postrun manifest differs from the manifest bound by HSG preflight")
    manifest = load_json(manifest_path)
    summary_path = _require_file(args.summary, "fixed-seed summary")
    summary = load_json(summary_path)
    if summary.get("gate", {}).get("status") != "PASS":
        raise GateError("fixed-seed policy-improvement gate did not pass")
    recomputed = fixed_seed_gate.summarize_results(
        manifest=manifest,
        results_dir=args.results_dir,
        min_success_delta=args.min_success_delta,
    )
    if _without_created_at(summary) != _without_created_at(recomputed):
        raise GateError("fixed-seed summary does not match recomputed results")

    wandb_path = _require_file(args.wandb_upload, "W&B upload marker")
    wandb = load_json(wandb_path)
    expected_wandb = {
        "schema": WANDB_SCHEMA,
        "status": "PASS",
        **team_contract.team_binding(),
        "manifest_sha256": manifest_hash,
        "summary_sha256": sha256_file(summary_path),
        "wandb_online_acceptance_sha256": preflight_wandb_hash,
        "wandb_online_acceptance_observed_keys": sorted(
            team_contract.REQUIRED_OBSERVED_KEYS
        ),
    }
    for key, expected in expected_wandb.items():
        if wandb.get(key) != expected:
            raise GateError(f"W&B upload marker {key} mismatch")
    if ENTITY_RE.fullmatch(str(wandb.get("entity", ""))) is None:
        raise GateError("W&B upload marker entity is malformed")
    if wandb.get("observed_checkpoint_ordinals") != [0, 1, 2, 3]:
        raise GateError("W&B upload marker lacks four-checkpoint history readback")
    if wandb.get("observed_history_keys") != list(
        (
            "eval/checkpoint_ordinal",
            "eval/checkpoint_label",
            "eval/goal_reached_rate",
            "eval/reward",
            "eval/episodes",
        )
    ):
        raise GateError("W&B upload marker lacks required checkpoint scalar keys")
    if wandb.get("observed_summary_keys") != list(
        (
            "policy_improvement_gate/status",
            "policy_improvement_gate/paired_success_delta",
            "policy_improvement_gate/paired_success_delta_ci95_low",
            "policy_improvement_gate/paired_success_delta_ci95_high",
        )
    ):
        raise GateError("W&B upload marker lacks required policy-gate summary keys")

    result_files = sorted(args.results_dir.glob("*/seed_*/result.json"))
    expected_count = len(manifest["checkpoint_roles"]) * len(manifest["seeds"])
    if len(result_files) != expected_count:
        raise GateError(f"expected {expected_count} fixed-seed result files, found {len(result_files)}")
    result_inventory = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in result_files
    ]
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "status": "PASS",
        "scientific_gate": "paired_final_minus_baseline_goal_reached_rate",
        "observation_mode": "simulator_rgb",
        "image3f_in_the_loop": False,
        "production_observation_distribution_validated": False,
        "readiness_scope": "secondary_policy_mechanics_gate",
        "preflight": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "fixed_seed_manifest": str(manifest_path),
        "fixed_seed_manifest_sha256": manifest_hash,
        "summary": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "wandb_upload": str(wandb_path.resolve()),
        "wandb_upload_sha256": sha256_file(wandb_path),
        "wandb_account_email": team_contract.AUTHORIZED_EMAIL,
        "wandb_entity": team_contract.AUTHORIZED_ENTITY,
        "wandb_authorization_scope": team_contract.AUTHORIZED_SCOPE,
        "wandb_authorization": team_contract.team_binding(),
        "wandb_online_acceptance_sha256": preflight_wandb_hash,
        "paired_success_delta": summary["gate"]["paired_success_delta"],
        "paired_success_delta_ci95_low": summary["gate"][
            "paired_success_delta_seed_block_bootstrap95_low"
        ],
        "results": result_inventory,
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pre = subparsers.add_parser("preflight")
    pre.add_argument("--root", type=Path, required=True)
    pre.add_argument("--gr00t-root", type=Path, required=True)
    pre.add_argument("--fast-revision", required=True)
    pre.add_argument("--gr00t-revision", required=True)
    pre.add_argument("--trainer-container", type=Path, required=True)
    pre.add_argument("--trainer-container-sha256", required=True)
    pre.add_argument("--fixed-seed-manifest", type=Path, required=True)
    pre.add_argument("--fixed-seed-manifest-sha256", required=True)
    pre.add_argument("--train-run-dir", type=Path, required=True)
    pre.add_argument("--expected-seeds", required=True)
    pre.add_argument("--expected-num-envs", type=int, required=True)
    pre.add_argument("--wandb-acceptance", type=Path, required=True)
    pre.add_argument("--wandb-acceptance-sha256", required=True)

    post = subparsers.add_parser("postrun")
    post.add_argument("--preflight", type=Path, required=True)
    post.add_argument("--preflight-sha256", required=True)
    post.add_argument("--fixed-seed-manifest", type=Path, required=True)
    post.add_argument("--fixed-seed-manifest-sha256", required=True)
    post.add_argument("--results-dir", type=Path, required=True)
    post.add_argument("--summary", type=Path, required=True)
    post.add_argument("--min-success-delta", type=float, required=True)
    post.add_argument("--wandb-upload", type=Path, required=True)
    post.add_argument("--expected-wandb-email", required=True)
    post.add_argument("--expected-wandb-entity", required=True)
    post.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            report = preflight(args)
        else:
            report = postrun(args)
            atomic_json(args.output, report)
    except (GateError, fixed_seed_gate.GateError, OSError, KeyError, TypeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
