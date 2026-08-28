#!/usr/bin/env python3
"""Upload verified student/expert rollout quality from sealed DAgger cohorts.

This observer never writes below the campaign root.  It deep-verifies each
new cohort with the campaign's sealed ``round_dagger.artifacts`` runtime,
derives action-agreement metrics from the first 11 learned controls, finishes
the W&B synchronization, and only then advances an external fsynced cursor.
Task success and safety are intentionally left to the paired closed-loop
evaluator; rollout shards do not contain those labels.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).absolute().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(REPOSITORY_ROOT))

from round_dagger.artifacts import (  # noqa: E402
    canonical_json_sha256,
    load_strict_json,
    sha256_file,
    verify_cohort,
)
from scripts.scaling.wandb_preflight import (  # noqa: E402
    WandbPreflightError,
    _load_wandb,
    query_identity,
)


STATE_SCHEMA = "fast_roborender.round_dagger.student_quality_wandb_state.v1"
RUN_SCHEMA = "fast_roborender.round_dagger.student_quality_wandb_run.v1"
LEARNED_ACTION_DIMENSIONS = 11
MACRO_RE = re.compile(r"macro_round_([0-9]{6})\Z")
COHORT_RE = re.compile(r"cohort_([0-9]{2})\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class QualityBridgeError(RuntimeError):
    """Campaign evidence or external observer state is invalid."""


def _absolute(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def _regular(path: Path, label: str) -> Path:
    path = _absolute(path)
    if path.is_symlink() or not path.is_file():
        raise QualityBridgeError(f"{label} is missing or unsafe")
    return path


def _campaign(campaign_root: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    root = _absolute(campaign_root)
    if root.is_symlink() or not root.is_dir():
        raise QualityBridgeError("campaign root is missing or unsafe")
    manifest = load_strict_json(_regular(root / "campaign.json", "campaign manifest"))
    if not isinstance(manifest, Mapping):
        raise QualityBridgeError("campaign manifest root is malformed")
    campaign_id = manifest.get("campaign_id")
    if not isinstance(campaign_id, str) or TOKEN_RE.fullmatch(campaign_id) is None:
        raise QualityBridgeError("campaign ID is malformed")
    return root, dict(manifest)


def _state_dir(campaign_root: Path, value: str | os.PathLike[str]) -> Path:
    path = _absolute(value)
    try:
        path.relative_to(campaign_root)
    except ValueError:
        pass
    else:
        raise QualityBridgeError("observer state must be outside the campaign")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise QualityBridgeError("observer state contains a symbolic link")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise QualityBridgeError("observer state directory is unsafe")
    return path


@contextlib.contextmanager
def _lock(state_dir: Path):
    descriptor = os.open(
        state_dir / "student_quality_wandb.lock",
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise QualityBridgeError("observer lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise QualityBridgeError("another quality observer owns this state") from None
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _initial_state(
    *, campaign_id: str, campaign_sha256: str, entity: str, project: str, run_id: str
) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "campaign_id": campaign_id,
        "campaign_sha256": campaign_sha256,
        "wandb_entity": entity,
        "wandb_project": project,
        "wandb_run_id": run_id,
        "uploaded_seal_sha256_by_event": {},
    }


def _load_state(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(expected)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise QualityBridgeError("observer cursor is unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise QualityBridgeError("observer cursor is unreadable") from None
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise QualityBridgeError("observer cursor schema differs")
    for key in set(expected) - {"uploaded_seal_sha256_by_event"}:
        if value[key] != expected[key]:
            raise QualityBridgeError(f"observer cursor {key} differs")
    uploaded = value["uploaded_seal_sha256_by_event"]
    if not isinstance(uploaded, Mapping):
        raise QualityBridgeError("observer uploaded cursor is malformed")
    clean: dict[str, str] = {}
    for event, digest in uploaded.items():
        if not isinstance(event, str) or not event.isdigit() or str(int(event)) != event:
            raise QualityBridgeError("observer event key is non-canonical")
        if not isinstance(digest, str) or SHA_RE.fullmatch(digest) is None:
            raise QualityBridgeError("observer seal digest is malformed")
        clean[event] = digest
    return {**dict(value), "uploaded_seal_sha256_by_event": clean}


def _cohort_seals(root: Path, campaign_id: str) -> list[tuple[int, int, Path]]:
    actors = root / "actors" / campaign_id
    if not actors.exists():
        return []
    if actors.is_symlink() or not actors.is_dir():
        raise QualityBridgeError("campaign actor directory is unsafe")
    result: list[tuple[int, int, Path]] = []
    for macro in actors.iterdir():
        match = MACRO_RE.fullmatch(macro.name)
        if match is None:
            continue
        if macro.is_symlink() or not macro.is_dir():
            raise QualityBridgeError("macro-round actor directory is unsafe")
        for cohort in macro.iterdir():
            cohort_match = COHORT_RE.fullmatch(cohort.name)
            if cohort_match is None:
                continue
            seal = cohort / "seal.json"
            if seal.exists():
                result.append((int(match.group(1)), int(cohort_match.group(1)), _regular(seal, "cohort seal")))
    return sorted(result)


def _cosine(student: np.ndarray, expert: np.ndarray) -> np.ndarray:
    dot = np.sum(student * expert, axis=-1, dtype=np.float64)
    student_norm = np.linalg.norm(student, axis=-1)
    expert_norm = np.linalg.norm(expert, axis=-1)
    denominator = student_norm * expert_norm
    result = np.zeros_like(dot, dtype=np.float64)
    nonzero = denominator > 0.0
    np.divide(dot, denominator, out=result, where=nonzero)
    result[(student_norm == 0.0) & (expert_norm == 0.0)] = 1.0
    return np.clip(result, -1.0, 1.0)


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values.astype(np.float64, copy=False), probability, method="linear"))


def summarize_cohort(cohort_dir: Path, *, macro_round: int, cohort_index: int) -> dict[str, Any]:
    """Deep-verify and summarize one immutable train cohort."""

    seal = verify_cohort(cohort_dir)
    if seal.get("purpose") != "train":
        raise QualityBridgeError("quality observer accepts only train cohorts")
    if seal.get("macro_round") != macro_round or seal.get("cohort_index") != cohort_index:
        raise QualityBridgeError("cohort directory identity differs from its seal")
    action_contract = seal.get("action_contract")
    action_shape = (
        action_contract.get("shape")
        if isinstance(action_contract, Mapping)
        else None
    )
    if (
        not isinstance(action_shape, list)
        or len(action_shape) != 1
        or type(action_shape[0]) is not int
        or action_shape[0] < LEARNED_ACTION_DIMENSIONS
    ):
        raise QualityBridgeError("cohort exposes fewer than 11 learned controls")
    squared_sum = absolute_sum = 0.0
    transitions = behavior_matches = interventions = dones = 0
    l2_rows: list[np.ndarray] = []
    cosine_rows: list[np.ndarray] = []
    episodes: set[tuple[int, int]] = set()
    for shard in seal["shards"]:
        payload = _regular(cohort_dir / shard["payload_path"], "rollout payload")
        if sha256_file(payload) != shard["payload_sha256"]:
            raise QualityBridgeError("rollout payload changed after cohort verification")
        with np.load(payload, allow_pickle=False) as archive:
            required = {
                "student_actions",
                "expert_actions",
                "behavior_actions",
                "expert_mask",
                "dones",
                "episode_ids",
                "global_env_ids",
            }
            if not required.issubset(archive.files):
                raise QualityBridgeError("verified rollout lacks quality arrays")
            student = archive["student_actions"][..., :LEARNED_ACTION_DIMENSIONS].astype(np.float64)
            expert = archive["expert_actions"][..., :LEARNED_ACTION_DIMENSIONS].astype(np.float64)
            behavior = archive["behavior_actions"][..., :LEARNED_ACTION_DIMENSIONS].astype(np.float64)
            if student.shape != expert.shape or behavior.shape != expert.shape:
                raise QualityBridgeError("rollout action shapes differ")
            if student.shape[-1] != LEARNED_ACTION_DIMENSIONS:
                raise QualityBridgeError("rollout learned-action slice is incomplete")
            difference = student - expert
            count = int(np.prod(student.shape[:-1]))
            if count != shard["transition_count"]:
                raise QualityBridgeError("rollout transition count differs")
            squared_sum += float(np.sum(difference * difference, dtype=np.float64))
            absolute_sum += float(np.sum(np.abs(difference), dtype=np.float64))
            l2_rows.append(np.linalg.norm(difference, axis=-1).reshape(-1))
            cosine_rows.append(_cosine(student, expert).reshape(-1))
            behavior_matches += int(np.count_nonzero(np.all(behavior == expert, axis=-1)))
            interventions += int(np.count_nonzero(archive["expert_mask"]))
            dones += int(np.count_nonzero(archive["dones"]))
            env_ids = archive["global_env_ids"]
            if env_ids.ndim == 2:
                env_ids = env_ids[0]
            episode_ids = archive["episode_ids"]
            for env_offset, env_id in enumerate(env_ids.tolist()):
                episodes.update((int(env_id), int(item)) for item in episode_ids[:, env_offset].tolist())
            transitions += count
    if transitions != seal["transition_count"] or transitions <= 0:
        raise QualityBridgeError("quality aggregation coverage differs from seal")
    l2 = np.concatenate(l2_rows)
    cosine = np.concatenate(cosine_rows)
    mse = squared_sum / (transitions * LEARNED_ACTION_DIMENSIONS)
    metrics: dict[str, int | float] = {
        "rollout_quality/action_mse": mse,
        "rollout_quality/action_mae": absolute_sum / (transitions * LEARNED_ACTION_DIMENSIONS),
        "rollout_quality/action_rmse": math.sqrt(mse),
        "rollout_quality/action_cosine_mean": float(np.mean(cosine)),
        "rollout_quality/action_cosine_p50": _quantile(cosine, 0.50),
        "rollout_quality/action_cosine_p90": _quantile(cosine, 0.90),
        "rollout_quality/action_l2_mean": float(np.mean(l2)),
        "rollout_quality/action_l2_p50": _quantile(l2, 0.50),
        "rollout_quality/action_l2_p90": _quantile(l2, 0.90),
        "rollout_quality/action_l2_p99": _quantile(l2, 0.99),
        "rollout_quality/expert_intervention_rate": interventions / transitions,
        "rollout_quality/expert_label_coverage": 1.0,
        "rollout_quality/student_behavior_rate": 1.0 - interventions / transitions,
        "rollout_quality/behavior_expert_match_rate": behavior_matches / transitions,
        "rollout_quality/done_rate": dones / transitions,
        "rollout_quality/transitions": transitions,
        "rollout_quality/episodes": len(episodes),
        "rollout_quality/macro_round": macro_round,
        "rollout_quality/cohort_index": cohort_index,
        "rollout_quality/event_index": macro_round * 100 + cohort_index,
    }
    if any(isinstance(value, float) and not math.isfinite(value) for value in metrics.values()):
        raise QualityBridgeError("derived rollout quality is non-finite")
    return {
        "event": macro_round * 100 + cohort_index,
        "seal_sha256": seal["cohort_sha256"],
        "metrics": metrics,
    }


def _run_id(campaign_id: str, campaign_sha256: str) -> str:
    value = hashlib.sha256(f"{RUN_SCHEMA}\n{campaign_id}\n{campaign_sha256}\n".encode()).hexdigest()
    return f"rrd-quality-{value}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--expected-email", required=True)
    parser.add_argument("--authorization-scope", choices=("team",), required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not math.isfinite(arguments.poll_seconds) or arguments.poll_seconds < 0:
        print("student quality bridge failed: invalid poll interval", file=sys.stderr)
        return 2
    try:
        root, campaign = _campaign(arguments.campaign_root)
        campaign_sha = canonical_json_sha256(campaign)
        run_id = _run_id(campaign["campaign_id"], campaign_sha)
        state_dir = _state_dir(root, arguments.state_dir)
        state_path = state_dir / "student_quality_wandb_state.json"
        expected = _initial_state(
            campaign_id=campaign["campaign_id"],
            campaign_sha256=campaign_sha,
            entity=arguments.entity,
            project=arguments.project,
            run_id=run_id,
        )
        with _lock(state_dir):
            wandb = _load_wandb()
            query_identity(
                wandb,
                expected_email=arguments.expected_email,
                requested_entity=arguments.entity,
                authorization_scope=arguments.authorization_scope,
                timeout_s=15.0,
            )
            while True:
                state = _load_state(state_path, expected)
                uploaded = dict(state["uploaded_seal_sha256_by_event"])
                pending: list[dict[str, Any]] = []
                for macro_round, cohort_index, seal_path in _cohort_seals(root, campaign["campaign_id"]):
                    event = macro_round * 100 + cohort_index
                    key = str(event)
                    seal_sha = sha256_file(seal_path)
                    if key in uploaded:
                        if uploaded[key] != seal_sha:
                            raise QualityBridgeError("uploaded cohort seal changed")
                        continue
                    pending.append(
                        summarize_cohort(
                            seal_path.parent,
                            macro_round=macro_round,
                            cohort_index=cohort_index,
                        )
                    )
                if pending:
                    wandb_dir = state_dir / "wandb"
                    wandb_dir.mkdir(exist_ok=True)
                    run = wandb.init(
                        project=arguments.project,
                        entity=arguments.entity,
                        id=run_id,
                        name=f"{campaign['campaign_id']}-student-quality",
                        resume="allow",
                        dir=str(wandb_dir),
                        config={
                            "round_dagger/campaign_id": campaign["campaign_id"],
                            "round_dagger/campaign_sha256": campaign_sha,
                            "round_dagger/quality_scope": "sealed-rollout-action-agreement",
                            "round_dagger/telemetry_read_only": True,
                        },
                        tags=("round-dagger", "student-quality", "read-only-bridge"),
                    )
                    exit_code = 1
                    try:
                        run.define_metric("rollout_quality/event_index")
                        run.define_metric(
                            "rollout_quality/*",
                            step_metric="rollout_quality/event_index",
                        )
                        for record in pending:
                            run.log(record["metrics"], step=record["event"])
                        exit_code = 0
                    finally:
                        # Cursor publication happens only after this synchronous
                        # sender drain returns successfully.
                        run.finish(exit_code=exit_code)
                    if exit_code != 0:
                        raise QualityBridgeError("W&B sender did not finish cleanly")
                    for record in pending:
                        uploaded[str(record["event"])] = record["seal_sha256"]
                    state = {
                        **state,
                        "uploaded_seal_sha256_by_event": dict(
                            sorted(uploaded.items(), key=lambda item: int(item[0]))
                        ),
                    }
                    _atomic_json(state_path, state)
                print(
                    f"STUDENT_QUALITY_WANDB_POLL_OK uploaded={len(pending)} run_id={run_id}",
                    flush=True,
                )
                if arguments.poll_seconds == 0:
                    break
                time.sleep(arguments.poll_seconds)
    except WandbPreflightError:
        print("student quality bridge failed: W&B identity/preflight rejected", file=sys.stderr)
        return 2
    except Exception as error:
        if isinstance(error, QualityBridgeError):
            detail = str(error)
        else:
            detail = f"unexpected {type(error).__name__}"
        print(f"student quality bridge failed: {detail}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
