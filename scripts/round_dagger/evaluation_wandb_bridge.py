#!/usr/bin/env python3
"""Publish verified round-DAgger evaluation quality to one W&B dashboard.

This is deliberately an *observer*, not part of the campaign coordinator.  It
only reads canonical paired-evaluation evidence, asks
``verify_evaluator_result_evidence`` to recompute the complete result/bundle,
and writes its resume cursor and W&B files beneath an explicitly separate
state directory.  It never writes below the campaign root.

One invocation can either poll once (useful from a scheduler) or keep polling
as later macro-rounds complete::

    python scripts/round_dagger/evaluation_wandb_bridge.py \
      --campaign-root /mnt/data2/.../campaign \
      --state-dir /mnt/data2/.../evaluation-wandb \
      --project lepton-dagger-production-1 --entity nvidia \
      --expected-email ldu@nvidia.com --authorization-scope team \
      --poll-seconds 60

The W&B run ID is a deterministic function of the immutable campaign identity,
so restarts append to the same dashboard.  A verified result is uploaded at
most once per local state cursor; a conflicting result for an already-uploaded
macro-round fails closed.
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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Support the documented direct invocation by absolute script path.  In that
# mode Python otherwise places only ``scripts/round_dagger`` on ``sys.path``.
REPOSITORY_ROOT = Path(__file__).absolute().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.artifacts import ArtifactError, canonical_json_sha256
from round_dagger.evaluation import (
    EvaluationError,
    load_contract,
    verify_evaluator_result_evidence,
)
from round_dagger.promotion import load_candidate_manifest
from round_dagger.state import _campaign_root, load_campaign
from scripts.scaling.wandb_preflight import (
    WandbPreflightError,
    _load_wandb,
    query_identity,
)


STATE_SCHEMA = "fast_roborender.round_dagger.evaluation_wandb_bridge_state.v1"
BRIDGE_SCHEMA = "fast_roborender.round_dagger.evaluation_wandb_bridge.v1"
RUN_ID_SCHEMA = "fast_roborender.round_dagger.evaluation_wandb_run_id.v1"
READY_SCHEMA = "fast_roborender.round_dagger.evaluation_wandb_ready.v1"
ROUND_RE = re.compile(r"macro_round_([0-9]{6})\Z")
RECOVERY_RE = re.compile(r"hardened-[0-9a-f]{20}\Z")
RUN_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
DRIVER_STATUS_SCHEMA = "fast_roborender.round_dagger.production_driver_status.v1"
DRIVER_STATES = {"LAUNCHING", "DRY_RUN_VALIDATED", "COMPLETE", "FAILED", "RETRYING"}


class BridgeError(RuntimeError):
    """Evaluation evidence, external state, or W&B violated the bridge contract."""


@dataclass(frozen=True)
class EvaluationRecord:
    """One deeply verified immutable evaluation ready for telemetry mapping."""

    macro_round: int
    result_path: Path
    result_sha256: str
    evidence_bundle_sha256: str
    passed: bool
    metrics: Mapping[str, float]
    parent_policy_version: int
    candidate_policy_version: int
    candidate_optimizer_steps: int


_METRIC_MAP = {
    "parent_success_rate": "eval/parent/success_rate",
    "candidate_success_rate": "eval/candidate/success_rate",
    "paired_success_rate_delta": "eval/delta/success_rate",
    "parent_safety_violation_rate": "eval/parent/safety_violation_rate",
    "candidate_safety_violation_rate": "eval/candidate/safety_violation_rate",
    "paired_safety_violation_rate_delta": "eval/delta/safety_violation_rate",
    "parent_action_imitation_mse": "eval/parent/action_imitation_mse",
    "candidate_action_imitation_mse": "eval/candidate/action_imitation_mse",
    "paired_action_imitation_mse_delta": "eval/delta/action_imitation_mse",
    "paired_action_imitation_mse_relative_change": (
        "eval/delta/action_imitation_mse_relative_change"
    ),
    "parent_episode_length_mean": "eval/parent/episode_length_mean",
    "candidate_episode_length_mean": "eval/candidate/episode_length_mean",
    "paired_episode_length_mean_delta": "eval/delta/episode_length_mean",
    "parent_lag_boundary_cases": "eval/parent/lag_boundary_cases",
    "candidate_lag_boundary_cases": "eval/candidate/lag_boundary_cases",
    "paired_initial_condition_matches": "eval/pair/initial_condition_matches",
    "paired_initial_generated_matches": "eval/pair/initial_generated_matches",
    "renderer_errors": "eval/renderer/errors",
}
_OPTIONAL_METRIC_MAP = {
    # Historical immutable paired outcomes predate retention of simulator
    # reward.  Continue uploading their established metrics while exposing
    # return only for evidence that actually contains it.
    "parent_episode_return_mean": "eval/parent/episode_return_mean",
    "candidate_episode_return_mean": "eval/candidate/episode_return_mean",
    "paired_episode_return_mean_delta": "eval/delta/episode_return_mean",
}


def dashboard_run_id(*, campaign_id: str, campaign_sha256: str) -> str:
    """Return a stable, bounded run ID for a campaign-wide eval dashboard."""

    if not isinstance(campaign_id, str) or not campaign_id:
        raise BridgeError("campaign ID is missing")
    if not isinstance(campaign_sha256, str) or SHA_RE.fullmatch(campaign_sha256) is None:
        raise BridgeError("campaign SHA-256 is malformed")
    payload = (
        f"{RUN_ID_SCHEMA}\ncampaign_id={campaign_id}\n"
        f"campaign_sha256={campaign_sha256}\n"
    ).encode("utf-8")
    run_id = f"rrd-eval-{hashlib.sha256(payload).hexdigest()}"
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise BridgeError("derived W&B run ID is malformed")
    return run_id


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _reject_symlink_chain(path: Path) -> None:
    current = Path(path.anchor)
    parts = path.parts[1:] if path.anchor else path.parts
    for index, component in enumerate(parts):
        current /= component
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise BridgeError(f"symbolic links are forbidden in bridge state: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise BridgeError(f"bridge state ancestor is not a directory: {current}")


def prepare_state_directory(campaign_root: Path, state_dir: str | Path) -> Path:
    """Create a safe observer-owned state directory outside the campaign."""

    state = _absolute(state_dir)
    _reject_symlink_chain(state)
    state.mkdir(parents=True, exist_ok=True)
    _reject_symlink_chain(state)
    if not state.is_dir():
        raise BridgeError("bridge state path is not a directory")
    try:
        state.relative_to(campaign_root)
    except ValueError:
        pass
    else:
        raise BridgeError("bridge state directory must be outside the campaign root")
    return state


def ready_marker_path(state_dir: Path, value: str | Path | None) -> Path | None:
    """Validate an optional process-owned readiness marker."""

    if value is None:
        return None
    path = _absolute(value)
    if path.parent != state_dir:
        raise BridgeError(
            "ready marker must be a direct child of bridge state directory"
        )
    if path in {
        state_dir / "evaluation_wandb_bridge.lock",
        state_dir / "evaluation_wandb_bridge_state.json",
    }:
        raise BridgeError("ready marker collides with bridge state")
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise BridgeError("ready marker is unsafe")
    return path


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BridgeError(f"duplicate bridge state key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                raise BridgeError("bridge state is not a bounded regular file")
            return json.load(
                handle,
                object_pairs_hook=_strict_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    BridgeError(f"bridge state contains non-finite {value}")
                ),
            )
    except BridgeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BridgeError(f"cannot read bridge state ({type(exc).__name__})") from None


def load_driver_state(path: str | Path, *, campaign_root: Path) -> str:
    """Read the operational lifecycle that owns this observer's termination."""

    document = _load_json(_absolute(path))
    if (
        not isinstance(document, Mapping)
        or document.get("schema") != DRIVER_STATUS_SCHEMA
        or document.get("campaign_root") != str(campaign_root)
        or document.get("state") not in DRIVER_STATES
    ):
        raise BridgeError("production driver status is malformed or mismatched")
    return str(document["state"])


def _initial_state(
    *,
    campaign_id: str,
    campaign_sha256: str,
    run_id: str,
    entity: str,
    project: str,
) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "campaign_id": campaign_id,
        "campaign_sha256": campaign_sha256,
        "wandb_entity": entity,
        "wandb_project": project,
        "wandb_run_id": run_id,
        "uploaded_result_sha256_by_macro_round": {},
    }


def load_bridge_state(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(expected)
    document = _load_json(path)
    if not isinstance(document, dict) or set(document) != set(expected):
        raise BridgeError("bridge state fields differ from the expected schema")
    for key in set(expected) - {"uploaded_result_sha256_by_macro_round"}:
        if document[key] != expected[key]:
            raise BridgeError(f"bridge state {key} conflicts with this campaign/run")
    uploaded = document["uploaded_result_sha256_by_macro_round"]
    if not isinstance(uploaded, dict):
        raise BridgeError("bridge uploaded-result cursor must be a mapping")
    clean: dict[str, str] = {}
    for key, value in uploaded.items():
        if not isinstance(key, str) or not key.isdigit() or str(int(key)) != key:
            raise BridgeError("bridge cursor macro-round key is non-canonical")
        if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
            raise BridgeError("bridge cursor result digest is malformed")
        clean[key] = value
    return {**document, "uploaded_result_sha256_by_macro_round": clean}


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


@contextlib.contextmanager
def exclusive_bridge_lock(state_dir: Path) -> Iterator[None]:
    lock_path = state_dir / "evaluation_wandb_bridge.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise BridgeError("bridge lock is not a regular file")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise BridgeError("another evaluation W&B bridge owns this state directory")
    try:
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _evaluation_roots(round_dir: Path) -> list[Path]:
    """Select canonical evidence, or one explicit hardened replacement.

    Hardened recovery intentionally leaves the old canonical result in place
    as diagnostic-only evidence.  A complete recovery therefore supersedes
    that legacy result for telemetry, while two recovery results are
    ambiguous.  Without recovery, the canonical result is the sole choice.
    """

    canonical = round_dir / "evaluation"
    canonical_complete = (canonical / "result.json").is_file()
    recovery_parent = round_dir / "evaluation_recovery"
    recoveries: list[Path] = []
    if recovery_parent.exists():
        if recovery_parent.is_symlink() or not recovery_parent.is_dir():
            raise BridgeError("evaluation recovery path is unsafe")
        for child in sorted(recovery_parent.iterdir(), key=lambda path: path.name):
            if (
                RECOVERY_RE.fullmatch(child.name) is not None
                and child.is_dir()
                and not child.is_symlink()
                and (child / "result.json").is_file()
            ):
                recoveries.append(child)
    if len(recoveries) > 1:
        raise BridgeError("macro-round has multiple complete hardened evaluations")
    if recoveries:
        return recoveries
    return [canonical] if canonical_complete else []


def discover_verified_evaluations(campaign_root: str | Path) -> list[EvaluationRecord]:
    """Deep-verify every complete paired result and return it in round order.

    Partial evaluation directories without ``result.json`` are ignored.  A
    present but invalid result fails the whole poll.  Two independently valid
    results for one macro-round are ambiguous and also fail closed.
    """

    root = _campaign_root(campaign_root)
    rounds_root = root / "rounds"
    if not rounds_root.exists():
        return []
    if rounds_root.is_symlink() or not rounds_root.is_dir():
        raise BridgeError("campaign rounds path is unsafe")
    records: list[EvaluationRecord] = []
    seen_rounds: set[int] = set()
    for round_dir in sorted(rounds_root.iterdir(), key=lambda path: path.name):
        match = ROUND_RE.fullmatch(round_dir.name)
        if match is None:
            continue
        macro_round = int(match.group(1))
        if round_dir.is_symlink() or not round_dir.is_dir():
            raise BridgeError(f"macro-round path is unsafe: {round_dir.name}")
        evaluation_roots = _evaluation_roots(round_dir)
        if len(evaluation_roots) > 1:
            raise BridgeError(
                f"macro-round {macro_round} has multiple complete evaluation results"
            )
        if not evaluation_roots:
            continue
        evaluation_root = evaluation_roots[0]
        contract_path = evaluation_root / "contract.json"
        result_path = evaluation_root / "result.json"
        results_dir = evaluation_root / "results"
        try:
            verified = verify_evaluator_result_evidence(
                root,
                contract_path=contract_path,
                results_dir=results_dir,
                result_path=result_path,
            )
            contract = load_contract(root, contract_path)
            candidate = load_candidate_manifest(
                root,
                contract["candidate_manifest_path"],
                expected_macro_round=macro_round,
                bind_state=False,
            )
        except (ArtifactError, EvaluationError, OSError, ValueError) as exc:
            raise BridgeError(
                f"macro-round {macro_round} evaluation verification failed: {exc}"
            ) from None
        if verified["macro_round"] != macro_round:
            raise BridgeError("evaluation result macro-round differs from its directory")
        policies = {item["role"]: item for item in contract["policies"]}
        if set(policies) != {"parent", "candidate"}:
            raise BridgeError("paired evaluation policy roles are incomplete")
        if macro_round in seen_rounds:
            raise BridgeError(f"duplicate verified macro-round {macro_round}")
        seen_rounds.add(macro_round)
        metrics = verified["metrics"]
        if not isinstance(metrics, Mapping):
            raise BridgeError("verified evaluation metrics are malformed")
        clean_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, bool):
                raise BridgeError("verified evaluation metric is boolean")
            number = float(value)
            if not math.isfinite(number):
                raise BridgeError("verified evaluation metric is non-finite")
            clean_metrics[key] = number
        records.append(
            EvaluationRecord(
                macro_round=macro_round,
                result_path=result_path,
                result_sha256=verified["sha256"],
                evidence_bundle_sha256=verified["evidence_bundle_sha256"],
                passed=verified["passed"],
                metrics=clean_metrics,
                parent_policy_version=int(policies["parent"]["version"]),
                candidate_policy_version=int(policies["candidate"]["version"]),
                candidate_optimizer_steps=int(candidate["optimizer_steps"]),
            )
        )
    return records


def wandb_metrics(record: EvaluationRecord) -> dict[str, int | float]:
    """Map only the verified, fixed-schema evaluation values to W&B scalars."""

    missing = (set(_METRIC_MAP) | {"paired_cases"}) - set(record.metrics)
    if missing:
        raise BridgeError(f"verified evaluation lacks metrics: {sorted(missing)}")
    paired_cases = record.metrics["paired_cases"]
    if paired_cases <= 0 or not paired_cases.is_integer():
        raise BridgeError("verified paired case count must be a positive integer")
    payload: dict[str, int | float] = {
        "eval/macro_round": record.macro_round,
        "eval/pass": int(record.passed),
        "eval/case_count": int(paired_cases),
        "eval/parent/policy_version": record.parent_policy_version,
        "eval/candidate/policy_version": record.candidate_policy_version,
        "eval/candidate/optimizer_steps": record.candidate_optimizer_steps,
    }
    for source, destination in _METRIC_MAP.items():
        if source == "paired_cases":
            continue
        payload[destination] = float(record.metrics[source])
    for source, destination in _OPTIONAL_METRIC_MAP.items():
        if source in record.metrics:
            payload[destination] = float(record.metrics[source])
    return payload


class EvaluationWandbBridge:
    """Resumable read-only campaign observer suitable for repeated polling."""

    def __init__(
        self,
        *,
        campaign_root: str | Path,
        state_dir: str | Path,
        entity: str,
        project: str,
    ) -> None:
        self.campaign_root = _campaign_root(campaign_root)
        campaign = load_campaign(self.campaign_root)
        self.campaign_id = campaign["campaign_id"]
        self.campaign_sha256 = canonical_json_sha256(campaign)
        self.run_id = dashboard_run_id(
            campaign_id=self.campaign_id,
            campaign_sha256=self.campaign_sha256,
        )
        self.entity = entity
        self.project = project
        self.state_dir = prepare_state_directory(self.campaign_root, state_dir)
        self.state_path = self.state_dir / "evaluation_wandb_bridge_state.json"
        self.expected_state = _initial_state(
            campaign_id=self.campaign_id,
            campaign_sha256=self.campaign_sha256,
            run_id=self.run_id,
            entity=entity,
            project=project,
        )
        # W&B ``log`` is asynchronous.  Keep this process's accepted records
        # in memory and advance the durable cursor only after ``run.finish``
        # returns successfully.  A crash can therefore replay a record, but it
        # cannot silently mark a record delivered before the SDK has drained
        # it.  The explicit ``eval/macro_round`` axis makes such replay safe to
        # inspect and prevents W&B's global-step monotonicity from discarding
        # backfilled rounds.
        self._session_uploaded: dict[str, str] = {}

    @property
    def wandb_config(self) -> dict[str, Any]:
        return {
            "round_dagger/campaign_id": self.campaign_id,
            "round_dagger/campaign_sha256": self.campaign_sha256,
            "round_dagger/telemetry_bridge_schema": BRIDGE_SCHEMA,
            "round_dagger/telemetry_read_only": True,
            "round_dagger/evaluation_protocol": "paired-fixed-seed",
        }

    def poll_once(self, wandb_run: Any) -> int:
        """Verify and enqueue newly completed rounds for at-least-once delivery."""

        state = load_bridge_state(self.state_path, self.expected_state)
        uploaded = dict(state["uploaded_result_sha256_by_macro_round"])
        for key, digest in self._session_uploaded.items():
            previous = uploaded.get(key)
            if previous is not None and previous != digest:
                raise BridgeError(
                    f"macro-round {key} result conflicts with durable bridge state"
                )
            uploaded[key] = digest
        count = 0
        for record in discover_verified_evaluations(self.campaign_root):
            key = str(record.macro_round)
            previous = uploaded.get(key)
            if previous is not None:
                if previous != record.result_sha256:
                    raise BridgeError(
                        f"macro-round {record.macro_round} result changed after upload"
                    )
                continue
            payload = wandb_metrics(record)
            # Do not use W&B's global ``step`` here.  A resumed bridge may need
            # to replay an older macro-round, and the SDK/server can discard a
            # record whose global step is lower than one it already observed.
            wandb_run.log(payload)
            uploaded[key] = record.result_sha256
            self._session_uploaded[key] = record.result_sha256
            count += 1
        return count

    def commit_session(self) -> None:
        """Persist records only after the caller has drained the W&B run."""

        state = load_bridge_state(self.state_path, self.expected_state)
        uploaded = dict(state["uploaded_result_sha256_by_macro_round"])
        for key, digest in self._session_uploaded.items():
            previous = uploaded.get(key)
            if previous is not None and previous != digest:
                raise BridgeError(
                    f"macro-round {key} result conflicts with durable bridge state"
                )
            uploaded[key] = digest
        committed = {
            **state,
            "uploaded_result_sha256_by_macro_round": dict(
                sorted(uploaded.items(), key=lambda item: int(item[0]))
            ),
        }
        _atomic_write_json(self.state_path, committed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--expected-email", required=True)
    parser.add_argument("--authorization-scope", choices=("team",), required=True)
    parser.add_argument(
        "--driver-status",
        required=True,
        help="production driver_status.json used to terminate this observer",
    )
    parser.add_argument(
        "--ready-marker",
        help=(
            "optional direct child of --state-dir written atomically after "
            "W&B initialization and the first successful telemetry poll"
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=0.0)
    parser.add_argument("--identity-timeout-seconds", type=float, default=15.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds < 0:
        print("evaluation W&B bridge failed: poll seconds must be finite and >= 0", file=sys.stderr)
        return 2
    try:
        bridge = EvaluationWandbBridge(
            campaign_root=args.campaign_root,
            state_dir=args.state_dir,
            entity=args.entity,
            project=args.project,
        )
        # Own the external observer state before opening the deterministic W&B
        # run.  A second bridge must not resume and finish that run before it
        # discovers that this process already owns the cursor.
        with exclusive_bridge_lock(bridge.state_dir):
            ready_path = ready_marker_path(bridge.state_dir, args.ready_marker)
            if ready_path is not None:
                # Never let a prior process's marker acknowledge this process.
                ready_path.unlink(missing_ok=True)
            wandb = _load_wandb()
            query_identity(
                wandb,
                expected_email=args.expected_email,
                requested_entity=args.entity,
                authorization_scope=args.authorization_scope,
                timeout_s=args.identity_timeout_seconds,
            )
            wandb_dir = bridge.state_dir / "wandb"
            wandb_dir.mkdir(exist_ok=True)
            run = wandb.init(
                project=args.project,
                entity=args.entity,
                id=bridge.run_id,
                name=f"{bridge.campaign_id}-{bridge.campaign_sha256[:8]}-evaluation",
                resume="allow",
                dir=str(wandb_dir),
                config=bridge.wandb_config,
                tags=("round-dagger", "evaluation", "read-only-bridge"),
                settings=wandb.Settings(x_disable_stats=True),
            )
            run.define_metric("eval/macro_round")
            run.define_metric("eval/*", step_metric="eval/macro_round")
            run.define_metric("campaign/status_event")
            run.define_metric("campaign/*", step_metric="campaign/status_event")
            exit_code = 0
            status_event = 0
            ready_written = False
            try:
                while True:
                    uploaded = bridge.poll_once(run)
                    driver_state = load_driver_state(
                        args.driver_status, campaign_root=bridge.campaign_root
                    )
                    run.log(
                        {
                            "campaign/status_event": status_event,
                            "campaign/driver_failed": int(driver_state == "FAILED"),
                            "campaign/driver_complete": int(driver_state == "COMPLETE"),
                        }
                    )
                    run.summary["campaign/driver_state"] = driver_state
                    if ready_path is not None and not ready_written:
                        _atomic_write_json(
                            ready_path,
                            {
                                "schema": READY_SCHEMA,
                                "campaign_id": bridge.campaign_id,
                                "wandb_run_id": bridge.run_id,
                                "process_id": os.getpid(),
                                "ready_unix_ns": time.time_ns(),
                            },
                        )
                        ready_written = True
                    print(
                        f"EVALUATION_WANDB_POLL_OK uploaded={uploaded} "
                        f"run_id={bridge.run_id}",
                        flush=True,
                    )
                    if driver_state == "FAILED":
                        # The observer successfully reported a failed owner;
                        # that is not itself an observer-process failure.
                        break
                    if driver_state == "COMPLETE":
                        break
                    if args.poll_seconds == 0:
                        break
                    status_event += 1
                    time.sleep(args.poll_seconds)
            except KeyboardInterrupt:
                print("EVALUATION_WANDB_BRIDGE_STOPPED", flush=True)
            except Exception:
                exit_code = 1
                raise
            finally:
                run.finish(exit_code=exit_code)
                # ``finish`` is the SDK's public drain boundary.  If it raises,
                # this line is intentionally not reached and a restart replays
                # the uncommitted session instead of losing telemetry.
                bridge.commit_session()
    except (BridgeError, EvaluationError, WandbPreflightError, OSError, ValueError) as exc:
        # Never interpolate third-party exceptions here: API errors can include
        # authentication settings.  Our own exceptions contain bounded labels.
        if isinstance(exc, WandbPreflightError):
            detail = "W&B identity/preflight rejected"
        else:
            detail = str(exc)
        print(f"evaluation W&B bridge failed: {detail}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Third-party SDK exceptions have occasionally included credential
        # settings.  Report only the exception class at this process boundary.
        print(
            "evaluation W&B bridge failed: unexpected "
            f"{type(exc).__name__}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
