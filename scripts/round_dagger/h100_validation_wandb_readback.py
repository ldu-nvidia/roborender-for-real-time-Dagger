#!/usr/bin/env python3
"""Authenticate the one canonical W&B lineage used by the H100 restart gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.artifacts import (  # noqa: E402
    ArtifactError,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
)
from round_dagger.learner import CHECKPOINT_SCHEMA, RESULT_SCHEMA  # noqa: E402
from round_dagger.learner_identity import (  # noqa: E402
    LearnerIdentityError,
    learner_identity_sha256 as derive_learner_identity,
    learner_wandb_run_id,
)
from scripts.scaling.wandb_preflight import (  # noqa: E402
    _load_wandb,
    query_identity,
)


SCHEMA = "fast_roborender.round_dagger.h100_validation_wandb_readback.v2"
RESUME_EVIDENCE_SCHEMA = (
    "fast_roborender.round_dagger.h100_validation_strict_resume.v2"
)
TELEMETRY_SCHEMA = "fast_roborender.round_dagger.learner_telemetry.v2"
TELEMETRY_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "optimizer_step",
    "loss",
    "replay_transitions_rank",
    "replay_transitions_per_rank",
    "replay_transitions_global",
    "world_size",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class H100ValidationReadbackError(RuntimeError):
    """The local or remote validation telemetry contract did not hold."""


def _digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise H100ValidationReadbackError(f"{label} must be a lowercase SHA-256")
    return value


def _regular(path: Path, label: str) -> Path:
    value = Path(os.path.abspath(path.expanduser()))
    if value.is_symlink() or not value.is_file():
        raise H100ValidationReadbackError(
            f"{label} must be a regular non-symlink file"
        )
    return value.resolve(strict=True)


def _strict_json_line(line: str, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise H100ValidationReadbackError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise H100ValidationReadbackError(
            f"{label} contains non-finite value {value!r}"
        )

    try:
        value = json.loads(line, object_pairs_hook=pairs, parse_constant=constant)
    except H100ValidationReadbackError:
        raise
    except json.JSONDecodeError as exc:
        raise H100ValidationReadbackError(f"{label} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise H100ValidationReadbackError(f"{label} must be an object")
    return value


def _semantic_identity(
    *,
    run_id: str,
    campaign_id: str,
    macro_round: int,
    learner_code_sha256: str,
    learner_config_sha256: str,
    model_lineage_sha256: str,
    replay_snapshot_file_sha256: str,
    replay_snapshot_sha256: str,
    parent_runtime_policy_sha256: str,
    learner_identity_sha256: str,
    runtime_container_sha256: str,
    validation_contract_sha256: str,
) -> dict[str, Any]:
    if type(macro_round) is not int or macro_round < 0:
        raise H100ValidationReadbackError("macro round is invalid")
    values = {
        "learner_code_sha256": _digest(
            learner_code_sha256, "learner code SHA-256"
        ),
        "learner_config_sha256": _digest(
            learner_config_sha256, "learner config SHA-256"
        ),
        "model_lineage_sha256": _digest(
            model_lineage_sha256, "model lineage SHA-256"
        ),
        "replay_snapshot_file_sha256": _digest(
            replay_snapshot_file_sha256, "replay snapshot file SHA-256"
        ),
        "replay_snapshot_sha256": _digest(
            replay_snapshot_sha256, "replay snapshot semantic SHA-256"
        ),
        "parent_runtime_policy_sha256": _digest(
            parent_runtime_policy_sha256, "parent runtime policy SHA-256"
        ),
        "learner_identity_sha256": _digest(
            learner_identity_sha256, "learner identity SHA-256"
        ),
        "runtime_container_sha256": _digest(
            runtime_container_sha256, "runtime container SHA-256"
        ),
        # The validation contract is deliberately used as the learner
        # contract.  It must never be substituted with the production
        # 8-node/500-step contract for this 1-node/2-step live gate.
        "learner_contract_sha256": _digest(
            validation_contract_sha256, "validation contract SHA-256"
        ),
    }
    try:
        derived = derive_learner_identity(
            campaign_id=campaign_id,
            macro_round=macro_round,
            learner_code_sha256=values["learner_code_sha256"],
            learner_config_sha256=values["learner_config_sha256"],
            replay_snapshot_file_sha256=values[
                "replay_snapshot_file_sha256"
            ],
            replay_snapshot_sha256=values["replay_snapshot_sha256"],
            parent_checkpoint_sha256=values["model_lineage_sha256"],
            parent_runtime_policy_sha256=values[
                "parent_runtime_policy_sha256"
            ],
            runtime_container_sha256=values["runtime_container_sha256"],
            learner_contract_sha256=values["learner_contract_sha256"],
        )
        canonical_run_id = learner_wandb_run_id(
            campaign_id=campaign_id,
            macro_round=macro_round,
            learner_identity_sha256=derived,
        )
    except LearnerIdentityError as exc:
        raise H100ValidationReadbackError(
            f"semantic learner identity is malformed: {exc}"
        ) from exc
    if derived != values["learner_identity_sha256"]:
        raise H100ValidationReadbackError(
            "learner identity does not bind the exact validation semantics"
        )
    if run_id != canonical_run_id:
        raise H100ValidationReadbackError(
            "W&B run ID is not canonical for the learner identity"
        )
    return {
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        **values,
    }


def _telemetry(
    path: Path, *, campaign_id: str, macro_round: int
) -> tuple[Path, list[dict[str, Any]]]:
    path = _regular(path, "learner telemetry")
    rows = [
        _strict_json_line(line, f"learner telemetry line {index}")
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.strip()
    ]
    if len(rows) != 2:
        raise H100ValidationReadbackError(
            "learner telemetry must contain exactly optimizer steps 1 and 2"
        )
    expected_transition_counts: tuple[int, ...] | None = None
    for index, row in enumerate(rows):
        if set(row) != TELEMETRY_KEYS or row.get("schema") != TELEMETRY_SCHEMA:
            raise H100ValidationReadbackError("learner telemetry fields differ")
        expected = {
            "campaign_id": campaign_id,
            "macro_round": macro_round,
            "optimizer_step": index + 1,
            "world_size": 8,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise H100ValidationReadbackError(
                "learner telemetry step or lineage mismatch"
            )
        loss = row.get("loss")
        transitions = row.get("replay_transitions_rank")
        transition_counts = row.get("replay_transitions_per_rank")
        global_transitions = row.get("replay_transitions_global")
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or float(loss) < 0.0
        ):
            raise H100ValidationReadbackError("learner telemetry loss is invalid")
        if (
            isinstance(transitions, bool)
            or not isinstance(transitions, int)
            or transitions < 1
        ):
            raise H100ValidationReadbackError(
                "learner telemetry replay transition count is invalid"
            )
        if (
            not isinstance(transition_counts, list)
            or len(transition_counts) != 8
            or any(type(value) is not int or value < 1 for value in transition_counts)
            or transitions != transition_counts[0]
            or type(global_transitions) is not int
            or global_transitions != sum(transition_counts)
        ):
            raise H100ValidationReadbackError(
                "learner telemetry replay transition balance is invalid"
            )
        observed_transition_counts = tuple(transition_counts)
        if expected_transition_counts is None:
            expected_transition_counts = observed_transition_counts
        elif observed_transition_counts != expected_transition_counts:
            raise H100ValidationReadbackError(
                "learner telemetry replay partition changed between steps"
            )
    return path, rows


def _load_checkpoint(
    path: Path,
    *,
    label: str,
    completed_steps: int,
    semantic: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], str]:
    path = _regular(path, label)
    try:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise H100ValidationReadbackError(f"cannot load {label}: {exc}") from exc
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_schema": "gr00t.trl.full_state.v2",
        "campaign_id": semantic["campaign_id"],
        "macro_round": semantic["macro_round"],
        "model_lineage_sha256": semantic["model_lineage_sha256"],
        "code_sha256": semantic["learner_code_sha256"],
        "learner_config_sha256": semantic["learner_config_sha256"],
        "learner_identity_sha256": semantic["learner_identity_sha256"],
        "runtime_container_sha256": semantic["runtime_container_sha256"],
        "learner_contract_sha256": semantic["learner_contract_sha256"],
        "replay_snapshot_sha256": semantic["replay_snapshot_sha256"],
        "replay_snapshot_file_sha256": semantic[
            "replay_snapshot_file_sha256"
        ],
        "completed_steps": completed_steps,
        "world_size": 8,
    }
    if not isinstance(value, Mapping) or any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise H100ValidationReadbackError(f"{label} v2 lineage mismatch")
    initial_loss = value.get("initial_loss")
    states = value.get("rng_states")
    if (
        isinstance(initial_loss, bool)
        or not isinstance(initial_loss, (int, float))
        or not math.isfinite(float(initial_loss))
        or float(initial_loss) < 0.0
        or not isinstance(states, list)
        or len(states) != 8
    ):
        raise H100ValidationReadbackError(f"{label} restart frontier is invalid")
    return path, dict(value), sha256_file(path)


def _result(
    path: Path,
    *,
    label: str,
    semantic: Mapping[str, Any],
    expected_checkpoint_sha256: str,
) -> tuple[Path, dict[str, Any], Path]:
    path = _regular(path, label)
    value = load_strict_json(path)
    expected = {
        "schema": RESULT_SCHEMA,
        "campaign_id": semantic["campaign_id"],
        "macro_round": semantic["macro_round"],
        "code_sha256": semantic["learner_code_sha256"],
        "learner_config_sha256": semantic["learner_config_sha256"],
        "model_lineage_sha256": semantic["model_lineage_sha256"],
        "replay_snapshot_sha256": semantic["replay_snapshot_sha256"],
        "optimizer_steps": 2,
        "checkpoint_sha256": expected_checkpoint_sha256,
    }
    if not isinstance(value, Mapping) or any(
        value.get(key) != item for key, item in expected.items()
    ):
        raise H100ValidationReadbackError(f"{label} lineage mismatch")
    checkpoint_name = value.get("checkpoint_path")
    if (
        not isinstance(checkpoint_name, str)
        or not checkpoint_name
        or Path(checkpoint_name).name != checkpoint_name
    ):
        raise H100ValidationReadbackError(f"{label} candidate path is unsafe")
    candidate, _, candidate_sha = _load_checkpoint(
        path.parent / checkpoint_name,
        label=f"{label} candidate",
        completed_steps=2,
        semantic=semantic,
    )
    if candidate_sha != expected_checkpoint_sha256:
        raise H100ValidationReadbackError(
            f"{label} candidate differs from its restart checkpoint"
        )
    for key in ("initial_loss", "final_loss"):
        observed = value.get(key)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) < 0.0
        ):
            raise H100ValidationReadbackError(f"{label} {key} is invalid")
    return path, dict(value), candidate


def _resume_evidence(
    path: Path,
    *,
    run_id: str,
    semantic: Mapping[str, Any],
    stopped_checkpoint_sha256: str,
    resumed_checkpoint_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    path = _regular(path, "strict resume evidence")
    if path.stat().st_mode & 0o222:
        raise H100ValidationReadbackError("strict resume evidence must be read-only")
    value = load_strict_json(path)
    expected = {
        "schema": RESUME_EVIDENCE_SCHEMA,
        "status": "PASS",
        "run_id": run_id,
        "first_invocation_resume_mode": "allow",
        "second_invocation_resume_mode": "must",
        "stop_step": 1,
        "target_step": 2,
        "learner_code_sha256": semantic["learner_code_sha256"],
        "learner_config_sha256": semantic["learner_config_sha256"],
        "replay_snapshot_file_sha256": semantic[
            "replay_snapshot_file_sha256"
        ],
        "replay_snapshot_sha256": semantic["replay_snapshot_sha256"],
        "parent_checkpoint_sha256": semantic["model_lineage_sha256"],
        "parent_runtime_policy_sha256": semantic[
            "parent_runtime_policy_sha256"
        ],
        "learner_identity_sha256": semantic["learner_identity_sha256"],
        "runtime_container_sha256": semantic["runtime_container_sha256"],
        "learner_contract_sha256": semantic["learner_contract_sha256"],
        "uninterrupted_comparator_wandb": False,
        "canonical_wandb_lineage": True,
        "stopped_checkpoint_sha256": stopped_checkpoint_sha256,
        "resumed_checkpoint_sha256": resumed_checkpoint_sha256,
    }
    if not isinstance(value, Mapping) or set(value) != set(expected) or any(
        value.get(key) != item for key, item in expected.items()
    ):
        raise H100ValidationReadbackError("strict resume invocation evidence mismatch")
    return path, dict(value)


def _config_value(config: object, key: str) -> object:
    if isinstance(config, Mapping):
        return config.get(key)
    try:
        return config[key]  # type: ignore[index]
    except (KeyError, TypeError, AttributeError, IndexError):
        return None


def _remote_rows(remote: object) -> list[dict[str, Any]]:
    try:
        raw_rows = list(
            remote.scan_history(keys=sorted(TELEMETRY_KEYS), page_size=1000)
        )
    except Exception as exc:
        raise H100ValidationReadbackError(
            f"remote W&B history query failed ({type(exc).__name__})"
        ) from None
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or raw.get("schema") != TELEMETRY_SCHEMA:
            continue
        rows.append({key: raw.get(key) for key in TELEMETRY_KEYS})
    return rows


def _remote_run(
    api: object,
    *,
    entity: str,
    project: str,
    run_id: str,
    local_rows: list[dict[str, Any]],
    semantic: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        remote = api.run(f"{entity}/{project}/{run_id}")
    except Exception as exc:
        raise H100ValidationReadbackError(
            f"remote W&B run query failed ({type(exc).__name__})"
        ) from None
    if getattr(remote, "id", run_id) != run_id:
        raise H100ValidationReadbackError("remote W&B run ID mismatch")
    if getattr(remote, "state", "finished") != "finished":
        raise H100ValidationReadbackError("remote W&B learner run is not finished")
    config = getattr(remote, "config", None)
    expected_config = {
        "campaign_id": semantic["campaign_id"],
        "macro_round": semantic["macro_round"],
        "learner_code_sha256": semantic["learner_code_sha256"],
        "learner_config_sha256": semantic["learner_config_sha256"],
        "model_lineage_sha256": semantic["model_lineage_sha256"],
        "replay_snapshot_file_sha256": semantic[
            "replay_snapshot_file_sha256"
        ],
        "replay_snapshot_sha256": semantic["replay_snapshot_sha256"],
        "parent_runtime_policy_sha256": semantic[
            "parent_runtime_policy_sha256"
        ],
        "learner_identity_sha256": semantic["learner_identity_sha256"],
        "runtime_container_sha256": semantic["runtime_container_sha256"],
        "learner_contract_sha256": semantic["learner_contract_sha256"],
    }
    if any(
        _config_value(config, key) != value
        for key, value in expected_config.items()
    ):
        raise H100ValidationReadbackError("remote W&B learner config mismatch")
    rows = _remote_rows(remote)
    if rows != local_rows:
        raise H100ValidationReadbackError(
            "remote W&B history does not exactly match resumed local telemetry"
        )
    steps = [row["optimizer_step"] for row in rows]
    if steps != [1, 2]:
        raise H100ValidationReadbackError(
            "remote W&B history is not contiguous optimizer steps 1 and 2"
        )
    return {
        "run_id": run_id,
        "history": rows,
        "history_steps": steps,
        "history_contiguous": True,
        "remote_state": "finished",
        "config_verified": True,
    }


def readback(
    *,
    run_id: str,
    uninterrupted_telemetry: Path,
    resumed_telemetry: Path,
    uninterrupted_result: Path,
    resumed_result: Path,
    resume_evidence: Path,
    uninterrupted_checkpoint: Path,
    stopped_checkpoint: Path,
    resumed_checkpoint: Path,
    campaign_id: str,
    macro_round: int,
    learner_code_sha256: str,
    learner_config_sha256: str,
    model_lineage_sha256: str,
    replay_snapshot_file_sha256: str,
    replay_snapshot_sha256: str,
    parent_runtime_policy_sha256: str,
    learner_identity_sha256: str,
    runtime_container_sha256: str,
    validation_contract_sha256: str,
    project: str,
    expected_email: str,
    entity: str,
    authorization_scope: str,
    timeout_s: float,
    poll_interval_s: float,
    wandb_module: object | None = None,
) -> dict[str, Any]:
    if timeout_s <= 0 or poll_interval_s <= 0:
        raise H100ValidationReadbackError("readback timing must be positive")
    semantic = _semantic_identity(
        run_id=run_id,
        campaign_id=campaign_id,
        macro_round=macro_round,
        learner_code_sha256=learner_code_sha256,
        learner_config_sha256=learner_config_sha256,
        model_lineage_sha256=model_lineage_sha256,
        replay_snapshot_file_sha256=replay_snapshot_file_sha256,
        replay_snapshot_sha256=replay_snapshot_sha256,
        parent_runtime_policy_sha256=parent_runtime_policy_sha256,
        learner_identity_sha256=learner_identity_sha256,
        runtime_container_sha256=runtime_container_sha256,
        validation_contract_sha256=validation_contract_sha256,
    )
    comparator_telemetry_path, comparator_rows = _telemetry(
        uninterrupted_telemetry,
        campaign_id=campaign_id,
        macro_round=macro_round,
    )
    resumed_telemetry_path, resumed_rows = _telemetry(
        resumed_telemetry,
        campaign_id=campaign_id,
        macro_round=macro_round,
    )
    if comparator_rows != resumed_rows:
        raise H100ValidationReadbackError(
            "uninterrupted and resumed local loss histories differ"
        )
    comparator_checkpoint_path, comparator_checkpoint_doc, comparator_checkpoint_sha = (
        _load_checkpoint(
            uninterrupted_checkpoint,
            label="uninterrupted checkpoint",
            completed_steps=2,
            semantic=semantic,
        )
    )
    stopped_checkpoint_path, stopped_checkpoint_doc, stopped_checkpoint_sha = (
        _load_checkpoint(
            stopped_checkpoint,
            label="intentional-stop checkpoint",
            completed_steps=1,
            semantic=semantic,
        )
    )
    resumed_checkpoint_path, resumed_checkpoint_doc, resumed_checkpoint_sha = (
        _load_checkpoint(
            resumed_checkpoint,
            label="resumed checkpoint",
            completed_steps=2,
            semantic=semantic,
        )
    )
    if stopped_checkpoint_path.stat().st_mode & 0o222:
        raise H100ValidationReadbackError(
            "intentional-stop checkpoint must be read-only"
        )
    comparator_result_path, comparator_result_doc, comparator_candidate = _result(
        uninterrupted_result,
        label="uninterrupted result",
        semantic=semantic,
        expected_checkpoint_sha256=comparator_checkpoint_sha,
    )
    resumed_result_path, resumed_result_doc, resumed_candidate = _result(
        resumed_result,
        label="resumed result",
        semantic=semantic,
        expected_checkpoint_sha256=resumed_checkpoint_sha,
    )
    if (
        comparator_result_doc["initial_loss"] != comparator_rows[0]["loss"]
        or resumed_result_doc["initial_loss"] != resumed_rows[0]["loss"]
        or comparator_checkpoint_doc["initial_loss"]
        != comparator_result_doc["initial_loss"]
        or stopped_checkpoint_doc["initial_loss"]
        != resumed_result_doc["initial_loss"]
        or resumed_checkpoint_doc["initial_loss"]
        != resumed_result_doc["initial_loss"]
    ):
        raise H100ValidationReadbackError(
            "checkpoint, result, and telemetry initial losses differ"
        )
    if (
        comparator_result_doc["initial_loss"]
        != resumed_result_doc["initial_loss"]
        or comparator_result_doc["final_loss"]
        != resumed_result_doc["final_loss"]
    ):
        raise H100ValidationReadbackError(
            "uninterrupted and resumed result losses differ"
        )
    resume_path, _ = _resume_evidence(
        resume_evidence,
        run_id=run_id,
        semantic=semantic,
        stopped_checkpoint_sha256=stopped_checkpoint_sha,
        resumed_checkpoint_sha256=resumed_checkpoint_sha,
    )

    wandb_module = _load_wandb() if wandb_module is None else wandb_module
    identity = query_identity(
        wandb_module,
        expected_email=expected_email,
        requested_entity=entity,
        authorization_scope=authorization_scope,
    )
    try:
        api = wandb_module.Api(timeout=30)
    except Exception as exc:
        raise H100ValidationReadbackError(
            f"remote W&B API initialization failed ({type(exc).__name__})"
        ) from None

    deadline = time.monotonic() + timeout_s
    last_error = ""
    remote_run: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            remote_run = _remote_run(
                api,
                entity=entity,
                project=project,
                run_id=run_id,
                local_rows=resumed_rows,
                semantic=semantic,
            )
            break
        except H100ValidationReadbackError as exc:
            last_error = type(exc).__name__
            remote_run = None
            time.sleep(poll_interval_s)
    if remote_run is None:
        raise H100ValidationReadbackError(
            f"remote W&B learner readback timed out ({last_error or 'no evidence'})"
        )

    return {
        "schema": SCHEMA,
        "status": "PASS",
        "account_email": identity.email,
        "entity": identity.entity,
        "authorization_scope": identity.authorization_scope,
        "team_member": identity.team_member,
        "entity_is_team": identity.entity_is_team,
        "personal": identity.personal,
        "project": project,
        **semantic,
        "validation_contract_sha256": semantic["learner_contract_sha256"],
        "wandb_run_id": run_id,
        "canonical_run": remote_run,
        "uninterrupted_comparator": {
            "wandb_enabled": False,
            "history_steps": [1, 2],
            "telemetry_sha256": sha256_file(comparator_telemetry_path),
            "result_sha256": sha256_file(comparator_result_path),
            "checkpoint_sha256": comparator_checkpoint_sha,
            "candidate_checkpoint_sha256": sha256_file(comparator_candidate),
        },
        "resumed_execution": {
            "wandb_enabled": True,
            "history_steps": [1, 2],
            "telemetry_sha256": sha256_file(resumed_telemetry_path),
            "result_sha256": sha256_file(resumed_result_path),
            "stopped_checkpoint_sha256": stopped_checkpoint_sha,
            "checkpoint_sha256": resumed_checkpoint_sha,
            "candidate_checkpoint_sha256": sha256_file(resumed_candidate),
        },
        "uninterrupted_telemetry_sha256": sha256_file(
            comparator_telemetry_path
        ),
        "resumed_telemetry_sha256": sha256_file(resumed_telemetry_path),
        "uninterrupted_result_sha256": sha256_file(comparator_result_path),
        "resumed_result_sha256": sha256_file(resumed_result_path),
        "uninterrupted_checkpoint_sha256": comparator_checkpoint_sha,
        "stopped_checkpoint_sha256": stopped_checkpoint_sha,
        "resumed_checkpoint_sha256": resumed_checkpoint_sha,
        "strict_resume_evidence_sha256": sha256_file(resume_path),
        "initial_loss": comparator_result_doc["initial_loss"],
        "final_loss": comparator_result_doc["final_loss"],
        "histories_identical": True,
        "resumed_history_contiguous": True,
        "strict_resume_mode": "must",
        "strict_resume_verified": True,
        "single_canonical_wandb_run": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--uninterrupted-telemetry", required=True, type=Path)
    parser.add_argument("--resumed-telemetry", required=True, type=Path)
    parser.add_argument("--uninterrupted-result", required=True, type=Path)
    parser.add_argument("--resumed-result", required=True, type=Path)
    parser.add_argument("--resume-evidence", required=True, type=Path)
    parser.add_argument("--uninterrupted-checkpoint", required=True, type=Path)
    parser.add_argument("--stopped-checkpoint", required=True, type=Path)
    parser.add_argument("--resumed-checkpoint", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--macro-round", required=True, type=int)
    parser.add_argument("--learner-code-sha256", required=True)
    parser.add_argument("--learner-config-sha256", required=True)
    parser.add_argument("--model-lineage-sha256", required=True)
    parser.add_argument("--replay-snapshot-file-sha256", required=True)
    parser.add_argument("--replay-snapshot-sha256", required=True)
    parser.add_argument("--parent-runtime-policy-sha256", required=True)
    parser.add_argument("--learner-identity-sha256", required=True)
    parser.add_argument("--runtime-container-sha256", required=True)
    parser.add_argument("--validation-contract-sha256", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--expected-email", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--authorization-scope", required=True, choices=("team",))
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = readback(
        run_id=arguments.run_id,
        uninterrupted_telemetry=arguments.uninterrupted_telemetry,
        resumed_telemetry=arguments.resumed_telemetry,
        uninterrupted_result=arguments.uninterrupted_result,
        resumed_result=arguments.resumed_result,
        resume_evidence=arguments.resume_evidence,
        uninterrupted_checkpoint=arguments.uninterrupted_checkpoint,
        stopped_checkpoint=arguments.stopped_checkpoint,
        resumed_checkpoint=arguments.resumed_checkpoint,
        campaign_id=arguments.campaign_id,
        macro_round=arguments.macro_round,
        learner_code_sha256=arguments.learner_code_sha256,
        learner_config_sha256=arguments.learner_config_sha256,
        model_lineage_sha256=arguments.model_lineage_sha256,
        replay_snapshot_file_sha256=arguments.replay_snapshot_file_sha256,
        replay_snapshot_sha256=arguments.replay_snapshot_sha256,
        parent_runtime_policy_sha256=arguments.parent_runtime_policy_sha256,
        learner_identity_sha256=arguments.learner_identity_sha256,
        runtime_container_sha256=arguments.runtime_container_sha256,
        validation_contract_sha256=arguments.validation_contract_sha256,
        project=arguments.project,
        expected_email=arguments.expected_email,
        entity=arguments.entity,
        authorization_scope=arguments.authorization_scope,
        timeout_s=arguments.timeout_s,
        poll_interval_s=arguments.poll_interval_s,
    )
    output = Path(os.path.abspath(arguments.output.expanduser()))
    if output.is_symlink():
        raise H100ValidationReadbackError("readback output must not be a symlink")
    digest = publish_immutable_json(output, report)
    os.chmod(output, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(f"OCI_NRT_H100_WANDB_READBACK_PASS output={output} sha256={digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArtifactError, H100ValidationReadbackError) as exc:
        print(f"OCI H100 W&B readback failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
