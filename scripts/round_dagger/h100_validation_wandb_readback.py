#!/usr/bin/env python3
"""Authenticate and read back both OCI H100 validation learner runs."""

from __future__ import annotations

import argparse
import json
import math
import os
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
from scripts.scaling.wandb_preflight import (  # noqa: E402
    _load_wandb,
    query_identity,
)


SCHEMA = "fast_roborender.round_dagger.h100_validation_wandb_readback.v1"
RESULT_SCHEMA = "fast_roborender.round_dagger.learner_result.v2"
RESUME_EVIDENCE_SCHEMA = (
    "fast_roborender.round_dagger.h100_validation_strict_resume.v1"
)
TELEMETRY_SCHEMA = "fast_roborender.round_dagger.learner_telemetry.v1"
TELEMETRY_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "optimizer_step",
    "loss",
    "replay_transitions_rank",
    "world_size",
}


class H100ValidationReadbackError(RuntimeError):
    """The local or remote validation telemetry contract did not hold."""


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
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or float(loss) < 0.0
        ):
            raise H100ValidationReadbackError("learner telemetry loss is invalid")
        if isinstance(transitions, bool) or not isinstance(transitions, int) or transitions < 1:
            raise H100ValidationReadbackError(
                "learner telemetry replay transition count is invalid"
            )
    return path, rows


def _result(
    path: Path,
    *,
    campaign_id: str,
    macro_round: int,
    learner_config_sha256: str,
    model_lineage_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    path = _regular(path, "learner result")
    value = load_strict_json(path)
    expected = {
        "schema": RESULT_SCHEMA,
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "learner_config_sha256": learner_config_sha256,
        "model_lineage_sha256": model_lineage_sha256,
        "optimizer_steps": 2,
    }
    if not isinstance(value, Mapping) or any(
        value.get(key) != item for key, item in expected.items()
    ):
        raise H100ValidationReadbackError("learner result lineage mismatch")
    for key in ("initial_loss", "final_loss"):
        observed = value.get(key)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) < 0.0
        ):
            raise H100ValidationReadbackError(f"learner result {key} is invalid")
    return path, dict(value)


def _resume_evidence(
    path: Path,
    *,
    resumed_run_id: str,
    learner_config_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    path = _regular(path, "strict resume evidence")
    if path.stat().st_mode & 0o222:
        raise H100ValidationReadbackError("strict resume evidence must be read-only")
    value = load_strict_json(path)
    expected = {
        "schema": RESUME_EVIDENCE_SCHEMA,
        "status": "PASS",
        "run_id": resumed_run_id,
        "first_invocation_resume_mode": "allow",
        "second_invocation_resume_mode": "must",
        "stop_step": 1,
        "target_step": 2,
        "learner_config_sha256": learner_config_sha256,
    }
    if not isinstance(value, Mapping) or any(
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
        row = {key: raw.get(key) for key in TELEMETRY_KEYS}
        rows.append(row)
    return rows


def _remote_run(
    api: object,
    *,
    entity: str,
    project: str,
    run_id: str,
    local_rows: list[dict[str, Any]],
    campaign_id: str,
    macro_round: int,
    learner_config_sha256: str,
    model_lineage_sha256: str,
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
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "learner_config_sha256": learner_config_sha256,
        "model_lineage_sha256": model_lineage_sha256,
    }
    if any(_config_value(config, key) != value for key, value in expected_config.items()):
        raise H100ValidationReadbackError("remote W&B learner config mismatch")
    rows = _remote_rows(remote)
    if rows != local_rows:
        raise H100ValidationReadbackError(
            "remote W&B history does not exactly match local step/loss telemetry"
        )
    return {
        "run_id": run_id,
        "history": rows,
        "history_steps": [row["optimizer_step"] for row in rows],
        "history_contiguous": [row["optimizer_step"] for row in rows] == [1, 2],
        "remote_state": "finished",
        "config_verified": True,
    }


def readback(
    *,
    full_run_id: str,
    resumed_run_id: str,
    full_telemetry: Path,
    resumed_telemetry: Path,
    full_result: Path,
    resumed_result: Path,
    resume_evidence: Path,
    campaign_id: str,
    macro_round: int,
    learner_config_sha256: str,
    model_lineage_sha256: str,
    project: str,
    expected_email: str,
    entity: str,
    authorization_scope: str,
    timeout_s: float,
    poll_interval_s: float,
    wandb_module: object | None = None,
) -> dict[str, Any]:
    if full_run_id == resumed_run_id:
        raise H100ValidationReadbackError("full and resumed W&B run IDs must differ")
    if timeout_s <= 0 or poll_interval_s <= 0:
        raise H100ValidationReadbackError("readback timing must be positive")
    full_telemetry_path, full_rows = _telemetry(
        full_telemetry, campaign_id=campaign_id, macro_round=macro_round
    )
    resumed_telemetry_path, resumed_rows = _telemetry(
        resumed_telemetry, campaign_id=campaign_id, macro_round=macro_round
    )
    if full_rows != resumed_rows:
        raise H100ValidationReadbackError(
            "uninterrupted and resumed local loss histories differ"
        )
    full_result_path, full_result_doc = _result(
        full_result,
        campaign_id=campaign_id,
        macro_round=macro_round,
        learner_config_sha256=learner_config_sha256,
        model_lineage_sha256=model_lineage_sha256,
    )
    resumed_result_path, resumed_result_doc = _result(
        resumed_result,
        campaign_id=campaign_id,
        macro_round=macro_round,
        learner_config_sha256=learner_config_sha256,
        model_lineage_sha256=model_lineage_sha256,
    )
    if full_result_doc["initial_loss"] != full_rows[0]["loss"]:
        raise H100ValidationReadbackError(
            "uninterrupted result initial loss differs from telemetry"
        )
    if resumed_result_doc["initial_loss"] != resumed_rows[0]["loss"]:
        raise H100ValidationReadbackError(
            "resumed result initial loss differs from telemetry"
        )
    if (
        full_result_doc["initial_loss"] != resumed_result_doc["initial_loss"]
        or full_result_doc["final_loss"] != resumed_result_doc["final_loss"]
    ):
        raise H100ValidationReadbackError("full/resumed result losses differ")
    resume_path, resume_doc = _resume_evidence(
        resume_evidence,
        resumed_run_id=resumed_run_id,
        learner_config_sha256=learner_config_sha256,
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
    full_remote: dict[str, Any] | None = None
    resumed_remote: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            full_remote = _remote_run(
                api,
                entity=entity,
                project=project,
                run_id=full_run_id,
                local_rows=full_rows,
                campaign_id=campaign_id,
                macro_round=macro_round,
                learner_config_sha256=learner_config_sha256,
                model_lineage_sha256=model_lineage_sha256,
            )
            resumed_remote = _remote_run(
                api,
                entity=entity,
                project=project,
                run_id=resumed_run_id,
                local_rows=resumed_rows,
                campaign_id=campaign_id,
                macro_round=macro_round,
                learner_config_sha256=learner_config_sha256,
                model_lineage_sha256=model_lineage_sha256,
            )
            break
        except H100ValidationReadbackError as exc:
            last_error = type(exc).__name__
            full_remote = resumed_remote = None
            time.sleep(poll_interval_s)
    if full_remote is None or resumed_remote is None:
        raise H100ValidationReadbackError(
            f"remote W&B learner readback timed out ({last_error or 'no evidence'})"
        )
    if not full_remote["history_contiguous"] or not resumed_remote["history_contiguous"]:
        raise H100ValidationReadbackError("remote W&B histories are not contiguous 1,2")

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
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "learner_config_sha256": learner_config_sha256,
        "model_lineage_sha256": model_lineage_sha256,
        "full_run": full_remote,
        "resumed_run": resumed_remote,
        "runs_distinct": True,
        "full_telemetry_sha256": sha256_file(full_telemetry_path),
        "resumed_telemetry_sha256": sha256_file(resumed_telemetry_path),
        "full_result_sha256": sha256_file(full_result_path),
        "resumed_result_sha256": sha256_file(resumed_result_path),
        "initial_loss": full_result_doc["initial_loss"],
        "final_loss": full_result_doc["final_loss"],
        "histories_identical": True,
        "resumed_history_contiguous": True,
        "strict_resume_mode": "must",
        "strict_resume_verified": True,
        "strict_resume_evidence_sha256": sha256_file(resume_path),
        "stopped_checkpoint_sha256": resume_doc["stopped_checkpoint_sha256"],
        "resumed_checkpoint_sha256": resume_doc["resumed_checkpoint_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-id", required=True)
    parser.add_argument("--resumed-run-id", required=True)
    parser.add_argument("--full-telemetry", required=True, type=Path)
    parser.add_argument("--resumed-telemetry", required=True, type=Path)
    parser.add_argument("--full-result", required=True, type=Path)
    parser.add_argument("--resumed-result", required=True, type=Path)
    parser.add_argument("--resume-evidence", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--macro-round", required=True, type=int)
    parser.add_argument("--learner-config-sha256", required=True)
    parser.add_argument("--model-lineage-sha256", required=True)
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
        full_run_id=arguments.full_run_id,
        resumed_run_id=arguments.resumed_run_id,
        full_telemetry=arguments.full_telemetry,
        resumed_telemetry=arguments.resumed_telemetry,
        full_result=arguments.full_result,
        resumed_result=arguments.resumed_result,
        resume_evidence=arguments.resume_evidence,
        campaign_id=arguments.campaign_id,
        macro_round=arguments.macro_round,
        learner_config_sha256=arguments.learner_config_sha256,
        model_lineage_sha256=arguments.model_lineage_sha256,
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
