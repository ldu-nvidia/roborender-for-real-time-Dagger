"""Bounded Lepton H200 validation Job for the exclusive-node backend.

This is deliberately a different request schema and entrypoint from the
1,200-step production learner.  Its execution shape is fixed in code: one
eight-rank CUDA worker, four optimizer steps, a split/restart at step two, and
256 held-out cases evaluated every step.  It proves DDP, replay, restart,
student-quality telemetry, W&B readback, and the stop/Job/restore lifecycle
without accepting any production learner environment variables.
"""

from __future__ import annotations

import json
import math
import os
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from round_dagger import artifacts
from round_dagger.lepton_job import (
    LeptonBackendContract,
    LeptonJobError,
    publish_checksum,
    require_finalized_supply_chain,
    sha256_file,
    verify_python_environment_freeze,
    verify_sealed_contract_source,
)


SMOKE_REQUEST_SCHEMA = (
    "fast_roborender.round_dagger.lepton_bounded_smoke_request.v1"
)
SMOKE_SUCCESS_SCHEMA = (
    "fast_roborender.round_dagger.lepton_bounded_smoke_success.v1"
)
SMOKE_ENTRYPOINT_RELATIVE = PurePosixPath(
    "scripts/round_dagger/run_lepton_h200_smoke_job.sh"
)
SMOKE_SCRIPT_RELATIVE = PurePosixPath(
    "scripts/round_dagger/lepton_h200_learner_smoke.py"
)
SMOKE_OUTPUT_BASE_RELATIVE = PurePosixPath("validation/lepton_h200_bounded_smoke")
SMOKE_CAMPAIGN_ID = "lepton-h200-learner-smoke"
SMOKE_WORLD_SIZE = 8
SMOKE_HELDOUT_CASES = 256
SMOKE_MAX_STEPS = 4
SMOKE_SPLIT_STEP = 2
SMOKE_SAVE_EVERY = 1
SMOKE_HELDOUT_EVAL_INTERVAL = 1
SMOKE_WANDB_RUN_PREFIX = "rrd-h200-bounded-smoke-id"

_REQUEST_KEYS = {
    "schema",
    "backend_contract_sha256",
    "smoke_identity_sha256",
    "execution",
    "wandb",
    "paths",
}
_EXECUTION_KEYS = {
    "campaign_id",
    "world_size",
    "device",
    "heldout_cases",
    "max_steps",
    "split_step",
    "save_every",
    "heldout_eval_interval",
}
_WANDB_KEYS = {"project", "entity", "run_id"}
_PATH_KEYS = {"source_root", "learner_output", "success_marker", "success_checksum"}
_SUCCESS_KEYS = {
    "schema",
    "status",
    "smoke_identity_sha256",
    "backend_contract_sha256",
    "request_sha256",
    "semantic_attempt",
    "world_size",
    "max_steps",
    "heldout_cases",
    "report_path",
    "report_file_sha256",
    "wandb_readback_path",
    "wandb_readback_file_sha256",
    "wandb_run_path",
}
_SHA = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class LeptonSmokeRequest:
    sha256: str
    document: Mapping[str, Any]
    identity_sha256: str
    execution: Mapping[str, Any]
    wandb: Mapping[str, str]
    paths: Mapping[str, PurePosixPath]

    @property
    def wandb_run_id(self) -> str:
        return self.wandb["run_id"]


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise LeptonJobError(f"{label} has an unexpected key-set")
    return dict(value)


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA for character in value)
    ):
        raise LeptonJobError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise LeptonJobError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle, object_pairs_hook=pairs)
    except LeptonJobError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LeptonJobError("cannot read bounded-smoke JSON") from exc
    if not isinstance(document, Mapping):
        raise LeptonJobError("bounded-smoke JSON must be an object")
    return dict(document)


def _regular_json(
    value: str | os.PathLike[str], *, immutable: bool
) -> tuple[Path, dict[str, Any], str]:
    path = Path(os.path.abspath(Path(value).expanduser()))
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise LeptonJobError("bounded-smoke request does not exist") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise LeptonJobError("bounded-smoke request must be a regular non-symlink")
    if immutable and stat.S_IMODE(info.st_mode) != 0o444:
        raise LeptonJobError("bounded-smoke request must have mode 0444")
    return path, _strict_json(path), sha256_file(path)


def _mkdir_beneath(root: Path, target: Path) -> None:
    """Create ``target`` beneath ``root`` without following any symlink."""

    root = Path(os.path.abspath(root))
    target = Path(os.path.abspath(target))
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise LeptonJobError("bounded-smoke output escapes the project root") from exc
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise LeptonJobError("bounded-smoke project root is unsafe") from exc
    try:
        for component in relative.parts:
            try:
                os.mkdir(component, mode=0o750, dir_fd=descriptor)
            except FileExistsError:
                pass
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise LeptonJobError(
                    "bounded-smoke output path contains an unsafe component"
                ) from exc
            os.close(descriptor)
            descriptor = child
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise LeptonJobError(
                    "bounded-smoke output path contains a non-directory"
                )
    finally:
        os.close(descriptor)


def smoke_entrypoint(contract: LeptonBackendContract) -> PurePosixPath:
    return contract.source_root / SMOKE_ENTRYPOINT_RELATIVE


def smoke_contract_view(contract: LeptonBackendContract) -> LeptonBackendContract:
    """Return a Job-building view with the inventory-bound smoke entrypoint."""

    return replace(contract, entrypoint=smoke_entrypoint(contract))


def _execution() -> dict[str, Any]:
    return {
        "campaign_id": SMOKE_CAMPAIGN_ID,
        "world_size": SMOKE_WORLD_SIZE,
        "device": "cuda",
        "heldout_cases": SMOKE_HELDOUT_CASES,
        "max_steps": SMOKE_MAX_STEPS,
        "split_step": SMOKE_SPLIT_STEP,
        "save_every": SMOKE_SAVE_EVERY,
        "heldout_eval_interval": SMOKE_HELDOUT_EVAL_INTERVAL,
    }


def _identity_body(contract: LeptonBackendContract) -> dict[str, Any]:
    return {
        "schema": SMOKE_REQUEST_SCHEMA,
        "backend_contract_sha256": contract.sha256,
        "execution": _execution(),
        "wandb": {
            "project": contract.document["wandb"]["project"],
            "entity": contract.document["wandb"]["entity"],
        },
        "entrypoint": str(smoke_entrypoint(contract)),
    }


def smoke_identity_sha256(contract: LeptonBackendContract) -> str:
    return _canonical_json_sha256(_identity_body(contract))


def _expected_document(contract: LeptonBackendContract) -> dict[str, Any]:
    identity = smoke_identity_sha256(contract)
    output = (
        contract.project_root / SMOKE_OUTPUT_BASE_RELATIVE / identity
    )
    marker = output / "lepton_learner_success.json"
    return {
        "schema": SMOKE_REQUEST_SCHEMA,
        "backend_contract_sha256": contract.sha256,
        "smoke_identity_sha256": identity,
        "execution": _execution(),
        "wandb": {
            "project": contract.document["wandb"]["project"],
            "entity": contract.document["wandb"]["entity"],
            "run_id": f"{SMOKE_WANDB_RUN_PREFIX}{identity}",
        },
        "paths": {
            "source_root": str(contract.source_root),
            "learner_output": str(output),
            "success_marker": str(marker),
            "success_checksum": f"{marker}.sha256",
        },
    }


def load_smoke_request(
    path: str | os.PathLike[str],
    *,
    contract: LeptonBackendContract,
    immutable: bool = False,
) -> LeptonSmokeRequest:
    _path, raw, digest = _regular_json(path, immutable=immutable)
    document = _exact(raw, _REQUEST_KEYS, "bounded-smoke request")
    expected = _expected_document(contract)
    if document != expected:
        raise LeptonJobError(
            "bounded-smoke request differs from the fixed four-step validation contract"
        )
    execution = _exact(document["execution"], _EXECUTION_KEYS, "smoke execution")
    wandb = _exact(document["wandb"], _WANDB_KEYS, "smoke W&B")
    paths_raw = _exact(document["paths"], _PATH_KEYS, "smoke paths")
    paths = {key: PurePosixPath(value) for key, value in paths_raw.items()}
    return LeptonSmokeRequest(
        sha256=digest,
        document=document,
        identity_sha256=_sha(
            document["smoke_identity_sha256"], "bounded-smoke identity"
        ),
        execution=execution,
        wandb={key: str(value) for key, value in wandb.items()},
        paths=paths,
    )


def materialize_smoke_request(
    output: str | os.PathLike[str], *, contract: LeptonBackendContract
) -> LeptonSmokeRequest:
    destination = Path(os.path.abspath(Path(output).expanduser()))
    if destination.exists() or destination.is_symlink():
        raise LeptonJobError("bounded-smoke request destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise LeptonJobError("bounded-smoke request parent is unsafe")
    payload = (
        json.dumps(_expected_document(contract), sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise LeptonJobError(
                "bounded-smoke request destination appeared concurrently"
            ) from exc
        os.chmod(destination, 0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return load_smoke_request(destination, contract=contract, immutable=True)


def _validate_smoke_report(
    path: Path, *, request: LeptonSmokeRequest
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise LeptonJobError("bounded-smoke report is missing, mutable, or unsafe")
    report = _strict_json(path)
    expected = {
        "schema": "fast_roborender.round_dagger.lepton_h200_learner_smoke.v1",
        "status": "PASS",
        "campaign_id": SMOKE_CAMPAIGN_ID,
        "world_size": SMOKE_WORLD_SIZE,
        "device": "cuda",
        "max_steps": SMOKE_MAX_STEPS,
        "split_step": SMOKE_SPLIT_STEP,
        "heldout_cases": SMOKE_HELDOUT_CASES,
        "restart_semantic_equivalent": True,
        "restart_results_equivalent": True,
        "wandb_enabled": True,
        "wandb_project": request.wandb["project"],
        "wandb_entity": request.wandb["entity"],
        "wandb_run_id": request.wandb_run_id,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise LeptonJobError(f"bounded-smoke report {key} mismatch")
    declared = _sha(report.get("report_sha256"), "bounded-smoke report identity")
    body = dict(report)
    body.pop("report_sha256", None)
    if _canonical_json_sha256(body) != declared:
        raise LeptonJobError("bounded-smoke report semantic identity mismatch")
    quality = report.get("student_quality")
    if not isinstance(quality, Mapping):
        raise LeptonJobError("bounded-smoke student quality is missing")
    try:
        first = float(quality["heldout_action_mse_first"])
        last = float(quality["heldout_action_mse_last"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LeptonJobError("bounded-smoke student quality is malformed") from exc
    if not math.isfinite(first) or not math.isfinite(last) or not last < first:
        raise LeptonJobError("bounded-smoke student quality did not improve")
    return report


def _launch_smoke(
    *,
    contract: LeptonBackendContract,
    request: LeptonSmokeRequest,
    output: Path,
    runner: Callable[..., Any] = subprocess.run,
) -> None:
    command = [
        str(contract.python),
        str(contract.source_root / SMOKE_SCRIPT_RELATIVE),
        "--output-root",
        str(output),
        "--world-size",
        str(SMOKE_WORLD_SIZE),
        "--device",
        "cuda",
        "--heldout-cases",
        str(SMOKE_HELDOUT_CASES),
        "--max-steps",
        str(SMOKE_MAX_STEPS),
        "--split-step",
        str(SMOKE_SPLIT_STEP),
        "--save-every",
        str(SMOKE_SAVE_EVERY),
        "--heldout-eval-interval",
        str(SMOKE_HELDOUT_EVAL_INTERVAL),
        "--wandb-project",
        request.wandb["project"],
        "--wandb-entity",
        request.wandb["entity"],
        "--wandb-run-id",
        request.wandb_run_id,
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(contract.source_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "WANDB_MODE": "online",
            "WANDB_SILENT": "true",
            "WANDB_DIR": str(output / "wandb-runtime"),
            "WANDB_CACHE_DIR": str(output / "wandb-cache"),
            "WANDB_CONFIG_DIR": str(output / "wandb-config"),
        }
    )
    completed = runner(
        command,
        cwd=str(contract.source_root),
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise LeptonJobError(
            f"bounded H200 learner smoke exited with status {completed.returncode}"
        )


def _verify_wandb_readback(
    *,
    request: LeptonSmokeRequest,
    output: Path,
    attempts: int = 12,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, Any]:
    from scripts.round_dagger.lepton_h200_learner_smoke import _sha as smoke_sha
    from scripts.round_dagger.validate_learner_wandb_readback import validate

    train = output / "resumed/telemetry.jsonl"
    heldout = output / "resumed/heldout_telemetry.jsonl"
    report_path = output / "wandb_readback.json"
    error: Exception | None = None
    for index in range(attempts):
        try:
            import wandb

            return validate(
                run_path=(
                    f"{request.wandb['entity']}/{request.wandb['project']}/"
                    f"{request.wandb_run_id}"
                ),
                train_telemetry=train,
                heldout_telemetry=heldout,
                expected_config={
                    "campaign_id": SMOKE_CAMPAIGN_ID,
                    "learner_code_sha256": smoke_sha(
                        "lepton-h200-smoke-learner-code-v1"
                    ),
                },
                output=report_path,
                api=wandb.Api(timeout=60),
            )
        except Exception as exc:
            error = exc
            if report_path.exists() or report_path.is_symlink():
                break
            if index + 1 < attempts:
                sleep(10)
    raise LeptonJobError("bounded-smoke W&B readback did not converge") from error


def _publish_success(
    *,
    contract: LeptonBackendContract,
    request: LeptonSmokeRequest,
    semantic_attempt: int,
    attempt_output: Path,
) -> dict[str, Any]:
    report_path = attempt_output / "smoke_report.json"
    _validate_smoke_report(report_path, request=request)
    readback_path = attempt_output / "wandb_readback.json"
    readback = _strict_json(readback_path)
    run_path = (
        f"{request.wandb['entity']}/{request.wandb['project']}/"
        f"{request.wandb_run_id}"
    )
    required_readback = {
        "schema": "fast_roborender.round_dagger.learner_wandb_readback.v1",
        "status": "PASS",
        "run_path": run_path,
        "run_state": "finished",
        "exact_training_match": True,
        "exact_heldout_match": True,
        "identity_config_match": True,
    }
    for key, value in required_readback.items():
        if readback.get(key) != value:
            raise LeptonJobError(f"bounded-smoke W&B readback {key} mismatch")
    marker = Path(str(request.paths["success_marker"]))
    if marker.exists() or marker.is_symlink():
        raise LeptonJobError("bounded-smoke success marker already exists")
    document = {
        "schema": SMOKE_SUCCESS_SCHEMA,
        "status": "PASS",
        "smoke_identity_sha256": request.identity_sha256,
        "backend_contract_sha256": contract.sha256,
        "request_sha256": request.sha256,
        "semantic_attempt": semantic_attempt,
        "world_size": SMOKE_WORLD_SIZE,
        "max_steps": SMOKE_MAX_STEPS,
        "heldout_cases": SMOKE_HELDOUT_CASES,
        "report_path": str(report_path.relative_to(marker.parent)),
        "report_file_sha256": sha256_file(report_path),
        "wandb_readback_path": str(readback_path.relative_to(marker.parent)),
        "wandb_readback_file_sha256": sha256_file(readback_path),
        "wandb_run_path": run_path,
    }
    artifacts.publish_immutable_json(marker, document)
    publish_checksum(marker, Path(str(request.paths["success_checksum"])))
    return document


def execute_smoke_request(
    *,
    contract_path: str | os.PathLike[str],
    contract_sha256: str,
    request_path: str | os.PathLike[str],
    request_sha256: str,
    semantic_attempt: int,
    smoke_runner: Callable[..., None] = _launch_smoke,
    readback_runner: Callable[..., Mapping[str, Any]] = _verify_wandb_readback,
) -> dict[str, Any]:
    from round_dagger.lepton_job import load_contract

    contract = load_contract(contract_path, immutable=True)
    if contract.sha256 != _sha(contract_sha256, "backend contract SHA-256"):
        raise LeptonJobError("backend contract file SHA-256 mismatch")
    require_finalized_supply_chain(contract)
    request = load_smoke_request(request_path, contract=contract, immutable=True)
    if request.sha256 != _sha(request_sha256, "bounded-smoke request SHA-256"):
        raise LeptonJobError("bounded-smoke request file SHA-256 mismatch")
    if semantic_attempt != 1:
        raise LeptonJobError("bounded smoke permits exactly semantic attempt 1")
    if not os.environ.get("WANDB_API_KEY"):
        raise LeptonJobError("bounded smoke requires the injected W&B secret")

    # These gates run before output creation, W&B initialization, or CUDA work.
    verify_sealed_contract_source(contract)
    verify_python_environment_freeze(contract)

    learner_output = Path(str(request.paths["learner_output"]))
    _mkdir_beneath(Path(str(contract.project_root)), learner_output)
    attempt_output = learner_output / "attempt_1"
    if attempt_output.exists() or attempt_output.is_symlink():
        raise LeptonJobError("bounded-smoke attempt output already exists")
    smoke_runner(contract=contract, request=request, output=attempt_output)
    _validate_smoke_report(attempt_output / "smoke_report.json", request=request)
    readback_runner(request=request, output=attempt_output)
    return _publish_success(
        contract=contract,
        request=request,
        semantic_attempt=semantic_attempt,
        attempt_output=attempt_output,
    )


def validate_smoke_completion_marker(
    marker: Path, *, request: LeptonSmokeRequest
) -> dict[str, Any]:
    document = _exact(_strict_json(marker), _SUCCESS_KEYS, "smoke completion marker")
    expected = {
        "schema": SMOKE_SUCCESS_SCHEMA,
        "status": "PASS",
        "smoke_identity_sha256": request.identity_sha256,
        "backend_contract_sha256": request.document["backend_contract_sha256"],
        "request_sha256": request.sha256,
        "semantic_attempt": 1,
        "world_size": SMOKE_WORLD_SIZE,
        "max_steps": SMOKE_MAX_STEPS,
        "heldout_cases": SMOKE_HELDOUT_CASES,
        "wandb_run_path": (
            f"{request.wandb['entity']}/{request.wandb['project']}/"
            f"{request.wandb_run_id}"
        ),
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise LeptonJobError(f"smoke completion marker {key} mismatch")
    for key in ("report_file_sha256", "wandb_readback_file_sha256"):
        _sha(document.get(key), f"smoke completion marker {key}")
    if document.get("report_path") != "attempt_1/smoke_report.json":
        raise LeptonJobError("smoke completion report path mismatch")
    if document.get("wandb_readback_path") != "attempt_1/wandb_readback.json":
        raise LeptonJobError("smoke completion W&B readback path mismatch")
    return document
