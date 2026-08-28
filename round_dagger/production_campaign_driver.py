"""Fail-closed unattended driver for the bounded five-round campaign.

The one-round orchestrator remains the owner of every durable campaign state
transition.  This module only selects the current durable round, publishes a
fresh immutable orchestration configuration for that exact active policy, and
invokes the already validated production launcher.  A restart therefore
resumes the phase recorded by the coordinator; it never guesses which work
finished and never advances state itself.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from round_dagger import artifacts, state
from round_dagger.orchestration import MacroRoundConfig


DRIVER_STATUS_SCHEMA = "fast_roborender.round_dagger.production_driver_status.v1"
DRIVER_BINDING_SCHEMA = "fast_roborender.round_dagger.production_driver_binding.v1"
ORCHESTRATION_SCHEMA = "fast_roborender.round_dagger.orchestration_config.v3"
PRODUCTION_MACRO_ROUNDS = 5
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_SENSITIVE_RE = re.compile(r"(?i)(api[_-]?key|password|secret|access[_-]?token|identity[_-]?token)")
_PHASES = {"READY", "COLLECTING", "COLLECTED", "INGESTED", "TRAINED", "EVALUATED"}


class ProductionDriverError(RuntimeError):
    """The unattended driver could not prove that continuing was safe."""


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ProductionDriverError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if not path.is_absolute():
        raise ProductionDriverError(f"{label} must be absolute")
    return path


def _lstat_regular(path: Path, label: str, *, immutable: bool = False) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise ProductionDriverError(f"{label} does not exist: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ProductionDriverError(f"{label} must be a regular non-symlink file: {path}")
    if immutable and info.st_mode & 0o222:
        raise ProductionDriverError(f"{label} must be non-writable: {path}")


def _lstat_directory(path: Path, label: str) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise ProductionDriverError(f"{label} does not exist: {path}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ProductionDriverError(f"{label} must be a non-symlink directory: {path}")


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    _lstat_regular(path, label, immutable=True)
    observed = artifacts.sha256_file(path)
    if observed != _sha(expected_sha256, f"{label} SHA-256"):
        raise ProductionDriverError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
        )


def _reject_secrets(value: Any, label: str = "input") -> None:
    """Reject credential-shaped keys before they can enter configs/status."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or _SENSITIVE_RE.search(key):
                raise ProductionDriverError(f"{label} contains a credential-shaped key")
            _reject_secrets(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_secrets(child, f"{label}[{index}]")


def _atomic_json(path: Path, document: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _publish_immutable(path: Path, document: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(path, 0o444)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return artifacts.sha256_file(path)


@dataclass(frozen=True)
class DriverInputs:
    campaign_root: Path
    campaign_contract: Path
    campaign_contract_sha256: str
    learner_contract: Path
    learner_contract_sha256: str
    sealed_source_root: Path
    sealed_source_inventory_sha256: str
    evaluation_sealed_source_root: Path
    evaluation_sealed_source_inventory_sha256: str
    launcher: Path
    launcher_sha256: str
    control_root: Path
    heartbeat_seconds: float = 60.0

    def validate(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if self.heartbeat_seconds <= 0:
            raise ProductionDriverError("heartbeat_seconds must be positive")
        _lstat_directory(self.campaign_root, "campaign root")
        if not self.evaluation_sealed_source_root.is_absolute():
            raise ProductionDriverError(
                "evaluation sealed source root must be absolute"
            )
        _lstat_directory(self.evaluation_sealed_source_root, "evaluation sealed source")
        if not self.sealed_source_root.is_absolute():
            raise ProductionDriverError("sealed source root must be absolute")
        for value, label in (
            (self.sealed_source_inventory_sha256, "sealed source inventory"),
            (
                self.evaluation_sealed_source_inventory_sha256,
                "evaluation sealed source inventory",
            ),
        ):
            _sha(value, label)
        _verify_file(
            self.campaign_contract,
            self.campaign_contract_sha256,
            "post-seal runtime campaign contract",
        )
        _verify_file(self.learner_contract, self.learner_contract_sha256, "learner contract")
        _verify_file(self.launcher, self.launcher_sha256, "production launcher")
        repository_root = Path(__file__).resolve().parents[1]
        expected_launcher = (
            repository_root / "scripts/round_dagger/launch_production_macro_round.sh"
        )
        expected_learner = (
            repository_root
            / "configs/round_dagger/production_learner_8xh100_1200_dagger_prod_run1.json"
        )
        # The driver and actor hooks deliberately execute from the live RTX
        # checkout whose revisions are matched to the remote learner seal by
        # production_launch.  Evaluation consumes a separate immutable local
        # seal, which production_launch verifies by inventory before any
        # orchestration begins.  Requiring these roots to be identical makes
        # that intended three-root deployment impossible: a source tree cannot
        # be both the live Git checkout and the read-only evaluation seal.
        if self.launcher != expected_launcher:
            raise ProductionDriverError("launcher must belong to the driver source root")
        if self.learner_contract != expected_learner:
            raise ProductionDriverError(
                "learner contract must belong to the driver source root"
            )
        try:
            learner_contract = artifacts.load_strict_json(self.learner_contract)
            campaign = state.load_campaign(self.campaign_root)
        except Exception as exc:
            raise ProductionDriverError(f"cannot validate immutable inputs: {exc}") from exc
        _reject_secrets(learner_contract, "learner contract")
        invariant_profile = (
            repository_root / "configs/round_dagger/production_campaign_256x512.json"
        )
        try:
            from scripts.round_dagger import production_launch

            materialized_contract, runtime_evidence, _lineage = (
                production_launch.materialize_campaign_contract(
                    invariant_profile,
                    runtime_contract_path=self.campaign_contract,
                    runtime_contract_sha256=self.campaign_contract_sha256,
                )
            )
            contract_campaign_id = materialized_contract["campaign_id"]
            contract_round_count = materialized_contract["topology"][
                "macro_round_count"
            ]
            evaluator_sha = materialized_contract["evaluation"][
                "hydra_config_sha256"
            ]
        except Exception as exc:
            raise ProductionDriverError(
                f"cannot materialize the post-seal runtime campaign contract: {exc}"
            ) from exc
        _reject_secrets(runtime_evidence, "runtime campaign evidence")
        _reject_secrets(materialized_contract, "materialized campaign contract")
        if contract_campaign_id != campaign["campaign_id"]:
            raise ProductionDriverError("runtime contract belongs to another campaign")
        if contract_round_count != PRODUCTION_MACRO_ROUNDS:
            raise ProductionDriverError("runtime contract must bind exactly five macro-rounds")
        if campaign.get("macro_round_count") != PRODUCTION_MACRO_ROUNDS:
            raise ProductionDriverError("durable campaign must bind exactly five macro-rounds")
        _sha(evaluator_sha, "runtime evaluator config")
        try:
            state._check_path_chain(self.control_root)
        except Exception as exc:
            raise ProductionDriverError(f"unsafe driver control root: {exc}") from exc
        self.control_root.mkdir(parents=True, exist_ok=True)
        try:
            state._check_path_chain(self.control_root)
        except Exception as exc:
            raise ProductionDriverError(f"unsafe driver control root: {exc}") from exc
        _lstat_directory(self.control_root, "driver control root")
        return dict(materialized_contract), dict(learner_contract), campaign


class ExclusiveCampaignLock:
    """Process-lifetime non-blocking exclusive lock shared by every driver."""

    def __init__(self, campaign_root: Path):
        self.path = campaign_root / ".production_five_round_driver.lock"
        self._descriptor: int | None = None

    def __enter__(self) -> "ExclusiveCampaignLock":
        try:
            info = os.lstat(self.path)
        except FileNotFoundError:
            info = None
        if info is not None and not stat.S_ISREG(info.st_mode):
            raise ProductionDriverError(f"unsafe driver lock path: {self.path}")
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise ProductionDriverError("another production driver holds the campaign lock") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        self._descriptor = descriptor
        return self

    def __exit__(self, *_: object) -> None:
        if self._descriptor is not None:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None


def _proc_start_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[21])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _refuse_live_orphan(status_path: Path) -> None:
    if not os.path.lexists(status_path):
        return
    try:
        previous = artifacts.load_strict_json(status_path)
    except Exception as exc:
        raise ProductionDriverError(f"cannot inspect previous driver status: {exc}") from exc
    if not isinstance(previous, Mapping) or previous.get("state") != "LAUNCHING":
        return
    pid = previous.get("launcher_pid")
    ticks = previous.get("launcher_start_ticks")
    if type(pid) is int and type(ticks) is int and _proc_start_ticks(pid) == ticks:
        raise ProductionDriverError(
            f"prior launcher process {pid} is still alive; refusing duplicate orchestration"
        )


def _snapshot(campaign_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        campaign = state.load_campaign(campaign_root)
        coordinator = state.load_state(campaign_root)
    except Exception as exc:
        raise ProductionDriverError(f"durable campaign state is invalid: {exc}") from exc
    if campaign.get("macro_round_count") != PRODUCTION_MACRO_ROUNDS:
        raise ProductionDriverError("durable campaign horizon is not five rounds")
    phase = coordinator.get("phase")
    next_round = coordinator.get("next_macro_round")
    if phase not in _PHASES or type(next_round) is not int:
        raise ProductionDriverError("coordinator phase/frontier is invalid")
    if not 0 <= next_round <= PRODUCTION_MACRO_ROUNDS:
        raise ProductionDriverError("coordinator frontier lies outside rounds 0..4")
    if next_round == PRODUCTION_MACRO_ROUNDS and phase != "READY":
        raise ProductionDriverError("completed campaign must be READY")
    expected_open = None if phase == "READY" else next_round
    if coordinator.get("open_macro_round") != expected_open:
        raise ProductionDriverError("coordinator open round is inconsistent with its phase")
    return campaign, coordinator


def _derive_learner_bundle_sha(
    *,
    campaign_root: Path,
    campaign: Mapping[str, Any],
    coordinator: Mapping[str, Any],
    macro_round: int,
    source_inventory_sha256: str,
    learner_contract_path: Path,
) -> str:
    # Import lazily so status inspection and unit tests do not import the heavy
    # production preflight module unless a round config is being built.
    from scripts.round_dagger import production_launch

    active = coordinator["active_policy"]
    checkpoint = campaign_root / active["checkpoint_path"]
    spec = Path(f"{checkpoint}.policy_spec.json")
    learner_contract = production_launch.load_learner_contract(learner_contract_path)
    try:
        return production_launch.derive_learner_bundle_sha256(
            base_policy_spec=spec,
            parent_checkpoint=checkpoint,
            expected_parent_checkpoint_sha256=active["checkpoint_sha256"],
            runtime_policy_sha256=active["runtime_policy_sha256"],
            campaign_id=campaign["campaign_id"],
            macro_round=macro_round,
            code_sha256=source_inventory_sha256,
            learner_contract=learner_contract,
        )
    except Exception as exc:
        raise ProductionDriverError(f"cannot derive per-round learner identity: {exc}") from exc


def _orchestration_document(
    inputs: DriverInputs,
    runtime_contract: Mapping[str, Any],
    campaign: Mapping[str, Any],
    coordinator: Mapping[str, Any],
) -> dict[str, Any]:
    macro_round = coordinator["next_macro_round"]
    if not 0 <= macro_round < PRODUCTION_MACRO_ROUNDS:
        raise ProductionDriverError("cannot materialize a config outside rounds 0..4")
    active = coordinator["active_policy"]
    repository_root = Path(__file__).resolve().parents[1]
    document = {
        "schema": ORCHESTRATION_SCHEMA,
        "campaign_root": str(inputs.campaign_root),
        "actor_root": str(inputs.campaign_root / "actors"),
        "macro_round": macro_round,
        "macro_round_count": PRODUCTION_MACRO_ROUNDS,
        "campaign_contract_sha256": inputs.campaign_contract_sha256,
        "runtime_policy_sha256": active["runtime_policy_sha256"],
        "learner_code_sha256": inputs.sealed_source_inventory_sha256,
        "learner_config_sha256": _derive_learner_bundle_sha(
            campaign_root=inputs.campaign_root,
            campaign=campaign,
            coordinator=coordinator,
            macro_round=macro_round,
            source_inventory_sha256=inputs.sealed_source_inventory_sha256,
            learner_contract_path=inputs.learner_contract,
        ),
        "learner_model_lineage_sha256": active["checkpoint_sha256"],
        "evaluator_config_sha256": runtime_contract["evaluation"]["hydra_config_sha256"],
        "hooks": {
            "collect": {
                "argv": [
                    str(repository_root / "scripts/round_dagger/run_production_actor_cohort_hook.sh")
                ],
                "timeout_seconds": 14400,
            },
            "transfer": None,
            "ingest": None,
            "train": {
                "argv": [
                    str(
                        repository_root
                        / "scripts/round_dagger/submit_oci_nrt_round_learner_and_wait.sh"
                    )
                ],
                "timeout_seconds": 172800,
            },
            "evaluate": {
                "argv": [
                    str(
                        inputs.evaluation_sealed_source_root
                        / "scripts/round_dagger/run_production_paired_evaluation_hook.sh"
                    )
                ],
                "timeout_seconds": 28800,
            },
            "promote": None,
        },
    }
    _reject_secrets(document, "orchestration config")
    return document


def _next_attempt_path(control_root: Path, coordinator: Mapping[str, Any]) -> Path:
    active = coordinator["active_policy"]
    stem = (
        f"macro_round_{coordinator['next_macro_round']:06d}"
        f"_policy_{active['version']:06d}"
        f"_{active['checkpoint_sha256'][:12]}"
    )
    directory = control_root / "orchestration_configs"
    directory.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 1_000_000):
        candidate = directory / f"{stem}_attempt_{attempt:06d}.json"
        if not os.path.lexists(candidate):
            return candidate
    raise ProductionDriverError("orchestration attempt namespace is exhausted")


def materialize_orchestration_config(
    inputs: DriverInputs,
    runtime_contract: Mapping[str, Any],
    campaign: Mapping[str, Any],
    coordinator: Mapping[str, Any],
) -> tuple[Path, str, Path]:
    document = _orchestration_document(inputs, runtime_contract, campaign, coordinator)
    config_path = _next_attempt_path(inputs.control_root, coordinator)
    config_sha = _publish_immutable(config_path, document)
    try:
        loaded = MacroRoundConfig.load(config_path)
    except Exception as exc:
        raise ProductionDriverError(f"published orchestration config is invalid: {exc}") from exc
    if loaded.sha256 != artifacts.canonical_json_sha256(document):
        raise ProductionDriverError("orchestration canonical identity changed after publication")
    binding_path = config_path.with_suffix(".binding.json")
    binding = {
        "schema": DRIVER_BINDING_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "campaign_contract_path": str(inputs.campaign_contract),
        "campaign_contract_sha256": inputs.campaign_contract_sha256,
        "learner_contract_path": str(inputs.learner_contract),
        "learner_contract_sha256": inputs.learner_contract_sha256,
        "launcher_path": str(inputs.launcher),
        "launcher_sha256": inputs.launcher_sha256,
        "macro_round": coordinator["next_macro_round"],
        "active_policy_version": coordinator["active_policy"]["version"],
        "active_policy_checkpoint_sha256": coordinator["active_policy"]["checkpoint_sha256"],
        "orchestration_config_path": str(config_path),
        "orchestration_config_sha256": config_sha,
    }
    _publish_immutable(binding_path, binding)
    return config_path, config_sha, binding_path


def _launcher_argv(inputs: DriverInputs, config_path: Path, *, dry_run: bool) -> list[str]:
    argv = [
        str(inputs.launcher),
        "--orchestration-config",
        str(config_path),
        "--campaign-contract",
        str(inputs.campaign_contract),
        "--campaign-contract-sha256",
        inputs.campaign_contract_sha256,
        "--learner-contract",
        str(inputs.learner_contract),
        "--learner-contract-sha256",
        inputs.learner_contract_sha256,
        "--sealed-source-root",
        str(inputs.sealed_source_root),
        "--sealed-source-inventory-sha256",
        inputs.sealed_source_inventory_sha256,
        "--evaluation-sealed-source-root",
        str(inputs.evaluation_sealed_source_root),
        "--evaluation-sealed-source-inventory-sha256",
        inputs.evaluation_sealed_source_inventory_sha256,
    ]
    if dry_run:
        argv.append("--dry-run")
    if any(_SENSITIVE_RE.search(item) for item in argv):
        raise ProductionDriverError("launcher argv unexpectedly contains a credential-shaped value")
    return argv


def _status_document(
    *,
    inputs: DriverInputs,
    coordinator: Mapping[str, Any],
    driver_state: str,
    config_path: Path | None = None,
    config_sha256: str | None = None,
    binding_path: Path | None = None,
    launcher_pid: int | None = None,
    launcher_start_ticks: int | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": DRIVER_STATUS_SCHEMA,
        "updated_unix_ns": time.time_ns(),
        "driver_pid": os.getpid(),
        "state": driver_state,
        "campaign_root": str(inputs.campaign_root),
        "campaign_contract_path": str(inputs.campaign_contract),
        "campaign_contract_sha256": inputs.campaign_contract_sha256,
        "coordinator_phase": coordinator["phase"],
        "next_macro_round": coordinator["next_macro_round"],
        "macro_round_count": PRODUCTION_MACRO_ROUNDS,
        "completed": (
            coordinator["phase"] == "READY"
            and coordinator["next_macro_round"] == PRODUCTION_MACRO_ROUNDS
        ),
        "active_policy_version": coordinator["active_policy"]["version"],
        "active_policy_checkpoint_sha256": coordinator["active_policy"]["checkpoint_sha256"],
        "orchestration_config_path": None if config_path is None else str(config_path),
        "orchestration_config_sha256": config_sha256,
        "binding_path": None if binding_path is None else str(binding_path),
        "launcher_pid": launcher_pid,
        "launcher_start_ticks": launcher_start_ticks,
        "detail": detail,
    }
    _reject_secrets(document, "driver status")
    return document


def _invoke_launcher(
    argv: Sequence[str],
    *,
    log_path: Path,
    heartbeat_seconds: float,
    on_heartbeat: Callable[[int, int | None], None],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        child = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        start_ticks = _proc_start_ticks(child.pid)
        on_heartbeat(child.pid, start_ticks)
        stop_signal: list[int] = []

        def forward(signum: int, _frame: object) -> None:
            stop_signal.append(signum)
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

        previous_handlers: dict[int, Any] = {}
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.signal(signum, forward)
        try:
            while True:
                try:
                    return_code = child.wait(timeout=heartbeat_seconds)
                    break
                except subprocess.TimeoutExpired:
                    on_heartbeat(child.pid, start_ticks)
            if stop_signal and return_code == 0:
                return 128 + stop_signal[-1]
            return return_code
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def run_driver(
    inputs: DriverInputs,
    *,
    dry_run: bool = False,
    launcher_runner: Callable[..., int] = _invoke_launcher,
) -> dict[str, Any]:
    """Run/resume rounds 0..4, or validate only the current round in dry-run."""

    runtime_contract, _learner_contract, _campaign = inputs.validate()
    status_path = inputs.control_root / "driver_status.json"
    with ExclusiveCampaignLock(inputs.campaign_root):
        _refuse_live_orphan(status_path)
        while True:
            campaign, coordinator = _snapshot(inputs.campaign_root)
            if coordinator["next_macro_round"] == PRODUCTION_MACRO_ROUNDS:
                result = _status_document(
                    inputs=inputs,
                    coordinator=coordinator,
                    driver_state="COMPLETE",
                    detail="all five macro-rounds are durably promoted",
                )
                _atomic_json(status_path, result)
                return result

            config_path, config_sha, binding_path = materialize_orchestration_config(
                inputs, runtime_contract, campaign, coordinator
            )
            initial_round = coordinator["next_macro_round"]
            log_path = config_path.with_suffix(".launcher.log")

            def heartbeat(pid: int, ticks: int | None) -> None:
                # Re-read only for reporting.  The launcher/orchestrator owns all
                # state transitions while it is alive.
                _, current = _snapshot(inputs.campaign_root)
                _atomic_json(
                    status_path,
                    _status_document(
                        inputs=inputs,
                        coordinator=current,
                        driver_state="LAUNCHING",
                        config_path=config_path,
                        config_sha256=config_sha,
                        binding_path=binding_path,
                        launcher_pid=pid,
                        launcher_start_ticks=ticks,
                        detail="production launcher is running",
                    ),
                )

            argv = _launcher_argv(inputs, config_path, dry_run=dry_run)
            return_code = launcher_runner(
                argv,
                log_path=log_path,
                heartbeat_seconds=inputs.heartbeat_seconds,
                on_heartbeat=heartbeat,
            )
            _, after = _snapshot(inputs.campaign_root)
            if return_code != 0:
                failed = _status_document(
                    inputs=inputs,
                    coordinator=after,
                    driver_state="FAILED",
                    config_path=config_path,
                    config_sha256=config_sha,
                    binding_path=binding_path,
                    detail=f"launcher exited nonzero ({return_code}); fail-closed stop",
                )
                _atomic_json(status_path, failed)
                raise ProductionDriverError(f"production launcher exited {return_code}")
            if dry_run:
                if after != coordinator:
                    raise ProductionDriverError("dry-run unexpectedly changed coordinator state")
                validated = _status_document(
                    inputs=inputs,
                    coordinator=after,
                    driver_state="DRY_RUN_VALIDATED",
                    config_path=config_path,
                    config_sha256=config_sha,
                    binding_path=binding_path,
                    detail="current durable round passed launcher preflight without submission",
                )
                _atomic_json(status_path, validated)
                return validated
            if not (
                after["phase"] == "READY"
                and after["next_macro_round"] == initial_round + 1
            ):
                failed = _status_document(
                    inputs=inputs,
                    coordinator=after,
                    driver_state="FAILED",
                    config_path=config_path,
                    config_sha256=config_sha,
                    binding_path=binding_path,
                    detail="launcher returned success without exactly one durable promotion",
                )
                _atomic_json(status_path, failed)
                raise ProductionDriverError(
                    "launcher success did not advance exactly one round to READY"
                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--campaign-contract", type=Path, required=True)
    parser.add_argument("--campaign-contract-sha256", required=True)
    parser.add_argument("--learner-contract", type=Path, required=True)
    parser.add_argument("--learner-contract-sha256", required=True)
    parser.add_argument("--sealed-source-root", type=Path, required=True)
    parser.add_argument("--sealed-source-inventory-sha256", required=True)
    parser.add_argument("--evaluation-sealed-source-root", type=Path, required=True)
    parser.add_argument(
        "--evaluation-sealed-source-inventory-sha256", required=True
    )
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        inputs = DriverInputs(
            campaign_root=_absolute(arguments.campaign_root, "campaign root"),
            campaign_contract=_absolute(arguments.campaign_contract, "campaign contract"),
            campaign_contract_sha256=arguments.campaign_contract_sha256,
            learner_contract=_absolute(arguments.learner_contract, "learner contract"),
            learner_contract_sha256=arguments.learner_contract_sha256,
            sealed_source_root=_absolute(arguments.sealed_source_root, "sealed source root"),
            sealed_source_inventory_sha256=arguments.sealed_source_inventory_sha256,
            evaluation_sealed_source_root=_absolute(
                arguments.evaluation_sealed_source_root,
                "evaluation sealed source root",
            ),
            evaluation_sealed_source_inventory_sha256=(
                arguments.evaluation_sealed_source_inventory_sha256
            ),
            launcher=_absolute(arguments.launcher, "launcher"),
            launcher_sha256=arguments.launcher_sha256,
            control_root=_absolute(arguments.control_root, "control root"),
            heartbeat_seconds=arguments.heartbeat_seconds,
        )
        result = run_driver(inputs, dry_run=arguments.dry_run)
    except ProductionDriverError as exc:
        print(f"production five-round driver failed: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


__all__ = [
    "DriverInputs",
    "ExclusiveCampaignLock",
    "ProductionDriverError",
    "materialize_orchestration_config",
    "run_driver",
]
