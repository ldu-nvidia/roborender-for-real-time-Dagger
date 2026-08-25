"""Fail-closed production orchestration for one round-based DAgger macro-round.

The actor, transfer, learner, and evaluator are deliberately external hooks:
their deployment details differ between RTX and H100 clusters.  Hooks only
produce artifacts.  This module owns every coordinator transition and verifies
the immutable artifact contract after each hook returns.

The production topology is intentionally not configurable: four synchronous,
sequential cohorts cover 64 logical environments each for 300 steps, yielding
exactly 256 environments and 76,800 transitions per macro-round.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from round_dagger import artifacts, handoff, policy_bundle, promotion, replay, state


ORCHESTRATION_CONFIG_SCHEMA = "fast_roborender.round_dagger.orchestration_config.v1"
EVALUATOR_RESULT_SCHEMA = "fast_roborender.round_dagger.evaluator_result.v2"
WANDB_PUBLIC_CONFIG_SCHEMA = "fast_roborender.round_dagger.wandb_public_config.v1"

COHORT_COUNT = 4
ENVIRONMENTS_PER_COHORT = 64
TARGET_ENVIRONMENTS = COHORT_COUNT * ENVIRONMENTS_PER_COHORT
STEPS_PER_ENVIRONMENT = 300
TRANSITIONS_PER_COHORT = ENVIRONMENTS_PER_COHORT * STEPS_PER_ENVIRONMENT
TRANSITIONS_PER_MACRO_ROUND = TARGET_ENVIRONMENTS * STEPS_PER_ENVIRONMENT
EXPECTED_ACTOR_RANKS = (0, 1)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_HOOK_NAMES = ("collect", "transfer", "ingest", "train", "evaluate", "promote")
_REQUIRED_HOOKS = {"collect", "train", "evaluate"}
_SENSITIVE_ARGUMENT_RE = re.compile(
    r"(?i)(?:^--?(?:api[_-]?key|authorization|credential|password|secret|token)"
    r"(?:$|[=:])|(?:api[_-]?key|authorization|credential|password|secret|token)"
    r"\s*(?:=|:))"
)
_LEARNER_RESULT_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "checkpoint_path",
    "checkpoint_sha256",
    "runtime_policy_sha256",
    "policy_spec_path",
    "policy_spec_sha256",
    "policy_bundle_manifest_path",
    "policy_bundle_manifest_sha256",
    "model_lineage_sha256",
    "replay_snapshot_sha256",
    "code_sha256",
    "learner_config_sha256",
    "optimizer_steps",
    "initial_loss",
    "final_loss",
}
_EVALUATOR_RESULT_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "candidate_manifest_sha256",
    "evaluator_config_sha256",
    "evidence_bundle_path",
    "evidence_bundle_sha256",
    "passed",
    "metrics",
}
_PUBLIC_CONFIG_KEYS = {
    "round_dagger/campaign_id",
    "round_dagger/macro_round",
    "round_dagger/cohort_index",
    "round_dagger/cohort_count",
    "round_dagger/environments_per_cohort",
    "round_dagger/target_environments",
    "round_dagger/steps_per_environment",
    "round_dagger/transitions_per_cohort",
    "round_dagger/transitions_per_macro_round",
    "round_dagger/policy_version",
    "round_dagger/policy_checkpoint_sha256",
    "round_dagger/runtime_policy_sha256",
    "round_dagger/teacher_sha256",
    "round_dagger/renderer_sha256",
    "round_dagger/camera_mapping_sha256",
    "round_dagger/training_config_sha256",
    "round_dagger/beta",
    "round_dagger/beta_sha256",
    "round_dagger/orchestration_config_sha256",
}
_PUBLIC_SHA_KEYS = {key for key in _PUBLIC_CONFIG_KEYS if key.endswith("_sha256")}
_PHASE_CODES = {
    "COLLECTING": 1,
    "COLLECTED": 2,
    "INGESTED": 3,
    "TRAINED": 4,
    "EVALUATED": 5,
    "READY": 6,
}


class OrchestrationError(RuntimeError):
    """A macro-round configuration, hook, or postcondition failed closed."""


def _exact_mapping(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise OrchestrationError(
            f"{label} has missing={sorted(keys - actual)} unknown={sorted(actual - keys)}"
        )
    return dict(value)


def _true_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise OrchestrationError(f"{label} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OrchestrationError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise OrchestrationError(f"{label} must be an absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise OrchestrationError(f"{label} must be an absolute path")
    return Path(os.path.abspath(path))


def _safe_relative(value: str, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise OrchestrationError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise OrchestrationError(f"{label} is unsafe")
    return path


@dataclass(frozen=True)
class HookSpec:
    """One argv-only hook. Credentials must be supplied through the environment."""

    argv: tuple[str, ...]
    timeout_seconds: int

    def __post_init__(self) -> None:
        if not isinstance(self.argv, tuple) or not self.argv:
            raise OrchestrationError("hook argv must be a non-empty tuple")
        for index, raw in enumerate(self.argv):
            if not isinstance(raw, str) or not raw or "\x00" in raw or "\n" in raw:
                raise OrchestrationError(f"hook argv[{index}] is invalid")
            if _SENSITIVE_ARGUMENT_RE.search(raw):
                raise OrchestrationError(
                    "hook argv appears to contain a credential; pass secrets "
                    "through the inherited environment"
                )
        if not Path(self.argv[0]).is_absolute():
            raise OrchestrationError("hook argv[0] must be an absolute executable path")
        _true_int(self.timeout_seconds, "hook timeout_seconds", minimum=1)

    @classmethod
    def from_mapping(cls, value: Any, *, label: str) -> "HookSpec":
        document = _exact_mapping(value, {"argv", "timeout_seconds"}, label)
        raw_argv = document["argv"]
        if not isinstance(raw_argv, list) or not raw_argv:
            raise OrchestrationError(f"{label}.argv must be a non-empty JSON list")
        argv: list[str] = []
        for index, raw in enumerate(raw_argv):
            if not isinstance(raw, str) or not raw or "\x00" in raw or "\n" in raw:
                raise OrchestrationError(f"{label}.argv[{index}] is invalid")
            if _SENSITIVE_ARGUMENT_RE.search(raw):
                raise OrchestrationError(
                    f"{label}.argv[{index}] appears to contain a credential; "
                    "pass secrets through the inherited environment"
                )
            argv.append(raw)
        if not Path(argv[0]).is_absolute():
            raise OrchestrationError(f"{label}.argv[0] must be an absolute executable path")
        timeout = _true_int(document["timeout_seconds"], f"{label}.timeout_seconds", minimum=1)
        return cls(tuple(argv), timeout)

    def public_digest(self) -> str:
        return artifacts.canonical_json_sha256(
            {"argv": list(self.argv), "timeout_seconds": self.timeout_seconds}
        )


@dataclass(frozen=True)
class MacroRoundConfig:
    """Strict operator-supplied inputs for one expected macro-round."""

    campaign_root: Path
    actor_root: Path
    macro_round: int
    runtime_policy_sha256: str
    learner_code_sha256: str
    learner_config_sha256: str
    learner_model_lineage_sha256: str
    evaluator_config_sha256: str
    hooks: Mapping[str, HookSpec | None]
    sha256: str

    def __post_init__(self) -> None:
        for path, label in (
            (self.campaign_root, "campaign_root"),
            (self.actor_root, "actor_root"),
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise OrchestrationError(f"{label} must be an absolute Path")
        _true_int(self.macro_round, "macro_round")
        for value, label in (
            (self.runtime_policy_sha256, "runtime_policy_sha256"),
            (self.learner_code_sha256, "learner_code_sha256"),
            (self.learner_config_sha256, "learner_config_sha256"),
            (self.learner_model_lineage_sha256, "learner_model_lineage_sha256"),
            (self.evaluator_config_sha256, "evaluator_config_sha256"),
            (self.sha256, "orchestration config sha256"),
        ):
            _sha256(value, label)
        if not isinstance(self.hooks, Mapping) or set(self.hooks) != set(_HOOK_NAMES):
            raise OrchestrationError("hooks must contain exactly the six orchestration hooks")
        frozen_hooks: dict[str, HookSpec | None] = {}
        for name in _HOOK_NAMES:
            hook = self.hooks[name]
            if hook is None and name in _REQUIRED_HOOKS:
                raise OrchestrationError(f"{name} hook is required")
            if hook is not None and not isinstance(hook, HookSpec):
                raise OrchestrationError(f"{name} hook must be HookSpec or null")
            frozen_hooks[name] = hook
        object.__setattr__(self, "hooks", MappingProxyType(frozen_hooks))

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "MacroRoundConfig":
        source = Path(path).expanduser()
        try:
            raw = artifacts.load_strict_json(source)
        except Exception as exc:
            raise OrchestrationError(
                f"cannot load strict orchestration config ({type(exc).__name__})"
            ) from exc
        document = _exact_mapping(
            raw,
            {
                "schema",
                "campaign_root",
                "actor_root",
                "macro_round",
                "runtime_policy_sha256",
                "learner_code_sha256",
                "learner_config_sha256",
                "learner_model_lineage_sha256",
                "evaluator_config_sha256",
                "hooks",
            },
            "orchestration config",
        )
        if document["schema"] != ORCHESTRATION_CONFIG_SCHEMA:
            raise OrchestrationError("orchestration config schema mismatch")
        raw_hooks = _exact_mapping(document["hooks"], set(_HOOK_NAMES), "hooks")
        hooks: dict[str, HookSpec | None] = {}
        for name in _HOOK_NAMES:
            value = raw_hooks[name]
            if value is None:
                if name in _REQUIRED_HOOKS:
                    raise OrchestrationError(f"{name} hook is required")
                hooks[name] = None
            else:
                hooks[name] = HookSpec.from_mapping(value, label=f"hooks.{name}")
        return cls(
            campaign_root=_absolute_path(document["campaign_root"], "campaign_root"),
            actor_root=_absolute_path(document["actor_root"], "actor_root"),
            macro_round=_true_int(document["macro_round"], "macro_round"),
            runtime_policy_sha256=_sha256(
                document["runtime_policy_sha256"], "runtime_policy_sha256"
            ),
            learner_code_sha256=_sha256(
                document["learner_code_sha256"], "learner_code_sha256"
            ),
            learner_config_sha256=_sha256(
                document["learner_config_sha256"], "learner_config_sha256"
            ),
            learner_model_lineage_sha256=_sha256(
                document["learner_model_lineage_sha256"],
                "learner_model_lineage_sha256",
            ),
            evaluator_config_sha256=_sha256(
                document["evaluator_config_sha256"], "evaluator_config_sha256"
            ),
            hooks=hooks,
            sha256=artifacts.canonical_json_sha256(document),
        )


class HookRunner(Protocol):
    def run(
        self,
        name: str,
        hook: HookSpec,
        *,
        environment: Mapping[str, str],
        workdir: Path,
    ) -> None: ...


class SubprocessHookRunner:
    """Run hooks without a shell and without echoing argv or environment."""

    def run(
        self,
        name: str,
        hook: HookSpec,
        *,
        environment: Mapping[str, str],
        workdir: Path,
    ) -> None:
        executable = Path(hook.argv[0])
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise OrchestrationError(f"{name} hook executable is missing or not executable")
        child_environment = dict(os.environ)
        child_environment.update(environment)
        try:
            completed = subprocess.run(
                list(hook.argv),
                cwd=workdir,
                env=child_environment,
                check=False,
                timeout=hook.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise OrchestrationError(f"{name} hook timed out") from exc
        except OSError as exc:
            raise OrchestrationError(
                f"{name} hook could not start ({type(exc).__name__})"
            ) from exc
        if completed.returncode != 0:
            raise OrchestrationError(
                f"{name} hook failed with exit status {completed.returncode}"
            )


def validate_production_topology(
    campaign: Mapping[str, Any], contract: Mapping[str, Any] | None = None
) -> None:
    """Reject any campaign that is not exactly 4 x 64 x 300."""

    expected = {
        "target_environments": TARGET_ENVIRONMENTS,
        "active_environments": ENVIRONMENTS_PER_COHORT,
        "cohort_count": COHORT_COUNT,
        "steps_per_environment": STEPS_PER_ENVIRONMENT,
    }
    for key, value in expected.items():
        if campaign.get(key) != value:
            raise OrchestrationError(
                f"production topology requires campaign.{key}={value}"
            )
    if contract is None:
        return
    if contract.get("steps_per_environment") != STEPS_PER_ENVIRONMENT:
        raise OrchestrationError("macro-round step count is not exactly 300")
    cohorts = contract.get("cohorts")
    expected_cohorts = [
        {
            "cohort_index": index,
            "environment_start": index * ENVIRONMENTS_PER_COHORT,
            "environment_count": ENVIRONMENTS_PER_COHORT,
        }
        for index in range(COHORT_COUNT)
    ]
    if cohorts != expected_cohorts:
        raise OrchestrationError(
            "macro-round cohorts do not exactly cover logical environments 0..255"
        )


def _public_wandb_config(
    *,
    config: MacroRoundConfig,
    campaign: Mapping[str, Any],
    contract: Mapping[str, Any],
    cohort_index: int | None,
) -> dict[str, Any]:
    """Return the only orchestration values permitted in W&B run config.

    Paths, hook argv, inherited environment values, account details, and
    credentials are intentionally absent.
    """

    public = {
        "round_dagger/campaign_id": campaign["campaign_id"],
        "round_dagger/macro_round": config.macro_round,
        "round_dagger/cohort_index": -1 if cohort_index is None else cohort_index,
        "round_dagger/cohort_count": COHORT_COUNT,
        "round_dagger/environments_per_cohort": ENVIRONMENTS_PER_COHORT,
        "round_dagger/target_environments": TARGET_ENVIRONMENTS,
        "round_dagger/steps_per_environment": STEPS_PER_ENVIRONMENT,
        "round_dagger/transitions_per_cohort": TRANSITIONS_PER_COHORT,
        "round_dagger/transitions_per_macro_round": TRANSITIONS_PER_MACRO_ROUND,
        "round_dagger/policy_version": contract["policy_version"],
        "round_dagger/policy_checkpoint_sha256": contract[
            "policy_checkpoint_sha256"
        ],
        "round_dagger/runtime_policy_sha256": contract["policy_runtime_sha256"],
        "round_dagger/teacher_sha256": contract["teacher_sha256"],
        "round_dagger/renderer_sha256": contract["renderer_sha256"],
        "round_dagger/camera_mapping_sha256": contract["camera_mapping_sha256"],
        "round_dagger/training_config_sha256": contract["training_config_sha256"],
        "round_dagger/beta": float(contract["beta"]),
        "round_dagger/beta_sha256": contract["beta_sha256"],
        "round_dagger/orchestration_config_sha256": config.sha256,
    }
    return {"schema": WANDB_PUBLIC_CONFIG_SCHEMA, "config": public}


def _validate_public_wandb_config(document: Mapping[str, Any]) -> None:
    wrapper = _exact_mapping(
        document, {"schema", "config"}, "public W&B config wrapper"
    )
    if wrapper["schema"] != WANDB_PUBLIC_CONFIG_SCHEMA:
        raise OrchestrationError("public W&B config schema mismatch")
    values = _exact_mapping(
        wrapper["config"], _PUBLIC_CONFIG_KEYS, "public W&B config"
    )
    campaign_id = values["round_dagger/campaign_id"]
    if (
        not isinstance(campaign_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", campaign_id) is None
    ):
        raise OrchestrationError("public W&B campaign_id is not a safe token")
    for key in _PUBLIC_SHA_KEYS:
        _sha256(values[key], key)
    exact_integers = {
        "round_dagger/cohort_count": COHORT_COUNT,
        "round_dagger/environments_per_cohort": ENVIRONMENTS_PER_COHORT,
        "round_dagger/target_environments": TARGET_ENVIRONMENTS,
        "round_dagger/steps_per_environment": STEPS_PER_ENVIRONMENT,
        "round_dagger/transitions_per_cohort": TRANSITIONS_PER_COHORT,
        "round_dagger/transitions_per_macro_round": TRANSITIONS_PER_MACRO_ROUND,
    }
    for key, expected in exact_integers.items():
        if _true_int(values[key], key, minimum=1) != expected:
            raise OrchestrationError(f"public W&B config {key} violates topology")
    for key in ("round_dagger/macro_round", "round_dagger/policy_version"):
        _true_int(values[key], key)
    cohort_index = values["round_dagger/cohort_index"]
    if type(cohort_index) is not int or cohort_index not in {-1, 0, 1, 2, 3}:
        raise OrchestrationError("public W&B cohort_index must be -1 or 0..3")
    beta = values["round_dagger/beta"]
    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not math.isfinite(float(beta))
        or not 0.0 <= float(beta) <= 1.0
    ):
        raise OrchestrationError("public W&B beta must be finite and in [0,1]")


class RoundTelemetry:
    """Deterministic scalar/config artifacts consumed by the W&B sidecar."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.telemetry_dir = self.run_dir / "telemetry"

    def publish_config(self, document: Mapping[str, Any]) -> Path:
        _validate_public_wandb_config(document)
        path = self.telemetry_dir / "WANDB_PUBLIC_CONFIG.json"
        artifacts.publish_immutable_json(path, document)
        return path

    def record(self, *, step: int, metrics: Mapping[str, int | float]) -> Path:
        clean: dict[str, int | float] = {}
        for key, value in metrics.items():
            if not isinstance(key, str) or not key:
                raise OrchestrationError("telemetry metric keys must be non-empty strings")
            if isinstance(value, bool):
                clean[key] = int(value)
            elif isinstance(value, int):
                clean[key] = value
            elif isinstance(value, float) and math.isfinite(value):
                clean[key] = value
            else:
                raise OrchestrationError(f"telemetry metric {key!r} is not finite numeric")
        payload = {"step": _true_int(step, "telemetry step"), "metrics": clean}
        path = self.telemetry_dir / "scalars" / f"step_{step:09d}_macro_round.json"
        artifacts.publish_immutable_json(path, payload)
        return path


class ProductionBackend:
    """Adapter that composes the existing immutable round_dagger primitives."""

    def __init__(self, config: MacroRoundConfig):
        self.config = config
        self.root = config.campaign_root

    def load_campaign(self) -> dict[str, Any]:
        return state.load_campaign(self.root)

    def load_state(self) -> dict[str, Any]:
        return state.load_state(self.root)

    def open_round(self) -> dict[str, Any]:
        current = self.load_state()
        return state.open_macro_round(self.root, expected_revision=current["revision"])

    def load_contract(self) -> dict[str, Any]:
        return state.load_round_contract(self.root, self.config.macro_round)

    def ensure_cohort_contract(
        self,
        path: Path,
        *,
        campaign: Mapping[str, Any],
        contract: Mapping[str, Any],
        cohort_index: int,
    ) -> None:
        start = cohort_index * ENVIRONMENTS_PER_COHORT
        artifacts.create_cohort(
            path,
            campaign_id=campaign["campaign_id"],
            macro_round=self.config.macro_round,
            cohort_index=cohort_index,
            purpose="train",
            expected_ranks=EXPECTED_ACTOR_RANKS,
            expected_global_env_ids=list(range(start, start + ENVIRONMENTS_PER_COHORT)),
            expected_transitions=TRANSITIONS_PER_COHORT,
            view_contract={
                "dtype": "uint8",
                "layout": "vertical",
                "shape": contract["policy_image_shape"],
                "views": contract["view_names"],
            },
            action_contract={"dtype": "float32", "shape": [contract["action_dim"]]},
            policy_sha256=contract["policy_checkpoint_sha256"],
            runtime_policy_sha256=contract["policy_runtime_sha256"],
            teacher_sha256=contract["teacher_sha256"],
            renderer_sha256=contract["renderer_sha256"],
            beta_sha256=contract["beta_sha256"],
            config_sha256=contract["training_config_sha256"],
        )

    def complete_cohort(self, path: Path) -> dict[str, Any] | None:
        if not (path / artifacts.SEAL_FILENAME).exists():
            return None
        return artifacts.verify_cohort(path)

    def seal_cohort(self, path: Path) -> dict[str, Any]:
        return artifacts.seal_cohort(path)

    def transfer_local(self, source: Path, destination: Path) -> dict[str, Any]:
        return handoff.transfer_local(source, destination)

    def seal_collection(self, cohort_paths: Sequence[Path]) -> dict[str, Any]:
        return state.seal_collection(
            self.root,
            macro_round=self.config.macro_round,
            cohort_manifests=list(cohort_paths),
        )

    def mark_collected(self, collection: Mapping[str, Any]) -> dict[str, Any]:
        current = self.load_state()
        return state.transition(
            self.root,
            expected_phase="COLLECTING",
            next_phase="COLLECTED",
            macro_round=self.config.macro_round,
            evidence_path=collection["manifest_path"],
            evidence_sha256=collection["manifest_sha256"],
            expected_revision=current["revision"],
        )

    def load_collection(self) -> dict[str, Any]:
        path = state.round_directory(self.root, self.config.macro_round) / "collection.json"
        return state.load_collection_manifest(
            self.root, path, expected_macro_round=self.config.macro_round
        )

    def ingest_collection(self, collection: Mapping[str, Any]) -> dict[str, Any]:
        campaign = self.load_campaign()
        ledger_path = self.root / "replay" / "replay.sqlite3"
        with replay.ReplayLedger(ledger_path) as ledger:
            for entry in collection["cohorts"]:
                cohort_path = self.root / entry["cohort_manifest_path"]
                report = artifacts.verify_cohort(cohort_path)
                ledger.ingest_cohort(report["manifest_path"])
            output = (
                self.root
                / "replay"
                / "snapshots"
                / f"through_macro_round_{self.config.macro_round:06d}.json"
            )
            return ledger.build_snapshot(
                campaign["campaign_id"],
                through_macro_round=self.config.macro_round,
                output=output,
                expected_train_cohorts=COHORT_COUNT,
            )

    def load_snapshot(self, *, expected_file_sha256: str | None = None) -> dict[str, Any]:
        path = (
            self.root
            / "replay"
            / "snapshots"
            / f"through_macro_round_{self.config.macro_round:06d}.json"
        )
        return replay.load_snapshot(path, expected_file_sha256=expected_file_sha256)

    def mark_ingested(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        current = self.load_state()
        return state.transition(
            self.root,
            expected_phase="COLLECTED",
            next_phase="INGESTED",
            macro_round=self.config.macro_round,
            evidence_path=snapshot["path"],
            evidence_sha256=snapshot["file_sha256"],
            expected_revision=current["revision"],
        )

    def learner_output_dir(self) -> Path:
        return state.round_directory(self.root, self.config.macro_round) / "learner"

    def learner_result_path(self) -> Path:
        return self.learner_output_dir() / (
            f"learner_result_round_{self.config.macro_round:06d}.json"
        )

    def load_learner_result(
        self,
        *,
        campaign: Mapping[str, Any],
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        path = self.learner_result_path()
        document = _exact_mapping(
            artifacts.load_strict_json(path), _LEARNER_RESULT_KEYS, "learner result"
        )
        if document["schema"] != "fast_roborender.round_dagger.learner_result.v2":
            raise OrchestrationError("learner result schema mismatch")
        expected = {
            "campaign_id": campaign["campaign_id"],
            "macro_round": self.config.macro_round,
            # The learner records the snapshot semantic digest.  Coordinator
            # state separately binds the immutable snapshot file digest.
            "replay_snapshot_sha256": snapshot["snapshot_sha256"],
            "code_sha256": self.config.learner_code_sha256,
            "learner_config_sha256": self.config.learner_config_sha256,
            "model_lineage_sha256": self.config.learner_model_lineage_sha256,
        }
        if any(document.get(key) != value for key, value in expected.items()):
            raise OrchestrationError("learner result lineage mismatch")
        checkpoint_relative = _safe_relative(
            document["checkpoint_path"], "learner checkpoint_path"
        )
        if len(checkpoint_relative.parts) != 1:
            raise OrchestrationError("learner checkpoint_path must be one basename")
        checkpoint = path.parent / checkpoint_relative.as_posix()
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise OrchestrationError("learner checkpoint is missing or unsafe")
        if artifacts.sha256_file(checkpoint) != _sha256(
            document["checkpoint_sha256"], "learner checkpoint_sha256"
        ):
            raise OrchestrationError("learner checkpoint SHA-256 mismatch")
        spec_relative = _safe_relative(
            document["policy_spec_path"], "learner policy_spec_path"
        )
        bundle_relative = _safe_relative(
            document["policy_bundle_manifest_path"],
            "learner policy_bundle_manifest_path",
        )
        if len(spec_relative.parts) != 1 or len(bundle_relative.parts) != 1:
            raise OrchestrationError("learner policy sidecars must be one basename")
        policy_spec = path.parent / spec_relative.as_posix()
        policy_bundle_manifest = path.parent / bundle_relative.as_posix()
        if (
            policy_spec != Path(f"{checkpoint}.policy_spec.json")
            or policy_bundle_manifest != Path(f"{checkpoint}.manifest.json")
            or policy_spec.is_symlink()
            or not policy_spec.is_file()
            or policy_bundle_manifest.is_symlink()
            or not policy_bundle_manifest.is_file()
        ):
            raise OrchestrationError(
                "learner policy sidecar is missing, unsafe, or non-canonical"
            )
        if artifacts.sha256_file(policy_spec) != _sha256(
            document["policy_spec_sha256"], "learner policy_spec_sha256"
        ) or artifacts.sha256_file(policy_bundle_manifest) != _sha256(
            document["policy_bundle_manifest_sha256"],
            "learner policy_bundle_manifest_sha256",
        ):
            raise OrchestrationError("learner policy sidecar SHA-256 mismatch")
        runtime_policy_sha = _sha256(
            document["runtime_policy_sha256"], "learner runtime_policy_sha256"
        )
        try:
            policy_bundle.validate_policy_bundle_identity(
                checkpoint=checkpoint,
                policy_spec=policy_spec,
                manifest=policy_bundle_manifest,
                expected_checkpoint_sha256=document["checkpoint_sha256"],
                expected_runtime_policy_sha256=runtime_policy_sha,
            )
        except policy_bundle.PolicyBundleError as exc:
            raise OrchestrationError(
                f"learner policy bundle identity mismatch: {exc}"
            ) from exc
        _true_int(document["optimizer_steps"], "optimizer_steps", minimum=1)
        for key in ("initial_loss", "final_loss"):
            value = document[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OrchestrationError(f"learner {key} must be finite numeric")
            if not math.isfinite(float(value)):
                raise OrchestrationError(f"learner {key} must be finite numeric")
        return {
            **document,
            "manifest_path": str(path),
            "manifest_sha256": artifacts.sha256_file(path),
            "absolute_checkpoint_path": str(checkpoint),
            "absolute_policy_spec_path": str(policy_spec),
            "absolute_policy_bundle_manifest_path": str(policy_bundle_manifest),
        }

    def stage_candidate(
        self, learner_result: Mapping[str, Any], snapshot: Mapping[str, Any]
    ) -> dict[str, Any]:
        return promotion.stage_candidate(
            self.root,
            macro_round=self.config.macro_round,
            checkpoint_source=learner_result["absolute_checkpoint_path"],
            replay_snapshot_sha256=snapshot["file_sha256"],
            code_sha256=self.config.learner_code_sha256,
            learner_config_sha256=self.config.learner_config_sha256,
            optimizer_steps=learner_result["optimizer_steps"],
        )

    def mark_trained(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        summary = {
            "version": candidate["policy_version"],
            "macro_round": candidate["macro_round"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "checkpoint_path": candidate["checkpoint_path"],
            "runtime_policy_sha256": candidate["runtime_policy_sha256"],
            "policy_spec_sha256": candidate["policy_spec_sha256"],
            "policy_bundle_manifest_sha256": candidate[
                "policy_bundle_manifest_sha256"
            ],
            "manifest_sha256": candidate["manifest_sha256"],
            "manifest_path": candidate["manifest_path"],
            "parent_policy_version": candidate["parent_policy_version"],
            "parent_checkpoint_sha256": candidate["parent_checkpoint_sha256"],
            "replay_snapshot_sha256": candidate["replay_snapshot_sha256"],
            "code_sha256": candidate["code_sha256"],
            "policy_config_sha256": candidate["learner_config_sha256"],
        }
        current = self.load_state()
        return state.transition(
            self.root,
            expected_phase="INGESTED",
            next_phase="TRAINED",
            macro_round=self.config.macro_round,
            evidence_path=candidate["manifest_path"],
            evidence_sha256=candidate["manifest_sha256"],
            candidate_policy=summary,
            expected_revision=current["revision"],
        )

    def load_candidate(self) -> dict[str, Any]:
        current = self.load_state()
        candidate = current.get("candidate_policy")
        if not isinstance(candidate, Mapping):
            raise OrchestrationError("coordinator has no staged candidate")
        return promotion.load_candidate_manifest(
            self.root,
            candidate["manifest_path"],
            expected_macro_round=self.config.macro_round,
            bind_state=True,
        )

    def evaluator_result_path(self) -> Path:
        return (
            state.round_directory(self.root, self.config.macro_round)
            / "evaluation"
            / "result.json"
        )

    def load_evaluator_result(
        self, *, campaign: Mapping[str, Any], candidate: Mapping[str, Any]
    ) -> dict[str, Any]:
        path = self.evaluator_result_path()
        result = _exact_mapping(
            artifacts.load_strict_json(path), _EVALUATOR_RESULT_KEYS, "evaluator result"
        )
        expected = {
            "schema": EVALUATOR_RESULT_SCHEMA,
            "campaign_id": campaign["campaign_id"],
            "macro_round": self.config.macro_round,
            "candidate_manifest_sha256": candidate["manifest_sha256"],
            "evaluator_config_sha256": self.config.evaluator_config_sha256,
        }
        if any(result.get(key) != value for key, value in expected.items()):
            raise OrchestrationError("evaluator result lineage mismatch")
        if not isinstance(result["evidence_bundle_path"], str):
            raise OrchestrationError("evaluation evidence bundle path is unsafe")
        expected_bundle = path.parent / "results" / "evaluation_bundle.json"
        try:
            bundle_path, bundle_relative = state._internal_path(
                self.root, result["evidence_bundle_path"], "evaluation evidence bundle"
            )
        except (artifacts.ArtifactError, OSError, ValueError) as exc:
            raise OrchestrationError("evaluation evidence bundle path is unsafe") from exc
        if (
            bundle_path != expected_bundle
            or result["evidence_bundle_path"] != bundle_relative
        ):
            raise OrchestrationError("evaluation evidence bundle path is non-canonical")
        if (
            not isinstance(result["evidence_bundle_sha256"], str)
            or _SHA256_RE.fullmatch(result["evidence_bundle_sha256"]) is None
        ):
            raise OrchestrationError("evaluation evidence bundle SHA is invalid")
        try:
            actual_bundle_sha = artifacts.sha256_file(bundle_path)
        except (artifacts.ArtifactError, OSError) as exc:
            raise OrchestrationError("evaluation evidence bundle is missing or unsafe") from exc
        if actual_bundle_sha != result["evidence_bundle_sha256"]:
            raise OrchestrationError("evaluation evidence bundle SHA mismatch")
        if type(result["passed"]) is not bool:
            raise OrchestrationError("evaluator result passed must be boolean")
        metrics = result["metrics"]
        if not isinstance(metrics, Mapping) or not metrics:
            raise OrchestrationError("evaluator result metrics must be non-empty")
        clean_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            if not isinstance(key, str) or not key:
                raise OrchestrationError("evaluation metric keys must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OrchestrationError("evaluation metrics must be finite numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise OrchestrationError("evaluation metrics must be finite numeric")
            clean_metrics[key] = numeric
        return {**result, "metrics": clean_metrics, "path": str(path)}

    def publish_evaluation(
        self,
        candidate: Mapping[str, Any],
        evaluator_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        return promotion.write_evaluation(
            self.root,
            candidate_manifest=candidate["manifest_path"],
            passed=evaluator_result["passed"],
            metrics=evaluator_result["metrics"],
            evaluator_config_sha256=self.config.evaluator_config_sha256,
            evidence_bundle_path=evaluator_result["evidence_bundle_path"],
            evidence_bundle_sha256=evaluator_result["evidence_bundle_sha256"],
        )

    def mark_evaluated(self, evidence: Mapping[str, Any]) -> dict[str, Any]:
        current = self.load_state()
        return state.transition(
            self.root,
            expected_phase="TRAINED",
            next_phase="EVALUATED",
            macro_round=self.config.macro_round,
            evidence_path=evidence["manifest_path"],
            evidence_sha256=evidence["evidence_sha256"],
            expected_revision=current["revision"],
        )

    def load_evaluation(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        path = (
            self.root
            / "policies"
            / f"policy_{candidate['policy_version']:06d}"
            / "evaluation.json"
        )
        return promotion.load_evaluation_manifest(
            self.root, path, candidate=candidate
        )

    def promote(
        self, candidate: Mapping[str, Any], evaluation: Mapping[str, Any]
    ) -> dict[str, Any]:
        return promotion.promote_candidate(
            self.root,
            macro_round=self.config.macro_round,
            candidate_manifest=candidate["manifest_path"],
            evaluation_manifest=evaluation["manifest_path"],
        )


def _cohort_environment(
    *,
    config: MacroRoundConfig,
    campaign: Mapping[str, Any],
    contract: Mapping[str, Any],
    cohort_index: int,
    source: Path,
    destination: Path,
    actor_run_dir: Path,
) -> dict[str, str]:
    offset = cohort_index * ENVIRONMENTS_PER_COHORT
    checkpoint = config.campaign_root / contract["policy_checkpoint_path"]
    return {
        "ROUND_DAGGER_CAMPAIGN_ROOT": str(config.campaign_root),
        "ROUND_DAGGER_CAMPAIGN_ID": str(campaign["campaign_id"]),
        "ROUND_DAGGER_MACRO_ROUND": str(config.macro_round),
        "ROUND_DAGGER_COHORT_INDEX": str(cohort_index),
        "ROUND_DAGGER_COHORT_DIR": str(source),
        "ROUND_DAGGER_SOURCE_COHORT": str(source),
        "ROUND_DAGGER_DESTINATION_COHORT": str(destination),
        "ROUND_DAGGER_GLOBAL_ENV_OFFSET": str(offset),
        "ROUND_DAGGER_LOGICAL_ENV_COUNT": str(ENVIRONMENTS_PER_COHORT),
        "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": str(STEPS_PER_ENVIRONMENT),
        "ROUND_DAGGER_EXPECTED_TRANSITIONS": str(TRANSITIONS_PER_COHORT),
        "ROUND_DAGGER_POLICY_CHECKPOINT": str(checkpoint),
        "ROUND_DAGGER_POLICY_SPEC": f"{checkpoint}.policy_spec.json",
        "ROUND_DAGGER_POLICY_MANIFEST": f"{checkpoint}.manifest.json",
        "ROUND_DAGGER_POLICY_SPEC_SHA256": contract["policy_spec_sha256"],
        "ROUND_DAGGER_POLICY_MANIFEST_SHA256": contract[
            "policy_bundle_manifest_sha256"
        ],
        "ROUND_DAGGER_POLICY_SHA256": str(contract["policy_checkpoint_sha256"]),
        "ROUND_DAGGER_RUNTIME_POLICY_SHA256": contract["policy_runtime_sha256"],
        "ROUND_DAGGER_TEACHER_SHA256": str(contract["teacher_sha256"]),
        "ROUND_DAGGER_RENDERER_SHA256": str(contract["renderer_sha256"]),
        "ROUND_DAGGER_BETA": repr(float(contract["beta"])),
        "ROUND_DAGGER_BETA_SHA256": str(contract["beta_sha256"]),
        "ROUND_DAGGER_CONFIG_SHA256": str(contract["training_config_sha256"]),
        "ROUND_DAGGER_ACTOR_RUN_DIR": str(actor_run_dir),
        "TRAIN_RUN_DIR": str(actor_run_dir),
        "ROBORENDER_TELEMETRY_DIR": str(actor_run_dir / "telemetry"),
        "ROLLOUT_STEPS": str(STEPS_PER_ENVIRONMENT),
    }


def _validate_cohort_report(
    report: Mapping[str, Any],
    *,
    campaign: Mapping[str, Any],
    contract: Mapping[str, Any],
    config: MacroRoundConfig,
    cohort_index: int,
) -> None:
    start = cohort_index * ENVIRONMENTS_PER_COHORT
    expected = {
        "campaign_id": campaign["campaign_id"],
        "macro_round": config.macro_round,
        "cohort_index": cohort_index,
        "purpose": "train",
        "sealed": True,
        "expected_ranks": list(EXPECTED_ACTOR_RANKS),
        "ranks": list(EXPECTED_ACTOR_RANKS),
        "expected_global_env_ids": list(
            range(start, start + ENVIRONMENTS_PER_COHORT)
        ),
        "global_env_ids": list(range(start, start + ENVIRONMENTS_PER_COHORT)),
        "expected_transitions": TRANSITIONS_PER_COHORT,
        "transition_count": TRANSITIONS_PER_COHORT,
        "policy_sha256": contract["policy_checkpoint_sha256"],
        "runtime_policy_sha256": contract["policy_runtime_sha256"],
        "teacher_sha256": contract["teacher_sha256"],
        "renderer_sha256": contract["renderer_sha256"],
        "beta_sha256": contract["beta_sha256"],
        "config_sha256": contract["training_config_sha256"],
    }
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    if mismatches:
        raise OrchestrationError(
            f"cohort {cohort_index} failed exact production postconditions: {mismatches}"
        )


class MacroRoundOrchestrator:
    """Run or resume one explicitly numbered production macro-round."""

    def __init__(
        self,
        config: MacroRoundConfig,
        *,
        backend: ProductionBackend | None = None,
        hook_runner: HookRunner | None = None,
    ):
        self.config = config
        self.backend = backend or ProductionBackend(config)
        self.hook_runner = hook_runner or SubprocessHookRunner()

    def _run_hook(self, name: str, environment: Mapping[str, str]) -> None:
        hook = self.config.hooks[name]
        if hook is None:
            return
        before = self.backend.load_state()
        self.hook_runner.run(
            name,
            hook,
            environment=environment,
            workdir=self.config.campaign_root,
        )
        after = self.backend.load_state()
        if after != before:
            raise OrchestrationError(
                f"{name} hook mutated coordinator state; hooks may only produce artifacts"
            )

    def _base_environment(
        self,
        *,
        campaign: Mapping[str, Any],
        contract: Mapping[str, Any],
    ) -> dict[str, str]:
        checkpoint = self.config.campaign_root / contract["policy_checkpoint_path"]
        return {
            "ROUND_DAGGER_CAMPAIGN_ROOT": str(self.config.campaign_root),
            "ROUND_DAGGER_CAMPAIGN_ID": str(campaign["campaign_id"]),
            "ROUND_DAGGER_MACRO_ROUND": str(self.config.macro_round),
            "ROUND_DAGGER_POLICY_CHECKPOINT": str(checkpoint),
            "ROUND_DAGGER_POLICY_SPEC": f"{checkpoint}.policy_spec.json",
            "ROUND_DAGGER_POLICY_MANIFEST": f"{checkpoint}.manifest.json",
            "ROUND_DAGGER_BASE_POLICY_SPEC": f"{checkpoint}.policy_spec.json",
            "ROUND_DAGGER_POLICY_SPEC_SHA256": contract["policy_spec_sha256"],
            "ROUND_DAGGER_POLICY_MANIFEST_SHA256": contract[
                "policy_bundle_manifest_sha256"
            ],
            "ROUND_DAGGER_POLICY_SHA256": str(contract["policy_checkpoint_sha256"]),
            "ROUND_DAGGER_RUNTIME_POLICY_SHA256": contract[
                "policy_runtime_sha256"
            ],
            "ROUND_DAGGER_BETA": repr(float(contract["beta"])),
            "ROUND_DAGGER_BETA_SHA256": str(contract["beta_sha256"]),
            "ROUND_DAGGER_TARGET_ENVIRONMENTS": str(TARGET_ENVIRONMENTS),
            "ROUND_DAGGER_COHORT_COUNT": str(COHORT_COUNT),
            "ROUND_DAGGER_ENVIRONMENTS_PER_COHORT": str(ENVIRONMENTS_PER_COHORT),
            "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": str(STEPS_PER_ENVIRONMENT),
            "ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND": str(
                TRANSITIONS_PER_MACRO_ROUND
            ),
        }

    def _completed_retry(
        self, current: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if current.get("phase") != "READY":
            return None
        next_round = current.get("next_macro_round")
        if next_round == self.config.macro_round:
            return None
        if next_round != self.config.macro_round + 1:
            raise OrchestrationError(
                "configured macro_round is stale or skips coordinator state"
            )
        version = current["active_policy"]["version"]
        candidate_path = (
            self.config.campaign_root
            / "policies"
            / f"policy_{version:06d}"
            / "candidate.json"
        )
        candidate = promotion.load_candidate_manifest(
            self.config.campaign_root,
            candidate_path,
            expected_macro_round=self.config.macro_round,
            bind_state=False,
        )
        evaluation = self.backend.load_evaluation(candidate)
        return self.backend.promote(candidate, evaluation)

    def run(self) -> dict[str, Any]:
        campaign = self.backend.load_campaign()
        validate_production_topology(campaign)
        current = self.backend.load_state()
        completed = self._completed_retry(current)
        if completed is not None:
            return completed

        if current["phase"] == "READY":
            if current["next_macro_round"] != self.config.macro_round:
                raise OrchestrationError("configured macro_round is not the next durable round")
            contract = self.backend.open_round()
        else:
            if current.get("open_macro_round") != self.config.macro_round:
                raise OrchestrationError("another macro-round is already open")
            contract = self.backend.load_contract()
        validate_production_topology(campaign, contract)
        if self.config.runtime_policy_sha256 != contract["policy_runtime_sha256"]:
            raise OrchestrationError(
                "orchestration runtime policy SHA-256 does not match the frozen round contract"
            )
        if (
            self.config.learner_model_lineage_sha256
            != contract["policy_checkpoint_sha256"]
        ):
            raise OrchestrationError(
                "learner model lineage does not match the frozen parent checkpoint"
            )

        round_dir = state.round_directory(
            self.config.campaign_root, self.config.macro_round
        )
        telemetry = RoundTelemetry(round_dir / "orchestration_run")
        telemetry.publish_config(
            _public_wandb_config(
                config=self.config,
                campaign=campaign,
                contract=contract,
                cohort_index=None,
            )
        )
        phase = self.backend.load_state()["phase"]
        telemetry.record(
            step=self.config.macro_round * 100,
            metrics={
                "round/macro_round": self.config.macro_round,
                # This is the immutable round-open baseline.  Keeping its
                # payload independent of the resume phase makes retries
                # byte-identical after a later phase has already committed.
                "round/phase_code": _PHASE_CODES["COLLECTING"],
                "round/cohorts_complete": 0,
                "round/logical_environments_complete": 0,
                "round/transitions_complete": 0,
                "round/beta": float(contract["beta"]),
                "round/policy_version": int(contract["policy_version"]),
            },
        )

        base_environment = self._base_environment(campaign=campaign, contract=contract)
        if phase == "COLLECTING":
            accepted_paths: list[Path] = []
            actor_round = (
                self.config.actor_root
                / str(campaign["campaign_id"])
                / f"macro_round_{self.config.macro_round:06d}"
            )
            accepted_round = round_dir / "cohorts"
            for cohort_index in range(COHORT_COUNT):
                source = actor_round / f"cohort_{cohort_index:02d}"
                actor_run_dir = actor_round / f"cohort_{cohort_index:02d}_run"
                destination = accepted_round / f"cohort_{cohort_index:02d}"
                self.backend.ensure_cohort_contract(
                    source,
                    campaign=campaign,
                    contract=contract,
                    cohort_index=cohort_index,
                )
                RoundTelemetry(actor_run_dir).publish_config(
                    _public_wandb_config(
                        config=self.config,
                        campaign=campaign,
                        contract=contract,
                        cohort_index=cohort_index,
                    )
                )
                environment = {
                    **base_environment,
                    **_cohort_environment(
                        config=self.config,
                        campaign=campaign,
                        contract=contract,
                        cohort_index=cohort_index,
                        source=source,
                        destination=destination,
                        actor_run_dir=actor_run_dir,
                    ),
                }
                source_report = self.backend.complete_cohort(source)
                if source_report is None:
                    self._run_hook("collect", environment)
                    source_report = self.backend.seal_cohort(source)
                _validate_cohort_report(
                    source_report,
                    campaign=campaign,
                    contract=contract,
                    config=self.config,
                    cohort_index=cohort_index,
                )

                accepted_report = self.backend.complete_cohort(destination)
                if accepted_report is None:
                    if self.config.hooks["transfer"] is None:
                        accepted_report = self.backend.transfer_local(source, destination)
                    else:
                        self._run_hook("transfer", environment)
                        accepted_report = self.backend.complete_cohort(destination)
                        if accepted_report is None:
                            raise OrchestrationError(
                                "transfer hook returned without publishing a sealed destination"
                            )
                _validate_cohort_report(
                    accepted_report,
                    campaign=campaign,
                    contract=contract,
                    config=self.config,
                    cohort_index=cohort_index,
                )
                if accepted_report["cohort_sha256"] != source_report["cohort_sha256"]:
                    raise OrchestrationError("transferred cohort digest differs from source")
                accepted_paths.append(destination)
                completed_cohorts = cohort_index + 1
                telemetry.record(
                    step=self.config.macro_round * 100 + completed_cohorts,
                    metrics={
                        "round/macro_round": self.config.macro_round,
                        "round/phase_code": _PHASE_CODES["COLLECTING"],
                        "round/cohort_index": cohort_index,
                        "round/cohorts_complete": completed_cohorts,
                        "round/logical_environments_complete": completed_cohorts
                        * ENVIRONMENTS_PER_COHORT,
                        "round/transitions_complete": completed_cohorts
                        * TRANSITIONS_PER_COHORT,
                    },
                )

            collection = self.backend.seal_collection(accepted_paths)
            if collection.get("cohort_count") != COHORT_COUNT or collection.get(
                "transition_count"
            ) != TRANSITIONS_PER_MACRO_ROUND:
                raise OrchestrationError("immutable collection aggregate is not exactly 256x300")
            self.backend.mark_collected(collection)
            telemetry.record(
                step=self.config.macro_round * 100 + 10,
                metrics={
                    "round/macro_round": self.config.macro_round,
                    "round/phase_code": _PHASE_CODES["COLLECTED"],
                    "round/cohorts_complete": COHORT_COUNT,
                    "round/logical_environments_complete": TARGET_ENVIRONMENTS,
                    "round/transitions_complete": TRANSITIONS_PER_MACRO_ROUND,
                },
            )

        current = self.backend.load_state()
        if current["phase"] == "COLLECTED":
            collection = self.backend.load_collection()
            ingest_environment = {
                **base_environment,
                "ROUND_DAGGER_COLLECTION_MANIFEST": str(
                    self.config.campaign_root / collection["manifest_path"]
                ),
                "ROUND_DAGGER_REPLAY_LEDGER": str(
                    self.config.campaign_root / "replay" / "replay.sqlite3"
                ),
            }
            self._run_hook("ingest", ingest_environment)
            snapshot = self.backend.ingest_collection(collection)
            expected_cumulative = (self.config.macro_round + 1) * TRANSITIONS_PER_MACRO_ROUND
            if snapshot.get("expected_train_cohorts") != COHORT_COUNT or snapshot.get(
                "transition_count"
            ) != expected_cumulative:
                raise OrchestrationError("cumulative replay snapshot coverage is incomplete")
            self.backend.mark_ingested(snapshot)
            telemetry.record(
                step=self.config.macro_round * 100 + 20,
                metrics={
                    "round/macro_round": self.config.macro_round,
                    "round/phase_code": _PHASE_CODES["INGESTED"],
                    "round/replay_transitions": int(snapshot["transition_count"]),
                    "round/replay_cohorts": int(snapshot["cohort_count"]),
                },
            )

        current = self.backend.load_state()
        if current["phase"] == "INGESTED":
            snapshot = self.backend.load_snapshot(
                expected_file_sha256=current["replay_snapshot_sha256"]
            )
            learner_output = self.backend.learner_output_dir()
            train_environment = {
                **base_environment,
                "ROUND_DAGGER_REPLAY_SNAPSHOT": str(snapshot["path"]),
                "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256": str(
                    snapshot["file_sha256"]
                ),
                "ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256": str(
                    snapshot["snapshot_sha256"]
                ),
                "ROUND_DAGGER_LEARNER_OUTPUT_DIR": str(learner_output),
                "ROUND_DAGGER_LEARNER_RESULT": str(
                    self.backend.learner_result_path()
                ),
                "ROUND_DAGGER_LEARNER_CODE_SHA256": self.config.learner_code_sha256,
                "ROUND_DAGGER_LEARNER_CONFIG_SHA256": self.config.learner_config_sha256,
                "ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256": self.config.learner_model_lineage_sha256,
            }
            if not self.backend.learner_result_path().exists():
                self._run_hook("train", train_environment)
            learner_result = self.backend.load_learner_result(
                campaign=campaign, snapshot=snapshot
            )
            candidate = self.backend.stage_candidate(learner_result, snapshot)
            self.backend.mark_trained(candidate)
            telemetry.record(
                step=self.config.macro_round * 100 + 30,
                metrics={
                    "round/macro_round": self.config.macro_round,
                    "round/phase_code": _PHASE_CODES["TRAINED"],
                    "round/policy_version": int(candidate["policy_version"]),
                    "round/optimizer_steps": int(candidate["optimizer_steps"]),
                    "round/initial_loss": float(learner_result["initial_loss"]),
                    "round/final_loss": float(learner_result["final_loss"]),
                },
            )

        current = self.backend.load_state()
        if current["phase"] == "TRAINED":
            candidate = self.backend.load_candidate()
            evaluator_result_path = self.backend.evaluator_result_path()
            evaluate_environment = {
                **base_environment,
                "ROUND_DAGGER_CANDIDATE_MANIFEST": str(
                    self.config.campaign_root / candidate["manifest_path"]
                ),
                "ROUND_DAGGER_CANDIDATE_SHA256": str(candidate["manifest_sha256"]),
                "ROUND_DAGGER_EVALUATOR_CONFIG_SHA256": self.config.evaluator_config_sha256,
                "ROUND_DAGGER_EVALUATOR_RESULT": str(evaluator_result_path),
            }
            if not evaluator_result_path.exists():
                self._run_hook("evaluate", evaluate_environment)
            evaluator_result = self.backend.load_evaluator_result(
                campaign=campaign, candidate=candidate
            )
            evidence = self.backend.publish_evaluation(candidate, evaluator_result)
            self.backend.mark_evaluated(evidence)
            telemetry.record(
                step=self.config.macro_round * 100 + 40,
                metrics={
                    "round/macro_round": self.config.macro_round,
                    "round/phase_code": _PHASE_CODES["EVALUATED"],
                    "round/evaluation_passed": int(evaluator_result["passed"]),
                    **{
                        f"evaluation/{key}": value
                        for key, value in evaluator_result["metrics"].items()
                    },
                },
            )

        current = self.backend.load_state()
        if current["phase"] != "EVALUATED":
            raise OrchestrationError(
                f"macro-round stopped in unexpected phase {current['phase']}"
            )
        candidate = self.backend.load_candidate()
        evaluation = self.backend.load_evaluation(candidate)
        promote_environment = {
            **base_environment,
            "ROUND_DAGGER_CANDIDATE_MANIFEST": str(
                self.config.campaign_root / candidate["manifest_path"]
            ),
            "ROUND_DAGGER_EVALUATION_MANIFEST": str(
                self.config.campaign_root / evaluation["manifest_path"]
            ),
            "ROUND_DAGGER_EVALUATION_SHA256": str(evaluation["evidence_sha256"]),
        }
        self._run_hook("promote", promote_environment)
        result = self.backend.promote(candidate, evaluation)
        final_state = self.backend.load_state()
        if (
            final_state.get("phase") != "READY"
            or final_state.get("next_macro_round") != self.config.macro_round + 1
            or final_state.get("active_policy", {}).get("version")
            != candidate["policy_version"]
        ):
            raise OrchestrationError("promotion did not atomically advance active policy")
        telemetry.record(
            step=self.config.macro_round * 100 + 50,
            metrics={
                "round/macro_round": self.config.macro_round,
                "round/phase_code": _PHASE_CODES["READY"],
                "round/evaluation_passed": 1,
                "round/policy_version": int(candidate["policy_version"]),
                "round/cohorts_complete": COHORT_COUNT,
                "round/logical_environments_complete": TARGET_ENVIRONMENTS,
                "round/transitions_complete": TRANSITIONS_PER_MACRO_ROUND,
            },
        )
        return result


__all__ = [
    "ORCHESTRATION_CONFIG_SCHEMA",
    "EVALUATOR_RESULT_SCHEMA",
    "WANDB_PUBLIC_CONFIG_SCHEMA",
    "COHORT_COUNT",
    "ENVIRONMENTS_PER_COHORT",
    "TARGET_ENVIRONMENTS",
    "STEPS_PER_ENVIRONMENT",
    "TRANSITIONS_PER_COHORT",
    "TRANSITIONS_PER_MACRO_ROUND",
    "EXPECTED_ACTOR_RANKS",
    "OrchestrationError",
    "HookSpec",
    "MacroRoundConfig",
    "HookRunner",
    "SubprocessHookRunner",
    "RoundTelemetry",
    "ProductionBackend",
    "validate_production_topology",
    "MacroRoundOrchestrator",
]
