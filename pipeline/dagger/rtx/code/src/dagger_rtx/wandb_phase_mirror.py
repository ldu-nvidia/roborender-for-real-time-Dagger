"""One-log-per-vector-step W&B mirror for phase-boundary telemetry.

Raven owns ``wandb.init``.  This observer only mirrors the already-small
phase end/error rows; it never receives arrays or changes transaction/wire
state.  W&B failures are remembered but cannot cause a committed simulator
transition to be replayed.
"""

from __future__ import annotations

from collections import defaultdict
import re
import threading
from typing import Any, Mapping


_SAFE_METRIC_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


class WandbPhaseMirrorError(RuntimeError):
    """The pre-initialized Raven W&B run does not match this collection."""


def production_wandb_identity(
    run_id: str,
    production_logging: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve the W&B identity without changing the coordinator run identity.

    Round 1 predates ``logging.wandb_run_id`` and therefore falls back to the
    coordinator ``run_id``.  Materialized later rounds carry an explicit,
    distinct W&B run ID/name while continuing to use the same coordinator run.
    """

    wandb_run_id = production_logging.get("wandb_run_id", run_id)
    wandb_run_name = production_logging.get("wandb_run_name")
    if (
        not isinstance(wandb_run_id, str)
        or not wandb_run_id
        or not isinstance(wandb_run_name, str)
        or not wandb_run_name
        or wandb_run_name != wandb_run_id
    ):
        raise WandbPhaseMirrorError(
            "production logging W&B run ID/name must be identical non-empty strings"
        )
    return wandb_run_id, wandb_run_name


def authorize_collection_wandb(
    *,
    mode: str,
    raw_enabled: str,
    run_id: str,
    production_logging: Mapping[str, Any] | None = None,
) -> bool:
    """Validate collection logging against the immutable execution contract."""

    if raw_enabled not in {"0", "1"}:
        raise WandbPhaseMirrorError(
            "DAGGER_COLLECTION_WANDB_ENABLED must be 0 or 1"
        )
    enabled = raw_enabled == "1"
    if mode == "production":
        if not isinstance(production_logging, Mapping) or type(
            production_logging.get("rtx_collection_wandb")
        ) is not bool:
            raise WandbPhaseMirrorError(
                "production logging.rtx_collection_wandb must be boolean"
            )
        configured = production_logging["rtx_collection_wandb"]
        if enabled != configured:
            raise WandbPhaseMirrorError(
                "DAGGER_COLLECTION_WANDB_ENABLED differs from immutable "
                "production config"
            )
        production_wandb_identity(run_id, production_logging)
        if enabled and production_logging.get("wandb_project") != "dagger":
            raise WandbPhaseMirrorError(
                "production collection W&B requires immutable project dagger"
            )
        return enabled
    if enabled and mode != "single-window":
        raise WandbPhaseMirrorError(
            "collection-side W&B is restricted to single-window or immutable "
            "production config authorization"
        )
    return enabled


def _metric_component(value: str) -> str:
    cleaned = _SAFE_METRIC_COMPONENT.sub("_", value).strip("_")
    return cleaned or "unnamed"


class WandbPhaseMirror:
    """Buffer scalar phase metrics and commit once at each vector barrier."""

    def __init__(
        self,
        wandb_client: Any,
        *,
        run_id: str,
        wandb_run_id: str | None = None,
        wandb_run_name: str | None = None,
    ) -> None:
        run = getattr(wandb_client, "run", None)
        if run is None:
            raise WandbPhaseMirrorError("Raven did not initialize a W&B run")
        expected_wandb_id = wandb_run_id or run_id
        expected_wandb_name = wandb_run_name or expected_wandb_id
        expected = {
            "id": expected_wandb_id,
            "name": expected_wandb_name,
            "project": "dagger",
            "entity": "nvidia",
        }
        for field, value in expected.items():
            if getattr(run, field, None) != value:
                raise WandbPhaseMirrorError(
                    f"Raven W&B {field} mismatch: expected {value!r}"
                )
        self.client = wandb_client
        self.run_id = run_id
        self._lock = threading.RLock()
        self._pending: dict[int, dict[str, int | float]] = defaultdict(dict)
        self._logged_steps: set[int] = set()
        self._error_counts: dict[tuple[int, str], int] = defaultdict(int)
        self._log_failures: list[str] = []
        self._finished = False
        self._finish_succeeded: bool | None = None

    def observe(self, row: Mapping[str, Any]) -> None:
        """Consume one phase row; start/progress records cost only a branch."""

        event = row.get("event")
        if event not in {"end", "error"}:
            return
        if row.get("run_id") != self.run_id:
            raise WandbPhaseMirrorError("phase row run_id does not match W&B run")
        step = int(row["vector_step"])
        phase = _metric_component(str(row["phase"]))
        with self._lock:
            if self._finished or step in self._logged_steps:
                return
            pending = self._pending[step]
            prefix = f"collection/phase/{phase}"
            elapsed_ms = row.get("elapsed_ms")
            if isinstance(elapsed_ms, (int, float)) and not isinstance(elapsed_ms, bool):
                suffix = "elapsed_ms" if event == "end" else "error_elapsed_ms"
                pending[f"{prefix}/{suffix}"] = float(elapsed_ms)
            pending[f"{prefix}/completed"] = int(row["completed"])
            pending[f"{prefix}/total"] = int(row["total"])
            if event == "end":
                pending[f"{prefix}/ended"] = 1
            else:
                token = (step, phase)
                self._error_counts[token] += 1
                pending[f"{prefix}/error_count"] = self._error_counts[token]
            for raw_name, value in row.get("counts", {}).items():
                if isinstance(value, bool):
                    value = int(value)
                if isinstance(value, (int, float)):
                    name = _metric_component(str(raw_name))
                    pending[f"{prefix}/count/{name}"] = value

            is_completed_barrier = (
                event == "end"
                and phase == "vector_step_barrier"
                and int(row["completed"]) == int(row["total"])
            )
            if not is_completed_barrier:
                return
            payload = dict(pending)
            payload["collection/vector_step_completed"] = 1
            payload["collection/vector_step"] = step
            completion_key = (
                "collection/phase/vector_step_barrier/count/completion_fraction"
            )
            if completion_key in payload:
                payload["collection/completion_fraction"] = payload[completion_key]
            for timing_phase in (
                "capture_teacher",
                "prepare_or_exchange",
                "env_step",
                "finalize",
            ):
                source_key = f"collection/phase/{timing_phase}/elapsed_ms"
                if source_key in payload:
                    payload[f"collection/timing/{timing_phase}_ms"] = payload[
                        source_key
                    ]
            payload["collection/errors"] = sum(
                int(value)
                for name, value in payload.items()
                if name.endswith("/error_count")
            )
            payload["collection/retries"] = sum(
                int(value)
                for name, value in payload.items()
                if name.endswith("/count/failed_attempt")
            )
            # Mark before the external call: a retry can never double-log a
            # physically completed vector step, even if W&B itself raises.
            self._logged_steps.add(step)
            self._pending.pop(step, None)
            try:
                self.client.log(payload, step=step, commit=True)
            except BaseException as exc:  # logging must not replay env.step
                self._log_failures.append(f"{type(exc).__name__}: {exc}"[:256])

    def finish(self, *, exit_code: int) -> None:
        """Finish Raven's W&B run exactly once, without changing run outcome."""

        with self._lock:
            if self._finished:
                return
            self._finished = True
            try:
                self.client.finish(exit_code=exit_code, quiet=True)
            except BaseException as exc:
                self._finish_succeeded = False
                self._log_failures.append(f"{type(exc).__name__}: {exc}"[:256])
            else:
                self._finish_succeeded = True

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "logged_vector_steps": len(self._logged_steps),
                "finish_succeeded": self._finish_succeeded,
                "failure_count": len(self._log_failures),
                "failures": list(self._log_failures),
            }


def active_raven_wandb_mirror(
    run_id: str,
    *,
    wandb_run_id: str | None = None,
    wandb_run_name: str | None = None,
) -> WandbPhaseMirror:
    """Bind to the run initialized earlier by Raven's train entry point."""

    import wandb  # imported only for the explicit opt-in path

    return WandbPhaseMirror(
        wandb,
        run_id=run_id,
        wandb_run_id=wandb_run_id,
        wandb_run_name=wandb_run_name,
    )
