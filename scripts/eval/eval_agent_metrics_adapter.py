#!/usr/bin/env python3
"""Run GR00T's evaluator while durably capturing ``trainer.eval()`` metrics.

The upstream evaluator currently discards the mapping returned by
``trainer.eval()`` and then calls ``os._exit``.  Some trainer variants happen
to write ``metrics_eval.json`` themselves and others do not.  This adapter
applies one deliberately narrow source transformation: the single standalone
``trainer.eval()`` call is replaced with a captured return value and an atomic
write callback.  Every other evaluator statement, including Isaac startup,
Hydra configuration, checkpoint loading, and shutdown, executes unchanged.

Any evaluator source that no longer matches the versioned structural contract
is rejected before Isaac is launched.  That makes evaluation fail closed
rather than silently accepting a missing or stale metrics file.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ADAPTER_SCHEMA_VERSION = 1
SAFE_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
# The accepted production camera/capture lineage pins the nested GR00T tree at
# 9dd746d.  That tree predates a small evaluator-only replay cache update: in
# paired first-condition replay the server binds the substituted previous-RGB
# triplet at step zero, whereas the trainer otherwise retains its live anchor
# for step one.  Keep the nested source immutable and install the exact
# compatibility behavior only in the outer fixed-evaluation adapter, after
# verifying the target source byte-for-byte.
PAIRED_REPLAY_TRAINER_TYPE = "TRLDistillTrainerVideoGen"
PAIRED_REPLAY_TRAINER_SHA256 = (
    "24a15d90c81d2aa813447c5cc6b6413566e6bda7849e4b62c5af484dd8f15db4"
)
EVAL_CALL_RE = re.compile(
    r"^(?P<indent>[ \t]*)trainer\.eval\(\)[ \t]*$", re.MULTILINE
)
REQUIRED_SOURCE_MARKERS = (
    "@hydra.main",
    "trainer = custom_instantiate(",
    'logger.info("Finished evaluation")',
    'if __name__ == "__main__":',
)


class AdapterError(RuntimeError):
    """The evaluator or returned metrics violated the fixed-eval contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def patch_evaluator_source(source: str) -> str:
    """Capture exactly one standalone trainer.eval() call, or fail closed."""

    missing = [marker for marker in REQUIRED_SOURCE_MARKERS if marker not in source]
    if missing:
        raise AdapterError(
            "unsupported evaluator source; missing contract marker(s): "
            + ", ".join(missing)
        )
    matches = list(EVAL_CALL_RE.finditer(source))
    if len(matches) != 1:
        raise AdapterError(
            "unsupported evaluator source; expected exactly one standalone "
            f"trainer.eval() call, found {len(matches)}"
        )
    match = matches[0]
    indent = match.group("indent")
    replacement = (
        f"{indent}# RoboRender fixed-eval metrics adapter schema "
        f"{ADAPTER_SCHEMA_VERSION}\n"
        f"{indent}__roborender_replay_anchor_compat = "
        f"__roborender_install_replay_anchor_compat(trainer)\n"
        f"{indent}__roborender_eval_metrics = trainer.eval()\n"
        f"{indent}__roborender_record_eval_metrics(\n"
        f"{indent}    __roborender_eval_metrics,\n"
        f"{indent}    replay_anchor_compat=__roborender_replay_anchor_compat,\n"
        f"{indent})"
    )
    return source[: match.start()] + replacement + source[match.end() :]


def _paired_replay_compat_requested() -> bool:
    """Return whether this exact evaluator invocation needs the 9dd shim."""

    return (
        os.environ.get("IMAGE_DAGGER_PAIRED_EVAL") == "1"
        and os.environ.get("IMAGE_DAGGER_FIRST_CONDITION_MODE") == "replay"
    )


def _trainer_source_sha256(trainer: Any) -> str:
    """Hash the concrete trainer implementation, rejecting ambiguous targets."""

    if type(trainer).__name__ != PAIRED_REPLAY_TRAINER_TYPE:
        raise AdapterError(
            "paired replay compatibility target is not "
            f"{PAIRED_REPLAY_TRAINER_TYPE}"
        )
    try:
        raw_path = inspect.getsourcefile(type(trainer))
    except (OSError, TypeError) as exc:
        raise AdapterError("cannot locate paired replay trainer source") from exc
    if not isinstance(raw_path, str) or not raw_path:
        raise AdapterError("cannot locate paired replay trainer source")
    source_path = Path(raw_path)
    if source_path.is_symlink() or not source_path.is_file():
        raise AdapterError("paired replay trainer source is unsafe or missing")
    try:
        digest = sha256_bytes(source_path.read_bytes())
    except OSError as exc:
        raise AdapterError("cannot read paired replay trainer source") from exc
    if digest != PAIRED_REPLAY_TRAINER_SHA256:
        raise AdapterError(
            "paired replay compatibility source digest differs from accepted "
            "GR00T 9dd implementation"
        )
    return digest


def _copy_replay_anchor_triplet(
    trainer: Any, *, environment_index: int
) -> dict[str, Any]:
    """Validate and copy one canonical previous-RGB triplet before commit."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - production requires NumPy
        raise AdapterError("paired replay compatibility requires NumPy") from exc
    bundle = getattr(trainer, "_first_condition_bundle", None)
    if not isinstance(bundle, Mapping):
        raise AdapterError("paired replay compatibility bundle is unavailable")
    requests = bundle.get("requests_by_env")
    if not isinstance(requests, Mapping):
        raise AdapterError("paired replay compatibility requests are malformed")
    try:
        request = requests[environment_index]
    except (KeyError, TypeError) as exc:
        raise AdapterError(
            "paired replay compatibility request is missing for environment "
            f"{environment_index}"
        ) from exc
    if not isinstance(request, (tuple, list)) or len(request) != 3:
        raise AdapterError("paired replay compatibility request has invalid groups")
    previous = request[2]
    view_names = tuple(getattr(trainer, "vg_view_names", ()))
    if not view_names or any(not isinstance(view, str) or not view for view in view_names):
        raise AdapterError("paired replay compatibility view names are invalid")
    if not isinstance(previous, Mapping) or set(previous) != set(view_names):
        raise AdapterError("paired replay compatibility previous-RGB view coverage differs")
    copied: dict[str, Any] = {}
    for view in view_names:
        value = np.asarray(previous[view])
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
            raise AdapterError(
                "paired replay compatibility previous-RGB payload is invalid "
                f"for view {view!r}"
            )
        copied[view] = np.ascontiguousarray(value).copy()
    return copied


def install_replay_anchor_compat(trainer: Any) -> dict[str, Any]:
    """Install the source-pinned, evaluator-only 9dd replay-anchor shim.

    Normal evaluation and every training rollout are strict no-ops.  In the
    paired first-condition replay mode, the wrapper commits canonical anchor
    bytes only after the original relabel method has completed its renderer
    batch and provenance validation.  That preserves its transaction boundary
    and keeps later requests byte-identical to the server-bound step-zero
    reference.
    """

    if not _paired_replay_compat_requested():
        return {"enabled": False}
    source_sha256 = _trainer_source_sha256(trainer)
    if bool(getattr(trainer, "_in_train_rollout", False)):
        raise AdapterError("paired replay compatibility cannot run in a training rollout")
    original = getattr(trainer, "_image_dagger_relabel", None)
    if not callable(original):
        raise AdapterError("paired replay compatibility target lacks relabel method")
    if getattr(trainer, "_roborender_replay_anchor_compat_installed", False):
        raise AdapterError("paired replay compatibility was installed more than once")

    def relabel_with_compatible_anchors(obs_dict: Any, cur_dones: Any) -> Any:
        # The adapter is installed by the transformed evaluator immediately
        # before ``trainer.eval()``.  GR00T deliberately initializes the
        # first-condition replay bundle at the start of that method, so
        # asserting this at installation time rejects every valid paired
        # evaluation.  Check immediately before the first relabel instead:
        # that remains before any renderer/cache mutation while respecting the
        # trainer's lifecycle.
        if getattr(trainer, "_first_condition_bundle_mode", None) != "replay":
            raise AdapterError(
                "paired replay compatibility bundle mode differs from replay"
            )
        before = frozenset(getattr(trainer, "_first_condition_replay_consumed", ()))
        result = original(obs_dict, cur_dones)
        try:
            after = frozenset(trainer._first_condition_replay_consumed)
            anchors = trainer._id_anchors
        except AttributeError as exc:
            raise AdapterError(
                "paired replay compatibility target lacks replay anchor state"
            ) from exc
        newly_consumed = after - before
        if not newly_consumed:
            return result
        if not isinstance(anchors, list) or not anchors:
            raise AdapterError("paired replay compatibility anchors are invalid")
        raytraced = getattr(trainer, "_id_use_raytraced", None) or {}
        if not isinstance(raytraced, Mapping):
            raise AdapterError("paired replay compatibility raytraced state is invalid")
        expected = {
            index
            for index in range(len(anchors))
            if not bool(raytraced.get(index, False))
        }
        if newly_consumed != expected:
            raise AdapterError(
                "paired replay compatibility consumed an incomplete or unexpected "
                "environment set"
            )
        # Validate every replacement before mutating one cache entry.  If any
        # bundle record is malformed, the original cache stays untouched.
        staged = {
            index: _copy_replay_anchor_triplet(
                trainer, environment_index=index
            )
            for index in sorted(newly_consumed)
        }
        for index, anchor in staged.items():
            anchors[index] = anchor
        return result

    trainer._image_dagger_relabel = relabel_with_compatible_anchors
    trainer._roborender_replay_anchor_compat_installed = True
    return {
        "enabled": True,
        "mode": "paired_eval_replay_anchor_9dd_compat_v1",
        "trainer_source_sha256": source_sha256,
    }


def _json_value(value: Any, *, path: str = "metrics") -> Any:
    """Convert common scalar/tensor containers without accepting opaque data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AdapterError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise AdapterError(f"{path} contains a non-string metric key")
            converted[key] = _json_value(item, path=f"{path}.{key}")
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, path=f"{path}[]") for item in value]

    # NumPy scalars and scalar torch tensors expose item(); arrays/tensors expose
    # tolist().  Convert only their public value, never object repr/pickle state.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError, RuntimeError):
            scalar = value
        if scalar is not value:
            return _json_value(scalar, path=path)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_value(tolist(), path=path)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise AdapterError(
                f"{path} cannot be converted to JSON ({type(exc).__name__})"
            ) from None
    raise AdapterError(f"{path} contains unsupported {type(value).__name__} data")


def write_metrics_atomic(
    metrics: Any,
    output: Path,
    *,
    evaluator_path: Path,
    evaluator_sha256: str,
    replay_anchor_compat: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(metrics, Mapping):
        raise AdapterError("trainer.eval() must return a metrics mapping")
    payload = _json_value(metrics)
    adapter = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": evaluator_sha256,
    }
    # Paired evaluation assigns a fresh stream namespace to every immutable
    # attempt.  Capture the namespace in the evaluator-produced metrics so the
    # final outcome can prove which stream identity the Image3F clients used.
    # Other users of this generic adapter need not set IMAGE3F_RUN_ID.
    image3f_run_id = os.environ.get("IMAGE3F_RUN_ID")
    if image3f_run_id is not None:
        if SAFE_RUN_ID_RE.fullmatch(image3f_run_id) is None:
            raise AdapterError("IMAGE3F_RUN_ID is not a safe runtime token")
        adapter["image3f_run_id"] = image3f_run_id
    if replay_anchor_compat is not None:
        adapter["paired_replay_anchor_compat"] = _json_value(
            replay_anchor_compat,
            path="paired_replay_anchor_compat",
        )
    payload["_roborender_eval_adapter"] = adapter
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def execute_evaluator(
    evaluator: Path, metrics_output: Path, evaluator_args: Sequence[str]
) -> None:
    evaluator = evaluator.resolve()
    if not evaluator.is_file():
        raise AdapterError(f"evaluator does not exist: {evaluator}")
    source_bytes = evaluator.read_bytes()
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise AdapterError("evaluator source is not UTF-8") from None
    patched = patch_evaluator_source(source)
    evaluator_sha256 = sha256_bytes(source_bytes)

    def record(
        metrics: Any, *, replay_anchor_compat: Mapping[str, Any] | None = None
    ) -> None:
        write_metrics_atomic(
            metrics,
            metrics_output,
            evaluator_path=evaluator,
            evaluator_sha256=evaluator_sha256,
            replay_anchor_compat=replay_anchor_compat,
        )

    namespace = {
        "__name__": "__main__",
        "__file__": str(evaluator),
        "__package__": None,
        "__cached__": None,
        "__roborender_record_eval_metrics": record,
        "__roborender_install_replay_anchor_compat": install_replay_anchor_compat,
    }
    original_argv = sys.argv
    original_path = list(sys.path)
    try:
        # Match direct execution closely enough for local imports and Hydra's
        # relative config-path resolution while retaining the original filename.
        sys.argv = [str(evaluator), *evaluator_args]
        sys.path.insert(0, str(evaluator.parents[2]))
        exec(compile(patched, str(evaluator), "exec"), namespace, namespace)
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("evaluator_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evaluator_args = list(args.evaluator_args)
    if evaluator_args[:1] == ["--"]:
        evaluator_args.pop(0)
    try:
        execute_evaluator(args.evaluator, args.metrics_output.resolve(), evaluator_args)
    except (AdapterError, OSError) as exc:
        print(f"fixed-eval metrics adapter failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
