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
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ADAPTER_SCHEMA_VERSION = 1
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
        f"{indent}__roborender_eval_metrics = trainer.eval()\n"
        f"{indent}__roborender_record_eval_metrics(__roborender_eval_metrics)"
    )
    return source[: match.start()] + replacement + source[match.end() :]


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
) -> None:
    if not isinstance(metrics, Mapping):
        raise AdapterError("trainer.eval() must return a metrics mapping")
    payload = _json_value(metrics)
    payload["_roborender_eval_adapter"] = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": evaluator_sha256,
    }
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

    def record(metrics: Any) -> None:
        write_metrics_atomic(
            metrics,
            metrics_output,
            evaluator_path=evaluator,
            evaluator_sha256=evaluator_sha256,
        )

    namespace = {
        "__name__": "__main__",
        "__file__": str(evaluator),
        "__package__": None,
        "__cached__": None,
        "__roborender_record_eval_metrics": record,
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
