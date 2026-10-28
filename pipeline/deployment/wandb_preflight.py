"""CPU-only W&B authentication and optional real remote-write qualification.

Credentials come only from WANDB_API_KEY, an explicitly supplied owner-only
file, or an explicit NETRC path, never from source/config manifests. This
command does not allocate GPUs.
The smoke mode writes a clearly named diagnostic run, not fabricated losses.
"""
from __future__ import annotations

import argparse
import json
import netrc
import os
from pathlib import Path
import re
import stat
import time

BASE_URL = "https://api.wandb.ai"


def read_key(path: Path | None = None) -> str:
    key = os.environ.get("WANDB_API_KEY", "").strip() if path is None else ""
    netrc_path = os.environ.get("NETRC") if path is None and not key else None
    if path is not None or netrc_path:
        path = path if path is not None else Path(netrc_path)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "r") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077
                    or info.st_uid != os.geteuid() or info.st_size > 4096):
                raise ValueError("W&B credential must be a bounded owner-only regular file")
            if netrc_path:
                # Parse the already-open, verified inode, not a second lookup
                # of the supplied path. Never copy the secret to a temp file.
                fd_root = "/proc/self/fd" if Path("/proc/self/fd").is_dir() else "/dev/fd"
                try:
                    auth = netrc.netrc(f"{fd_root}/{stream.fileno()}").hosts.get("api.wandb.ai")
                    key = auth[2].strip() if auth else ""
                except (netrc.NetrcParseError, OSError, UnicodeError, TypeError, AttributeError):
                    raise ValueError("Approved W&B netrc credential is malformed") from None
            else:
                key = stream.read(4097).strip()
    if len(key) < 20 or len(key) > 4096 or any(c.isspace() for c in key):
        raise ValueError("W&B credential is absent or malformed; provide an approved credential file or environment")
    return key


def authenticate(key: str, entity: str, *, api_factory=None):
    if not isinstance(entity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", entity):
        raise ValueError("an explicit W&B entity is required")
    if api_factory is None:
        from wandb import Api
        api_factory = Api
    try:
        api = api_factory(api_key=key, timeout=15,
                          overrides={"base_url": BASE_URL, "entity": entity, "project": "dagger"})
        viewer = api.viewer
        username = getattr(viewer, "username", None)
        if not username:
            raise ValueError("W&B did not return an authenticated identity")
        # A readable viewer is authentication evidence, not proof of project
        # write permission. Only a completed smoke round-trip proves that.
        return api, {"authenticated": True, "entity": entity, "project": "dagger",
                     "username": username, "project_write_verified": False}
    except Exception:
        # SDK exception text may embed request details; never echo it or keys.
        raise RuntimeError("W&B authentication failed; check the approved credential and connectivity") from None


def admit_logging(mode: str, entity: str | None, key_file: Path | None = None, *, api_factory=None):
    """Authenticate before collection; offline is explicit and never verified."""
    if mode == "offline":
        return None, {"mode": mode, "authenticated": False, "project_write_verified": False}
    if mode != "online":
        raise ValueError("logging mode must be explicitly online or offline")
    key = read_key(key_file)
    api, identity = authenticate(key, entity, api_factory=api_factory)
    os.environ.update(WANDB_API_KEY=key, WANDB_BASE_URL=BASE_URL, WANDB_ENTITY=entity)
    return api, {**identity, "mode": mode}


def _summary_matches(summary, name, expected):
    """Compare JSON values across W&B's nested/dotted summary encodings.

    The public API flattens logged mappings and wraps long arrays. These are
    representation changes, not permission to omit or approximate a value.
    """
    if isinstance(expected, dict):
        if name in summary:
            nested = summary[name]
            if hasattr(nested, "keys"):
                return all(_summary_matches(nested, key, value)
                           for key, value in expected.items())
            return False
        return bool(expected) and all(
            _summary_matches(summary, name + "." + key, value)
            for key, value in expected.items())
    if name not in summary:
        return False
    actual = summary[name]
    if isinstance(expected, list) and hasattr(actual, "keys"):
        if actual.get("_type") != "large-array" or "value" not in actual:
            return False
        actual = actual["value"]
    return actual == expected


def verify_remote_summary(api, entity: str, run_id: str, expected: dict, *, sleep=time.sleep):
    """Bounded cloud readback, once after finish, not in the optimizer hot path."""
    for attempt in range(5):
        try:
            api.flush()
            remote = api.run(f"{entity}/dagger/{run_id}")
            if all(_summary_matches(remote.summary, name, value)
                   for name, value in expected.items()):
                return
        except Exception:
            pass  # Never expose SDK request details or credentials.
        if attempt < 4:
            sleep(2)
    raise RuntimeError("W&B remote metric readback failed; online logging is not verified")


def smoke(key: str, entity: str, run_id: str, output: Path, *, wandb_module=None):
    if not re.fullmatch(r"smoke-dagger-admission-[a-z0-9-]{1,40}", run_id):
        raise ValueError("a fresh smoke-dagger-admission-* identity is required")
    if output.exists():
        raise ValueError("smoke evidence directory already exists")
    if wandb_module is None:
        import wandb as wandb_module
    api, result = authenticate(key, entity, api_factory=wandb_module.Api)
    output.mkdir(parents=True, mode=0o700)
    os.environ["WANDB_API_KEY"] = key
    os.environ["WANDB_BASE_URL"] = BASE_URL
    run = None
    try:
        run = wandb_module.init(
            entity=entity, project="dagger", id=run_id, name=run_id,
            mode="online", resume="never", dir=str(output),
            config={"purpose": "production_logging_qualification", "training_performed": False,
                    "success_evaluation_performed": False, "gpu_allocation": False},
            settings=wandb_module.Settings(init_timeout=45),
        )
        run.log({"admission/online_logging_probe": 1}, step=0)
        run.summary["admission/round_trip_expected"] = True
        url = run.url
        run.finish()
        run = None
        verify_remote_summary(api, entity, run_id, {
            "admission/online_logging_probe": 1, "admission/round_trip_expected": True})
        result.update(project_write_verified=True, run_id=run_id, url=url)
        return result
    except Exception:
        raise RuntimeError("W&B online write/readback failed; production remains blocked") from None
    finally:
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass  # Preserve the sanitized primary error.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "smoke"])
    parser.add_argument("--entity", required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        key = read_key(args.api_key_file)
        if args.mode == "smoke":
            if args.run_id is None or args.output is None:
                raise ValueError("smoke requires --run-id and --output")
            result = smoke(key, args.entity, args.run_id, args.output)
        else:
            _, result = authenticate(key, args.entity)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"authenticated": False, "production_ready": False, "reason": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
