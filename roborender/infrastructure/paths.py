"""Explicit, CPU-only storage locations and immutable per-run snapshots.

This registry describes locations, NOT asset identity, physical filesystem
attestation, source admission, or permission to launch/stop/release compute.
Resolution is lexical: it does not follow symlinks, canonicalize a physical
filesystem, require remote paths to exist, expand variables, or guess a host.

Usage::

    registry = load_registry("configs/storage_paths.json")
    path = registry.resolve("leo", "source_releases")
    registry.freeze("leo", run_directory / "storage-paths.json")

The snapshot binds the exact registry bytes loaded, not a later version of the
file. Its output parent must already exist. Repeating an identical freeze is
safe; a different or malformed existing snapshot is never overwritten.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from types import MappingProxyType

SCHEMA = "roborender.storage-paths.v1"
SNAPSHOT_SCHEMA = "roborender.storage-paths-snapshot.v1"
MAX_REGISTRY_BYTES = 1 << 20
_KEY = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")


def _identifier(value, label):
    if not isinstance(value, str) or not _KEY.fullmatch(value):
        raise ValueError(f"explicit lowercase {label} identifier required")
    return value


def _path(value, *, absolute):
    if (not isinstance(value, str) or not value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or any(c in value for c in ("$", "~", "\\"))):
        raise ValueError("literal POSIX paths without templates or control characters required")
    parsed = PurePosixPath(value)
    if (parsed.is_absolute() is not absolute or value.startswith("//")
            or ".." in parsed.parts or value in (".", "..")
            or parsed.as_posix() != value):
        raise ValueError("canonical absolute root or safe relative artifact path required")
    return parsed


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate registry JSON key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("nonfinite JSON value")


def _decode(raw):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("storage registry/snapshot must be strict UTF-8 JSON") from error


def _read_regular(path, limit):
    """Bounded local file read; refuse final-component links and special files."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("bounded regular registry/snapshot file required")
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("registry/snapshot exceeded byte limit")
    return raw


def _validated_locations(document):
    if (not isinstance(document, dict) or set(document) != {"schema", "hosts"}
            or document["schema"] != SCHEMA
            or not isinstance(document["hosts"], dict) or not document["hosts"]):
        raise ValueError("exact versioned storage registry with explicit hosts required")
    hosts = {}
    for host, value in document["hosts"].items():
        _identifier(host, "host")
        if (not isinstance(value, dict) or not {"roots"} <= set(value) <= {"roots", "artifacts"}
                or not isinstance(value["roots"], dict) or not value["roots"]
                or not isinstance(value.get("artifacts", {}), dict)):
            raise ValueError("host requires roots and optional artifacts mappings")
        roots = {}
        for key, location in value["roots"].items():
            _identifier(key, "path key")
            roots[key] = _path(location, absolute=True)
        locations = dict(roots)
        for key, locator in value.get("artifacts", {}).items():
            _identifier(key, "path key")
            if key in roots:
                raise ValueError("root and artifact keys must be distinct")
            if not isinstance(locator, dict) or set(locator) != {"root", "relative"}:
                raise ValueError("artifact requires exactly root and relative")
            root = _identifier(locator["root"], "root reference")
            if root not in roots:
                raise ValueError("artifact references an unknown root")
            locations[key] = roots[root] / _path(locator["relative"], absolute=False)
        hosts[host] = MappingProxyType(locations)
    return MappingProxyType(hosts)


@dataclass(frozen=True)
class Registry:
    """Validated locations pinned to the original registry's raw SHA-256."""

    registry_path: Path
    raw_sha256: str
    _locations: Mapping[str, Mapping[str, PurePosixPath]]

    def _host(self, host):
        _identifier(host, "host")
        if host not in self._locations:
            raise ValueError("unknown registry host: " + host)
        return self._locations[host]

    def resolve(self, host: str, key: str) -> Path:
        """Return a lexical location; never infer host or attest stored bytes."""
        locations = self._host(host)
        _identifier(key, "path key")
        if key not in locations:
            raise ValueError("unknown path key for host: " + host + "/" + key)
        return Path(locations[key])

    def snapshot(self, host: str, *, keys: Iterable[str] | None = None) -> dict:
        """Return deterministic, detached JSON data for all or selected keys."""
        locations = self._host(host)
        if keys is None:
            selected = sorted(locations)
        else:
            if isinstance(keys, (str, bytes, Mapping)):
                raise ValueError("keys must be an explicit nonempty iterable of identifiers")
            selected = list(keys)
            for key in selected:
                _identifier(key, "path key")
            if not selected or len(set(selected)) != len(selected):
                raise ValueError("snapshot keys must be nonempty and unique")
            selected.sort()
        return dict(schema=SNAPSHOT_SCHEMA, registry_schema=SCHEMA,
                    registry_path=str(self.registry_path), registry_raw_sha256=self.raw_sha256,
                    host=host, paths={key: str(self.resolve(host, key)) for key in selected})

    def freeze(self, host: str, output_path: str | os.PathLike, *,
               keys: Iterable[str] | None = None) -> dict:
        """Publish one atomic, no-clobber snapshot at an explicit local path.

        This writes only the requested snapshot and a short-lived sibling
        temporary file. It never creates roots, copies assets, or changes a
        host's configuration. The caller creates the run directory explicitly.
        """
        value = self.snapshot(host, keys=keys)
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        if len(raw) > MAX_REGISTRY_BYTES:
            raise ValueError("resolved snapshot exceeds byte limit")
        target = Path(output_path)
        if not target.name or not target.parent.is_dir():
            raise ValueError("explicit snapshot file and existing parent directory required")
        if os.path.lexists(target):
            if _read_regular(target, MAX_REGISTRY_BYTES) != raw:
                raise FileExistsError("existing storage snapshot differs; refusing overwrite")
            return value
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".storage-paths-", dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
                stream.flush()
                os.fchmod(stream.fileno(), 0o444)
                os.fsync(stream.fileno())
            try:
                # An atomic hard-link publication cannot replace an existing
                # path; interrupted writes never become visible at the target.
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError:
                if _read_regular(target, MAX_REGISTRY_BYTES) != raw:
                    raise FileExistsError("concurrent storage snapshot differs; refusing overwrite")
            directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink()
        return value


    def verify_snapshot(self, snapshot_path: str | os.PathLike) -> dict:
        """Check a snapshot against these explicitly loaded registry bytes.

        Historical verification uses the preserved original registry, which may
        now reside elsewhere. The snapshot's registry_path is provenance only;
        it is never opened or used to select the current machine's registry.
        """
        raw = _read_regular(Path(snapshot_path), MAX_REGISTRY_BYTES)
        value = _decode(raw)
        if (not isinstance(value, dict)
                or set(value) != {"schema", "registry_schema", "registry_path", "registry_raw_sha256", "host", "paths"}
                or value["schema"] != SNAPSHOT_SCHEMA or value["registry_schema"] != SCHEMA
                or not isinstance(value["registry_path"], str)
                or not PurePosixPath(value["registry_path"]).is_absolute()
                or any(ord(c) < 32 or ord(c) == 127 for c in value["registry_path"])
                or not isinstance(value["paths"], dict) or not value["paths"]):
            raise ValueError("exact versioned resolved-path snapshot required")
        if value["registry_raw_sha256"] != self.raw_sha256:
            raise ValueError("registry digest differs; supply the preserved original registry bytes")
        expected = self.snapshot(value["host"], keys=value["paths"].keys())
        if value["paths"] != expected["paths"]:
            raise ValueError("snapshot paths differ from the hash-bound registry")
        return dict(verified=True, host=value["host"], registry_raw_sha256=self.raw_sha256,
                    snapshot_raw_sha256=hashlib.sha256(raw).hexdigest(), resolved_keys=len(value["paths"]))


def load_registry(path: str | os.PathLike) -> Registry:
    """Read a bounded UTF-8 JSON registry; there is deliberately no default."""
    path = Path(path).absolute()
    raw = _read_regular(path, MAX_REGISTRY_BYTES)
    return Registry(path, hashlib.sha256(raw).hexdigest(), _validated_locations(_decode(raw)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    resolve = commands.add_parser("resolve", help="print one explicit host/key location; no filesystem probe")
    resolve.add_argument("--registry", type=Path, required=True)
    resolve.add_argument("--host", required=True)
    resolve.add_argument("--key", required=True)
    snapshot = commands.add_parser("snapshot", help="freeze selected or all resolved paths without overwriting")
    snapshot.add_argument("--registry", type=Path, required=True)
    snapshot.add_argument("--host", required=True)
    snapshot.add_argument("--key", action="append", dest="keys", help="repeat to select keys; default is all")
    snapshot.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="verify using explicitly supplied original registry bytes")
    verify.add_argument("--registry", type=Path, required=True)
    verify.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        registry = load_registry(args.registry)
        if args.action == "resolve":
            print(registry.resolve(args.host, args.key))
        elif args.action == "snapshot":
            print(json.dumps(registry.freeze(args.host, args.output, keys=args.keys), sort_keys=True))
        else:
            print(json.dumps(registry.verify_snapshot(args.snapshot), sort_keys=True))
    except (OSError, ValueError) as error:
        parser.exit(2, "storage paths: " + str(error) + "\n")


if __name__ == "__main__":
    main()
