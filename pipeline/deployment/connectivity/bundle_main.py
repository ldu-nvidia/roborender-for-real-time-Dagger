"""Source-only bootstrap; executed from the generated, hash-checked zipapp."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import pwd
import runpy
import socket
import sys
import zipfile

FILES = {"cloud_proxy.py", "cpu_endpoint.py", "beta_access.py", "public-config.json"}
BASE = Path("/home/ldu/.local/share/roborender-beta-access")


def unpack(archive, base):
    """Reject traversal, unknown members, or altered/overwritten source files."""
    base = Path(base)
    if base.is_symlink():
        raise RuntimeError("refusing symlink install directory")
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if set(names) != FILES | {"__main__.py", "MANIFEST.json"} or len(names) != 6:
            raise ValueError("unexpected bootstrap archive members")
        manifest = json.loads(source.read("MANIFEST.json"))
        if set(manifest) != FILES or any(len(v) != 64 for v in manifest.values()):
            raise ValueError("invalid bootstrap manifest")
        content = {name: source.read(name) for name in sorted(FILES)}
        if any(hashlib.sha256(content[name]).hexdigest() != manifest[name] for name in FILES):
            raise ValueError("bootstrap source hash mismatch")
    release = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    target = base / "versions" / release
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.parent.is_symlink() or target.is_symlink():
        raise RuntimeError("refusing symlink release directory")
    target.mkdir(mode=0o700, exist_ok=True)
    for name, data in content.items():
        path = target / name
        if path.is_symlink():
            raise RuntimeError("refusing symlink source file")
        if path.exists():
            if path.read_bytes() != data:
                raise RuntimeError("installed release changed; refusing overwrite")
        else:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
    return target


def main():
    if socket.gethostname() != "465d6b0-lcedt" or pwd.getpwuid(os.getuid()).pw_name != "ldu":
        raise RuntimeError("run this bootstrap as ldu on RTX Beta, not on the Mac or GPU pod")
    if not (3, 9) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("use the already authenticated venv-compatible Python")
    os.umask(0o077)
    if sys.argv[1:] not in ([], ["--upgrade"]):
        raise ValueError("only the explicit --upgrade option is supported")
    action = "upgrade" if sys.argv[1:] else "install"
    source = unpack(sys.argv[0], BASE)
    sys.path.insert(0, str(source))
    sys.argv = [str(source / "beta_access.py"), action, "--config", str(source / "public-config.json")]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
