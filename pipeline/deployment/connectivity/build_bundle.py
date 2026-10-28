"""Create a public-source-only Beta installer. No credentials/assets included."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

SOURCE = Path(__file__).resolve().parent


def build(output):
    files = {name: (SOURCE / name).read_bytes() for name in ("cloud_proxy.py", "cpu_endpoint.py", "beta_access.py")}
    config = {"cpu_endpoint_path": "/workspace/project/roborender/pipeline/deployment/connectivity/cpu_endpoint.py",
              "cpu_endpoint_sha256": hashlib.sha256(files["cpu_endpoint.py"]).hexdigest()}
    files["public-config.json"] = (json.dumps(config, sort_keys=True) + "\n").encode()
    manifest = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())}
    files["MANIFEST.json"] = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    files["__main__.py"] = (SOURCE / "bundle_main.py").read_bytes()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100600 << 16
            archive.writestr(info, data)
    return {"path": str(output.resolve()), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "members": sorted(files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args().output), sort_keys=True))
