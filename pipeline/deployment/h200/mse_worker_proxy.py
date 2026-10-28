"""CPU launcher for inference on an already-owned H200 allocation, never allocate."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
import subprocess


def command(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.host):
        raise ValueError("an explicit pinned allocation SSH alias is required")
    for name in ("ssh_config", "source_root", "asset_root", "netrc_file", "request", "result"):
        if not Path(getattr(args, name)).is_absolute():
            raise ValueError("worker paths must be absolute")
    if not Path(args.ssh_config).is_file():
        raise ValueError("owned allocation SSH configuration is absent")
    remote = ["bash", str(Path(args.source_root) / "pipeline/dagger/run_heldout_mse_worker_h200.sh"),
              "--run-id", args.run_id, "--asset-root", str(args.asset_root),
              "--netrc-file", str(args.netrc_file), "--request", str(args.request), "--result", str(args.result)]
    return ["ssh", "-F", str(args.ssh_config), "-o", "BatchMode=yes", "-o", "ControlMaster=no",
            "-o", "ControlPath=none", "-o", "ConnectTimeout=15", args.host, shlex.join(remote)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ssh-config", "host", "run-id", "source-root", "asset-root", "netrc-file", "request", "result"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    subprocess.run(command(args), check=True, timeout=3600)


if __name__ == "__main__":
    main()
