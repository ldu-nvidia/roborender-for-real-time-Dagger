#!/usr/bin/env python3
"""CPU-only receipt worker used solely by service integration tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    output = args.result.parent / "test_output.pt"
    output.write_bytes(json.dumps(job, sort_keys=True).encode())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    args.result.write_text(json.dumps({
        "output_checkpoint_path": str(output.resolve()),
        "output_checkpoint_sha256": digest,
    }, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
