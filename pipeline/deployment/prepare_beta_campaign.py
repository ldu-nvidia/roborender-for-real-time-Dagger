"""Freeze a beta campaign plan and deployment bindings; never launch workloads."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from pipeline.dagger.beta_campaign import make_first_round_config, immutable_json
from pipeline.dagger.services.contracts import file_sha256
from pipeline.deployment.beta_operations import PHASES
from pipeline.deployment.allocation_wait import PREPARE_TIMEOUT_SECONDS
from pipeline.deployment.beta_site import validate_site, site_host
from pipeline.deployment.source_snapshot import verify


def stage_round_parent(checkpoint: Path, checkpoint_root: Path) -> dict:
    """Admit exact parent bytes without widening the registry's trusted root.

    BC smoke outputs live under runtime/migrations, outside register-local's
    allowlist. Stage once on shared storage before allocating any GPUs. This
    is an immutable input artifact, not promotion to a production/best policy.
    """
    source = checkpoint.resolve(strict=True)
    digest = file_sha256(source)
    root = checkpoint_root.resolve(strict=True)
    directory = root / "campaign_parents"
    if directory.is_symlink():
        raise ValueError("parent staging directory must not be a symlink")
    directory.mkdir(exist_ok=True)
    target = directory / f"{digest}.pt"
    if target.is_symlink():
        raise ValueError("staged parent must not be a symlink")
    if not target.exists():
        try:
            os.link(source, target)
        except FileExistsError:
            pass
        except OSError:
            # A different source filesystem requires one cold-path copy. Link
            # the verified temporary file exclusively; never replace a target.
            fd, name = tempfile.mkstemp(prefix=f".{digest}.", suffix=".part", dir=directory)
            temporary = Path(name)
            try:
                with os.fdopen(fd, "wb") as output, source.open("rb") as incoming:
                    shutil.copyfileobj(incoming, output, 1 << 20)
                    output.flush()
                    os.fsync(output.fileno())
                if file_sha256(temporary) != digest:
                    raise ValueError("parent changed while staging")
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    pass
            finally:
                temporary.unlink(missing_ok=True)
    if target.is_symlink() or not target.is_file() or file_sha256(target) != digest:
        raise ValueError("staged parent bytes differ; refusing overwrite")
    return {"schema": "roborender.campaign-parent-admission.v1", "source": str(source),
            "path": str(target), "sha256": digest, "bytes": target.stat().st_size,
            "production_promoted": False}


def prepare(site, run_id, seed, checkpoint, panel, output, *, smoke=False):
    validate_site(site)
    manifest = json.loads(Path(site["source_manifest"]).read_bytes())
    verify(Path(site["repository_root"]), manifest)
    if output.exists():
        raise ValueError("campaign plan output already exists")
    config = make_first_round_config(run_id, seed=seed, checkpoint=checkpoint.resolve(),
        checkpoint_sha=file_sha256(checkpoint), panel=panel.resolve(), panel_sha=file_sha256(panel),
        source_sha=manifest["source_sha256"], gateway_pids=[], smoke=smoke, deployment_host=site_host(site))
    from pipeline.dagger.beta_validation import load_panel
    load_panel(config)
    output.mkdir(parents=True)
    admission = stage_round_parent(checkpoint, Path(site["repository_root"]) / "checkpoints")
    if admission["sha256"] != config["student"]["initial_checkpoint_sha256"]:
        raise ValueError("parent changed after campaign validation")
    config["student"]["initial_checkpoint"] = admission["path"]
    immutable_json(output / "parent-checkpoint-admission.json", admission)
    site_path, config_path = output / "site.json", output / "first-round-template.json"
    site_sha = immutable_json(site_path, site)
    immutable_json(config_path, config)
    entrypoint = Path(site["repository_root"]) / "pipeline/deployment/beta_site.py"
    argv = [site["cpu_python"], str(entrypoint), "--site", str(site_path.resolve()), "--site-sha256", site_sha]
    deadlines = {"prepare": PREPARE_TIMEOUT_SECONDS, "collect": 22200, "release": 600,
                 "evaluate": 18600, "control_only": 300, "abort": 600}
    commands = {phase: {"argv": argv, "entrypoint": str(entrypoint), "entrypoint_sha256": file_sha256(entrypoint),
                        "timeout_seconds": deadlines[phase]} for phase in sorted(PHASES)}
    immutable_json(output / "operations.json", {"schema": "roborender.beta-campaign-operations.v1", "commands": commands})
    return {"config": str(config_path.resolve()), "operations": str((output / "operations.json").resolve()),
            "state_root": str((output / "active").resolve()), "run_id": run_id,
            "allocated_gpus": 0, "workloads_started": False, "source_sha256": manifest["source_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Three real updates and ten-case validation in each of four rounds")
    args = parser.parse_args()
    print(json.dumps(prepare(json.loads(args.site.read_bytes()), args.run_id, args.seed,
                             args.checkpoint, args.panel, args.output.resolve(), smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
