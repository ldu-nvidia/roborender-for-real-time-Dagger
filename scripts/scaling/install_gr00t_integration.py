#!/usr/bin/env python3
"""Install/check the versioned Image3F client in Raven's external GR00T tree.

The core trainer is intentionally a separate repository and is not present in
public NVlabs GR00T.  This command fails loudly unless Raven's Image3F DAgger
files are present; it never pretends that the orchestration repository alone is
an end-to-end implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import sys


REQUIRED_TRAINER_MARKERS = (
    "_image_dagger_relabel",
    "_id_env_clients",
    "ThreadPoolExecutor",
    "raytraced_mix_prob",
)
SCALING_MARKER = "FAST_ROBORENDER_SCALING_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def inspect(gr00t_root: Path, source: Path) -> dict[str, object]:
    trainer = gr00t_root / "gr00t/rl/trl/trainer/distill_trainer_videogen.py"
    config = gr00t_root / "gr00t/rl/config/exp/loco_manip/wsdpt_image_dagger.yaml"
    target = gr00t_root / "gr00t/rl/utils/image3f_frame_client.py"
    revision, dirty = git_revision(gr00t_root)
    trainer_text = trainer.read_text(encoding="utf-8") if trainer.is_file() else ""
    config_text = config.read_text(encoding="utf-8") if config.is_file() else ""
    missing_markers = [marker for marker in REQUIRED_TRAINER_MARKERS if marker not in trainer_text]
    pure_bc = (
        "value_loss_coef" in config_text
        and "entropy_coef" in config_text
        and ("value_loss_coef: 0" in config_text or "value_loss_coef: 0.0" in config_text)
        and ("entropy_coef: 0" in config_text or "entropy_coef: 0.0" in config_text)
    )
    source_hash = sha256(source)
    target_hash = sha256(target) if target.is_file() else None
    trainer_syntax_error = None
    if trainer_text:
        try:
            compile(trainer_text, str(trainer), "exec")
        except SyntaxError as exc:
            trainer_syntax_error = str(exc)
    return {
        "gr00t_root": str(gr00t_root),
        "gr00t_revision": revision,
        "gr00t_dirty": dirty,
        "trainer": str(trainer),
        "trainer_exists": trainer.is_file(),
        "trainer_syntax_ok": trainer.is_file() and trainer_syntax_error is None,
        "trainer_syntax_error": trainer_syntax_error,
        "missing_trainer_markers": missing_markers,
        "trainer_scaling_integrated": SCALING_MARKER in trainer_text,
        "config": str(config),
        "config_exists": config.is_file(),
        "pure_bc_config": pure_bc,
        "client": str(target),
        "client_exists": target.is_file(),
        "client_source_sha256": source_hash,
        "client_installed_sha256": target_hash,
        "client_matches": target_hash == source_hash,
    }


def validate_base(report: dict[str, object], *, allow_dirty: bool) -> list[str]:
    errors = []
    if not report["trainer_exists"]:
        errors.append("Raven distill_trainer_videogen.py is missing")
    elif not report["trainer_syntax_ok"]:
        errors.append(f"trainer has invalid Python syntax: {report['trainer_syntax_error']}")
    if report["missing_trainer_markers"]:
        errors.append(
            "trainer lacks required parallel Image3F markers: "
            + ", ".join(report["missing_trainer_markers"])
        )
    if not report["config_exists"]:
        errors.append("wsdpt_image_dagger.yaml is missing")
    elif not report["pure_bc_config"]:
        errors.append("DAgger config does not visibly set value and entropy losses to zero")
    if report["gr00t_dirty"] and not allow_dirty:
        errors.append("GR00T worktree is dirty (pass --allow-dirty only for an intentional run)")
    return errors


def install_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copy2(source, tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gr00t-root", type=Path, default=repo_root / "GR00T-VisualSim2Real"
    )
    parser.add_argument("--install", action="store_true", help="install the tracked client")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--expected-revision", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    source = repo_root / "integrations/gr00t/image3f_frame_client.py"
    report = inspect(args.gr00t_root.resolve(), source)
    errors = validate_base(report, allow_dirty=args.allow_dirty)
    if args.expected_revision and report["gr00t_revision"] != args.expected_revision:
        errors.append(
            f"GR00T revision {report['gr00t_revision']} does not match "
            f"{args.expected_revision}"
        )
    if args.install and not errors:
        sys.path.insert(0, str(repo_root))
        from integrations.gr00t.patch_trainer_scaling import PatchError, patch_source

        trainer = Path(report["trainer"])
        try:
            patched = patch_source(trainer.read_text(encoding="utf-8"))
            compile(patched, str(trainer), "exec")
        except (PatchError, SyntaxError) as exc:
            errors.append(f"trainer integration patch failed: {exc}")
        else:
            fd, tmp_name = tempfile.mkstemp(prefix=f".{trainer.name}.", dir=trainer.parent)
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                tmp.write_text(patched, encoding="utf-8")
                os.replace(tmp, trainer)
            finally:
                if tmp.exists():
                    tmp.unlink()
            install_file(source, Path(report["client"]))
        report = inspect(args.gr00t_root.resolve(), source)
    if not report["trainer_scaling_integrated"]:
        errors.append("explicit episode/step trainer integration is not installed")
    if not report["client_matches"]:
        errors.append("versioned production Image3F client is not installed")

    report["status"] = "ok" if not errors else "error"
    report["errors"] = errors
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GR00T revision: {report['gr00t_revision'] or 'unknown'}")
        print(f"trainer: {'OK' if report['trainer_exists'] else 'MISSING'}")
        print(
            "scaling: "
            + ("EXPLICIT EPISODE/STEPS" if report["trainer_scaling_integrated"] else "NOT INSTALLED")
        )
        print(f"config:  {'OK' if report['config_exists'] else 'MISSING'}")
        print(f"client:  {'MATCH' if report['client_matches'] else 'NOT INSTALLED'}")
        for error in errors:
            print(f"ERROR: {error}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
