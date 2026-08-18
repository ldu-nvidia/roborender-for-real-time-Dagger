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
import re
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
SCALING_TRAINER_MARKERS = (
    "FAST_ROBORENDER_SCALING_V1: explicit ordered stream metadata",
    'self.vg_reference_lag = int(vg.get("reference_lag", 0))',
    'self._id_rank = int(getattr(self.accelerator, "process_index", 0))',
    "self._id_episode_ids = np.zeros(B, dtype=np.int64)",
    "self._id_episode_steps = np.zeros(B, dtype=np.int64)",
    "self._id_episode_ids[b] += 1",
    "self._id_episode_steps[b] = 0",
    "stream_id=(f\"{os.environ.get('IMAGE3F_RUN_ID', 'run')}:\"",
    "episode_id=int(self._id_episode_ids[b])",
    "step=int(self._id_episode_steps[b])",
    "reference_lag=self.vg_reference_lag,",
    "self._id_episode_steps += 1",
)
RUNTIME_GATES_MARKER = "FAST_ROBORENDER_PRODUCTION_GATES_V2"
RUNTIME_GATE_TRAINER_MARKERS = (
    RUNTIME_GATES_MARKER,
    "IMAGE_DAGGER_EXPECT_PRIMARY_VIEW",
    "production gate failed: primary view mismatch",
    "IMAGE_DAGGER_EXPECT_ROLLOUT_ACTION_SOURCE",
    "production gate failed: rollout action source mismatch",
    "IMAGE_DAGGER_EXPECT_BETA_SCHEDULE",
    "production gate failed: beta schedule mismatch",
    "IMAGE_DAGGER_EXPECT_BETA_START",
    "production gate failed: beta start mismatch",
    "IMAGE_DAGGER_EXPECT_BETA_END",
    "production gate failed: beta end mismatch",
    "IMAGE_DAGGER_EXPECT_TEACHER_WARMUP_STEPS",
    "production gate failed: teacher warmup mismatch",
    "IMAGE_DAGGER_EXPECT_ENFORCE_TEACHER",
    "production gate failed: behavior source mismatch",
    "IMAGE_DAGGER_EXPECT_TEACHER_RATIO",
    "production gate failed: teacher ratio mismatch",
    "IMAGE_DAGGER_EXPECT_LEARNING_EPOCHS",
    "production gate failed: learning epochs mismatch",
    "IMAGE_DAGGER_EXPECT_VALUE_LOSS_COEF",
    "production gate failed: value loss coefficient mismatch",
    "IMAGE_DAGGER_EXPECT_ENTROPY_COEF",
    "production gate failed: entropy coefficient mismatch",
)


def yaml_numeric_scalar(document: str, key: str) -> float | None:
    """Read one unquoted top-level/nested YAML numeric scalar exactly.

    This intentionally handles only the numeric form required by the production
    configuration.  It rejects duplicate keys, expressions, quoted values, and
    prefix matches such as ``0.1`` being mistaken for zero.
    """

    matches = re.findall(
        rf"(?m)^\s*{re.escape(key)}\s*:\s*([^\s#]+)\s*(?:#.*)?$", document
    )
    if len(matches) != 1:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None
TELEMETRY_MARKER = "FAST_ROBORENDER_DAGGER_TELEMETRY_V1"
TELEMETRY_TRAINER_MARKERS = (
    TELEMETRY_MARKER,
    "DaggerTelemetry.from_environment(rank=self._id_rank)",
    "trainer_step=int(self._id_telemetry_step))",
    "self._id_telemetry.record(",
    "depth=reqs[0][0], mask=reqs[0][1]",
    "client_anchor=reqs[0][2], generated=gens[0]",
    "IMAGE_DAGGER_TELEMETRY_REQUIRED",
    "self._id_telemetry_step += 1",
)


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


def inspect(
    gr00t_root: Path, source: Path, telemetry_source: Path | None = None
) -> dict[str, object]:
    telemetry_source = telemetry_source or source.with_name("dagger_telemetry.py")
    trainer = gr00t_root / "gr00t/rl/trl/trainer/distill_trainer_videogen.py"
    config = gr00t_root / "gr00t/rl/config/exp/loco_manip/wsdpt_image_dagger.yaml"
    target = gr00t_root / "gr00t/rl/utils/image3f_frame_client.py"
    telemetry_target = gr00t_root / "gr00t/rl/utils/dagger_telemetry.py"
    revision, dirty = git_revision(gr00t_root)
    trainer_text = trainer.read_text(encoding="utf-8") if trainer.is_file() else ""
    config_text = config.read_text(encoding="utf-8") if config.is_file() else ""
    missing_markers = [marker for marker in REQUIRED_TRAINER_MARKERS if marker not in trainer_text]
    missing_scaling_markers = [
        marker for marker in SCALING_TRAINER_MARKERS if trainer_text.count(marker) != 1
    ]
    missing_telemetry_markers = [
        marker for marker in TELEMETRY_TRAINER_MARKERS if trainer_text.count(marker) != 1
    ]
    missing_runtime_gate_markers = [
        marker for marker in RUNTIME_GATE_TRAINER_MARKERS if trainer_text.count(marker) != 1
    ]
    value_loss_coef = yaml_numeric_scalar(config_text, "value_loss_coef")
    entropy_coef = yaml_numeric_scalar(config_text, "entropy_coef")
    pure_bc = value_loss_coef == 0.0 and entropy_coef == 0.0
    source_hash = sha256(source)
    target_hash = sha256(target) if target.is_file() else None
    telemetry_source_hash = sha256(telemetry_source)
    telemetry_target_hash = sha256(telemetry_target) if telemetry_target.is_file() else None
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
        "trainer_scaling_integrated": not missing_scaling_markers,
        "missing_trainer_scaling_markers": missing_scaling_markers,
        "trainer_runtime_gates_integrated": not missing_runtime_gate_markers,
        "missing_trainer_runtime_gate_markers": missing_runtime_gate_markers,
        "trainer_telemetry_integrated": not missing_telemetry_markers,
        "missing_trainer_telemetry_markers": missing_telemetry_markers,
        "config": str(config),
        "config_exists": config.is_file(),
        "pure_bc_config": pure_bc,
        "value_loss_coef": value_loss_coef,
        "entropy_coef": entropy_coef,
        "client": str(target),
        "client_exists": target.is_file(),
        "client_source_sha256": source_hash,
        "client_installed_sha256": target_hash,
        "client_matches": target_hash == source_hash,
        "telemetry_utility": str(telemetry_target),
        "telemetry_utility_exists": telemetry_target.is_file(),
        "telemetry_source_sha256": telemetry_source_hash,
        "telemetry_installed_sha256": telemetry_target_hash,
        "telemetry_utility_matches": telemetry_target_hash == telemetry_source_hash,
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
        errors.append("DAgger config does not set value and entropy losses exactly to zero")
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
    telemetry_source = repo_root / "integrations/gr00t/dagger_telemetry.py"
    report = inspect(args.gr00t_root.resolve(), source, telemetry_source)
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
            install_file(telemetry_source, Path(report["telemetry_utility"]))
        report = inspect(args.gr00t_root.resolve(), source, telemetry_source)
    if not report["trainer_scaling_integrated"]:
        errors.append(
            "explicit episode/step trainer integration is incomplete: "
            + ", ".join(report["missing_trainer_scaling_markers"])
        )
    if not report["trainer_runtime_gates_integrated"]:
        errors.append("production runtime configuration gates are not installed")
    if not report["trainer_telemetry_integrated"]:
        errors.append("rank-zero DAgger telemetry integration is not installed")
    if not report["client_matches"]:
        errors.append("versioned production Image3F client is not installed")
    if not report["telemetry_utility_matches"]:
        errors.append("versioned DAgger telemetry utility is not installed")

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
        print(
            "runtime gates: "
            + ("INSTALLED" if report["trainer_runtime_gates_integrated"] else "NOT INSTALLED")
        )
        print(
            "telemetry: "
            + (
                "INSTALLED"
                if report["trainer_telemetry_integrated"]
                and report["telemetry_utility_matches"]
                else "NOT INSTALLED"
            )
        )
        print(f"config:  {'OK' if report['config_exists'] else 'MISSING'}")
        print(f"client:  {'MATCH' if report['client_matches'] else 'NOT INSTALLED'}")
        for error in errors:
            print(f"ERROR: {error}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
