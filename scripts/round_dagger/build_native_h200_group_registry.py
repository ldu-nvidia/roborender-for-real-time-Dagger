#!/usr/bin/env python3
"""Build and audit the native-H200 33-motion group registry.

Only the original ``record_2025-10-14_*.npz`` files are eligible.  The runtime
registry deliberately contains no outcome, lighting, material, or friction
labels: the source files do not provide audited values for those concepts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np


REGISTRY_SCHEMA = "fast_roborender.round_dagger.group_registry.v2"
AUDIT_SCHEMA = "fast_roborender.round_dagger.group_registry_audit.v1"
MOTION_DIRECTORY = (
    "GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775"
)
MOTION_GLOB = "record_2025-10-14_*.npz"
MOTION_NAME = re.compile(
    r"record_2025-10-14_\d{2}-\d{2}-\d{2}_(\d\.\d{3})m\.npz\Z"
)
EXPECTED_COUNT = 33
EXPECTED_NPZ_MEMBERS = {
    "robot.npy",
    "table_grid.npy",
    "bottle_table.npy",
    "bottle_hand.npy",
    "tray_table.npy",
    "homie_command.npy",
    "primitive_action.npy",
}

# These are fixed midpoints between ranks 11/12 and 22/23 in the original
# 33-motion inventory.  The audit records the raw values needed to recompute
# them.  Values exactly on a cut enter the higher-numbered bin.
DISTANCE_CUTS_M = (1.4761623740196228, 1.6032079458236694)
WORLD_BEARING_CUTS_DEG = (-8.443572677951135, 11.08288340098878)
TRAJECTORY_FRAME_CUTS = (501.0, 591.5)


class RegistryBuildError(RuntimeError):
    """The source inventory or a derived measurement is not auditable."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path, expected: os.stat_result) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RegistryBuildError(f"cannot open motion for hashing: {path}") from exc
    try:
        before = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        expected_identity = (
            expected.st_dev,
            expected.st_ino,
            expected.st_size,
            expected.st_mtime_ns,
        )
        if identity != expected_identity or not stat.S_ISREG(before.st_mode):
            raise RegistryBuildError(f"motion changed before hashing: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
            raise RegistryBuildError(f"motion changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryBuildError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise RegistryBuildError(f"{label} is outside its finite range")
    return result


def _finite_array(value: Any, label: str) -> np.ndarray:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result = np.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive format boundary
        raise RegistryBuildError(f"{label} cannot be converted to an array") from exc
    if result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise RegistryBuildError(f"{label} must be a finite numeric array")
    return result


def _object_mapping(archive: Any, key: str, source: str) -> dict[str, Any]:
    try:
        value = archive[key]
        if value.shape != () or value.dtype.kind != "O":
            raise RegistryBuildError(f"{source}:{key} is not a scalar object")
        item = value.item()
    except RegistryBuildError:
        raise
    except Exception as exc:
        raise RegistryBuildError(f"cannot read {source}:{key}") from exc
    if not isinstance(item, Mapping):
        raise RegistryBuildError(f"{source}:{key} is not a mapping")
    return dict(item)


def _asset_positions(
    asset: Mapping[str, Any], *, asset_name: str, source: str, frames: int
) -> tuple[list[str], np.ndarray]:
    names = asset.get("body_names")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(item, str) or not item for item in names)
        or len(names) != len(set(names))
    ):
        raise RegistryBuildError(f"{source}:{asset_name}.body_names are invalid")
    expected_shapes = {
        "body_positions": (frames, len(names), 3),
        "body_rotations": (frames, len(names), 4),
        "body_linear_velocities": (frames, len(names), 3),
        "body_angular_velocities": (frames, len(names), 3),
    }
    arrays: dict[str, np.ndarray] = {}
    for field, expected_shape in expected_shapes.items():
        value = _finite_array(asset.get(field), f"{source}:{asset_name}.{field}")
        if value.shape != expected_shape:
            raise RegistryBuildError(f"{source}:{asset_name}.{field} shape differs")
        arrays[field] = value
    return names, arrays["body_positions"]


def _bin(value: float, cuts: tuple[float, float], labels: tuple[str, str, str]) -> str:
    if value < cuts[0]:
        return labels[0]
    if value < cuts[1]:
        return labels[1]
    return labels[2]


def _measure(path: Path, source_root: Path) -> dict[str, Any]:
    match = MOTION_NAME.fullmatch(path.name)
    if match is None:
        raise RegistryBuildError(f"unexpected motion basename: {path.name}")
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise RegistryBuildError(f"motion is not a regular non-symlink file: {path}")

    try:
        with ZipFile(path) as zipped:
            members = zipped.namelist()
            if len(members) != len(set(members)) or set(members) != EXPECTED_NPZ_MEMBERS:
                raise RegistryBuildError(f"{path.name} NPZ member set differs")
            if any(item.flag_bits & 0x1 for item in zipped.infolist()):
                raise RegistryBuildError(f"{path.name} contains an encrypted member")
    except BadZipFile as exc:
        raise RegistryBuildError(f"{path.name} is not a valid NPZ") from exc

    # The repository's motion-loader format stores dictionaries containing
    # torch tensors in scalar NumPy object arrays, so pickle loading is required.
    # The strict path/member checks above constrain the accepted input surface.
    try:
        with np.load(path, allow_pickle=True) as archive:
            if set(archive.files) != {item.removesuffix(".npy") for item in EXPECTED_NPZ_MEMBERS}:
                raise RegistryBuildError(f"{path.name} logical NPZ keys differ")
            homie = _finite_array(archive["homie_command"], f"{path.name}:homie_command")
            primitive = _finite_array(
                archive["primitive_action"], f"{path.name}:primitive_action"
            )
            if homie.ndim != 2 or homie.shape[1] != 7 or homie.shape[0] < 2:
                raise RegistryBuildError(f"{path.name} homie_command shape differs")
            frames = int(homie.shape[0])
            if primitive.shape != (frames, 2):
                raise RegistryBuildError(f"{path.name} primitive_action shape differs")

            assets = {
                key: _object_mapping(archive, key, path.name)
                for key in (
                    "robot",
                    "table_grid",
                    "bottle_table",
                    "bottle_hand",
                    "tray_table",
                )
            }
    except RegistryBuildError:
        raise
    except Exception as exc:
        raise RegistryBuildError(f"cannot load motion archive: {path.name}") from exc

    robot = assets["robot"]
    table = assets["table_grid"]
    bottle = assets["bottle_table"]
    positions = {
        key: _asset_positions(value, asset_name=key, source=path.name, frames=frames)
        for key, value in assets.items()
    }
    robot_names, robot_positions = positions["robot"]
    bottle_names, bottle_positions = positions["bottle_table"]
    if robot_names.count("pelvis") != 1:
        raise RegistryBuildError(f"{path.name} must identify one pelvis body")
    if bottle_names.count("BottleTable") != 1:
        raise RegistryBuildError(f"{path.name} must identify one BottleTable body")

    fps = _finite_number(robot.get("fps"), f"{path.name}:robot.fps", positive=True)
    for asset_name, asset in assets.items():
        asset_fps = _finite_number(
            asset.get("fps"), f"{path.name}:{asset_name}.fps", positive=True
        )
        if asset_fps != fps:
            raise RegistryBuildError(f"{path.name} asset FPS values differ")

    dof_names = robot.get("dof_names")
    if (
        not isinstance(dof_names, list)
        or not dof_names
        or any(not isinstance(item, str) or not item for item in dof_names)
        or len(dof_names) != len(set(dof_names))
    ):
        raise RegistryBuildError(f"{path.name}:robot.dof_names are invalid")
    for field in (
        "dof_positions",
        "dof_velocities",
        "dof_position_targets",
        "dof_velocity_targets",
        "dof_effort_targets",
    ):
        value = _finite_array(robot.get(field), f"{path.name}:robot.{field}")
        if value.shape != (frames, len(dof_names)):
            raise RegistryBuildError(f"{path.name}:robot.{field} shape differs")

    filename_height_m = _finite_number(float(match.group(1)), "filename table height")
    custom_data = table.get("custom_data")
    if not isinstance(custom_data, Mapping) or not custom_data:
        raise RegistryBuildError(f"{path.name} has no table custom_data")
    if any(
        not isinstance(key, str) or not key or not isinstance(value, Mapping)
        for key, value in custom_data.items()
    ):
        raise RegistryBuildError(f"{path.name} table custom_data entries are invalid")
    embedded_heights = sorted({
        _finite_number(value.get("table_height"), f"{path.name}:{key}.table_height")
        for key, value in custom_data.items()
    })
    if not embedded_heights or any(
        abs(value - filename_height_m) > 1e-9 for value in embedded_heights
    ):
        raise RegistryBuildError(f"{path.name} filename/embedded table heights differ")
    height_mm = filename_height_m * 1000.0
    if abs(height_mm - round(height_mm)) > 1e-9:
        raise RegistryBuildError(f"{path.name} table height is not integral millimetres")

    pelvis = robot_positions[0, robot_names.index("pelvis")].astype(np.float64)
    bottle_table = bottle_positions[0, bottle_names.index("BottleTable")].astype(np.float64)
    delta_xy = bottle_table[:2] - pelvis[:2]
    planar_distance = float(np.linalg.norm(delta_xy))
    world_bearing = math.degrees(math.atan2(float(delta_xy[1]), float(delta_xy[0])))
    if not math.isfinite(planar_distance) or not math.isfinite(world_bearing):
        raise RegistryBuildError(f"{path.name} initial geometry is nonfinite")

    strata = {
        "table_height": f"{round(height_mm):d}mm",
        "initial_planar_distance_bin": _bin(
            planar_distance,
            DISTANCE_CUTS_M,
            ("q1_near", "q2_middle", "q3_far"),
        ),
        "initial_world_bearing_bin": _bin(
            world_bearing,
            WORLD_BEARING_CUTS_DEG,
            ("q1_low", "q2_middle", "q3_high"),
        ),
        "trajectory_length_bin": _bin(
            float(frames),
            TRAJECTORY_FRAME_CUTS,
            ("q1_short", "q2_middle", "q3_long"),
        ),
    }
    relative = path.relative_to(source_root).as_posix()
    return {
        "group_id": path.name,
        "relative_path": relative,
        "size_bytes": info.st_size,
        "motion_sha256": _file_sha256(path, info),
        "table_height_m": filename_height_m,
        "embedded_table_heights_m": embedded_heights,
        "trajectory_frames": frames,
        "fps": fps,
        "duration_seconds": (frames - 1) / fps,
        "initial_pelvis_xyz_m": pelvis.tolist(),
        "initial_bottle_table_xyz_m": bottle_table.tolist(),
        "initial_planar_distance_m": planar_distance,
        "initial_world_bearing_degrees": world_bearing,
        "strata": strata,
    }


def build(source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_root = source_root.resolve(strict=True)
    motion_root = source_root / MOTION_DIRECTORY
    if not motion_root.is_dir() or motion_root.is_symlink():
        raise RegistryBuildError("motion directory is missing or is a symlink")
    paths = sorted(motion_root.glob(MOTION_GLOB), key=lambda item: item.name)
    if len(paths) != EXPECTED_COUNT:
        raise RegistryBuildError(
            f"expected {EXPECTED_COUNT} original motions, found {len(paths)}"
        )
    rows = [_measure(path, source_root) for path in paths]
    if len({row["group_id"] for row in rows}) != EXPECTED_COUNT:
        raise RegistryBuildError("motion basenames are not unique")
    if len({row["relative_path"] for row in rows}) != EXPECTED_COUNT:
        raise RegistryBuildError("motion paths are not unique")
    if len({row["motion_sha256"] for row in rows}) != EXPECTED_COUNT:
        raise RegistryBuildError("motion content hashes are not unique")

    registry = {
        "schema": REGISTRY_SCHEMA,
        "groups": [
            {
                "group_id": row["group_id"],
                "relative_path": row["relative_path"],
                "size_bytes": row["size_bytes"],
                "motion_sha256": row["motion_sha256"],
                "strata": row["strata"],
            }
            for row in rows
        ],
    }
    stratum_counts = {
        field: dict(sorted(Counter(row["strata"][field] for row in rows).items()))
        for field in (
            "table_height",
            "initial_planar_distance_bin",
            "initial_world_bearing_bin",
            "trajectory_length_bin",
        )
    }
    for field in (
        "initial_planar_distance_bin",
        "initial_world_bearing_bin",
        "trajectory_length_bin",
    ):
        if sorted(stratum_counts[field].values()) != [11, 11, 11]:
            raise RegistryBuildError(f"fixed {field} cuts no longer form 11/11/11 bins")

    inventory_commitment = {
        "motion_directory": MOTION_DIRECTORY,
        "members": [
            {
                "relative_path": row["relative_path"],
                "size_bytes": row["size_bytes"],
                "motion_sha256": row["motion_sha256"],
            }
            for row in rows
        ],
    }
    audit = {
        "schema": AUDIT_SCHEMA,
        "registry_schema": REGISTRY_SCHEMA,
        "selection": {
            "source_root_basis": "repository_root",
            "motion_directory": MOTION_DIRECTORY,
            "basename_glob": MOTION_GLOB,
            "required_count": EXPECTED_COUNT,
            "required_basename_date": "2025-10-14",
        },
        "measurement_definitions": {
            "table_height_m": (
                "decimal metres parsed from the exact NPZ basename and checked against "
                "every table_grid.custom_data table_height"
            ),
            "trajectory_frames": (
                "homie_command axis-0 length, cross-checked against primitive_action and "
                "every robot/object body-state array"
            ),
            "duration_seconds": "(trajectory_frames - 1) / robot.fps",
            "initial_planar_distance_m": (
                "Euclidean XY distance at frame 0 from robot pelvis to BottleTable"
            ),
            "initial_world_bearing_degrees": (
                "atan2(BottleTable_y - pelvis_y, BottleTable_x - pelvis_x) at frame 0; "
                "world frame, degrees"
            ),
        },
        "binning": {
            "boundary_rule": "lower if value < cut; values on a cut enter the higher bin",
            "initial_planar_distance_bin": {
                "cuts_m": list(DISTANCE_CUTS_M),
                "labels": ["q1_near", "q2_middle", "q3_far"],
            },
            "initial_world_bearing_bin": {
                "cuts_degrees": list(WORLD_BEARING_CUTS_DEG),
                "labels": ["q1_low", "q2_middle", "q3_high"],
            },
            "trajectory_length_bin": {
                "cuts_frames": list(TRAJECTORY_FRAME_CUTS),
                "labels": ["q1_short", "q2_middle", "q3_long"],
            },
            "cut_derivation": (
                "fixed midpoint between sorted ranks 11/12 and 22/23 in this original "
                "33-motion inventory"
            ),
        },
        "inventory": rows,
        "summary": {
            "motion_count": len(rows),
            "total_size_bytes": sum(row["size_bytes"] for row in rows),
            "stratum_counts": stratum_counts,
            "inventory_sha256": _digest(inventory_commitment),
            "group_registry_sha256": _digest(registry),
            "limitations": [
                "All source motions have the same 2025-10-14 recording-date prefix; "
                "representativeness beyond this fixed inventory is not established.",
                "Only filename table height, initial geometry, and trajectory length are "
                "stratified; the files provide no audited lighting, material, friction, "
                "task-outcome, or difficulty labels.",
                "Bearing is measured in the world XY frame, not relative to robot heading.",
                "The audit contains no train, validation, or test assignment and no test "
                "case identity or test seed.",
            ],
        },
    }
    return registry, audit


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RegistryBuildError(f"cannot read JSON: {path}") from exc


def _write_atomic(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    staged = Path(staged_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        try:
            staged.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=repository_root)
    parser.add_argument(
        "--registry",
        type=Path,
        default=repository_root / "configs/round_dagger/native_h200_group_registry.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=repository_root / "configs/round_dagger/native_h200_group_registry_audit.json",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    registry, audit = build(args.source_root)
    if args.check:
        if _read_json(args.registry) != registry:
            raise RegistryBuildError(f"registry is stale: {args.registry}")
        if _read_json(args.audit) != audit:
            raise RegistryBuildError(f"audit is stale: {args.audit}")
    else:
        _write_atomic(args.registry, registry)
        _write_atomic(args.audit, audit)
    print(
        json.dumps(
            {
                "motion_count": len(registry["groups"]),
                "registry_sha256": _digest(registry),
                "inventory_sha256": audit["summary"]["inventory_sha256"],
                "mode": "check" if args.check else "write",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RegistryBuildError as exc:
        print(f"native-H200 registry build failed: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
