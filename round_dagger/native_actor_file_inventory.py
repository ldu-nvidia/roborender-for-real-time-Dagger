"""Strict native-actor file inventory verification.

The v1 row shape remains ``path,size_bytes,sha256``.  Regular rows hash file
bytes.  An explicitly allowlisted symlink row hashes the raw link-target bytes.
A regular-file target must have its own normal row; a directory target must
have every file below its canonical path inventoried.  Launchers may impose a
stricter absolute-link policy for executable entry points.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence


SCHEMA = "fast_roborender.round_dagger.native_actor_file_inventory.v1"


class NativeActorInventoryError(RuntimeError):
    """A runtime/source inventory is malformed or differs from disk."""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _local_round_dagger_module_paths(
    actor_root: Path, module_name: str
) -> tuple[Path, ...]:
    """Resolve one local ``round_dagger`` module and its package initializers."""

    parts = module_name.split(".")
    if not parts or parts[0] != "round_dagger" or any(not part for part in parts):
        return ()
    module_stem = actor_root.joinpath(*parts)
    module_file = module_stem.with_suffix(".py")
    package_file = module_stem / "__init__.py"
    if module_file.is_file():
        resolved = module_file
    elif package_file.is_file():
        resolved = package_file
    else:
        return ()
    package_initializers = tuple(
        actor_root.joinpath(*parts[:index], "__init__.py")
        for index in range(1, len(parts))
        if actor_root.joinpath(*parts[:index], "__init__.py").is_file()
    )
    return (*package_initializers, resolved)


def actor_source_import_closure(
    *, actor_root: Path, entry_paths: Sequence[Path]
) -> frozenset[Path]:
    """Return the deterministic local-Python import closure for actor entries.

    Only imports in the repository's ``round_dagger`` package belong to this
    source inventory; third-party and GR00T imports are sealed independently.
    The scan includes imports inside functions and guarded branches because
    they remain executable actor code.
    """

    actor_root = Path(actor_root)
    package_init = actor_root / "round_dagger/__init__.py"
    pending = {Path(path) for path in entry_paths if Path(path).suffix == ".py"}
    if pending and not package_init.is_file():
        raise NativeActorInventoryError(
            f"actor source package initializer is missing: {package_init}"
        )
    pending.add(package_init)
    closure: set[Path] = set()

    def require_local_module(module_name: str, *, imported_from: Path) -> None:
        if module_name != "round_dagger" and not module_name.startswith(
            "round_dagger."
        ):
            return
        resolved = _local_round_dagger_module_paths(actor_root, module_name)
        if not resolved:
            raise NativeActorInventoryError(
                f"actor source import cannot be resolved: {module_name} "
                f"from {imported_from}"
            )
        pending.update(resolved)

    while pending:
        source = min(pending, key=os.fspath)
        pending.remove(source)
        if source in closure:
            continue
        if not _inside(source, actor_root) or not source.is_file():
            raise NativeActorInventoryError(
                f"actor source import entry is unsafe: {source}"
            )
        try:
            tree = ast.parse(
                source.read_text(encoding="utf-8"), filename=os.fspath(source)
            )
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise NativeActorInventoryError(
                f"cannot parse actor source import entry: {source}"
            ) from exc
        closure.add(source)

        relative = source.relative_to(actor_root)
        module_parts = list(relative.with_suffix("").parts)
        if module_parts[-1] == "__init__":
            package_parts = module_parts[:-1]
        else:
            package_parts = module_parts[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    require_local_module(alias.name, imported_from=source)
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                if relative.parts[0] != "round_dagger":
                    raise NativeActorInventoryError(
                        f"actor script uses an invalid relative import: {source}"
                    )
                trim = node.level - 1
                if trim >= len(package_parts):
                    raise NativeActorInventoryError(
                        f"actor source relative import leaves its package: {source}"
                    )
                base_parts = package_parts[: len(package_parts) - trim]
                if node.module:
                    base_parts.extend(node.module.split("."))
                module_name = ".".join(base_parts)
            else:
                module_name = node.module or ""
            require_local_module(module_name, imported_from=source)
            for alias in node.names:
                if alias.name == "*" or not module_name:
                    continue
                # ``from round_dagger import artifacts`` names a submodule,
                # while ``from round_dagger.artifacts import ArtifactError``
                # names an attribute.  Include the former when it exists.
                child_name = f"{module_name}.{alias.name}"
                resolved = _local_round_dagger_module_paths(actor_root, child_name)
                if resolved:
                    pending.update(resolved)
    return frozenset(closure)


def inventory_row(path: Path, *, symlink_target: Path | None = None) -> dict:
    """Return the canonical v1 row for a regular file or allowed symlink."""

    path = Path(path)
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        raw_target = os.readlink(path)
        raw_path = Path(raw_target)
        resolved_target = (
            raw_path if raw_path.is_absolute() else path.parent / raw_path
        ).resolve(strict=True)
        if (
            symlink_target is None
            or Path(os.path.normpath(raw_target)) != raw_path
            or resolved_target != Path(symlink_target)
        ):
            raise NativeActorInventoryError(
                f"inventory symlink is not explicitly allowed: {path}"
            )
        payload = os.fsencode(raw_target)
        size_bytes = len(payload)
        sha256 = hashlib.sha256(payload).hexdigest()
    elif stat.S_ISREG(info.st_mode) and path.resolve(strict=True) == path:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        digest = hashlib.sha256()
        size_bytes = 0
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)
            ):
                raise NativeActorInventoryError(
                    f"inventory file changed while opening: {path}"
                )
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
        if size_bytes != opened.st_size:
            raise NativeActorInventoryError(
                f"inventory file changed while hashing: {path}"
            )
        sha256 = digest.hexdigest()
    else:
        raise NativeActorInventoryError(f"inventory path is not a safe file: {path}")
    return {
        "path": str(path),
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def inventory_tree_contract(
    root: Path, *, allowed_target_roots: Sequence[Path] | None = None
) -> tuple[frozenset[Path], dict[Path, Path]]:
    """Enumerate every file in a sealed import/library tree, fail closed.

    File and directory symlinks are accepted only when their canonical target
    is in one of the explicitly supplied roots.  The raw link bytes are part
    of the inventory.  A regular-file target gets its own normal row; a
    directory target is traversed through its canonical path without following
    the link.  This binds common runtime aliases such as ``lib64 -> lib`` while
    preventing a sealed tree from importing bytes through an external link.
    """

    root = Path(root)
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
    ):
        raise NativeActorInventoryError(f"inventory tree root is unsafe: {root}")
    target_roots = tuple(Path(item) for item in (allowed_target_roots or (root,)))
    if any(
        not item.is_absolute()
        or item.is_symlink()
        or not item.is_dir()
        or item.resolve(strict=True) != item
        for item in target_roots
    ):
        raise NativeActorInventoryError(
            "inventory allowed-target root is unsafe"
        )
    paths: set[Path] = set()
    symlinks: dict[Path, Path] = {}
    pending = [root]
    visited: set[Path] = set()
    while pending:
        tree = pending.pop()
        tree = tree.resolve(strict=True)
        if tree in visited:
            continue
        if not any(_inside(tree, allowed) for allowed in target_roots):
            raise NativeActorInventoryError(
                f"inventory tree target leaves its allowed roots: {tree}"
            )
        visited.add(tree)
        for directory, directory_names, file_names in os.walk(
            tree, followlinks=False
        ):
            directory_path = Path(directory)
            for name in directory_names:
                candidate = directory_path / name
                if not candidate.is_symlink():
                    continue
                target = candidate.resolve(strict=True)
                target_info = os.lstat(target)
                if (
                    not stat.S_ISDIR(target_info.st_mode)
                    or target.is_symlink()
                    or not any(
                        _inside(target, allowed) for allowed in target_roots
                    )
                ):
                    raise NativeActorInventoryError(
                        f"inventory tree directory symlink target is unsafe: {candidate}"
                    )
                paths.add(candidate)
                symlinks[candidate] = target
                pending.append(target)
            for name in file_names:
                candidate = directory_path / name
                info = os.lstat(candidate)
                if stat.S_ISREG(info.st_mode):
                    if candidate.resolve(strict=True) != candidate:
                        raise NativeActorInventoryError(
                            f"inventory tree file is non-canonical: {candidate}"
                        )
                    paths.add(candidate)
                elif stat.S_ISLNK(info.st_mode):
                    target = candidate.resolve(strict=True)
                    target_info = os.lstat(target)
                    if (
                        not stat.S_ISREG(target_info.st_mode)
                        or target.is_symlink()
                        or not any(
                            _inside(target, allowed) for allowed in target_roots
                        )
                    ):
                        raise NativeActorInventoryError(
                            f"inventory tree file symlink target is unsafe: {candidate}"
                        )
                    paths.update((candidate, target))
                    symlinks[candidate] = target
                else:
                    raise NativeActorInventoryError(
                        f"inventory tree contains a non-file entry: {candidate}"
                    )
    if not paths:
        raise NativeActorInventoryError(f"inventory tree is empty: {root}")
    return frozenset(paths), symlinks


def verify_file_inventory(
    manifest: Path,
    *,
    kind: str,
    allowed_roots: Sequence[Path],
    required_paths: Sequence[Path],
    allowed_symlinks: Mapping[Path, Path] | None = None,
    rehash_paths: Sequence[Path] | None = None,
) -> int:
    """Rehash a strict inventory and require all named live inputs."""

    allowed_symlinks = {
        Path(link): Path(target) for link, target in (allowed_symlinks or {}).items()
    }
    roots = tuple(Path(root) for root in allowed_roots)
    rehash = None if rehash_paths is None else {Path(path) for path in rehash_paths}
    try:
        document = json.loads(Path(manifest).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeActorInventoryError(f"cannot load {kind} inventory") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "kind", "entries"}
        or document["schema"] != SCHEMA
        or document["kind"] != kind
    ):
        raise NativeActorInventoryError(f"{kind} inventory identity differs")
    entries = document["entries"]
    if not isinstance(entries, list) or not entries:
        raise NativeActorInventoryError(f"{kind} inventory is empty")
    names: list[str] = []
    rows_by_name: dict[str, dict] = {}
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict) or set(raw) != {
            "path", "size_bytes", "sha256"
        }:
            raise NativeActorInventoryError(
                f"{kind} inventory row {index} fields differ"
            )
        if not isinstance(raw["path"], str):
            raise NativeActorInventoryError(
                f"{kind} inventory row {index} path is malformed"
            )
        candidate = Path(raw["path"])
        if (
            not candidate.is_absolute()
            or Path(os.path.normpath(os.fspath(candidate))) != candidate
            or not any(_inside(candidate, root) for root in roots)
        ):
            raise NativeActorInventoryError(
                f"{kind} inventory path is outside its sealed roots: {candidate}"
            )
        expected_target = allowed_symlinks.get(candidate)
        if rehash is None or candidate in rehash:
            observed = inventory_row(candidate, symlink_target=expected_target)
            if observed != raw:
                raise NativeActorInventoryError(
                    f"{kind} inventory row differs: {candidate}"
                )
        if expected_target is not None and (rehash is None or candidate in rehash):
            target_info = os.lstat(expected_target)
            if (
                not (
                    stat.S_ISREG(target_info.st_mode)
                    or stat.S_ISDIR(target_info.st_mode)
                )
                or expected_target.is_symlink()
                or expected_target.resolve(strict=True) != expected_target
                or not any(_inside(expected_target, root) for root in roots)
            ):
                raise NativeActorInventoryError(
                    f"{kind} inventory symlink target is unsafe: {candidate}"
                )
        names.append(str(candidate))
        rows_by_name[str(candidate)] = raw
    if names != sorted(set(names)):
        raise NativeActorInventoryError(
            f"{kind} inventory paths must be sorted and unique"
        )
    required_names = {str(Path(path)) for path in required_paths}
    missing = sorted(required_names - set(names))
    if missing:
        raise NativeActorInventoryError(
            f"{kind} inventory omits required live inputs: {missing}"
        )
    if rehash is not None:
        invalid_rehash = sorted(str(path) for path in rehash if str(path) not in required_names)
        if invalid_rehash:
            raise NativeActorInventoryError(
                f"{kind} rehash set is not a required live-input subset: {invalid_rehash}"
            )
    for link, target in allowed_symlinks.items():
        if str(link) not in names:
            raise NativeActorInventoryError(
                f"{kind} inventory omits allowed symlink: {link}"
            )
        if inventory_row(link, symlink_target=target) != rows_by_name[str(link)]:
            raise NativeActorInventoryError(
                f"{kind} inventory symlink row differs: {link}"
            )
        target_info = os.lstat(target)
        if (
            not (
                stat.S_ISREG(target_info.st_mode)
                or stat.S_ISDIR(target_info.st_mode)
            )
            or target.is_symlink()
            or target.resolve(strict=True) != target
            or not any(_inside(target, root) for root in roots)
        ):
            raise NativeActorInventoryError(
                f"{kind} inventory symlink target is unsafe: {link}"
            )
        if stat.S_ISREG(target_info.st_mode) and str(target) not in names:
            raise NativeActorInventoryError(
                f"{kind} inventory omits resolved symlink target: {target}"
            )
        if stat.S_ISDIR(target_info.st_mode) and rehash is None:
            target_paths, target_links = inventory_tree_contract(
                target, allowed_target_roots=roots
            )
            missing_targets = sorted(
                str(path) for path in target_paths if str(path) not in names
            )
            if missing_targets:
                raise NativeActorInventoryError(
                    f"{kind} inventory omits directory-link target files: "
                    f"{missing_targets[:3]}"
                )
            inconsistent_links = sorted(
                str(path)
                for path, resolved in target_links.items()
                if allowed_symlinks.get(path) != resolved
            )
            if inconsistent_links:
                raise NativeActorInventoryError(
                    f"{kind} inventory omits nested allowed symlinks: "
                    f"{inconsistent_links[:3]}"
                )
    return len(entries)


def verify_native_actor_bundle(
    *,
    actor_manifest: Path,
    gr00t_manifest: Path,
    isaac_manifest: Path,
    python_manifest: Path,
    actor_root: Path,
    gr00t_root: Path,
    gr00t_python: Path,
    isaac_root: Path,
    python_root: Path,
    isaac_setup: Path,
    headless_experience: Path,
    cudnn_root: Path,
    isaaclab_source_root: Path,
    dependency_root: Path,
    system_library_root: Path,
    full_runtime_trees: bool,
) -> dict[str, int]:
    """Verify the one canonical native-actor source/runtime bundle.

    The trusted fleet controller performs the full tree rehash once before it
    launches training.  Each jailed rank then reparses the same hash-pinned
    manifests and rehashes only live entry points after QROOT has been
    bind-remounted read-only.  This avoids re-reading a 1.9 GiB package tree on
    every rank and probe while the training process still cannot alter or
    replace any verified runtime byte.
    """

    paths = {
        "actor_manifest": Path(actor_manifest),
        "gr00t_manifest": Path(gr00t_manifest),
        "isaac_manifest": Path(isaac_manifest),
        "python_manifest": Path(python_manifest),
        "actor_root": Path(actor_root),
        "gr00t_root": Path(gr00t_root),
        "gr00t_python": Path(gr00t_python),
        "isaac_root": Path(isaac_root),
        "python_root": Path(python_root),
        "isaac_setup": Path(isaac_setup),
        "headless_experience": Path(headless_experience),
        "cudnn_root": Path(cudnn_root),
        "isaaclab_source_root": Path(isaaclab_source_root),
        "dependency_root": Path(dependency_root),
        "system_library_root": Path(system_library_root),
    }
    actor_root = paths["actor_root"]
    gr00t_root = paths["gr00t_root"]
    gr00t_python = paths["gr00t_python"]
    gr00t_python_target = Path(os.readlink(gr00t_python))
    actor_required = {
        actor_root / "round_dagger/__init__.py",
        actor_root / "round_dagger/actor.py",
        actor_root / "round_dagger/actor_checkpoint_compat.py",
        actor_root / "round_dagger/actor_identity.py",
        actor_root / "round_dagger/artifacts.py",
        actor_root / "round_dagger/checkpoint_compat.py",
        actor_root / "round_dagger/conditioning_artifacts.py",
        actor_root / "round_dagger/conditioning_capture.py",
        actor_root / "round_dagger/first_condition_bundle.py",
        actor_root / "round_dagger/gr00t_policy.py",
        actor_root / "round_dagger/native_actor_file_inventory.py",
        actor_root / "round_dagger/native_actor_protocol.py",
        actor_root / "round_dagger/native_actor_saturation.py",
        actor_root / "round_dagger/policy_identity.py",
        actor_root / "round_dagger/production_data_lineage.py",
        actor_root / "round_dagger/scenario_capture.py",
        actor_root / "round_dagger/seed_policy.py",
        actor_root / "scripts/round_dagger/actor_training_entrypoint.py",
        actor_root / "scripts/round_dagger/native_h200_bc_wave_artifact.py",
        actor_root / "scripts/round_dagger/run_native_h200_actor_fleet.py",
        actor_root / "scripts/round_dagger/run_native_h200_bc_collection.sh",
        actor_root / "scripts/round_dagger/run_native_h200_actor_rank.sh",
        paths["isaac_setup"],
    }
    actor_import_closure = actor_source_import_closure(
        actor_root=actor_root,
        entry_paths=tuple(actor_required),
    )
    missing_actor_imports = sorted(
        actor_import_closure.difference(actor_required), key=os.fspath
    )
    if missing_actor_imports:
        raise NativeActorInventoryError(
            "actor source inventory required set omits reachable local imports: "
            + ", ".join(os.fspath(path) for path in missing_actor_imports)
        )
    gr00t_required = {
        gr00t_root / "gr00t/rl/train_agent_trl.py",
        gr00t_root / "gr00t/rl/agents/modules/modules.py",
        gr00t_root / "gr00t/rl/trl/trainer/ppo_trainer.py",
        gr00t_root / "gr00t/rl/trl/trainer/distill_trainer_native_dagger.py",
        gr00t_root / "gr00t/rl/utils/native_dagger_contract.py",
        gr00t_root / "gr00t/rl/simulator/isaacsim/isaacsim.py",
        gr00t_root / "gr00t/rl/envs/legged_base_task/legged_robot_base.py",
        gr00t_root
        / "gr00t/rl/config/exp/loco_manip/wsdpt_round_actor_native_3view.yaml",
    }
    # Bind every importable GR00T source/config byte and every local asset the
    # native task can load.  The full motion library is intentionally excluded:
    # the actor sees a separately hashed, train-only bind mount in its place.
    # Unrelated robot/object libraries are excluded as well; this task loads the
    # G1 tree, bottle, and tray only.
    gr00t_package = gr00t_root / "gr00t"
    motion_tree = gr00t_package / "rl/data/motions"
    object_tree = gr00t_package / "rl/data/objects"
    robot_tree = gr00t_package / "rl/data/robots"
    g1_tree = robot_tree / "g1"
    exact_objects = {
        object_tree / "grab/bottle.usd",
        object_tree / "simple/tray-28-28-10-thick-15mm.usd",
    }
    for item in gr00t_package.rglob("*"):
        if not item.is_file() or item.is_symlink():
            continue
        if _inside(item, motion_tree):
            continue
        if _inside(item, object_tree) and item not in exact_objects:
            continue
        if _inside(item, robot_tree) and not _inside(item, g1_tree):
            continue
        gr00t_required.add(item)
    critical_isaac = {
        paths["isaac_setup"],
        paths["headless_experience"],
        gr00t_python,
        gr00t_python_target,
        paths["isaaclab_source_root"] / "isaaclab/isaaclab/app/app_launcher.py",
        paths["isaaclab_source_root"] / "isaaclab_rl/isaaclab_rl/__init__.py",
    }
    critical_python = {
        gr00t_python,
        gr00t_python_target,
        paths["dependency_root"] / "hydra/__init__.py",
        paths["dependency_root"] / "omegaconf/__init__.py",
    }
    isaac_links = {gr00t_python: gr00t_python_target}
    isaac_required = set(critical_isaac)
    python_required = set(critical_python)
    if full_runtime_trees:
        isaac_files, isaac_tree_links = inventory_tree_contract(
            paths["isaac_root"],
            allowed_target_roots=[paths["isaac_root"], paths["python_root"]],
        )
        isaac_required.update(isaac_files)
        isaac_links.update(isaac_tree_links)
        for root_name in (
            "isaaclab_source_root",
            "system_library_root",
            "cudnn_root",
        ):
            files, links = inventory_tree_contract(
                paths[root_name], allowed_target_roots=[paths[root_name]]
            )
            isaac_required.update(files)
            isaac_links.update(links)
        dependency_files, dependency_links = inventory_tree_contract(
            paths["dependency_root"],
            allowed_target_roots=[paths["dependency_root"]],
        )
        if dependency_links:
            raise NativeActorInventoryError(
                "qualified Python dependency root contains a symlink"
            )
        python_required.update(dependency_files)
    counts = {
        "actor_source": verify_file_inventory(
            paths["actor_manifest"],
            kind="actor_source",
            allowed_roots=[actor_root],
            required_paths=actor_required,
        ),
        "gr00t_source": verify_file_inventory(
            paths["gr00t_manifest"],
            kind="gr00t_source",
            allowed_roots=[gr00t_root],
            required_paths=gr00t_required,
        ),
        "isaac_runtime": verify_file_inventory(
            paths["isaac_manifest"],
            kind="isaac_runtime",
            allowed_roots=[
                actor_root,
                paths["isaac_root"],
                paths["python_root"],
                paths["cudnn_root"],
                paths["isaaclab_source_root"],
                paths["system_library_root"],
            ],
            required_paths=isaac_required,
            allowed_symlinks=isaac_links,
            rehash_paths=None if full_runtime_trees else critical_isaac,
        ),
        "python_runtime": verify_file_inventory(
            paths["python_manifest"],
            kind="python_runtime",
            allowed_roots=[
                paths["isaac_root"],
                paths["python_root"],
                paths["dependency_root"],
            ],
            required_paths=python_required,
            allowed_symlinks={gr00t_python: gr00t_python_target},
            rehash_paths=None if full_runtime_trees else critical_python,
        ),
    }
    return counts
