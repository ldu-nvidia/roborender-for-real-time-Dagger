from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest import mock

from round_dagger.native_actor_file_inventory import (
    NativeActorInventoryError,
    SCHEMA,
    actor_source_import_closure,
    inventory_row,
    inventory_tree_contract,
    verify_native_actor_bundle,
    verify_file_inventory,
)
from round_dagger.native_actor_saturation import (
    ProcessWaveRunner,
    sanitize_actor_child_environment,
)


def _device_namespaces(inventory):
    return [
        {
            "schema": "fast_roborender.round_dagger.native_actor_gpu_device_namespace.v1",
            "rank": rank,
            "physical_index": rank,
            "uuid": gpu["uuid"],
            "pci_bus_id": gpu.get("pci_bus_id", "00000000:01:00.0"),
            "pci_function": f"0000:{rank + 1:02x}:00.0",
            "nvidia_device": {
                "path": f"/dev/nvidia{(rank + 4) % 8}",
                "major": 195,
                "minor": (rank + 4) % 8,
            },
            "drm_card_device": {
                "path": f"/dev/dri/card{(rank + 4) % 8}",
                "major": 226,
                "minor": (rank + 4) % 8,
            },
            "drm_render_device": {
                "path": f"/dev/dri/renderD{128 + (rank + 4) % 8}",
                "major": 226,
                "minor": 128 + (rank + 4) % 8,
            },
            "common_control_devices": [],
        }
        for rank, gpu in enumerate(inventory)
    ]


class NativeActorSourceJailTest(unittest.TestCase):
    def test_rank_launcher_gpu_namespace_shell_is_syntactically_valid(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts/round_dagger/run_native_h200_actor_rank.sh"
        )

    def test_rank_launcher_has_exact_seven_unique_pinned_inputs_at_each_stage(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts/round_dagger/run_native_h200_actor_rank.sh"
        )
        source = launcher.read_text(encoding="utf-8")
        blocks = re.findall(r"(?ms)^\s*pinned_inputs=\(\n(.*?)^\s*\)$", source)
        self.assertEqual(len(blocks), 3)
        expected = (
            "ROUND_DAGGER_POLICY_CHECKPOINT",
            "ROUND_DAGGER_TEACHER_CHECKPOINT",
            "ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST",
            "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST",
            "ROUND_DAGGER_ACTOR_SOURCE_INVENTORY",
            "ROUND_DAGGER_GR00T_SOURCE_INVENTORY",
            "native_config",
        )
        for block in blocks:
            names = tuple(re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)", block))
            self.assertEqual(names, expected)
            self.assertEqual(len(names), len(set(names)))
        subprocess.run(
            ["bash", "-n", str(launcher)],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_actor_source_import_closure_is_recursive_and_binds_package_init(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            package = root / "round_dagger"
            package.mkdir()
            initializer = package / "__init__.py"
            entry = package / "entry.py"
            helper = package / "helper.py"
            leaf = package / "leaf.py"
            unrelated = package / "unrelated.py"
            initializer.write_text("", encoding="utf-8")
            entry.write_text("from .helper import helper\n", encoding="utf-8")
            helper.write_text(
                "from round_dagger import leaf\n\ndef helper():\n    return leaf\n",
                encoding="utf-8",
            )
            leaf.write_text("VALUE = 1\n", encoding="utf-8")
            unrelated.write_text("VALUE = 2\n", encoding="utf-8")

            self.assertEqual(
                actor_source_import_closure(
                    actor_root=root,
                    entry_paths=(entry,),
                ),
                frozenset((initializer, entry, helper, leaf)),
            )

    def test_actor_source_import_closure_rejects_unresolved_local_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            package = root / "round_dagger"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            entry = package / "entry.py"
            entry.write_text("import round_dagger.missing\n", encoding="utf-8")

            with self.assertRaisesRegex(
                NativeActorInventoryError,
                "actor source import cannot be resolved: round_dagger.missing",
            ):
                actor_source_import_closure(
                    actor_root=root,
                    entry_paths=(entry,),
                )

    def test_python_inventory_binds_absolute_link_and_regular_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = root / "runtime/python/bin/python3.11"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"qualified-python")
            link = root / "runtime/isaac/bin/python"
            link.parent.mkdir(parents=True)
            link.symlink_to(target)
            rows = sorted(
                (
                    inventory_row(link, symlink_target=target),
                    inventory_row(target),
                ),
                key=lambda row: row["path"],
            )
            manifest = root / "python-runtime.json"
            manifest.write_text(
                json.dumps({
                    "schema": SCHEMA,
                    "kind": "python_runtime",
                    "entries": rows,
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                verify_file_inventory(
                    manifest,
                    kind="python_runtime",
                    allowed_roots=(root,),
                    required_paths=(link, target),
                    allowed_symlinks={link: target},
                ),
                2,
            )

            document = json.loads(manifest.read_text(encoding="utf-8"))
            document["entries"] = [row for row in rows if row["path"] != str(target)]
            manifest.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                NativeActorInventoryError, "omits required live inputs"
            ):
                verify_file_inventory(
                    manifest,
                    kind="python_runtime",
                    allowed_roots=(root,),
                    required_paths=(link, target),
                    allowed_symlinks={link: target},
                )

    def test_symlink_identity_is_bound_and_python_absolute_policy_is_declared(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = root / "python3.11"
            target.write_bytes(b"python")
            relative = root / "relative-python"
            relative.symlink_to("python3.11")
            relative_row = inventory_row(relative, symlink_target=target)
            self.assertEqual(relative_row["size_bytes"], len(b"python3.11"))

            other = root / "other-python"
            other.write_bytes(b"other")
            absolute = root / "absolute-python"
            absolute.symlink_to(target)
            row = inventory_row(absolute, symlink_target=target)
            absolute.unlink()
            absolute.symlink_to(other)
            manifest = root / "runtime.json"
            manifest.write_text(json.dumps({
                "schema": SCHEMA,
                "kind": "python_runtime",
                "entries": sorted(
                    (row, inventory_row(target)), key=lambda item: item["path"]
                ),
            }), encoding="utf-8")
            with self.assertRaisesRegex(
                NativeActorInventoryError, "not explicitly allowed"
            ):
                verify_file_inventory(
                    manifest,
                    kind="python_runtime",
                    allowed_roots=(root,),
                    required_paths=(absolute, target),
                    allowed_symlinks={absolute: target},
                )
            launcher = (
                Path(__file__).resolve().parents[1]
                / "scripts/round_dagger/run_native_h200_actor_rank.sh"
            ).read_text(encoding="utf-8")
            self.assertIn("gr00t_python_link_target == /*", launcher)

    def test_tree_contract_covers_regular_files_and_safe_directory_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = root / "libactual.so"
            target.write_bytes(b"library")
            link = root / "libname.so"
            link.symlink_to(target.name)
            paths, links = inventory_tree_contract(root)
            self.assertEqual(paths, frozenset((target, link)))
            self.assertEqual(links, {link: target})

            directory = root / "real-package"
            directory.mkdir()
            nested = directory / "module.py"
            nested.write_bytes(b"module")
            directory_link = root / "package"
            directory_link.symlink_to(directory.name)
            paths, links = inventory_tree_contract(root)
            self.assertIn(directory_link, paths)
            self.assertIn(nested, paths)
            self.assertEqual(links[directory_link], directory)

            rows = sorted(
                (
                    inventory_row(path, symlink_target=links.get(path))
                    for path in paths
                ),
                key=lambda row: row["path"],
            )
            manifest = root / "runtime.json"
            manifest.write_text(json.dumps({
                "schema": SCHEMA,
                "kind": "isaac_runtime",
                "entries": rows,
            }), encoding="utf-8")
            self.assertEqual(
                verify_file_inventory(
                    manifest,
                    kind="isaac_runtime",
                    allowed_roots=(root,),
                    required_paths=paths,
                    allowed_symlinks=links,
                ),
                len(paths),
            )

    def test_tree_contract_rejects_directory_link_outside_allowed_roots(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as other:
            root = Path(temporary).resolve()
            external = Path(other).resolve()
            (external / "module.py").write_bytes(b"external")
            (root / "package").symlink_to(external)
            with self.assertRaisesRegex(
                NativeActorInventoryError, "directory symlink target is unsafe"
            ):
                inventory_tree_contract(root)

    def test_directory_link_manifest_requires_complete_canonical_target_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory = root / "lib"
            directory.mkdir()
            nested = directory / "module.py"
            nested.write_bytes(b"module")
            link = root / "lib64"
            link.symlink_to(directory.name)
            manifest = root / "runtime.json"
            manifest.write_text(json.dumps({
                "schema": SCHEMA,
                "kind": "isaac_runtime",
                "entries": [inventory_row(link, symlink_target=directory)],
            }), encoding="utf-8")
            with self.assertRaisesRegex(
                NativeActorInventoryError, "directory-link target files"
            ):
                verify_file_inventory(
                    manifest,
                    kind="isaac_runtime",
                    allowed_roots=(root,),
                    required_paths=(link,),
                    allowed_symlinks={link: directory},
                )

    def test_lightweight_rank_verification_still_rehashes_live_entrypoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            critical = root / "entrypoint.py"
            other = root / "package.py"
            critical.write_bytes(b"trusted-entrypoint")
            other.write_bytes(b"trusted-package")
            manifest = root / "source.json"
            manifest.write_text(json.dumps({
                "schema": SCHEMA,
                "kind": "actor_source",
                "entries": sorted(
                    (inventory_row(critical), inventory_row(other)),
                    key=lambda row: row["path"],
                ),
            }), encoding="utf-8")
            critical.write_bytes(b"changed-entrypoint")
            with self.assertRaisesRegex(
                NativeActorInventoryError, "inventory row differs"
            ):
                verify_file_inventory(
                    manifest,
                    kind="actor_source",
                    allowed_roots=(root,),
                    required_paths=(critical, other),
                    rehash_paths=(critical,),
                )

    def test_process_runner_attests_runtime_before_querying_gpus(self):
        order = []
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch(
                    "round_dagger.native_actor_saturation.verify_actor_runtime_before_launch",
                    side_effect=lambda environment: (
                        order.append("runtime") or ({"python_runtime": 2}, "a" * 64)
                    ),
                ), \
                mock.patch(
                    "round_dagger.native_actor_saturation.query_physical_gpu_inventory",
                    side_effect=lambda: order.append("gpu") or [{"uuid": "GPU-0"}],
                ), \
                mock.patch(
                    "round_dagger.native_actor_saturation.query_gpu_device_namespaces",
                    side_effect=lambda inventory: order.append("devices") or [],
                ):
            runner = ProcessWaveRunner(("/bin/true",), Path(temporary))
        self.assertEqual(order, ["runtime", "gpu", "devices"])
        self.assertEqual(runner.host_runtime_attestation_sha256, "a" * 64)

    def test_process_runner_exposes_all_admitted_h200s_one_per_rank(self):
        inventory = [
            {
                "rank": rank,
                "physical_index": rank,
                "uuid": f"GPU-{rank:08x}-0000-0000-0000-{rank:012x}",
                "pci_bus_id": f"00000000:{rank + 1:02X}:00.0",
                "name": "NVIDIA H200",
                "compute_capability": "9.0",
                "total_memory_mib": 143_771,
                "mig_mode": "Disabled",
            }
            for rank in range(8)
        ]
        captured = []
        protocol = []

        class FinishedProcess:
            returncode = 0

            def poll(self):
                return 0

            def wait(self):
                return 0

        def fake_popen(command, **kwargs):
            environment = dict(kwargs["env"])
            captured.append(environment)
            protocol.append(("spawn", int(environment["ROUND_DAGGER_ACTOR_RANK"])))
            return FinishedProcess()

        def kit_ready(**kwargs):
            protocol.append(("kit_ready", kwargs["rank"]))

        def runtime_ready(**kwargs):
            protocol.append(("runtime_ready", len(kwargs["processes"]) - 1))

        def publish_release(path):
            protocol.append(("publish_release", None))
            path.write_bytes(b"release\n")

        def fleet_released(**_kwargs):
            protocol.append(("fleet_released", None))

        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary).resolve()
            campaign = temporary_root / "campaign"
            generation = campaign / "BC/isaac/collection"
            generation.mkdir(parents=True)
            log_root = generation / "logs/test-attempt"
            with mock.patch.dict(
                "os.environ",
                {
                    "CUDA_VISIBLE_DEVICES": "GPU-ambient",
                    "ACCELERATE_TORCH_DEVICE": "cuda:0",
                    "ROUND_DAGGER_COHORT_DIR": "/ambient/collect",
                    "ROUND_DAGGER_ACTOR_SEED_CONTRACT_JSON": "ambient-secret",
                },
                clear=False,
            ), mock.patch(
                "round_dagger.native_actor_saturation.CANONICAL_CAMPAIGN_ROOT",
                campaign,
            ), mock.patch(
                "round_dagger.native_actor_saturation.query_physical_gpu_inventory",
                return_value=inventory,
            ), mock.patch(
                "round_dagger.native_actor_saturation.query_gpu_device_namespaces",
                return_value=_device_namespaces(inventory),
            ), mock.patch(
                "round_dagger.native_actor_saturation.subprocess.Popen",
                side_effect=fake_popen,
            ), mock.patch(
                "round_dagger.native_actor_saturation._wait_rank_kit_ready",
                side_effect=kit_ready,
            ), mock.patch(
                "round_dagger.native_actor_saturation._wait_all_ranks_runtime_ready",
                side_effect=runtime_ready,
            ), mock.patch(
                "round_dagger.native_actor_saturation._publish_fleet_release",
                side_effect=publish_release,
            ), mock.patch(
                "round_dagger.native_actor_saturation._wait_all_ranks_fleet_released",
                side_effect=fleet_released,
            ), mock.patch(
                "round_dagger.native_actor_saturation.wait_for_admitted_gpu_quiescence",
            ), mock.patch(
                "round_dagger.native_actor_saturation.rank_log_is_healthy",
                return_value=True,
            ):
                runner = object.__new__(ProcessWaveRunner)
                runner.command = ("/bin/true",)
                runner.log_root = log_root
                runner.attempt_id = "test-attempt"
                runner.physical_gpu_inventory = inventory
                runner.physical_gpu_inventory_sha256 = "b" * 64
                runner.gpu_device_namespaces = _device_namespaces(inventory)
                runner.host_runtime_attestation_sha256 = "a" * 64
                runner.admitted_physical_gpu_indices = tuple(range(8))
                runner.virtual_actor_world_size = 8
                runner.virtual_rank_cohorts = (
                    tuple((rank, rank) for rank in range(8)),
                )
                runner(
                    envs_per_gpu=1,
                    steps=16,
                    phase="smoke",
                    wave_index=0,
                    timeout=1,
                    environment={
                        "ROUND_DAGGER_GENERATION_ROOT": str(generation),
                        "ROUND_DAGGER_ISAAC_RUNTIME_SHA256": "c" * 64,
                        "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256": "d" * 64,
                    },
                )
        self.assertEqual(len(captured), 8)
        for rank, environment in enumerate(captured):
            self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)
            self.assertNotIn("NVIDIA_VISIBLE_DEVICES", environment)
            self.assertEqual(
                environment["ROUND_DAGGER_ASSIGNED_GPU_UUID"],
                inventory[rank]["uuid"],
            )
            self.assertEqual(
                environment["ROUND_DAGGER_ASSIGNED_PHYSICAL_INDEX"], str(rank)
            )
            namespace = _device_namespaces(inventory)[rank]
            self.assertEqual(
                json.loads(environment["ROUND_DAGGER_GPU_DEVICE_NAMESPACE_JSON"]),
                namespace,
            )
            self.assertEqual(
                environment["ROUND_DAGGER_NVIDIA_DEVICE"],
                namespace["nvidia_device"]["path"],
            )
            self.assertNotIn("ACCELERATE_TORCH_DEVICE", environment)
            self.assertNotIn("ROUND_DAGGER_COHORT_DIR", environment)
            self.assertNotIn(
                "ROUND_DAGGER_ACTOR_SEED_CONTRACT_JSON", environment
            )
            self.assertEqual(environment["ROUND_DAGGER_ACTOR_RANK"], str(rank))
            self.assertEqual(environment["ROUND_DAGGER_ACTOR_WORLD_SIZE"], "8")
            self.assertEqual(environment["ROUND_DAGGER_ATTEMPT_ID"], "test-attempt")
            self.assertEqual(
                json.loads(environment["ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON"]),
                inventory[rank],
            )
            self.assertEqual(
                environment["ROUND_DAGGER_FLEET_RELEASE_FILE"],
                captured[0]["ROUND_DAGGER_FLEET_RELEASE_FILE"],
            )
            self.assertEqual(
                Path(environment["ROUND_DAGGER_FLEET_RELEASE_FILE"]).name,
                "kit-ready.release",
            )
        self.assertEqual(
            protocol[:24],
            [
                event
                for rank in range(8)
                for event in (
                    ("spawn", rank),
                    ("kit_ready", rank),
                    ("runtime_ready", rank),
                )
            ],
        )
        self.assertEqual(
            protocol[24:],
            [
                ("publish_release", None),
                ("fleet_released", None),
            ],
        )

    def test_full_bundle_verification_enumerates_complete_isaac_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            actor = root / "actor"
            gr00t = actor / "GR00T-VisualSim2Real"
            isaac = root / "runtime/isaacsim-5.1"
            python_root = root
            python_target = root / "runtime/python/bin/python3.11"
            for directory in (
                actor,
                gr00t / "gr00t/rl/config",
                isaac / "bin",
                python_target.parent,
                root / "runtime/IsaacLab/source",
                root / "runtime/dependencies",
                root / "runtime/system",
                root / "runtime/cudnn",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            python_target.write_bytes(b"python")
            gr00t_python = isaac / "bin/python"
            gr00t_python.symlink_to(python_target)

            seen = []

            def tree_contract(path, *, allowed_target_roots=None):
                seen.append(
                    (
                        Path(path),
                        tuple(Path(item) for item in allowed_target_roots),
                    )
                )
                return frozenset(), {}

            with mock.patch(
                "round_dagger.native_actor_file_inventory.inventory_tree_contract",
                side_effect=tree_contract,
            ), mock.patch(
                "round_dagger.native_actor_file_inventory.verify_file_inventory",
                return_value=1,
            ), mock.patch(
                "round_dagger.native_actor_file_inventory.actor_source_import_closure",
                return_value=frozenset(),
            ):
                verify_native_actor_bundle(
                    actor_manifest=root / "actor.json",
                    gr00t_manifest=root / "gr00t.json",
                    isaac_manifest=root / "isaac.json",
                    python_manifest=root / "python.json",
                    actor_root=actor,
                    gr00t_root=gr00t,
                    gr00t_python=gr00t_python,
                    isaac_root=isaac,
                    python_root=python_root,
                    isaac_setup=actor / "setup.sh",
                    headless_experience=root / "headless.kit",
                    cudnn_root=root / "runtime/cudnn",
                    isaaclab_source_root=root / "runtime/IsaacLab/source",
                    dependency_root=root / "runtime/dependencies",
                    system_library_root=root / "runtime/system",
                    full_runtime_trees=True,
                )
            self.assertIn((isaac, (isaac, python_root)), seen)

    def test_actor_handoff_strips_all_heldout_and_namespace_state(self):
        environment = {
            "ROUND_DAGGER_VALIDATION_ROOT": "/secret/validation",
            "ROUND_DAGGER_VALIDATION_CASE_REGISTRY": "/secret/cases.json",
            "ROUND_DAGGER_TEST_ROOT": "/secret/test",
            "ROUND_DAGGER_TEST_TOKEN": "secret",
            "ROUND_DAGGER_HELDOUT_ROOT": "/secret/heldout",
            "ROUND_DAGGER_PRIVATE_MOUNT_STAGE": "ready",
            "ROUND_DAGGER_JAIL_HANDLE_ROOT": "/run/escape",
            "ROUND_DAGGER_TRAINER_SUCCESS_MARKER": "/tmp/forged-success",
            "ROUND_DAGGER_SIMULATION_APP_SKIP_CLEANUP": "0",
            "LD_PRELOAD": "/tmp/inject.so",
            "LD_LIBRARY_PATH": "/tmp/inject-libs",
            "LD_AUDIT": "/tmp/inject-audit.so",
            "PYTHONINSPECT": "1",
            "PYTHONUSERBASE": "/tmp/inject-python",
            "CUDA_VISIBLE_DEVICES": "GPU-hidden",
            "ACCELERATE_TORCH_DEVICE": "cuda:0",
            "HOME": "/tmp/inject-home",
            "PATH": "/tmp/inject-bin",
            "BASH_FUNC_mount%%": "() { :; }",
            "SHELLOPTS": "xtrace",
            "PS4": "$(touch /tmp/injected)",
            "ROUND_DAGGER_PURPOSE": "train",
        }
        clean = sanitize_actor_child_environment(environment)
        self.assertEqual(clean, {
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
            "ROUND_DAGGER_PURPOSE": "train",
        })

    def test_launcher_declares_broad_masks_git_masks_and_input_pins(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts/round_dagger/run_native_h200_actor_rank.sh"
        ).read_text(encoding="utf-8")
        for marker in (
            'native-actor-workspace "$WORKSPACE_ROOT"',
            'native-actor-storage "$DAGGER_STORAGE_ROOT"',
            'native-git-store "$git_store"',
            'mount --bind "$jail_handles/checkpoints" "$checkpoint_input_dir"',
            'mount --bind "$jail_handles/pins/$index" "${pinned_inputs[$index]}"',
            "ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT",
            "native-validation",
            "native-test",
            "--bounding-set=-sys_admin,-dac_override,-dac_read_search",
            'PYTHONPATH="$ROOT" python3 -',
            "--portable-root $physical_cache/kit-portable",
            "--/log/file=$process_state/kit-logs/kit.log",
            "--/crashreporter/dumpDir=$process_state/crash",
            "ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT",
            "ROUND_DAGGER_ISAACLAB_SOURCE_ROOT",
            "ROUND_DAGGER_SYSTEM_LIBRARY_DIR",
            "NATIVE_ACTOR_IMPORT_ROOTS_OK",
            "verify_native_actor_bundle",
            "runtime/physical-cache/gpu-",
            "production_heldout_motion_plan_8x32.json",
            "native_h200_bc_dagger4_protocol.v1.json",
            "production_paired_scenarios_2x8.json",
            "33demos_675_775_heldout_256_v1.json",
            'mount --bind /dev/null "$sensitive"',
            'public split detail is not masked: $sensitive',
            "native H200 rank forbids CUDA/NVIDIA visibility remapping",
            "ROUND_DAGGER_GPU_DEVICE_NAMESPACE_JSON",
            "ROUND_DAGGER_ASSIGNED_PHYSICAL_INDEX",
            "ROUND_DAGGER_ASSIGNED_GPU_UUID",
            "native-actor-dri /dev/dri",
            'mount --bind /dev/null "$gpu_device"',
            'mount --bind "$jail_handles/devices/nvidia"',
            "native_actor_gpu_namespace_marker",
            "ROUND_DAGGER_LOCAL_GPU_INDEX=0",
            "torch.cuda.device_count() != 1",
            '"CUDA_VISIBLE_DEVICES" in os.environ',
            "verify_native_actor_kit_log",
            "ROUND_DAGGER_TRAINER_SUCCESS_MARKER",
            "NATIVE_ACTOR_TRAINER_BODY_OK",
            "runtime/private-inputs/attempts/$ROUND_DAGGER_ATTEMPT_ID",
            "runtime/attempts/$ROUND_DAGGER_ATTEMPT_ID",
            "export ROUND_DAGGER_SIMULATION_APP_SKIP_CLEANUP=1",
            "readonly ROUND_DAGGER_SIMULATION_APP_SKIP_CLEANUP",
        ):
            self.assertIn(marker, source)
        self.assertNotIn("--portable-root=", source)
        self.assertLess(
            source.index('"${command[@]}"'),
            source.index('echo "$expected_trainer_success"'),
        )
        self.assertLess(
            source.index('echo "$expected_trainer_success"'),
            source.index("verify_native_actor_kit_log"),
        )
        self.assertLess(
            source.index("verify_native_actor_kit_log"),
            source.index('echo "NATIVE_ACTOR_RANK_OK'),
        )

        setup = (
            Path(__file__).resolve().parents[1]
            / "scripts/round_dagger/setup_native_h200_isaac_runtime.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("/workspaces/", setup)
        self.assertNotIn("/bootstrap", setup)
        self.assertNotIn("${PYTHONPATH:+", setup)
        self.assertNotIn("${LD_LIBRARY_PATH:+", setup)
        self.assertIn("runtime/actor-python-site-packages", setup)


if __name__ == "__main__":
    unittest.main()
