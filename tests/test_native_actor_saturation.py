from __future__ import annotations

import math
import contextlib
import fcntl
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from round_dagger import artifacts
from round_dagger.actor import (
    ActorExportError,
    _native_generation_shard_ids_from_environment,
    _observation_provenance_from_environment,
)
from round_dagger import native_actor_saturation as saturation
from round_dagger.native_actor_saturation import (
    CANDIDATES,
    GPU_COUNT,
    MEMORY_LIMIT,
    NOMINAL_TRANSITIONS,
    PROBE_STEPS,
    SMOKE_STEPS,
    WAVE_STEPS,
    VIRTUAL_ACTOR_WORLD_SIZE,
    SaturationError,
    expected_shard_names,
    group_rotation,
    run_tuner,
    select_candidate,
    validate_probe,
    verify_shard_inventory,
    wave_shape,
    native_actor_gpu_namespace_marker,
    native_actor_kit_isolation_marker,
    query_gpu_device_namespaces,
    rank_log_is_healthy,
    verify_native_actor_kit_log,
)
from round_dagger.seed_policy import derive_actor_seed_contract
from scripts.round_dagger import native_h200_bc_wave_artifact as wave_artifact


def _result(
    envs,
    steps,
    throughput,
    memory,
    *,
    bad_rank=None,
    duration=None,
    startup=0.0,
):
    transitions = VIRTUAL_ACTOR_WORLD_SIZE * envs * steps
    wall_seconds = duration or transitions / throughput
    return {
        "envs_per_gpu": envs,
        "steps": steps,
        "wall_seconds": wall_seconds,
        "startup_seconds": startup,
        "execution_seconds": wall_seconds - startup,
        "rank_results": [
            {"rank": rank, "healthy": rank != bad_rank, "exit_code": 0 if rank != bad_rank else 1}
            for rank in range(VIRTUAL_ACTOR_WORLD_SIZE)
        ],
        "peak_memory_fraction": memory,
        "aggregate_transitions_per_second": throughput,
    }


def _gpu_inventory():
    return [
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


def _portable_root(rank, *, physical_index=None):
    if physical_index is None:
        physical_index = rank
    gpu = _gpu_inventory()[physical_index]
    return (
        saturation.CANONICAL_CAMPAIGN_ROOT
        / "BC/collection/runtime/physical-cache"
        / f"gpu-{physical_index:02d}"
        / gpu["uuid"]
        / f"{'0' * 64}-{'f' * 64}/kit-portable"
    )


def _gpu_device_namespaces(inventory=None):
    inventory = _gpu_inventory() if inventory is None else inventory
    rows = []
    for rank, gpu in enumerate(inventory):
        device_minor = (rank + 4) % GPU_COUNT
        rows.append({
            "schema": saturation.GPU_DEVICE_NAMESPACE_SCHEMA,
            "rank": rank,
            "physical_index": rank,
            "uuid": gpu["uuid"],
            "pci_bus_id": gpu["pci_bus_id"],
            "pci_function": f"0000:{rank + 1:02x}:00.0",
            "nvidia_device": {
                "path": f"/dev/nvidia{device_minor}", "major": 195,
                "minor": device_minor,
            },
            "drm_card_device": {
                "path": f"/dev/dri/card{device_minor}", "major": 226,
                "minor": device_minor,
            },
            "drm_render_device": {
                "path": f"/dev/dri/renderD{128 + device_minor}", "major": 226,
                "minor": 128 + device_minor,
            },
            "common_control_devices": [
                {"path": path, "major": 195, "minor": 240 + index}
                for index, path in enumerate(
                    saturation.COMMON_NVIDIA_CONTROL_DEVICES
                )
            ],
        })
    return rows


def _healthy_kit_log(rank, *, separator="=", quote="'"):
    portable_root = _portable_root(rank)
    command_root = f"{quote}{portable_root}{quote}" if quote else str(portable_root)
    return (
        "2026-09-04T10:00:00Z [1ms] [Info] [omni.kit.app.plugin] Cmd: "
        "/sealed/simulation_app.py /sealed/headless.kit "
        "--/app/tokens/exe-path=/sealed/kit "
        "--/rtx/hydra/materialSyncLoads=True "
        f"--/renderer/activeGpu{separator}0 "
        f"--/physics/cudaDevice{separator}0 "
        f"--/renderer/multiGpu/enabled{separator}False "
        f"--portable --portable-root{separator}{command_root} --no-window\n"
        "2026-09-04T10:00:00Z [1ms] [Info] [omni.kit.app.plugin] "
        f"Portable Root: {quote}{portable_root}{quote}\n"
    )


def _vulkan_gpu_table(*, active_rank, inventory=None, ecc_rank=0, active=True):
    """Build the exact seven-column table emitted by Kit 107.3."""

    inventory = _gpu_inventory() if inventory is None else inventory
    lines = [
        "| GPU | Name                             | Active | LDA | GPU Memory | Vendor-ID | LUID       |",
        "|     |                                  |        |     |            | Device-ID | UUID       |",
        "|     |                                  |        |     |            | Bus-ID    |            |",
        "|---------------------------------------------------------------------------------------------|",
    ]
    gpu = inventory[active_rank]
    active_value = "Yes: 0" if active else ""
    uuid_prefix = gpu["uuid"].removeprefix("GPU-")[:8].lower()
    bus = gpu["pci_bus_id"].split(":")[-2].lower()
    lines.extend(
        (
            f"| 0   | NVIDIA H200                      | {active_value:<6} |     | 143771  MB | 10de      | 0          |",
            f"|     |                                  |        |     |            | 2335      | {uuid_prefix}.. |",
            f"|     |                                  |        |     |            | {bus:<9} |            |",
            "|---------------------------------------------------------------------------------------------|",
        )
    )
    lines.extend(
        (
            "| OS: 24.04.4 LTS (Noble Numbat) ubuntu, Version: 24.04.4, Kernel: test",
            f"ECC is enabled on physical device {ecc_rank}",
        )
    )
    return "\n".join(lines) + "\n"


def _startup_protocol_markers(rank, gpu=None, gpu_namespace=None):
    gpu = _gpu_inventory()[rank] if gpu is None else gpu
    gpu_namespace = (
        _gpu_device_namespaces()[rank]
        if gpu_namespace is None
        else gpu_namespace
    )
    return (
        native_actor_gpu_namespace_marker(
            rank=rank, gpu_namespace=gpu_namespace
        )
        + "\n"
        f"NATIVE_ACTOR_KIT_READY rank={rank} renderer_active_gpu=0 "
        "physics_cuda_device=0 logical_cuda_devices=1\n"
        f"NATIVE_ACTOR_RUNTIME_READY rank={rank} uuid={gpu['uuid']} "
        "renderer_active_gpu=0 physics_cuda_device=0 "
        "logical_cuda_devices=1\n"
        f"NATIVE_ACTOR_FLEET_RELEASED rank={rank}\n"
    )


class _Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class NativeActorSaturationTest(unittest.TestCase):
    def test_physical_cache_identity_uses_device_uuid_and_full_runtime_hashes(self):
        gpu = _gpu_inventory()[2]
        root = Path("/sealed/BC/collection")
        cache = saturation.physical_gpu_cache_root(
            generation_root=root,
            physical_gpu=gpu,
            isaac_runtime_sha256="1" * 64,
            actor_python_sha256="2" * 64,
        )
        self.assertEqual(
            cache,
            root
            / "runtime/physical-cache/gpu-02"
            / gpu["uuid"]
            / f"{'1' * 64}-{'2' * 64}",
        )

    def test_quiescence_waits_for_zero_processes_and_memory(self):
        gpu_uuid = _gpu_inventory()[0]["uuid"]
        clock = _Clock()
        app_rows = iter((
            ({"uuid": gpu_uuid, "pid": 123, "used_memory_mib": 8},),
            (),
        ))
        memory_rows = iter((
            {gpu_uuid: (8, 143_771)},
            {gpu_uuid: (0, 143_771)},
        ))
        saturation.wait_for_admitted_gpu_quiescence(
            admitted_gpu_uuids=(gpu_uuid,),
            deadline=1.0,
            query_compute_apps=lambda: next(app_rows),
            query_memory=lambda: next(memory_rows),
            clock=clock,
            sleep=clock.advance,
        )
        self.assertEqual(clock.value, 0.25)

    def test_quiescence_fails_closed_when_admitted_gpu_stays_busy(self):
        gpu_uuid = _gpu_inventory()[0]["uuid"]
        clock = _Clock()
        compute_calls = []

        def apps():
            compute_calls.append(clock())
            return ({"uuid": gpu_uuid, "pid": 456, "used_memory_mib": 9},)

        with self.assertRaisesRegex(
            SaturationError, rf"{re.escape(gpu_uuid)}:pid=456"
        ):
            saturation.wait_for_admitted_gpu_quiescence(
                admitted_gpu_uuids=(gpu_uuid,),
                deadline=0.5,
                query_compute_apps=apps,
                query_memory=lambda: {gpu_uuid: (9, 143_771)},
                clock=clock,
                sleep=clock.advance,
            )
        self.assertEqual(compute_calls, [0.0, 0.25, 0.5])

    def test_virtual_rank_cohorts_bind_single_rank_to_admitted_gpu_zero(self):
        self.assertEqual(
            saturation.virtual_rank_cohorts((0,)),
            (((0, 0),),),
        )
        for value in ((), (0, 0), (8,), (0, 2), "0,1", 3):
            with self.subTest(value=value), self.assertRaises(SaturationError):
                saturation.virtual_rank_cohorts(value)
        self.assertEqual(saturation.parse_admitted_gpu_indices("7,3"), (7, 3))
        for raw in ("", "0,", "0,0", "8", "00", "0, 1"):
            with self.subTest(raw=raw), self.assertRaises(SaturationError):
                saturation.parse_admitted_gpu_indices(raw)

    def test_gpu_device_namespace_uses_proc_minor_and_pci_resolved_drm(self):
        inventory = _gpu_inventory()
        device_minors = (4, 5, 6, 7, 0, 1, 2, 3)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            proc_root = root / "proc"
            pci_root = root / "pci"
            drm_root = root / "drm-class"
            dev_root = root / "dev"
            for rank, (gpu, minor) in enumerate(zip(
                inventory, device_minors, strict=True
            )):
                pci_function = f"0000:{rank + 1:02x}:00.0"
                information = proc_root / pci_function / "information"
                information.parent.mkdir(parents=True)
                information.write_text(
                    "\n".join((
                        "Model: NVIDIA H200",
                        f"GPU UUID: {gpu['uuid']}",
                        f"Bus Location: {pci_function}",
                        f"Device Minor: {minor}",
                    )) + "\n",
                    encoding="utf-8",
                )
                pci_path = pci_root / pci_function
                (pci_path / "drm" / f"card{minor}").mkdir(parents=True)
                (pci_path / "drm" / f"renderD{128 + minor}").mkdir()
                for name, device_minor in (
                    (f"card{minor}", minor),
                    (f"renderD{128 + minor}", 128 + minor),
                ):
                    class_path = drm_root / name
                    class_path.mkdir(parents=True)
                    (class_path / "device").symlink_to(
                        pci_path, target_is_directory=True
                    )
                    (class_path / "dev").write_text(
                        f"226:{device_minor}\n", encoding="ascii"
                    )

            def fake_device_record(path, _label):
                name = path.name
                if name.startswith("nvidia") and name[6:].isdigit():
                    major, minor = 195, int(name[6:])
                elif name.startswith("card"):
                    major, minor = 226, int(name[4:])
                elif name.startswith("renderD"):
                    major, minor = 226, int(name[7:])
                else:
                    major = 195
                    minor = {
                        "nvidiactl": 255,
                        "nvidia-modeset": 254,
                        "nvidia-uvm": 0,
                        "nvidia-uvm-tools": 1,
                    }[name]
                return {"path": str(path), "major": major, "minor": minor}

            namespaces = query_gpu_device_namespaces(
                inventory,
                proc_gpu_root=proc_root,
                sysfs_pci_root=pci_root,
                sysfs_drm_root=drm_root,
                dev_root=dev_root,
                device_record=fake_device_record,
            )
            self.assertEqual(
                namespaces[1]["nvidia_device"]["path"],
                str(dev_root / "nvidia5"),
            )
            self.assertEqual(
                namespaces[1]["drm_card_device"]["path"],
                str(dev_root / "dri/card5"),
            )
            self.assertEqual(
                namespaces[1]["drm_render_device"]["path"],
                str(dev_root / "dri/renderD133"),
            )

            rank_one = proc_root / "0000:02:00.0/information"
            rank_one.write_text(
                rank_one.read_text(encoding="utf-8").replace(
                    inventory[1]["uuid"], inventory[0]["uuid"]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SaturationError, "differs"):
                query_gpu_device_namespaces(
                    inventory,
                    proc_gpu_root=proc_root,
                    sysfs_pci_root=pci_root,
                    sysfs_drm_root=drm_root,
                    dev_root=dev_root,
                    device_record=fake_device_record,
                )

    def test_serial_kit_ready_timeout_is_charged_to_existing_wave_clock(self):
        clock = _Clock()

        class Process:
            returncode = None

            def poll(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            rank_log = root / "rank.log"
            kit_log = root / "kit.log"
            rank_log.write_text("", encoding="utf-8")
            kit_log.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(SaturationError, "did not become Kit-ready"):
                saturation._wait_rank_kit_ready(
                    processes=[Process()],
                    rank=0,
                    gpu=_gpu_inventory()[0],
                    gpu_namespace=_gpu_device_namespaces()[0],
                    rank_log=rank_log,
                    kit_log=kit_log,
                    portable_root=_portable_root(0),
                    started=0.0,
                    timeout=100.0,
                    clock=clock,
                    sleep=clock.advance,
                )
            self.assertEqual(clock(), 100.0)

            clock.value = 99.0
            with self.assertRaisesRegex(SaturationError, "did not become Kit-ready"):
                saturation._wait_rank_kit_ready(
                    processes=[Process()],
                    rank=0,
                    gpu=_gpu_inventory()[0],
                    gpu_namespace=_gpu_device_namespaces()[0],
                    rank_log=rank_log,
                    kit_log=kit_log,
                    portable_root=_portable_root(0),
                    started=0.0,
                    timeout=100.0,
                    clock=clock,
                    sleep=clock.advance,
                )
            self.assertEqual(clock(), 100.0)

    def test_cold_startup_bounds_cover_observed_rank_and_smoke(self):
        # smoke-20260904j rank 0 required 125.46 seconds from Popen through the
        # exact Kit-ready marker.  The per-rank bound includes margin, while
        # the smoke wave retains time after eight serialized observed starts.
        self.assertEqual(saturation.RANK_KIT_READY_TIMEOUT_SECONDS, 240.0)
        self.assertGreater(saturation.RANK_KIT_READY_TIMEOUT_SECONDS, 125.46)
        self.assertEqual(saturation.SMOKE_TIMEOUT_SECONDS, 1_800.0)
        self.assertGreater(
            saturation.SMOKE_TIMEOUT_SECONDS,
            VIRTUAL_ACTOR_WORLD_SIZE * 125.46,
        )
        self.assertEqual(
            saturation.PROBE_TIMEOUT_CAP_SECONDS,
            saturation.PROBE_BUDGET_SECONDS,
        )

    def test_kit_ready_gate_rereads_marker_after_telemetry_crosses_deadline(self):
        clock = _Clock()
        rank = 0
        gpu = _gpu_inventory()[rank]

        class Process:
            returncode = None

            def poll(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            rank_log = root / "rank.log"
            kit_log = root / "kit.log"
            gpu_marker = (
                f"NATIVE_ACTOR_GPU_OK rank={rank} uuid={gpu['uuid']} "
                f"pci={gpu['pci_bus_id']} logical_cuda=0 local_nvml=0 "
                "visible_devices=1 cuda_visible_devices=unset\n"
            )
            ready_marker = (
                f"NATIVE_ACTOR_KIT_READY rank={rank} "
                "renderer_active_gpu=0 physics_cuda_device=0 "
                "logical_cuda_devices=1\n"
            )
            namespace_marker = native_actor_gpu_namespace_marker(
                rank=rank, gpu_namespace=_gpu_device_namespaces()[rank]
            ) + "\n"
            rank_log.write_text(namespace_marker + gpu_marker, encoding="utf-8")
            kit_log.write_text(
                _healthy_kit_log(rank) + _vulkan_gpu_table(active_rank=rank),
                encoding="utf-8",
            )

            sampled = False

            def sample_telemetry():
                nonlocal sampled
                self.assertFalse(sampled)
                sampled = True
                rank_log.write_text(
                    namespace_marker + gpu_marker + ready_marker,
                    encoding="utf-8",
                )
                clock.advance(2.0)

            saturation._wait_rank_kit_ready(
                processes=[Process()],
                rank=rank,
                gpu=gpu,
                gpu_namespace=_gpu_device_namespaces()[rank],
                rank_log=rank_log,
                kit_log=kit_log,
                portable_root=_portable_root(rank),
                started=0.0,
                timeout=1.0,
                sample_telemetry=sample_telemetry,
                clock=clock,
                sleep=lambda _seconds: self.fail("deadline sleep was unexpected"),
            )
            self.assertTrue(sampled)

    def test_vulkan_gpu_table_binds_active_rank_uuid_bus_memory_and_ecc(self):
        inventory = _gpu_inventory()
        saturation.verify_vulkan_gpu_table(
            text=_vulkan_gpu_table(active_rank=4, inventory=inventory),
            rank=4,
            gpu=inventory[4],
        )

        corruptions = {
            "duplicate table": (
                _vulkan_gpu_table(active_rank=4, inventory=inventory) * 2,
                "one Vulkan GPU table",
            ),
            "wrong assigned physical GPU": (
                _vulkan_gpu_table(active_rank=3, inventory=inventory),
                "attested H200 inventory",
            ),
            "wrong selected UUID": (
                _vulkan_gpu_table(active_rank=4, inventory=inventory).replace(
                    inventory[4]["uuid"].removeprefix("GPU-")[:8].lower(),
                    "deadbeef",
                    1,
                ),
                "attested H200 inventory",
            ),
            "wrong selected bus": (
                _vulkan_gpu_table(active_rank=4, inventory=inventory).replace(
                    "| 05        |", "| ff        |", 1
                ),
                "attested H200 inventory",
            ),
            "wrong selected memory": (
                _vulkan_gpu_table(active_rank=4, inventory=inventory).replace(
                    "143771  MB", "140000  MB", 5
                ),
                "attested H200 inventory",
            ),
            "wrong ECC rank": (
                _vulkan_gpu_table(
                    active_rank=4, inventory=inventory, ecc_rank=4
                ),
                "ECC marker",
            ),
        }
        for name, (text, message) in corruptions.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                SaturationError, message
            ):
                saturation.verify_vulkan_gpu_table(
                    text=text, rank=4, gpu=inventory[4]
                )

    def test_serial_kit_ready_gate_requires_complete_post_launcher_evidence(self):
        rank = 0
        gpu = _gpu_inventory()[rank]

        class AliveProcess:
            returncode = None

            def poll(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            rank_log = root / "rank.log"
            kit_log = root / "kit.log"
            portable_root = _portable_root(rank)
            gpu_marker = (
                f"NATIVE_ACTOR_GPU_OK rank={rank} uuid={gpu['uuid']} "
                f"pci={gpu['pci_bus_id']} logical_cuda=0 local_nvml=0 "
                "visible_devices=1 cuda_visible_devices=unset\n"
            )
            ready_marker = (
                f"NATIVE_ACTOR_KIT_READY rank={rank} "
                "renderer_active_gpu=0 physics_cuda_device=0 "
                "logical_cuda_devices=1\n"
            )
            namespace_marker = native_actor_gpu_namespace_marker(
                rank=rank, gpu_namespace=_gpu_device_namespaces()[rank]
            ) + "\n"
            valid = namespace_marker + gpu_marker + ready_marker
            rank_log.write_text(valid, encoding="utf-8")
            kit_log.write_text(
                _healthy_kit_log(rank) + _vulkan_gpu_table(active_rank=rank),
                encoding="utf-8",
            )
            saturation._wait_rank_kit_ready(
                processes=[AliveProcess()],
                rank=rank,
                gpu=gpu,
                gpu_namespace=_gpu_device_namespaces()[rank],
                rank_log=rank_log,
                kit_log=kit_log,
                portable_root=portable_root,
                started=0.0,
                timeout=10.0,
                clock=lambda: 0.0,
                sleep=lambda _seconds: None,
            )

            failures = {
                "missing logical CUDA attestation": valid.replace(gpu_marker, ""),
                "duplicate Kit-ready marker": valid + ready_marker,
                "startup fatal": valid + "ERROR_DEVICE_LOST\n",
            }
            for name, text in failures.items():
                with self.subTest(name=name):
                    rank_log.write_text(text, encoding="utf-8")
                    with self.assertRaises(SaturationError):
                        saturation._wait_rank_kit_ready(
                            processes=[AliveProcess()],
                            rank=rank,
                            gpu=gpu,
                            gpu_namespace=_gpu_device_namespaces()[rank],
                            rank_log=rank_log,
                            kit_log=kit_log,
                            portable_root=portable_root,
                            started=0.0,
                            timeout=10.0,
                            clock=lambda: 0.0,
                            sleep=lambda _seconds: None,
                        )

            rank_log.write_text(valid, encoding="utf-8")
            exited = AliveProcess()
            exited.returncode = 0
            with self.assertRaisesRegex(
                SaturationError, "exited 0 before fleet release"
            ):
                saturation._wait_rank_kit_ready(
                    processes=[exited],
                    rank=rank,
                    gpu=gpu,
                    gpu_namespace=_gpu_device_namespaces()[rank],
                    rank_log=rank_log,
                    kit_log=kit_log,
                    portable_root=portable_root,
                    started=0.0,
                    timeout=10.0,
                    clock=lambda: 0.0,
                    sleep=lambda _seconds: None,
                )

    def test_all_rank_runtime_ready_gate_requires_every_exact_fresh_reset(self):
        inventory = _gpu_inventory()

        class AliveProcess:
            returncode = None

            def poll(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            logs = []
            for rank, gpu in enumerate(inventory):
                path = root / f"gpu-{rank:02d}.log"
                path.write_text(
                    f"NATIVE_ACTOR_RUNTIME_READY rank={rank} uuid={gpu['uuid']} "
                    "renderer_active_gpu=0 physics_cuda_device=0 "
                    "logical_cuda_devices=1\n",
                    encoding="utf-8",
                )
                logs.append(path)
            processes = [AliveProcess() for _ in range(GPU_COUNT)]
            saturation._wait_all_ranks_runtime_ready(
                processes=processes,
                logs=logs,
                gpus=inventory,
                started=time.monotonic(),
                timeout=10.0,
            )

            original = logs[7].read_text(encoding="utf-8")
            failures = {
                "missing rank": "",
                "wrong UUID": original.replace(inventory[7]["uuid"], "GPU-wrong"),
                "duplicate marker": original + original,
                "startup fatal": original + "Traceback\n",
            }
            for name, text in failures.items():
                with self.subTest(name=name):
                    logs[7].write_text(text, encoding="utf-8")
                    with mock.patch.object(
                        saturation.time, "monotonic", return_value=2.0
                    ), self.assertRaises(SaturationError):
                        saturation._wait_all_ranks_runtime_ready(
                            processes=processes,
                            logs=logs,
                            gpus=inventory,
                            started=0.0,
                            timeout=1.0,
                        )
            logs[7].write_text(original, encoding="utf-8")

    def test_runtime_ready_gate_accepts_launch_prefix_and_rechecks_prior_ranks(self):
        inventory = _gpu_inventory()

        class AliveProcess:
            returncode = None

            def poll(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            logs = []
            processes = []
            for rank, gpu in enumerate(inventory):
                path = root / f"gpu-{rank:02d}.log"
                path.write_text(
                    f"NATIVE_ACTOR_RUNTIME_READY rank={rank} uuid={gpu['uuid']} "
                    "renderer_active_gpu=0 physics_cuda_device=0 "
                    "logical_cuda_devices=1\n",
                    encoding="utf-8",
                )
                logs.append(path)
                processes.append(AliveProcess())
                saturation._wait_all_ranks_runtime_ready(
                    processes=processes,
                    logs=logs,
                    gpus=inventory[: rank + 1],
                    started=0.0,
                    timeout=10.0,
                    clock=lambda: 0.0,
                    sleep=lambda _seconds: None,
                )

            logs[0].write_text(
                logs[0].read_text(encoding="utf-8") + "ERROR_DEVICE_LOST\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                SaturationError, "rank 0 failed before runtime-ready"
            ):
                saturation._wait_all_ranks_runtime_ready(
                    processes=processes[:2],
                    logs=logs[:2],
                    gpus=inventory[:2],
                    started=0.0,
                    timeout=10.0,
                    clock=lambda: 0.0,
                    sleep=lambda _seconds: None,
                )

    def test_wave_launch_serializes_each_rank_through_runtime_ready(self):
        inventory = _gpu_inventory()

        class CompletedProcess:
            returncode = 0

            def __init__(self, pid):
                self.pid = pid

            def poll(self):
                return self.returncode

            def wait(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            runner = object.__new__(saturation.ProcessWaveRunner)
            runner.command = ("/bin/false",)
            runner.log_root = root / "logs"
            runner.attempt_id = "retry-a"
            runner.physical_gpu_inventory = inventory
            runner.physical_gpu_inventory_sha256 = (
                artifacts.canonical_json_sha256(inventory)
            )
            runner.gpu_device_namespaces = _gpu_device_namespaces(inventory)
            runner.host_runtime_attestation_sha256 = "3" * 64
            runtime_prefixes = []
            kit_logs = []

            def fake_popen(_command, *, stdout, env, **_kwargs):
                rank = int(env["ROUND_DAGGER_ACTOR_RANK"])
                # A subsequent Popen is forbidden until the preceding
                # launched prefix has passed the runtime-ready wait.
                self.assertEqual(rank, len(runtime_prefixes))
                stdout.flush()
                return CompletedProcess(10_000 + rank)

            def fake_runtime_wait(*, processes, logs, gpus, **_kwargs):
                prefix = len(processes)
                self.assertEqual(len(logs), prefix)
                self.assertEqual(list(gpus), inventory[:prefix])
                runtime_prefixes.append(prefix)

            def fake_kit_wait(**kwargs):
                kit_logs.append(kwargs["kit_log"])

            def fake_publish(_path):
                self.assertEqual(
                    runtime_prefixes,
                    list(range(1, VIRTUAL_ACTOR_WORLD_SIZE + 1)),
                )

            environment = {
                "ROUND_DAGGER_GENERATION_ROOT": str(
                    root / "BC/collection"
                ),
                "ROUND_DAGGER_ISAAC_RUNTIME_SHA256": "1" * 64,
                "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256": "2" * 64,
            }
            with mock.patch.object(
                saturation, "CANONICAL_CAMPAIGN_ROOT", root
            ), mock.patch.object(
                saturation, "query_physical_gpu_inventory", return_value=inventory
            ), mock.patch.object(
                saturation,
                "query_gpu_device_namespaces",
                return_value=runner.gpu_device_namespaces,
            ), mock.patch.object(
                saturation.subprocess, "Popen", side_effect=fake_popen
            ), mock.patch.object(
                saturation, "_wait_rank_kit_ready", side_effect=fake_kit_wait
            ), mock.patch.object(
                saturation,
                "_wait_all_ranks_runtime_ready",
                side_effect=fake_runtime_wait,
            ), mock.patch.object(
                saturation, "_publish_fleet_release", side_effect=fake_publish
            ), mock.patch.object(
                saturation, "_wait_all_ranks_fleet_released"
            ), mock.patch.object(
                saturation, "wait_for_admitted_gpu_quiescence"
            ), mock.patch.object(
                saturation, "rank_log_is_healthy", return_value=True
            ):
                result = runner(
                    envs_per_gpu=1,
                    steps=SMOKE_STEPS,
                    phase="smoke",
                    wave_index=0,
                    timeout=10.0,
                    environment=environment,
                )

            self.assertEqual(
                runtime_prefixes,
                list(range(1, VIRTUAL_ACTOR_WORLD_SIZE + 1)),
            )
            self.assertEqual(
                len(result["rank_results"]), VIRTUAL_ACTOR_WORLD_SIZE
            )
            self.assertEqual(
                kit_logs,
                [
                    root
                    / "BC/collection"
                    / "runtime/attempts/retry-a/smoke/wave-000/gpu-00/"
                    "process-state/kit-logs/kit.log"
                ],
            )

    def test_wave_executes_single_virtual_rank_on_admitted_gpu_zero(self):
        inventory = _gpu_inventory()

        class CompletedProcess:
            returncode = 0

            def __init__(self, pid):
                self.pid = pid

            def poll(self):
                return self.returncode

            def wait(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            runner = object.__new__(saturation.ProcessWaveRunner)
            runner.command = ("/bin/false",)
            runner.log_root = root / "logs"
            runner.attempt_id = "production-a"
            runner.physical_gpu_inventory = inventory
            runner.physical_gpu_inventory_sha256 = (
                artifacts.canonical_json_sha256(inventory)
            )
            runner.gpu_device_namespaces = _gpu_device_namespaces(inventory)
            runner.host_runtime_attestation_sha256 = "3" * 64
            runner.admitted_physical_gpu_indices = (0,)
            runner.virtual_actor_world_size = VIRTUAL_ACTOR_WORLD_SIZE
            runner.virtual_rank_cohorts = saturation.virtual_rank_cohorts((0,))
            launches = []
            releases = []
            quiescence = mock.Mock()

            def fake_popen(_command, *, stdout, env, **_kwargs):
                rank = int(env["ROUND_DAGGER_ACTOR_RANK"])
                physical = int(env["ROUND_DAGGER_ASSIGNED_PHYSICAL_INDEX"])
                expected = json.loads(env["ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON"])
                self.assertEqual(expected, inventory[physical])
                launches.append((rank, physical))
                stdout.flush()
                return CompletedProcess(20_000 + rank)

            environment = {
                "ROUND_DAGGER_GENERATION_ROOT": str(
                    root / "BC/collection"
                ),
                "ROUND_DAGGER_ISAAC_RUNTIME_SHA256": "1" * 64,
                "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256": "2" * 64,
            }
            with mock.patch.object(
                saturation, "CANONICAL_CAMPAIGN_ROOT", root
            ), mock.patch.object(
                saturation, "query_physical_gpu_inventory", return_value=inventory
            ), mock.patch.object(
                saturation,
                "query_gpu_device_namespaces",
                return_value=runner.gpu_device_namespaces,
            ), mock.patch.object(
                saturation.subprocess, "Popen", side_effect=fake_popen
            ), mock.patch.object(
                saturation, "_wait_rank_kit_ready"
            ), mock.patch.object(
                saturation, "_wait_all_ranks_runtime_ready"
            ), mock.patch.object(
                saturation,
                "_publish_fleet_release",
                side_effect=lambda path: releases.append(path),
            ), mock.patch.object(
                saturation, "_wait_all_ranks_fleet_released"
            ), mock.patch.object(
                saturation,
                "wait_for_admitted_gpu_quiescence",
                side_effect=quiescence,
            ), mock.patch.object(
                saturation, "rank_log_is_healthy", return_value=True
            ):
                result = runner(
                    envs_per_gpu=128,
                    steps=WAVE_STEPS,
                    phase="collect",
                    wave_index=0,
                    timeout=10.0,
                    environment=environment,
                )

            self.assertEqual(
                launches,
                [(0, 0)],
            )
            self.assertEqual(len(releases), 1)
            self.assertEqual(quiescence.call_count, 2)
            self.assertEqual(
                [row["physical_gpu_index"] for row in result["rank_results"]],
                [0],
            )
            self.assertEqual(
                len(result["rank_results"]), VIRTUAL_ACTOR_WORLD_SIZE
            )

    def test_wave_aborts_live_above_memory_limit_and_rechecks_quiescence(self):
        inventory = _gpu_inventory()

        class FinishedProcess:
            pid = 31_000
            returncode = 0

            def poll(self):
                return self.returncode

            def wait(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            generation = root / "BC/collection"
            generation.mkdir(parents=True)
            runner = object.__new__(saturation.ProcessWaveRunner)
            runner.command = ("/bin/false",)
            runner.log_root = generation / "logs/test-memory-abort"
            runner.admitted_physical_gpu_indices = (0,)
            runner.virtual_actor_world_size = VIRTUAL_ACTOR_WORLD_SIZE
            runner.virtual_rank_cohorts = (((0, 0),),)
            runner.physical_gpu_inventory = inventory
            runner.physical_gpu_inventory_sha256 = (
                artifacts.canonical_json_sha256(inventory)
            )
            runner.gpu_device_namespaces = _gpu_device_namespaces(inventory)
            runner.host_runtime_attestation_sha256 = "3" * 64

            memory = {
                gpu["uuid"]: (0, gpu["total_memory_mib"])
                for gpu in inventory
            }
            memory[inventory[0]["uuid"]] = (
                math.floor(inventory[0]["total_memory_mib"] * 0.86),
                inventory[0]["total_memory_mib"],
            )
            runner._memory = lambda: memory
            quiescence = mock.Mock()

            def sample_during_startup(**kwargs):
                kwargs["sample_telemetry"]()

            with mock.patch.object(
                saturation, "CANONICAL_CAMPAIGN_ROOT", root
            ), mock.patch.object(
                saturation, "query_physical_gpu_inventory", return_value=inventory
            ), mock.patch.object(
                saturation,
                "query_gpu_device_namespaces",
                return_value=runner.gpu_device_namespaces,
            ), mock.patch.object(
                saturation.subprocess,
                "Popen",
                return_value=FinishedProcess(),
            ), mock.patch.object(
                saturation,
                "_wait_rank_kit_ready",
                side_effect=sample_during_startup,
            ), mock.patch.object(
                saturation,
                "wait_for_admitted_gpu_quiescence",
                side_effect=quiescence,
            ), self.assertRaisesRegex(
                SaturationError, "live memory limit exceeded"
            ):
                runner(
                    envs_per_gpu=128,
                    steps=PROBE_STEPS,
                    phase="probe",
                    wave_index=0,
                    timeout=10.0,
                    environment={
                        "ROUND_DAGGER_GENERATION_ROOT": str(generation),
                        "ROUND_DAGGER_ISAAC_RUNTIME_SHA256": "1" * 64,
                        "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256": "2" * 64,
                    },
                )
            self.assertEqual(quiescence.call_count, 2)

    def test_fleet_release_token_and_all_rank_consumption_are_exact(self):
        class AliveProcess:
            returncode = None

            def poll(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            release = root / "kit-ready.release"
            saturation._publish_fleet_release(release)
            self.assertEqual(release.read_bytes(), b"release\n")
            mode = release.stat().st_mode & 0o777
            self.assertEqual(mode & 0o222, 0)
            self.assertNotEqual(mode & 0o400, 0)
            with self.assertRaisesRegex(SaturationError, "already exists"):
                saturation._publish_fleet_release(release)

            logs = []
            for rank in range(GPU_COUNT):
                path = root / f"gpu-{rank:02d}.log"
                path.write_text(
                    f"NATIVE_ACTOR_FLEET_RELEASED rank={rank}\n",
                    encoding="utf-8",
                )
                logs.append(path)
            processes = [AliveProcess() for _ in range(GPU_COUNT)]
            saturation._wait_all_ranks_fleet_released(
                processes=processes,
                logs=logs,
                started=time.monotonic(),
                timeout=10.0,
            )

            original = logs[7].read_text(encoding="utf-8")
            for name, text in {
                "missing marker": "",
                "duplicate marker": original + original,
                "fatal before marker": original + "Fatal Python error\n",
            }.items():
                with self.subTest(name=name):
                    logs[7].write_text(text, encoding="utf-8")
                    with mock.patch.object(
                        saturation.time, "monotonic", return_value=2.0
                    ), self.assertRaises(SaturationError):
                        saturation._wait_all_ranks_fleet_released(
                            processes=processes,
                            logs=logs,
                            started=0.0,
                            timeout=1.0,
                        )
            logs[7].write_text(original, encoding="utf-8")
            saturation._retract_fleet_release(release)
            self.assertFalse(release.exists())

    def test_rank_health_requires_each_camera_content_marker(self):
        gpu = _gpu_inventory()[4]
        gpu_namespace = _gpu_device_namespaces()[4]
        base = (
            "NATIVE_ACTOR_TRAINER_BODY_OK rank=4 phase=smoke\n"
            "NATIVE_ACTOR_RANK_OK rank=4 phase=smoke\n"
            f"NATIVE_ACTOR_GPU_OK rank=4 uuid={gpu['uuid']} "
            f"pci={gpu['pci_bus_id']} logical_cuda=0 local_nvml=0 "
            "visible_devices=1 cuda_visible_devices=unset\n"
            + _startup_protocol_markers(4, gpu)
            + native_actor_kit_isolation_marker(
                rank=4, portable_root=_portable_root(4)
            )
            + "\n"
        )
        camera = (
            "NATIVE_CAMERA_CONTENT_OK rank=4 "
            "views=head_color,hand_left,hand_right spatial=1 temporal=1 distinct=1\n"
            "NATIVE_CAMERA_CADENCE_OK rank=4 control_hz=50 render_hz=5 "
            "stride=10 fresh_reset_step0=1 source_mapping=1\n"
        )
        self.assertTrue(rank_log_is_healthy(
            text=base + camera, rank=4, gpu=gpu,
            gpu_namespace=gpu_namespace, returncode=0,
        ))

        isolation_marker = native_actor_kit_isolation_marker(
            rank=4, portable_root=_portable_root(4)
        )
        self.assertFalse(rank_log_is_healthy(
            text=base, rank=4, gpu=gpu,
            gpu_namespace=gpu_namespace, returncode=0,
        ))
        self.assertFalse(rank_log_is_healthy(
            text=base + camera.splitlines(keepends=True)[0],
            rank=4,
            gpu=gpu,
            gpu_namespace=gpu_namespace,
            returncode=0,
        ))
        self.assertFalse(rank_log_is_healthy(
            text=(base + camera).replace(isolation_marker + "\n", ""),
            rank=4,
            gpu=gpu,
            gpu_namespace=gpu_namespace,
            returncode=0,
        ))
        self.assertFalse(rank_log_is_healthy(
            text=base + camera + isolation_marker + "\n",
            rank=4,
            gpu=gpu,
            gpu_namespace=gpu_namespace,
            returncode=0,
        ))
        self.assertFalse(rank_log_is_healthy(
            text=base + camera + "Traceback", rank=4, gpu=gpu,
            gpu_namespace=gpu_namespace, returncode=0,
        ))
        self.assertFalse(rank_log_is_healthy(
            text=(base + camera).replace(
                "NATIVE_ACTOR_TRAINER_BODY_OK rank=4 phase=smoke\n", "", 1
            ),
            rank=4,
            gpu=gpu,
            gpu_namespace=gpu_namespace,
            returncode=0,
        ))
        self.assertFalse(rank_log_is_healthy(
            text=base
            + camera
            + "NATIVE_ACTOR_TRAINER_BODY_OK rank=4 phase=probe\n",
            rank=4,
            gpu=gpu,
            gpu_namespace=gpu_namespace,
            returncode=0,
        ))
        for marker in _startup_protocol_markers(4, gpu).splitlines(keepends=True):
            with self.subTest(missing_protocol_marker=marker.strip()):
                self.assertFalse(
                    rank_log_is_healthy(
                        text=(base + camera).replace(marker, "", 1),
                        rank=4,
                        gpu=gpu,
                        gpu_namespace=gpu_namespace,
                        returncode=0,
                    )
                )

    def test_rank_health_accepts_virtual_rank_on_different_physical_cache(self):
        rank = 7
        gpu = _gpu_inventory()[2]
        gpu_namespace = _gpu_device_namespaces()[2]
        text = (
            f"NATIVE_ACTOR_TRAINER_BODY_OK rank={rank} phase=collect\n"
            f"NATIVE_ACTOR_RANK_OK rank={rank} phase=collect\n"
            f"NATIVE_ACTOR_GPU_OK rank={rank} uuid={gpu['uuid']} "
            f"pci={gpu['pci_bus_id']} logical_cuda=0 local_nvml=0 "
            "visible_devices=1 cuda_visible_devices=unset\n"
            + _startup_protocol_markers(rank, gpu, gpu_namespace)
            + native_actor_kit_isolation_marker(
                rank=rank,
                portable_root=_portable_root(rank, physical_index=2),
            )
            + "\n"
            + f"NATIVE_CAMERA_CONTENT_OK rank={rank} "
            "views=head_color,hand_left,hand_right spatial=1 temporal=1 distinct=1\n"
            + f"NATIVE_CAMERA_CADENCE_OK rank={rank} control_hz=50 render_hz=5 "
            "stride=10 fresh_reset_step0=1 source_mapping=1\n"
        )
        self.assertTrue(
            rank_log_is_healthy(
                text=text,
                rank=rank,
                gpu=gpu,
                gpu_namespace=gpu_namespace,
                returncode=0,
            )
        )

    def test_kit_isolation_accepts_actual_equals_or_space_cmd_formatting(self):
        for separator, quote in (("=", "'"), (" ", '"'), (" ", "")):
            with self.subTest(separator=separator, quote=quote):
                marker = verify_native_actor_kit_log(
                    text=_healthy_kit_log(4, separator=separator, quote=quote),
                    rank=4,
                    expected_portable_root=_portable_root(4),
                )
                self.assertEqual(
                    marker,
                    native_actor_kit_isolation_marker(
                        rank=4, portable_root=_portable_root(4)
                    ),
                )

    def test_kit_isolation_rejects_wrong_root_gpu_and_hydra_leakage(self):
        valid = _healthy_kit_log(4)
        corruptions = {
            "non-rank-local portable root": valid.replace(
                "Portable Root: '", "Portable Root: '/sealed/wrong/", 1
            ),
            "renderer GPU": valid.replace("activeGpu=0", "activeGpu=4"),
            "physics GPU": valid.replace("cudaDevice=0", "cudaDevice=4"),
            "Hydra CLI leakage": valid.replace(
                " --no-window", " num_envs=32 --no-window"
            ),
            "implicit startup stage": (
                valid
                + "[Info] [isaacsim.simulation_app] "
                + "SimulationApp.__init__: Creating new stage\n"
            ),
        }
        for reason, text in corruptions.items():
            with self.subTest(reason=reason), self.assertRaises(SaturationError):
                verify_native_actor_kit_log(
                    text=text,
                    rank=4,
                    expected_portable_root=_portable_root(4),
                )

    def test_kit_and_rank_health_reject_isolation_and_remote_access_records(self):
        gpu = _gpu_inventory()[4]
        gpu_namespace = _gpu_device_namespaces()[4]
        camera = (
            "NATIVE_CAMERA_CONTENT_OK rank=4 "
            "views=head_color,hand_left,hand_right spatial=1 temporal=1 distinct=1\n"
            "NATIVE_CAMERA_CADENCE_OK rank=4 control_hz=50 render_hz=5 "
            "stride=10 fresh_reset_step0=1 source_mapping=1\n"
        )
        rank_base = (
            "NATIVE_ACTOR_TRAINER_BODY_OK rank=4 phase=smoke\n"
            "NATIVE_ACTOR_RANK_OK rank=4 phase=smoke\n"
            f"NATIVE_ACTOR_GPU_OK rank=4 uuid={gpu['uuid']} "
            f"pci={gpu['pci_bus_id']} logical_cuda=0 local_nvml=0 "
            "visible_devices=1 cuda_visible_devices=unset\n"
            + _startup_protocol_markers(4, gpu)
            + native_actor_kit_isolation_marker(
                rank=4, portable_root=_portable_root(4)
            )
            + "\n"
            + camera
        )
        forbidden_records = (
            "Failed to acquire exclusive lock to data store",
            "OSError: [Errno 30] Read-only file system",
            "activeGpu 0 is not compatible with current foundation settings",
            "Failed to create any GPU devices",
            "unable to get a valid CUDA device id from the renderer",
            "Successfully created S3 provider for https://example.s3-us-west-2.amazonaws.com",
            "Using CloudFront http://example.cloudfront.net/",
            "Opened asset 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/a.usd'",
            "Opened asset 'omniverse://server/path/robot.usd' for read",
        )
        for record in forbidden_records:
            with self.subTest(record=record), self.assertRaises(SaturationError):
                verify_native_actor_kit_log(
                    text=_healthy_kit_log(4) + record + "\n",
                    rank=4,
                    expected_portable_root=_portable_root(4),
                )
            with self.subTest(rank_record=record):
                self.assertFalse(
                    rank_log_is_healthy(
                        text=rank_base + record + "\n",
                        rank=4,
                        gpu=gpu,
                        gpu_namespace=gpu_namespace,
                        returncode=0,
                    )
                )

        for namespace_warning in (
            "Skipping NVIDIA GPU due CUDA being in bad state",
            "Note CUDA device enumeration and Omniverse device enumeration are different.",
            "CUDA_VISIBLE_DEVICES environment variable is set.",
        ):
            with self.subTest(namespace_warning=namespace_warning), \
                    self.assertRaisesRegex(SaturationError, "device-namespace mismatch"):
                verify_native_actor_kit_log(
                    text=_healthy_kit_log(4) + namespace_warning + "\n",
                    rank=4,
                    expected_portable_root=_portable_root(4),
                )
            self.assertFalse(
                rank_log_is_healthy(
                    text=rank_base + namespace_warning + "\n",
                    rank=4,
                    gpu=gpu,
                    gpu_namespace=gpu_namespace,
                    returncode=0,
                )
            )

    def test_selection_requires_the_rank_at_the_fixed_production_density(self):
        settings = {128: (1570, 0.75, None)}
        probes = [
            validate_probe(_result(envs, PROBE_STEPS, *settings[envs][:2]), envs)
            for envs in CANDIDATES
        ]
        self.assertEqual(select_candidate(probes)["envs_per_gpu"], 128)

        broken = validate_probe(
            _result(128, PROBE_STEPS, 1800, 0.5, bad_rank=0), 128
        )
        self.assertFalse(broken["eligible"])
        self.assertIn("rank_unhealthy", broken["rejection_reasons"])
        with self.assertRaisesRegex(SaturationError, "all saturation"):
            select_candidate(probes[:-1])

    def test_wave_math_is_ceil_to_target_with_less_than_one_wave_overshoot(self):
        expected = {128: 2}
        for envs, waves in expected.items():
            with self.subTest(envs=envs):
                shape = wave_shape(envs)
                self.assertEqual(shape["planned_waves"], waves)
                self.assertGreaterEqual(shape["planned_transitions"], NOMINAL_TRANSITIONS)
                self.assertLess(shape["planned_overshoot"], shape["transitions_per_wave"])

    def test_group_surplus_is_balanced_and_rotates_between_waves(self):
        groups = tuple(f"group-{index:02d}" for index in range(23))
        first = group_rotation(groups, 128, 0)
        second = group_rotation(groups, 128, 1)
        self.assertLessEqual(max(first["counts"].values()) - min(first["counts"].values()), 1)
        self.assertNotEqual(first["start_group_offset"], second["start_group_offset"])
        self.assertNotEqual(first["rank_group_offsets"], second["rank_group_offsets"])

    def test_shard_names_are_generation_wave_gpu_and_global_index_bound(self):
        names = expected_shard_names(
            wave_index=1, envs_per_gpu=128, environments_per_shard=4
        )
        self.assertEqual(len(names), 32 * 2)
        self.assertEqual(
            names[0], "gen-000-bc__wave-001__gpu-00__shard-000032.npz"
        )
        self.assertEqual(
            names[-1], "gen-000-bc__wave-001__gpu-00__shard-000063.json"
        )
        self.assertEqual(len(names), len(set(names)))

    def test_actor_export_derives_only_the_exact_generation_inventory(self):
        campaign = Path("/workspace/project/roborender/data")
        generation = campaign / "BC/collection"
        cohort = generation / "conditioning/waves/wave-001"
        names = expected_shard_names(
            wave_index=1, envs_per_gpu=128, environments_per_shard=4
        )
        environment = {
            "ROUND_DAGGER_CAMPAIGN_ROOT": str(campaign),
            "ROUND_DAGGER_GENERATION_ROOT": str(generation),
            "ROUND_DAGGER_COHORT_DIR": str(cohort),
            "ROUND_DAGGER_GENERATION_ID": "gen-000-bc",
            "ROUND_DAGGER_MACRO_ROUND": "0",
            "ROUND_DAGGER_ACTOR_ROLE": "bc_collect",
            "ROUND_DAGGER_BETA": "1.0",
            "ROUND_DAGGER_SATURATION_PHASE": "collect",
            "ROUND_DAGGER_SATURATION_WAVE_INDEX": "1",
            "ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON": json.dumps(names),
            "ROUND_DAGGER_OBSERVATION_SOURCE": "native",
            "ROUND_DAGGER_PURPOSE": "train",
            "ROUND_DAGGER_ACTOR_RANK": "0",
            "ROUND_DAGGER_ACTOR_WORLD_SIZE": "1",
        }
        shard_ids = _native_generation_shard_ids_from_environment(
            environment, rank=0, batch_size=128, environments_per_shard=4
        )
        self.assertEqual(len(shard_ids), 32)
        self.assertEqual(
            shard_ids[0], "gen-000-bc__wave-001__gpu-00__shard-000032"
        )

        changed = dict(environment)
        changed["ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON"] = json.dumps(names[:-1])
        with self.assertRaisesRegex(ActorExportError, "inventory differs"):
            _native_generation_shard_ids_from_environment(
                changed, rank=0, batch_size=128, environments_per_shard=4
            )
        partial = dict(environment)
        partial.pop("ROUND_DAGGER_GENERATION_ROOT")
        with self.assertRaisesRegex(ActorExportError, "incomplete"):
            _native_generation_shard_ids_from_environment(
                partial, rank=0, batch_size=128, environments_per_shard=4
            )

    def test_artifact_native_shard_id_and_purposes_fail_closed(self):
        exact = "gen-000-bc__wave-001__gpu-03__shard-000176"
        self.assertEqual(
            artifacts._shard_id(3, 123, 7, native_shard_id=exact), exact
        )
        for invalid in (
            "gen-001-dagger__wave-001__gpu-03__shard-000176",
            "gen-000-bc__wave-001__gpu-04__shard-000176",
            "rank-0003_rollout-00000001_part-0007",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(artifacts.ArtifactError):
                artifacts._shard_id(3, 123, 7, native_shard_id=invalid)

        contract_values = {
            "campaign_id": "native-purpose-test",
            "macro_round": 0,
            "cohort_index": 0,
            "expected_ranks": [0],
            "expected_global_env_ids": [0],
            "expected_transitions": 1,
            "view_contract": {
                "dtype": "uint8", "layout": "vertical", "shape": [6, 4, 3],
                "views": ["head_color", "hand_left", "hand_right"],
            },
            "action_contract": {"dtype": "float32", "shape": [2]},
            "policy_sha256": "1" * 64,
            "runtime_policy_sha256": "2" * 64,
            "teacher_sha256": "3" * 64,
            "renderer_sha256": "4" * 64,
            "beta_sha256": "5" * 64,
            "config_sha256": "6" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for purpose in ("train", "validation", "test"):
                contract = artifacts.create_cohort(
                    root / purpose, purpose=purpose, **contract_values
                )
                self.assertEqual(contract["purpose"], purpose)
            with self.assertRaisesRegex(artifacts.ArtifactError, "train, validation, or test"):
                artifacts.create_cohort(
                    root / "eval", purpose="eval", **contract_values
                )

    def test_native_artifact_provenance_is_direct_h200_and_has_no_renderer_alias(self):
        config_sha = "6" * 64
        runtime = {
            "schema": "fast_roborender.round_dagger.native_h200_observation_runtime.v1",
            "observation_source": "native",
            "isaac_runtime_sha256": "7" * 64,
            "actor_python_freeze_sha256": "8" * 64,
            "config_sha256": config_sha,
            "actor_source_inventory_sha256": "9" * 64,
            "gr00t_revision": "a" * 40,
            "gr00t_source_tree_sha256": "b" * 64,
            "gpu_name": "NVIDIA H200",
            "compute_capability": "9.0",
            "gpu_count": 8,
            "physical_gpu_inventory": _gpu_inventory(),
            "physical_gpu_inventory_sha256": artifacts.canonical_json_sha256(
                _gpu_inventory()
            ),
        }
        runtime_sha = artifacts.canonical_json_sha256(runtime)
        environment = {
            "ROUND_DAGGER_OBSERVATION_SOURCE": "native",
            "ROUND_DAGGER_OBSERVATION_RUNTIME_SHA256": runtime_sha,
            "ROUND_DAGGER_ISAAC_RUNTIME_SHA256": runtime["isaac_runtime_sha256"],
            "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256": runtime[
                "actor_python_freeze_sha256"
            ],
            "ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256": runtime[
                "actor_source_inventory_sha256"
            ],
            "ROUND_DAGGER_GR00T_REVISION": runtime["gr00t_revision"],
            "ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256": runtime[
                "gr00t_source_tree_sha256"
            ],
            "ROUND_DAGGER_GPU_NAME": "NVIDIA H200",
            "ROUND_DAGGER_COMPUTE_CAPABILITY": "9.0",
            "ROUND_DAGGER_GPU_COUNT": "8",
            "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON": json.dumps(
                _gpu_inventory()
            ),
            "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256": runtime[
                "physical_gpu_inventory_sha256"
            ],
        }
        provenance = _observation_provenance_from_environment(environment)
        with tempfile.TemporaryDirectory() as temporary:
            contract = artifacts.create_cohort(
                Path(temporary).resolve() / "native",
                campaign_id="native-provenance-test",
                macro_round=0,
                cohort_index=0,
                purpose="train",
                expected_ranks=[0],
                expected_global_env_ids=[0],
                expected_transitions=1,
                view_contract={
                    "dtype": "uint8", "layout": "vertical", "shape": [6, 4, 3],
                    "views": ["head_color", "hand_left", "hand_right"],
                },
                action_contract={"dtype": "float32", "shape": [2]},
                policy_sha256="1" * 64,
                runtime_policy_sha256="2" * 64,
                teacher_sha256="3" * 64,
                beta_sha256="5" * 64,
                config_sha256=config_sha,
                **provenance,
            )
            self.assertNotIn("renderer_sha256", contract)
            self.assertNotIn("actor_runtime_attestation_path", contract)
            self.assertEqual(contract["gpu_count"], 8)
            with self.assertRaisesRegex(artifacts.ArtifactError, "forbids renderer"):
                artifacts.create_cohort(
                    Path(temporary).resolve() / "native-with-renderer",
                    campaign_id="native-provenance-test",
                    macro_round=0,
                    cohort_index=1,
                    purpose="train",
                    expected_ranks=[0],
                    expected_global_env_ids=[0],
                    expected_transitions=1,
                    view_contract={
                        "dtype": "uint8", "layout": "vertical", "shape": [6, 4, 3],
                        "views": ["head_color", "hand_left", "hand_right"],
                    },
                    action_contract={"dtype": "float32", "shape": [2]},
                    policy_sha256="1" * 64,
                    runtime_policy_sha256="2" * 64,
                    teacher_sha256="3" * 64,
                    renderer_sha256=runtime_sha,
                    beta_sha256="5" * 64,
                    config_sha256=config_sha,
                    **provenance,
                )

    def test_seal_inventory_rejects_old_or_extra_shard_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            names = [
                "gen-000-bc__wave-000__gpu-00__shard-000000.npz",
                "gen-000-bc__wave-000__gpu-00__shard-000000.json",
            ]
            for name in names:
                (root / name).write_bytes(name.encode())
            inventory = verify_shard_inventory(
                root, names, deadline=time.monotonic() + 10
            )
            self.assertEqual([row["name"] for row in inventory], sorted(names))
            (root / "rank-0000_rollout-00000000_part-0000.npz").write_bytes(b"old")
            with self.assertRaisesRegex(SaturationError, "differs"):
                verify_shard_inventory(root, names, deadline=time.monotonic() + 10)

    def test_two_nominal_waves_fit_the_shared_collection_deadline(self):
        clock = _Clock()
        partition = {
            "path": "/immutable/train.json",
            "manifest_sha256": "1" * 64,
            "split_sha256": "2" * 64,
            "group_ids": tuple(f"group-{index:02d}" for index in range(23)),
        }

        def runner(*, envs_per_gpu, steps, phase, **_):
            if phase == "probe":
                duration = 10
                throughput = 2000
                memory = 0.75
            else:
                duration = 1800
                throughput = (
                    VIRTUAL_ACTOR_WORLD_SIZE * envs_per_gpu * steps / duration
                )
                memory = 0.80
            clock.advance(duration)
            return _result(
                envs_per_gpu, steps, throughput, memory, duration=duration
            )

        sealed = []

        def seal(environment, timeout):
            sealed.append((environment, timeout))
            return {
                "path": "/sealed/seal.json",
                "sha256": "a" * 64,
                "shards": [],
                "image_count": 6_656,
                "episode_count": 128,
            }

        report = run_tuner(
            partition=partition,
            wave_runner=runner,
            seal_runner=seal,
            clock=clock,
        )
        self.assertEqual(saturation.WALL_SECONDS, 5400.0)
        self.assertEqual(report["selection"]["environments_per_gpu"], 128)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["stop_reason"], "nominal_target_met")
        self.assertEqual(len(report["waves"]), 2)
        self.assertEqual(
            report["completed_transitions"],
            2 * VIRTUAL_ACTOR_WORLD_SIZE * 128 * WAVE_STEPS,
        )
        self.assertEqual(report["completed_image_count"], 13_312)
        self.assertEqual(report["completed_episode_count"], 256)
        self.assertEqual(len(sealed), 2)
        self.assertTrue(all(timeout > 0 for _, timeout in sealed))

    def test_collection_eta_scales_execution_but_not_serial_startup(self):
        clock = _Clock()
        partition = {
            "path": "/immutable/train.json",
            "manifest_sha256": "1" * 64,
            "split_sha256": "2" * 64,
            "group_ids": ("group",),
        }
        phases = []
        deadlines = []

        def runner(*, envs_per_gpu, steps, phase, **_):
            phases.append(phase)
            if phase == "probe":
                # A full 900-second probe is admissible: 800 seconds of
                # serialized startup is fixed, while only 100 seconds of
                # execution scales with the four-times-longer collection.
                clock.advance(900.0)
                return _result(
                    envs_per_gpu,
                    steps,
                    1000,
                    0.5,
                    duration=900.0,
                    startup=800.0,
                )
            clock.advance(1000.0)
            return _result(
                envs_per_gpu,
                steps,
                1000,
                0.5,
                duration=1000.0,
                startup=600.0,
            )

        report = run_tuner(
            partition=partition,
            wave_runner=runner,
            seal_runner=lambda _environment, _timeout: {
                "path": "/sealed/seal.json",
                "sha256": "a" * 64,
                "shards": [],
                "image_count": 13_312,
                "episode_count": 256,
            },
            clock=clock,
        )
        self.assertEqual(phases, ["probe", "collect", "collect"])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(
            report["selection"]["conservative_wave_eta"],
            (800.0 + 4 * 100.0) * 1.25,
        )

    def test_probe_rejects_inexact_startup_execution_decomposition(self):
        result = _result(128, PROBE_STEPS, 1000, 0.5, duration=10.0)
        result["execution_seconds"] = 9.0
        validated = validate_probe(result, 128)
        self.assertFalse(validated["eligible"])
        self.assertIn("invalid_performance", validated["rejection_reasons"])

    def test_absolute_deadline_rejects_every_late_production_stage(self):
        partition = {
            "path": "/immutable/train.json",
            "manifest_sha256": "1" * 64,
            "split_sha256": "2" * 64,
            "group_ids": ("group",),
        }
        for late_stage in ("prepare", "collect", "seal"):
            clock = _Clock()
            # This explicit deadline injection exercises collection stages;
            # probe overruns use the retryable probe_budget_exhausted result.
            clock.value = 500.0

            def runner(*, envs_per_gpu, steps, phase, **_):
                if phase == late_stage:
                    clock.value = 5400.001
                return _result(
                    envs_per_gpu, steps, 2000, 0.5, duration=1.0
                )

            def prepare(_environment, _timeout):
                if late_stage == "prepare":
                    clock.value = 5400.001

            def seal(_environment, _timeout):
                if late_stage == "seal":
                    clock.value = 5400.001
                return {
                    "path": "/sealed",
                    "sha256": "a" * 64,
                    "shards": [],
                    "image_count": 13_312,
                    "episode_count": 256,
                }

            with self.subTest(stage=late_stage), self.assertRaisesRegex(
                SaturationError, "hard deadline expired"
            ):
                run_tuner(
                    partition=partition,
                    wave_runner=runner,
                    prepare_runner=prepare,
                    seal_runner=seal,
                    clock=clock,
                    hard_deadline=5400.0,
                )

    def test_completion_exactly_at_absolute_deadline_is_allowed(self):
        clock = _Clock()
        clock.value = 500.0
        partition = {
            "path": "/immutable/train.json",
            "manifest_sha256": "1" * 64,
            "split_sha256": "2" * 64,
            "group_ids": ("group",),
        }

        def runner(*, envs_per_gpu, steps, phase, **_):
            return _result(envs_per_gpu, steps, 2000, 0.5, duration=1.0)

        seal_calls = 0

        def seal(_environment, _timeout):
            nonlocal seal_calls
            seal_calls += 1
            if seal_calls == 2:
                clock.value = 5400.0
            return {
                "path": "/sealed",
                "sha256": "a" * 64,
                "shards": [],
                "image_count": 13_312,
                "episode_count": 256,
            }

        report = run_tuner(
            partition=partition,
            wave_runner=runner,
            seal_runner=seal,
            clock=clock,
            hard_deadline=5400.0,
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["elapsed_seconds"], 5400.0)

    def test_deadline_binds_file_hash_and_retracts_late_final_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            payload = root / "payload.bin"
            payload.write_bytes(b"payload")
            hash_times = iter((0.0, 2.0))
            with self.assertRaisesRegex(SaturationError, "file hashing"):
                saturation._sha_file(
                    payload,
                    deadline=1.0,
                    clock=lambda: next(hash_times),
                )

            manifest = root / "saturation_manifest.json"
            publish_times = iter((0.0, 0.0, 2.0))
            with self.assertRaisesRegex(SaturationError, "final manifest publication"):
                saturation._publish_json_create_only(
                    manifest,
                    {"status": "complete"},
                    deadline=1.0,
                    clock=lambda: next(publish_times),
                )
            self.assertFalse(manifest.exists())

    def test_nominal_wave_has_exact_global_ids_and_seal_evidence(self):
        clock = _Clock()
        partition = {
            "path": "/immutable/train.json",
            "manifest_sha256": "1" * 64,
            "split_sha256": "2" * 64,
            "group_ids": tuple(f"group-{index:02d}" for index in range(23)),
        }
        collect_environments = []

        def runner(*, envs_per_gpu, steps, phase, environment, **_):
            if phase == "collect":
                collect_environments.append(dict(environment))
            throughput = 2000
            return _result(envs_per_gpu, steps, throughput, 0.5, duration=1)

        def seal(environment, _timeout):
            wave = int(environment["ROUND_DAGGER_SATURATION_WAVE_INDEX"])
            return {
                "path": f"/sealed/wave-{wave:03d}/seal.json",
                "sha256": f"{wave + 1:064x}",
                "shards": [{"name": f"wave-{wave:03d}"}],
                "image_count": 6_656,
                "episode_count": 128,
            }

        report = run_tuner(
            partition=partition, wave_runner=runner, seal_runner=seal, clock=clock
        )
        self.assertEqual(len(collect_environments), 2)
        self.assertEqual(
            [env["ROUND_DAGGER_GLOBAL_ENV_OFFSET"] for env in collect_environments],
            ["0", "128"],
        )
        self.assertEqual(len(report["shard_inventory"]), 2)
        self.assertTrue(all(wave["seal"]["sha256"] for wave in report["waves"]))

    def test_probe_budget_is_part_of_same_wall_clock_and_fails_closed(self):
        clock = _Clock()
        partition = {
            "path": "/immutable/train.json",
            "manifest_sha256": "1" * 64,
            "split_sha256": "2" * 64,
            "group_ids": ("group",),
        }

        def runner(*, envs_per_gpu, steps, **_):
            clock.advance(901)
            return _result(envs_per_gpu, steps, 1000, 0.4, duration=901)

        report = run_tuner(partition=partition, wave_runner=runner, clock=clock)
        self.assertEqual(report["status"], "budget_capped")
        self.assertEqual(report["stop_reason"], "probe_budget_exhausted")
        self.assertEqual(report["waves"], [])
        self.assertLessEqual(len(report["probes"]), len(CANDIDATES))

    def test_measured_5121_7_second_timing_envelope_admits_both_waves(self):
        clock = _Clock()
        partition = {
            "path": "/immutable/train.json",
            "manifest_sha256": "1" * 64,
            "split_sha256": "2" * 64,
            "group_ids": ("group",),
        }
        phases = []
        deadlines = []
        probe_startup = 169.01333333333335
        collection_observed_seconds = 1_598.16

        def runner(*, envs_per_gpu, steps, phase, **_):
            phases.append(phase)
            if phase == "probe":
                clock.advance(526.3)
                return _result(
                    envs_per_gpu,
                    steps,
                    100.0,
                    0.75,
                    duration=526.3,
                    startup=probe_startup,
                )
            clock.advance(collection_observed_seconds)
            return _result(
                envs_per_gpu,
                steps,
                100.0,
                0.75,
                duration=collection_observed_seconds,
            )

        report = run_tuner(
            partition=partition,
            wave_runner=runner,
            seal_runner=lambda _environment, _timeout: {
                "path": "/sealed/seal.json",
                "sha256": "a" * 64,
                "shards": [],
                "image_count": 6_656,
                "episode_count": 128,
            },
            clock=clock,
            collection_deadline_sink=deadlines.append,
        )
        conservative_wave_eta = 1_997.7
        measured_envelope = (
            526.3
            + 2 * conservative_wave_eta
            + saturation.SEAL_RESERVE_SECONDS
        )
        self.assertAlmostEqual(measured_envelope, 5_121.7)
        self.assertEqual(len(deadlines), 1)
        self.assertAlmostEqual(
            deadlines[0], 526.3 + saturation.WALL_SECONDS
        )
        self.assertLess(
            2 * conservative_wave_eta + saturation.SEAL_RESERVE_SECONDS,
            saturation.WALL_SECONDS,
        )
        self.assertEqual(phases, ["probe", "collect", "collect"])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["waves"]), 2)
        self.assertAlmostEqual(
            report["selection"]["conservative_wave_eta"],
            conservative_wave_eta,
        )

    def test_deadline_admission_rejects_when_both_wave_slots_exceed_5400(self):
        clock = _Clock()
        partition = {
            "path": "/immutable/train.json",
            "manifest_sha256": "1" * 64,
            "split_sha256": "2" * 64,
            "group_ids": ("group",),
        }
        phases = []

        def runner(*, envs_per_gpu, steps, phase, **_):
            phases.append(phase)
            # 500 probe seconds scales to a conservative 2,500 seconds per
            # 512-step collection wave. One wave plus the seal reserve fits;
            # both committed waves do not.
            return _result(
                envs_per_gpu,
                steps,
                1.0,
                0.75,
                duration=500.0,
            )

        report = run_tuner(
            partition=partition,
            wave_runner=runner,
            clock=clock,
        )
        self.assertEqual(phases, ["probe"])
        self.assertEqual(report["status"], "budget_capped")
        self.assertEqual(
            report["stop_reason"],
            "all_remaining_wave_eta_crosses_seal_deadline",
        )
        self.assertEqual(report["waves"], [])

    def test_fixed_density_probe_timeout_is_bounded(self):
        clock = _Clock()
        partition = {
            "path": "/immutable/train.json",
            "manifest_sha256": "1" * 64,
            "split_sha256": "2" * 64,
            "group_ids": ("group",),
        }
        timeouts = []

        def runner(*, envs_per_gpu, steps, timeout, phase, **_):
            if phase == "probe":
                timeouts.append(timeout)
            duration = min(timeout, 1.0)
            clock.advance(duration)
            return _result(envs_per_gpu, steps, 1000, 0.4, duration=duration)

        run_tuner(partition=partition, wave_runner=runner, clock=clock)
        self.assertEqual(len(timeouts), 1)
        self.assertTrue(
            all(
                value <= saturation.PROBE_TIMEOUT_CAP_SECONDS
                for value in timeouts
            )
        )

    def test_cli_dry_run_builds_complete_manifest_without_launching(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            partition = root / "train.json"
            train_motion_root = root / "train-motions"
            train_motion_root.mkdir()
            motion = train_motion_root / "record_2025-10-14_fixture.npz"
            motion.write_bytes(b"x")
            document = {
                "schema": "fast_roborender.round_dagger.motion_partition.v1",
                "split_sha256": "2" * 64,
                "purpose": "train",
                "groups": [{
                    "group_id": "record_2025-10-14_fixture.npz",
                    "relative_path": "record_2025-10-14_fixture.npz",
                    "size_bytes": 1,
                    "motion_sha256": hashlib.sha256(b"x").hexdigest(),
                }],
            }
            partition.write_text(json.dumps(document), encoding="utf-8")
            digest = hashlib.sha256(partition.read_bytes()).hexdigest()
            output = io.StringIO()
            with mock.patch.object(saturation, "CANONICAL_CAMPAIGN_ROOT", root):
                with contextlib.redirect_stdout(output):
                    status = saturation.main([
                        "--campaign-root", str(root),
                        "--train-partition", str(partition),
                        "--train-partition-sha256", digest,
                        "--train-motion-root", str(train_motion_root),
                        "--dry-run",
                        "--gpu-indices", "0",
                        "--rank-command", "/bin/false",
                    ])
            report = json.loads(output.getvalue())
            self.assertEqual(status, 0)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["generation_id"], "gen-000-bc")
            self.assertEqual(report["selection"]["environments_per_gpu"], 128)
            self.assertEqual(len(report["probes"]), 1)
            self.assertEqual(len(report["waves"]), 2)
            self.assertEqual(report["virtual_actor_world_size"], 1)
            self.assertEqual(report["admitted_physical_gpu_indices"], [0])
            self.assertEqual(report["physical_execution_concurrency"], 1)
            self.assertEqual(
                [len(cohort) for cohort in report["virtual_rank_cohorts"]],
                [1],
            )

    def test_cli_smoke_runs_gpu_zero_without_replay_or_production_timer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            partition = root / "train.json"
            train_motion_root = root / "train-motions"
            train_motion_root.mkdir()
            motion = train_motion_root / "record_2025-10-14_fixture.npz"
            motion.write_bytes(b"x")
            document = {
                "schema": "fast_roborender.round_dagger.motion_partition.v1",
                "split_sha256": "2" * 64,
                "purpose": "train",
                "groups": [{
                    "group_id": "record_2025-10-14_fixture.npz",
                    "relative_path": "record_2025-10-14_fixture.npz",
                    "size_bytes": 1,
                    "motion_sha256": hashlib.sha256(b"x").hexdigest(),
                }],
            }
            partition.write_text(json.dumps(document), encoding="utf-8")
            digest = hashlib.sha256(partition.read_bytes()).hexdigest()
            calls = []

            class FakeRunner:
                def __init__(
                    self, command, log_root, admitted_indices, virtual_world_size,
                    attempt_id,
                ):
                    calls.append(
                        (
                            command,
                            log_root,
                            admitted_indices,
                            virtual_world_size,
                            attempt_id,
                        )
                    )
                    self.physical_gpu_inventory = _gpu_inventory()

                def __call__(self, **kwargs):
                    calls.append(kwargs)
                    return _result(1, SMOKE_STEPS, 10, 0.1, duration=1)

            output = io.StringIO()
            with mock.patch.object(saturation, "CANONICAL_CAMPAIGN_ROOT", root), \
                    mock.patch.object(saturation, "ProcessWaveRunner", FakeRunner), \
                    contextlib.redirect_stdout(output):
                status = saturation.main([
                    "--campaign-root", str(root),
                    "--train-partition", str(partition),
                    "--train-partition-sha256", digest,
                    "--train-motion-root", str(train_motion_root),
                    "--smoke-only",
                ])
            report = json.loads(output.getvalue())
            self.assertEqual(status, 0)
            self.assertEqual(report["status"], "smoke_complete")
            self.assertFalse(report["replay_published"])
            self.assertEqual(calls[1]["phase"], "smoke")
            self.assertEqual(calls[1]["envs_per_gpu"], 1)
            self.assertEqual(calls[1]["steps"], SMOKE_STEPS)
            self.assertEqual(
                calls[1]["timeout"], saturation.SMOKE_TIMEOUT_SECONDS
            )

    def test_production_deadline_is_anchored_after_runtime_attestation_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            clock = _Clock()
            clock.value = 100.0
            observed = {}

            class FakeRunner:
                def __init__(
                    self, _command, _log_root, admitted_indices, virtual_world_size,
                    _attempt_id,
                ):
                    if admitted_indices != (0,) or virtual_world_size != 1:
                        raise AssertionError("production actor topology differs")
                    observed["attestation_started"] = clock()
                    clock.advance(662.2)
                    self.physical_gpu_inventory = _gpu_inventory()

            def fake_tuner(**kwargs):
                clock.advance(526.3)
                anchored = clock() + saturation.WALL_SECONDS
                kwargs["collection_deadline_sink"](anchored)
                observed["tuner_deadline"] = anchored
                clock.advance(100.0)
                return {
                    "status": "complete",
                    "completed_transitions": NOMINAL_TRANSITIONS,
                    "waves": [{}, {}],
                }

            def fake_publish(_path, _report, **kwargs):
                observed["publication_deadline"] = kwargs["deadline"]

            partition = {
                "path": "/sealed/train.json",
                "manifest_sha256": "1" * 64,
                "split_sha256": "2" * 64,
                "group_ids": ("group",),
            }
            with mock.patch.dict(
                    os.environ,
                    {"ROUND_DAGGER_CONTROLLER_STARTED_MONOTONIC": "999999999.0"},
                ), \
                    mock.patch.object(saturation, "CANONICAL_CAMPAIGN_ROOT", root), \
                    mock.patch.object(saturation, "load_train_partition", return_value=partition), \
                    mock.patch.object(saturation, "validate_train_motion_root", return_value=Path("/sealed/motions")), \
                    mock.patch.object(saturation, "ProcessWaveRunner", FakeRunner), \
                    mock.patch.object(saturation, "run_tuner", side_effect=fake_tuner), \
                    mock.patch.object(saturation, "_publish_json_create_only", side_effect=fake_publish), \
                    mock.patch.object(saturation.time, "monotonic", side_effect=clock), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = saturation.main([
                    "--campaign-root", str(root),
                    "--train-partition", "/sealed/train.json",
                    "--train-partition-sha256", "1" * 64,
                    "--train-motion-root", "/sealed/motions",
                ])
            self.assertEqual(status, 0)
            self.assertEqual(observed["attestation_started"], 100.0)
            expected_deadline = (
                100.0 + 662.2 + 526.3 + saturation.WALL_SECONDS
            )
            self.assertAlmostEqual(observed["tuner_deadline"], expected_deadline)
            self.assertAlmostEqual(
                observed["publication_deadline"], expected_deadline
            )

    def test_bc_launcher_generation_lock_rejects_overlap_and_auto_releases(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts/round_dagger/run_native_h200_bc_collection.sh"
        )
        source = launcher.read_text(encoding="utf-8")
        lock_index = source.index('flock -n "$CONTROLLER_LOCK_FD"')
        self.assertLess(lock_index, source.index('require_hash "$POLICY"'))
        self.assertLess(lock_index, source.index("mapfile -t physical_binding"))
        self.assertLess(lock_index, source.index("exec /usr/bin/python3"))
        self.assertNotIn("ROUND_DAGGER_CONTROLLER_STARTED_MONOTONIC", source)
        self.assertIn(
            "RUNTIME_INPUT=$CHECKPOINT_INPUT/gen-000-bc__runtime-v008",
            source,
        )
        self.assertIn("--gpu-indices 0", source)
        self.assertNotIn("--gpu-count N", source)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            campaign = root / "campaign"
            generation = campaign / "BC/collection"
            generation.mkdir(parents=True)
            copied = root / "scripts/round_dagger/run_native_h200_bc_collection.sh"
            copied.parent.mkdir(parents=True)
            copied.write_text(
                source.replace(
                    "CAMPAIGN=/workspace/project/roborender/data",
                    f"CAMPAIGN={campaign}",
                    1,
                ),
                encoding="utf-8",
            )

            descriptor = os.open(
                generation,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                competing = subprocess.run(
                    ["bash", str(copied), "production", "lock-test"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            self.assertEqual(competing.returncode, 2)
            self.assertIn("another smoke/production controller", competing.stderr)
            self.assertNotIn("immutable input", competing.stderr)

            exited = subprocess.run(
                ["bash", str(copied), "smoke", "release-test"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(exited.returncode, 2)
            self.assertIn("immutable input", exited.stderr)
            descriptor = os.open(
                generation,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def test_runtime_attestation_is_a_readiness_gate_before_collection_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            clock = _Clock()
            clock.value = 100.0

            class LateRunner:
                def __init__(
                    self, _command, _log_root, _admitted_indices,
                    _virtual_world_size, _attempt_id,
                ):
                    clock.advance(5400.001)
                    self.physical_gpu_inventory = _gpu_inventory()

            partition = {
                "path": "/sealed/train.json",
                "manifest_sha256": "1" * 64,
                "split_sha256": "2" * 64,
                "group_ids": ("group",),
            }
            observed = {}

            def tuner(**kwargs):
                observed["deadline"] = clock() + saturation.WALL_SECONDS
                kwargs["collection_deadline_sink"](observed["deadline"])
                return {
                    "status": "complete",
                    "completed_transitions": NOMINAL_TRANSITIONS,
                    "waves": [{}, {}],
                }

            publisher = mock.Mock()
            with mock.patch.object(saturation, "CANONICAL_CAMPAIGN_ROOT", root), \
                    mock.patch.object(saturation, "load_train_partition", return_value=partition), \
                    mock.patch.object(saturation, "validate_train_motion_root", return_value=Path("/sealed/motions")), \
                    mock.patch.object(saturation, "ProcessWaveRunner", LateRunner), \
                    mock.patch.object(saturation, "run_tuner", side_effect=tuner), \
                    mock.patch.object(saturation, "_publish_json_create_only", publisher), \
                    mock.patch.object(saturation.time, "monotonic", side_effect=clock), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = saturation.main([
                        "--campaign-root", str(root),
                        "--train-partition", "/sealed/train.json",
                        "--train-partition-sha256", "1" * 64,
                        "--train-motion-root", "/sealed/motions",
                    ])
            self.assertEqual(status, 0)
            self.assertAlmostEqual(
                observed["deadline"], 100.0 + 5400.001 + saturation.WALL_SECONDS
            )
            publisher.assert_called_once()

    def test_budget_capped_production_never_publishes_authoritative_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()

            class FakeRunner:
                def __init__(
                    self, _command, _log_root, admitted_indices, virtual_world_size,
                    _attempt_id,
                ):
                    if admitted_indices != (0,) or virtual_world_size != 1:
                        raise AssertionError("production actor topology differs")
                    self.physical_gpu_inventory = _gpu_inventory()

            partition = {
                "path": "/sealed/train.json",
                "manifest_sha256": "1" * 64,
                "split_sha256": "2" * 64,
                "group_ids": ("group",),
            }
            publisher = mock.Mock()
            output = io.StringIO()
            with mock.patch.object(
                saturation, "CANONICAL_CAMPAIGN_ROOT", root
            ), mock.patch.object(
                saturation, "load_train_partition", return_value=partition
            ), mock.patch.object(
                saturation,
                "validate_train_motion_root",
                return_value=Path("/sealed/motions"),
            ), mock.patch.object(
                saturation, "ProcessWaveRunner", FakeRunner
            ), mock.patch.object(
                saturation,
                "run_tuner",
                return_value={
                    "status": "budget_capped",
                    "stop_reason": "all_remaining_wave_eta_crosses_seal_deadline",
                },
            ), mock.patch.object(
                saturation, "_publish_json_create_only", publisher
            ), contextlib.redirect_stdout(output):
                status = saturation.main([
                    "--campaign-root", str(root),
                    "--train-partition", "/sealed/train.json",
                    "--train-partition-sha256", "1" * 64,
                    "--train-motion-root", "/sealed/motions",
                ])
            self.assertEqual(status, 3)
            publisher.assert_not_called()
            terminal = json.loads(output.getvalue())
            self.assertFalse(terminal["authoritative"])
            self.assertIsNone(terminal["manifest"])

    def test_saturation_manifest_publication_is_create_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "saturation_manifest.json"
            saturation._publish_json_create_only(path, {"status": "complete"})
            inode = path.stat().st_ino
            saturation._publish_json_create_only(path, {"status": "complete"})
            self.assertEqual(path.stat().st_ino, inode)
            with self.assertRaisesRegex(SaturationError, "different bytes"):
                saturation._publish_json_create_only(path, {"status": "changed"})

    def test_wave_artifact_preflight_binds_exact_conditioning_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary).resolve()
            generation = campaign / "BC/collection"
            cohort = generation / "conditioning/waves/wave-001"
            names = expected_shard_names(
                wave_index=1, envs_per_gpu=128, environments_per_shard=4
            )
            seed_contract = derive_actor_seed_contract(
                campaign_id="dagger-redo-v001",
                macro_round=0,
                cohort_index=1,
                expected_ranks=[0],
                expected_global_env_ids=list(range(128, 256)),
            )
            train_group_ids = [f"train-{index:02d}.npz" for index in range(23)]
            rotation = group_rotation(train_group_ids, 128, 1)
            environment = {
                "ROUND_DAGGER_CAMPAIGN_ROOT": str(campaign),
                "ROUND_DAGGER_GENERATION_ROOT": str(generation),
                "ROUND_DAGGER_GENERATION_ID": "gen-000-bc",
                "ROUND_DAGGER_CAMPAIGN_ID": "dagger-redo-v001",
                "ROUND_DAGGER_MACRO_ROUND": "0",
                "ROUND_DAGGER_SATURATION_WAVE_INDEX": "1",
                "ROUND_DAGGER_COHORT_INDEX": "1",
                "ROUND_DAGGER_ROLLOUT_ID": "1",
                "ROUND_DAGGER_COHORT_DIR": str(cohort),
                "ROUND_DAGGER_ENVS_PER_GPU": "128",
                "ROUND_DAGGER_ENVS_PER_SHARD": "4",
                "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": "512",
                "ROUND_DAGGER_ACTOR_WORLD_SIZE": "1",
                "ROUND_DAGGER_GLOBAL_ENV_OFFSET": "128",
                "ROUND_DAGGER_ACTOR_ROLE": "bc_collect",
                "ROUND_DAGGER_PURPOSE": "train",
                "ROUND_DAGGER_BETA": "1.0",
                "ROUND_DAGGER_TRAIN_PARTITION_SHA256": "1" * 64,
                "ROUND_DAGGER_SPLIT_SHA256": "2" * 64,
                "ROUND_DAGGER_TRAIN_GROUP_IDS_JSON": json.dumps(train_group_ids),
                "ROUND_DAGGER_GROUP_ROTATION_JSON": json.dumps(rotation),
                "ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON": json.dumps(names),
                "ROUND_DAGGER_ACTOR_SEED_CONTRACT_JSON": json.dumps(
                    seed_contract
                ),
                "ROUND_DAGGER_ACTOR_SEED_CONTRACT_SHA256": (
                    artifacts.canonical_json_sha256(seed_contract)
                ),
                "ROUND_DAGGER_SIMULATOR_BASE_SEED": str(
                    seed_contract["simulator_base_seed"]
                ),
            }
            with mock.patch.object(wave_artifact, "CAMPAIGN_ROOT", campaign), \
                    mock.patch.dict(os.environ, environment, clear=True):
                values = wave_artifact.context()
                self.assertEqual(values["cohort"], cohort)
                self.assertEqual(values["offset"], 128)
                self.assertEqual(values["environments_per_shard"], 4)
                self.assertEqual(values["train_group_ids"], train_group_ids)
                self.assertEqual(values["group_rotation"], rotation)
                changed = dict(environment)
                changed["ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON"] = json.dumps(names[:-1])
                with mock.patch.dict(os.environ, changed, clear=True), \
                        self.assertRaisesRegex(SystemExit, "exact wave naming"):
                    wave_artifact.context()

    def test_wave_seal_rejects_rank_part_shape_and_motion_provenance_drift(self):
        train_group_ids = [f"train-{index:02d}.npz" for index in range(23)]
        rotation = group_rotation(train_group_ids, 128, 0)
        names = expected_shard_names(
            wave_index=0, envs_per_gpu=128, environments_per_shard=4
        )
        environment = {
            "ROUND_DAGGER_CAMPAIGN_ID": "dagger-redo-v001",
            "ROUND_DAGGER_MACRO_ROUND": "0",
            "ROUND_DAGGER_POLICY_SHA256": "1" * 64,
            "ROUND_DAGGER_RUNTIME_POLICY_SHA256": "2" * 64,
            "ROUND_DAGGER_TEACHER_SHA256": "3" * 64,
            "ROUND_DAGGER_BETA_SHA256": "4" * 64,
            "ROUND_DAGGER_CONFIG_SHA256": "5" * 64,
        }

        def run_seal(mutation=None):
            with tempfile.TemporaryDirectory() as temporary:
                cohort = Path(temporary).resolve() / "conditioning/waves/wave-000"
                shard_dir = cohort / "shards"
                shard_dir.mkdir(parents=True)
                for name in names:
                    (shard_dir / name).touch()
                values = {
                    "wave": 0,
                    "cohort": cohort,
                    "envs": 128,
                    "environments_per_shard": 4,
                    "offset": 0,
                    "expected_names": names,
                    "actor_seed_contract": {},
                    "split_manifest_sha256": "6" * 64,
                    "split_sha256": "7" * 64,
                    "train_group_ids": train_group_ids,
                    "group_rotation": rotation,
                }
                contract = {
                    "control_contract": {"identity": "control"},
                    "split_manifest_sha256": values["split_manifest_sha256"],
                    "split_sha256": values["split_sha256"],
                    "train_group_ids": train_group_ids,
                    "group_rotation": rotation,
                }
                first_stem = next(
                    name[:-5] for name in names if name.endswith(".json")
                )

                def load(path):
                    stem = path.stem
                    match = wave_artifact.SHARD_NAME.fullmatch(stem)
                    self.assertIsNotNone(match)
                    rank = int(match.group("rank"))
                    shard_index = int(match.group("shard"))
                    part_index = shard_index - rank * 32
                    ids = list(
                        range(
                            rank * 128 + part_index * 4,
                            rank * 128 + (part_index + 1) * 4,
                        )
                    )
                    motion_offset = rotation["rank_group_offsets"][rank]
                    motion_ids = [
                        train_group_ids[(motion_offset + part_index * 4 + index) % len(train_group_ids)]
                        for index in range(4)
                    ]
                    metadata = {
                        "campaign_id": "dagger-redo-v001",
                        "macro_round": 0,
                        "cohort_index": 0,
                        "rank": rank,
                        "rollout_id": 0,
                        "part_index": part_index,
                        "shard_id": stem,
                        "policy_sha256": "1" * 64,
                        "runtime_policy_sha256": "2" * 64,
                        "teacher_sha256": "3" * 64,
                        "beta_sha256": "4" * 64,
                        "config_sha256": "5" * 64,
                        "split_manifest_sha256": "6" * 64,
                        "split_sha256": "7" * 64,
                        "group_ids": train_group_ids,
                        "motion_ids": motion_ids,
                        "panel_sha256": None,
                        "observation_source": "native",
                    }
                    num_envs = 4
                    if stem == first_stem and mutation == "swapped_ids":
                        ids = list(range(4, 8))
                    elif stem == first_stem and mutation == "wrong_part":
                        metadata["part_index"] = 1
                    elif stem == first_stem and mutation == "wrong_size":
                        ids = ids[:3]
                        num_envs = 3
                    elif stem == first_stem and mutation == "wrong_motion":
                        metadata["motion_ids"] = ["heldout-motion.npz", *motion_ids[1:]]
                    elif stem == first_stem and mutation == "wrong_split_manifest":
                        metadata["split_manifest_sha256"] = "0" * 64
                    elif stem == first_stem and mutation == "wrong_split":
                        metadata["split_sha256"] = "0" * 64
                    elif stem == first_stem and mutation == "wrong_groups":
                        metadata["group_ids"] = ["heldout-motion.npz"]
                    report = {
                        "render": {"metadata": metadata},
                        "purpose": "train",
                        "learner_replay_ready": False,
                        "num_steps": 512,
                        "num_envs": num_envs,
                        "action_dim": wave_artifact.REPLAY_ACTION_DIM,
                        "control_contract": contract["control_contract"],
                        "transition_count": 512 * num_envs,
                        "episode_count": num_envs,
                        "image_count": num_envs,
                        "manifest_sha256": "8" * 64,
                        "payload": {"sha256": "9" * 64, "size_bytes": 0},
                    }
                    arrays = {
                        "global_env_ids": np.broadcast_to(
                            np.asarray(ids, dtype=np.int64)[None, :],
                            (512, len(ids)),
                        )
                    }
                    return report, arrays

                with mock.patch.dict(os.environ, environment, clear=True), \
                        mock.patch.object(
                            wave_artifact.artifacts,
                            "load_strict_json",
                            return_value=contract,
                        ), mock.patch.object(
                            wave_artifact, "prepare", return_value=contract
                        ), mock.patch.object(
                            wave_artifact.artifacts,
                            "sha256_file",
                            return_value="a" * 64,
                        ), mock.patch.object(
                            wave_artifact, "provenance", return_value={"observation_source": "native"}
                        ), mock.patch.object(
                            wave_artifact,
                            "load_raw_conditioning_artifact",
                            side_effect=load,
                        ), mock.patch.object(
                            wave_artifact.artifacts,
                            "publish_immutable_json",
                            return_value="b" * 64,
                        ):
                    return wave_artifact.seal(values)

        sealed = run_seal()
        self.assertEqual(sealed["global_env_ids"], list(range(128)))
        self.assertEqual(sealed["shard_count"], 32)
        cases = {
            "swapped_ids": "environment ownership differs",
            "wrong_part": "provenance differs",
            "wrong_size": "contract differs",
            "wrong_motion": "provenance differs",
            "wrong_split_manifest": "provenance differs",
            "wrong_split": "provenance differs",
            "wrong_groups": "provenance differs",
        }
        for mutation, message in cases.items():
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                SystemExit, message
            ):
                run_seal(mutation)

    def test_actual_rank_launcher_uses_compat_wrapper_and_isolated_process_state(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts/round_dagger/run_native_h200_actor_rank.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("actor_training_entrypoint.py", source)
        self.assertIn("33demos_675_775", source)
        self.assertIn("motion_assignment_offset", source)
        for name in (
            "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
            "XDG_RUNTIME_DIR", "HF_HOME", "MPLCONFIGDIR", "TORCH_HOME", "TMPDIR",
        ):
            self.assertIn(name, source)
        for required in (
            "unshare --mount --propagation private --pid --fork --kill-child",
            "ROUND_DAGGER_TRAIN_MOTION_ROOT",
            "native-validation",
            "native-test",
            "--bounding-set=-sys_admin,-dac_override,-dac_read_search",
            "ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON",
            "ROBORENDER_HEADLESS_EXPERIENCE",
            "ROUND_DAGGER_ISAAC_SETUP_SCRIPT",
            "ROUND_DAGGER_CUDNN_LIBRARY_DIR",
            "EXPECTED_TRAIN_CHECKPOINT_SHA256",
            "layer_config.pretrained=false",
            "algo.config.num_steps_per_env=512",
            "DAGGER_STORAGE_ROOT=/workspace/project/roborender/data",
            "native_h200_validation_panel_audit.json",
            "export ROUND_DAGGER_SIMULATION_APP_SKIP_CLEANUP=1",
            "readonly ROUND_DAGGER_SIMULATION_APP_SKIP_CLEANUP",
        ):
            self.assertIn(required, source)
        native_config = (
            Path(__file__).resolve().parents[1]
            / "GR00T-VisualSim2Real/gr00t/rl/config/exp/loco_manip/"
            "wsdpt_round_actor_native_3view.yaml"
        ).read_text(encoding="utf-8")
        self.assertRegex(native_config, r"(?m)^\s+pretrained: false$")

    def test_train_motion_materialization_rejects_extra_or_changed_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            motion = root / "train.npz"
            motion.write_bytes(b"train")
            partition = {
                "groups": ({
                    "group_id": motion.name,
                    "relative_path": motion.name,
                    "size_bytes": 5,
                    "motion_sha256": hashlib.sha256(b"train").hexdigest(),
                },),
            }
            self.assertEqual(
                saturation.validate_train_motion_root(root, partition), root
            )
            extra = root / "validation.npz"
            extra.write_bytes(b"heldout")
            with self.assertRaisesRegex(SaturationError, "inventory differs"):
                saturation.validate_train_motion_root(root, partition)
            extra.unlink()
            motion.write_bytes(b"wrong")
            with self.assertRaisesRegex(SaturationError, "SHA-256 mismatch"):
                saturation.validate_train_motion_root(root, partition)

    def test_wave_contract_contains_derived_seed_and_gpu_mapping(self):
        clock = _Clock()
        partition = {
            "path": "/immutable/train.json",
            "manifest_sha256": "1" * 64,
            "split_sha256": "2" * 64,
            "group_ids": tuple(f"group-{index:02d}" for index in range(23)),
        }
        prepared = []

        def runner(*, envs_per_gpu, steps, environment, **_):
            throughput = 2000
            return _result(envs_per_gpu, steps, throughput, 0.5, duration=1)

        report = run_tuner(
            partition=partition,
            wave_runner=runner,
            prepare_runner=lambda environment, _timeout: prepared.append(environment),
            seal_runner=lambda environment, _timeout: {
                "path": "/sealed/seal.json",
                "sha256": "a" * 64,
                "shards": [],
                "image_count": 13_312,
                "episode_count": 256,
            },
            clock=clock,
            train_motion_root=Path("/immutable/train-motions"),
            physical_gpu_inventory=_gpu_inventory(),
        )
        self.assertEqual(len(prepared), 2)
        first = json.loads(prepared[0]["ROUND_DAGGER_ACTOR_SEED_CONTRACT_JSON"])
        self.assertEqual(
            first["simulator_base_seed"],
            int(prepared[0]["ROUND_DAGGER_SIMULATOR_BASE_SEED"]),
        )
        self.assertEqual(
            sum(len(row["global_env_ids"]) for row in first["assignments"]),
            128,
        )
        self.assertEqual([row["rank"] for row in first["assignments"]], [0])
        self.assertEqual(
            report["physical_gpu_inventory_sha256"],
            artifacts.canonical_json_sha256(_gpu_inventory()),
        )


if __name__ == "__main__":
    unittest.main()
