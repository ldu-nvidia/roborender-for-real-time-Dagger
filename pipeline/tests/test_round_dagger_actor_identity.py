"""Unit gates for independent round-actor process identities."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from round_dagger.actor_identity import (
    ActorIdentityError,
    ActorProcessIdentity,
    actor_process_identity,
    native_actor_cuda_device,
    renderer_slot,
    wait_for_native_actor_fleet_release,
)


def _native_environment(rank=6, physical_index=None):
    if physical_index is None:
        physical_index = rank
    assigned_uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    pci_bus_id = "00000000:91:00.0"
    gpu_namespace = {
        "schema": "fast_roborender.round_dagger.native_actor_gpu_device_namespace.v1",
        "rank": physical_index,
        "physical_index": physical_index,
        "uuid": assigned_uuid,
        "pci_bus_id": pci_bus_id,
        "pci_function": "0000:91:00.0",
        "nvidia_device": {"path": "/dev/nvidia5", "major": 195, "minor": 5},
        "drm_card_device": {"path": "/dev/dri/card5", "major": 226, "minor": 5},
        "drm_render_device": {
            "path": "/dev/dri/renderD133", "major": 226, "minor": 133,
        },
        "common_control_devices": [],
    }
    raw_namespace = json.dumps(gpu_namespace, sort_keys=True, separators=(",", ":"))
    return {
        "ROUND_DAGGER_ACTOR_RANK": str(rank),
        "ROUND_DAGGER_ACTOR_WORLD_SIZE": "8",
        "ROUND_DAGGER_OBSERVATION_SOURCE": "native",
        "ROUND_DAGGER_ASSIGNED_PHYSICAL_INDEX": str(physical_index),
        "ROUND_DAGGER_ASSIGNED_GPU_UUID": assigned_uuid,
        "ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON": json.dumps({
            "rank": physical_index,
            "physical_index": physical_index,
            "uuid": assigned_uuid,
            "pci_bus_id": pci_bus_id,
        }),
        "ROUND_DAGGER_GPU_DEVICE_NAMESPACE_JSON": raw_namespace,
        "ROUND_DAGGER_GPU_DEVICE_NAMESPACE_SHA256": hashlib.sha256(
            raw_namespace.encode("utf-8")
        ).hexdigest(),
    }


class ActorProcessIdentityTest(unittest.TestCase):
    def test_absent_identity_is_not_inferred_from_torch_rank(self) -> None:
        self.assertIsNone(
            actor_process_identity({"RANK": "7", "WORLD_SIZE": "8", "LOCAL_RANK": "3"})
        )

    def test_complete_identity_is_exact(self) -> None:
        self.assertEqual(
            actor_process_identity(
                {
                    "ROUND_DAGGER_ACTOR_RANK": "1",
                    "ROUND_DAGGER_ACTOR_WORLD_SIZE": "2",
                }
            ),
            ActorProcessIdentity(rank=1, world_size=2),
        )

    def test_native_cuda_device_is_isolated_logical_zero(self) -> None:
        environment = _native_environment()
        self.assertEqual(native_actor_cuda_device(environment), "cuda:0")
        self.assertIsNone(native_actor_cuda_device({}))
        environment["CUDA_VISIBLE_DEVICES"] = environment[
            "ROUND_DAGGER_ASSIGNED_GPU_UUID"
        ]
        with self.assertRaisesRegex(ActorIdentityError, "forbids"):
            native_actor_cuda_device(environment)

        del environment["CUDA_VISIBLE_DEVICES"]
        environment["NVIDIA_VISIBLE_DEVICES"] = ""
        with self.assertRaisesRegex(ActorIdentityError, "forbids"):
            native_actor_cuda_device(environment)
        del environment["NVIDIA_VISIBLE_DEVICES"]
        del environment["ROUND_DAGGER_ASSIGNED_GPU_UUID"]
        with self.assertRaisesRegex(ActorIdentityError, "UUID is malformed"):
            native_actor_cuda_device(environment)

    def test_virtual_rank_can_map_to_a_different_admitted_physical_gpu(self) -> None:
        environment = _native_environment(rank=7, physical_index=2)
        self.assertEqual(native_actor_cuda_device(environment), "cuda:0")
        environment["ROUND_DAGGER_ASSIGNED_PHYSICAL_INDEX"] = "7"
        with self.assertRaisesRegex(ActorIdentityError, "namespace differs"):
            native_actor_cuda_device(environment)

    def test_native_cuda_device_rejects_non_singleton_or_unbound_visibility(self) -> None:
        base = _native_environment()
        cases = {
            "CUDA visibility remapping": {
                **base,
                "CUDA_VISIBLE_DEVICES": base["ROUND_DAGGER_ASSIGNED_GPU_UUID"],
            },
            "wrong physical index": {
                **base,
                "ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON": json.dumps(
                    {"rank": 6, "physical_index": 0,
                     "uuid": base["ROUND_DAGGER_ASSIGNED_GPU_UUID"],
                     "pci_bus_id": "00000000:91:00.0"}
                ),
            },
            "wrong logical rank": {
                **base,
                "ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON": json.dumps(
                    {"rank": 0, "physical_index": 6,
                     "uuid": base["ROUND_DAGGER_ASSIGNED_GPU_UUID"],
                     "pci_bus_id": "00000000:91:00.0"}
                ),
            },
            "malformed namespace": {
                **base,
                "ROUND_DAGGER_GPU_DEVICE_NAMESPACE_JSON": "not-json",
            },
            "wrong namespace digest": {
                **base,
                "ROUND_DAGGER_GPU_DEVICE_NAMESPACE_SHA256": "0" * 64,
            },
        }
        for name, environment in cases.items():
            with self.subTest(name=name), self.assertRaises(ActorIdentityError):
                native_actor_cuda_device(environment)

    def test_native_actor_waits_for_exact_controller_release_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generation = Path(temporary).resolve() / "BC/isaac/collection"
            release = generation / "logs/smoke/000-1/kit-ready.release"
            release.parent.mkdir(parents=True)
            sleeps = []

            def publish_after_first_wait(seconds):
                sleeps.append(seconds)
                release.write_bytes(b"release\n")

            output = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {
                    "ROUND_DAGGER_GENERATION_ROOT": str(generation),
                    "ROUND_DAGGER_FLEET_RELEASE_FILE": str(release),
                },
                clear=True,
            ), mock.patch(
                "round_dagger.actor_identity.time.sleep",
                side_effect=publish_after_first_wait,
            ), contextlib.redirect_stdout(output):
                wait_for_native_actor_fleet_release(6)
            self.assertEqual(sleeps, [0.05])
            self.assertEqual(
                output.getvalue(), "NATIVE_ACTOR_FLEET_RELEASED rank=6\n"
            )

    def test_native_actor_release_path_and_token_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generation = Path(temporary).resolve() / "BC/isaac/collection"
            logs = generation / "logs/smoke/000-1"
            logs.mkdir(parents=True)
            valid = logs / "kit-ready.release"
            outside = generation / "kit-ready.release"
            wrong_name = logs / "release.txt"
            for name, path, payload in (
                ("outside logs", outside, b"release\n"),
                ("wrong filename", wrong_name, b"release\n"),
                ("wrong token", valid, b"released\n"),
            ):
                with self.subTest(name=name):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload)
                    with mock.patch.dict(
                        os.environ,
                        {
                            "ROUND_DAGGER_GENERATION_ROOT": str(generation),
                            "ROUND_DAGGER_FLEET_RELEASE_FILE": str(path),
                        },
                        clear=True,
                    ), self.assertRaises(ActorIdentityError):
                        wait_for_native_actor_fleet_release(0)
                    path.unlink()

            target = logs / "target"
            target.write_bytes(b"release\n")
            valid.symlink_to(target)
            with mock.patch.dict(
                os.environ,
                {
                    "ROUND_DAGGER_GENERATION_ROOT": str(generation),
                    "ROUND_DAGGER_FLEET_RELEASE_FILE": str(valid),
                },
                clear=True,
            ), self.assertRaisesRegex(ActorIdentityError, "symlink"):
                wait_for_native_actor_fleet_release(0)

    def test_non_native_independent_actor_keeps_generic_cuda_visibility(self) -> None:
        environment = {
            "ROUND_DAGGER_ACTOR_RANK": "1",
            "ROUND_DAGGER_ACTOR_WORLD_SIZE": "2",
            "ROUND_DAGGER_OBSERVATION_SOURCE": "image3f",
            "CUDA_VISIBLE_DEVICES": "GPU-generic",
        }
        self.assertIsNone(native_actor_cuda_device(environment))

    def test_partial_noncanonical_and_out_of_range_values_fail(self) -> None:
        cases = (
            {"ROUND_DAGGER_ACTOR_RANK": "0"},
            {"ROUND_DAGGER_ACTOR_WORLD_SIZE": "2"},
            {
                "ROUND_DAGGER_ACTOR_RANK": "01",
                "ROUND_DAGGER_ACTOR_WORLD_SIZE": "2",
            },
            {
                "ROUND_DAGGER_ACTOR_RANK": "2",
                "ROUND_DAGGER_ACTOR_WORLD_SIZE": "2",
            },
            {
                "ROUND_DAGGER_ACTOR_RANK": "0",
                "ROUND_DAGGER_ACTOR_WORLD_SIZE": "0",
            },
        )
        for environment in cases:
            with self.subTest(environment=environment), self.assertRaises(
                ActorIdentityError
            ):
                actor_process_identity(environment)

    def test_renderer_slots_cover_ordered_global_actor_environments(self) -> None:
        self.assertEqual(
            [
                renderer_slot(
                    rank=rank,
                    local_env_count=3,
                    local_env_index=local,
                    renderer_count=4,
                )
                for rank in range(2)
                for local in range(3)
            ],
            [0, 1, 2, 3, 0, 1],
        )


if __name__ == "__main__":
    unittest.main()
