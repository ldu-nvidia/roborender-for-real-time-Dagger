"""Focused gates for the create-only Image3F renderer warm-up report."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np

from round_dagger.artifacts import ArtifactError, sha256_file
from scripts.round_dagger import warm_image3f_renderer as warmup


def _generated_views() -> dict[str, np.ndarray]:
    return {
        view: np.full((240, 416, 3), 101 + index, dtype=np.uint8)
        for index, view in enumerate(warmup.VIEWS)
    }


def _fake_client_type(
    *,
    metadata: dict[str, object] | None = None,
    generated: dict[str, np.ndarray] | None = None,
    constructor_error: BaseException | None = None,
    close_error: BaseException | None = None,
) -> type:
    response_metadata = dict(
        metadata
        if metadata is not None
        else {
            "replayed": False,
            "inference_ms": 12.5,
            "client_wall_ms": 14.25,
            "reference_source": "anchor",
        }
    )
    response_views = generated if generated is not None else _generated_views()

    class FakeImage3FFrameClient:
        instances: list[object] = []
        constructor_environments: list[str | None] = []

        def __init__(self, host: str, port: int, *, stream_slot: int):
            self.__class__.constructor_environments.append(
                os.environ.get("ROUND_DAGGER_RENDERER_SHA256")
            )
            if constructor_error is not None:
                raise constructor_error
            self.host = host
            self.port = port
            self.stream_slot = stream_slot
            self.last_metadata = dict(response_metadata)
            self.render_args: tuple[object, ...] | None = None
            self.render_kwargs: dict[str, object] | None = None
            self.closed = False
            self.__class__.instances.append(self)

        def render_frame(self, *args: object, **kwargs: object) -> dict[str, np.ndarray]:
            self.render_args = args
            self.render_kwargs = kwargs
            return response_views

        def close(self) -> None:
            self.closed = True
            if close_error is not None:
                raise close_error

    return FakeImage3FFrameClient


@contextmanager
def _installed_fake_client(client_type: type):
    gr00t = types.ModuleType("gr00t")
    gr00t.__path__ = []  # type: ignore[attr-defined]
    rl = types.ModuleType("gr00t.rl")
    rl.__path__ = []  # type: ignore[attr-defined]
    utils = types.ModuleType("gr00t.rl.utils")
    utils.__path__ = []  # type: ignore[attr-defined]
    client = types.ModuleType("gr00t.rl.utils.image3f_frame_client")
    client.Image3FFrameClient = client_type
    gr00t.rl = rl
    rl.utils = utils
    utils.image3f_frame_client = client
    with mock.patch.dict(
        sys.modules,
        {
            "gr00t": gr00t,
            "gr00t.rl": rl,
            "gr00t.rl.utils": utils,
            "gr00t.rl.utils.image3f_frame_client": client,
        },
    ):
        yield


class Image3FWarmupTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="image3f_warmup_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.endpoint = self.root / "renderer.txt"
        self.endpoint.write_text("127.0.0.1 23456\n", encoding="utf-8")
        self.anchor = self.root / "anchor.npz"
        self.anchor_views = {
            view: np.full((1, 240, 416, 3), index + 1, dtype=np.uint8)
            for index, view in enumerate(warmup.VIEWS)
        }
        np.savez(
            self.anchor,
            **{
                **{f"gen__{view}": value for view, value in self.anchor_views.items()},
                "prompt": np.array("lift the colored cube"),
            },
        )
        self.output = self.root / "warmup.json"
        self.renderer_sha256 = "a" * 64

    def _run(self, client_type: type, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "endpoint": self.endpoint,
            "anchor": self.anchor,
            "output": self.output,
            "renderer_sha256": self.renderer_sha256,
            "stream_id": "runtime-ab-warmup",
        }
        arguments.update(overrides)
        with _installed_fake_client(client_type):
            return warmup.warm_renderer(**arguments)  # type: ignore[arg-type]

    def test_publishes_exact_canonical_immutable_inference_report(self) -> None:
        client_type = _fake_client_type()
        previous_identity = "b" * 64
        with mock.patch.dict(
            os.environ,
            {"ROUND_DAGGER_RENDERER_SHA256": previous_identity},
            clear=False,
        ):
            report = self._run(client_type)
            self.assertEqual(
                os.environ["ROUND_DAGGER_RENDERER_SHA256"], previous_identity
            )

        self.assertEqual(
            set(report),
            {
                "schema",
                "endpoint",
                "endpoint_sha256",
                "anchor",
                "anchor_sha256",
                "renderer_sha256",
                "stream_id",
                "seed",
                "reference_lag",
                "replayed",
                "inference_ms",
                "client_wall_ms",
                "reference_source",
                "generated_sha256",
            },
        )
        self.assertEqual(report["schema"], warmup.SCHEMA)
        self.assertEqual(report["endpoint"], str(self.endpoint))
        self.assertEqual(report["endpoint_sha256"], sha256_file(self.endpoint))
        self.assertEqual(report["anchor"], str(self.anchor))
        self.assertEqual(report["anchor_sha256"], sha256_file(self.anchor))
        self.assertEqual(report["renderer_sha256"], self.renderer_sha256)
        self.assertEqual(report["stream_id"], "runtime-ab-warmup")
        self.assertEqual(report["seed"], 42)
        self.assertEqual(report["reference_lag"], 256)
        self.assertIs(report["replayed"], False)
        self.assertEqual(report["inference_ms"], 12.5)
        self.assertEqual(report["client_wall_ms"], 14.25)
        self.assertEqual(report["reference_source"], "anchor")
        self.assertEqual(
            report["generated_sha256"],
            {
                view: warmup._array_sha256(image)
                for view, image in _generated_views().items()
            },
        )

        raw = self.output.read_bytes()
        expected = json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        self.assertEqual(raw, expected)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), sha256_file(self.output))
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o444)

        self.assertEqual(client_type.constructor_environments, [self.renderer_sha256])
        self.assertEqual(len(client_type.instances), 1)
        instance = client_type.instances[0]
        self.assertEqual(
            (instance.host, instance.port, instance.stream_slot),
            ("127.0.0.1", 23456, 0),
        )
        self.assertTrue(instance.closed)
        depth, mask, previous = instance.render_args
        self.assertEqual(tuple(depth), warmup.VIEWS)
        self.assertEqual(tuple(mask), warmup.VIEWS)
        self.assertEqual(tuple(previous), warmup.VIEWS)
        for view in warmup.VIEWS:
            self.assertEqual(depth[view].shape, (240, 416))
            self.assertEqual(depth[view].dtype, np.uint8)
            self.assertFalse(depth[view].any())
            self.assertEqual(mask[view].shape, (240, 416, 3))
            self.assertEqual(mask[view].dtype, np.uint8)
            self.assertFalse(mask[view].any())
            np.testing.assert_array_equal(previous[view], self.anchor_views[view][0])
        self.assertEqual(
            instance.render_kwargs,
            {
                "prompt": "lift the colored cube",
                "seed": 42,
                "stream_id": "runtime-ab-warmup",
                "episode_id": "warmup",
                "step": 0,
                "reference_lag": 256,
                "trainer_step": 0,
            },
        )

    def test_rejects_replayed_response_without_publishing(self) -> None:
        client_type = _fake_client_type(
            metadata={
                "replayed": True,
                "inference_ms": 0.0,
                "client_wall_ms": 1.0,
                "reference_source": "anchor",
            }
        )
        with self.assertRaisesRegex(RuntimeError, "unexpectedly served from replay"):
            self._run(client_type)
        self.assertFalse(self.output.exists())
        self.assertTrue(client_type.instances[0].closed)

    def test_rejects_malformed_generated_view_without_publishing(self) -> None:
        generated = _generated_views()
        generated["hand_right"] = np.zeros((240, 415, 3), dtype=np.uint8)
        client_type = _fake_client_type(generated=generated)
        with self.assertRaisesRegex(RuntimeError, "unexpected shape or dtype"):
            self._run(client_type)
        self.assertFalse(self.output.exists())
        self.assertTrue(client_type.instances[0].closed)

    def test_rejects_missing_zero_and_nonfinite_inference_timing(self) -> None:
        cases: tuple[tuple[str, object], ...] = (
            ("missing", None),
            ("zero", 0.0),
            ("negative", -1.0),
            ("nan", float("nan")),
            ("infinity", float("inf")),
        )
        for label, timing in cases:
            with self.subTest(timing=label):
                metadata: dict[str, object] = {
                    "replayed": False,
                    "client_wall_ms": 2.0,
                    "reference_source": "anchor",
                }
                if timing is not None:
                    metadata["inference_ms"] = timing
                client_type = _fake_client_type(metadata=metadata)
                with self.assertRaisesRegex(RuntimeError, "real renderer inference"):
                    self._run(client_type)
                self.assertFalse(self.output.exists())

    def test_rejects_malformed_anchor_before_opening_client(self) -> None:
        np.savez(
            self.anchor,
            **{
                **{
                    f"gen__{view}": np.zeros((1, 240, 416, 3), dtype=np.uint8)
                    for view in warmup.VIEWS[:-1]
                },
                "prompt": np.array("scene"),
            },
        )
        client_type = _fake_client_type()
        with self.assertRaisesRegex(ValueError, "required three RGB views"):
            self._run(client_type)
        self.assertEqual(client_type.constructor_environments, [])
        self.assertFalse(self.output.exists())

    def test_rejects_bad_identity_stream_output_and_endpoint_inputs(self) -> None:
        client_type = _fake_client_type()
        cases: tuple[tuple[str, dict[str, object], str], ...] = (
            ("identity", {"renderer_sha256": "A" * 64}, "lowercase SHA-256"),
            ("stream", {"stream_id": "bad stream"}, "non-empty token"),
            ("output", {"output": Path("relative.json")}, "absolute non-symlink"),
        )
        for label, overrides, error in cases:
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, error):
                    self._run(client_type, **overrides)
        self.assertEqual(client_type.constructor_environments, [])

        endpoint_cases = (
            ("renderer.example 1234\n", "local host and port"),
            ("127.0.0.1 nope\n", "must be an integer"),
            ("127.0.0.1 0\n", "within 1..65535"),
            ("127.0.0.1 65536\n", "within 1..65535"),
        )
        for contents, error in endpoint_cases:
            with self.subTest(endpoint=contents.strip()):
                self.endpoint.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    self._run(client_type)
        self.assertEqual(client_type.constructor_environments, [])
        self.assertFalse(self.output.exists())

    def test_existing_different_report_is_never_overwritten(self) -> None:
        sentinel = b'{"existing":"must-survive"}'
        self.output.write_bytes(sentinel)
        before = self.output.stat()
        with self.assertRaisesRegex(ArtifactError, "immutable artifact conflict"):
            self._run(_fake_client_type())
        after = self.output.stat()
        self.assertEqual(self.output.read_bytes(), sentinel)
        self.assertEqual((after.st_dev, after.st_ino), (before.st_dev, before.st_ino))

    def test_renderer_identity_is_restored_on_constructor_and_close_failures(self) -> None:
        previous_identity = "c" * 64
        for label, client_type in (
            (
                "constructor",
                _fake_client_type(constructor_error=RuntimeError("connect failed")),
            ),
            ("close", _fake_client_type(close_error=RuntimeError("close failed"))),
        ):
            with self.subTest(failure=label):
                with mock.patch.dict(
                    os.environ,
                    {"ROUND_DAGGER_RENDERER_SHA256": previous_identity},
                    clear=False,
                ):
                    with self.assertRaisesRegex(RuntimeError, f"{label}|connect|close"):
                        self._run(client_type)
                    self.assertEqual(
                        os.environ["ROUND_DAGGER_RENDERER_SHA256"], previous_identity
                    )
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
