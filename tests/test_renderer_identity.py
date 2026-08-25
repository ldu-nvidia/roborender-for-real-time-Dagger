import tempfile
import unittest
from pathlib import Path

from scripts.server.renderer_identity import (
    canonical_json_sha256,
    describe_renderer_identity,
    regular_file_sha256,
    validate_renderer_identity,
)


class RendererIdentityTest(unittest.TestCase):
    def test_script_bytes_change_renderer_but_not_runtime_identity(self):
        runtime = {"release": "same", "engine": {"sha256": "a" * 64}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.py"
            second = root / "second.py"
            first.write_text("VALUE = 1\n", encoding="utf-8")
            second.write_text("VALUE = 2\n", encoding="utf-8")
            one = describe_renderer_identity(runtime, server_script=first)
            two = describe_renderer_identity(runtime, server_script=second)
            self.assertEqual(one["runtime_identity_sha256"], two["runtime_identity_sha256"])
            self.assertNotEqual(one["server_script_sha256"], two["server_script_sha256"])
            self.assertNotEqual(canonical_json_sha256(one), canonical_json_sha256(two))

    def test_plan_to_server_script_mismatch_fails_closed(self):
        runtime = {"release": "fixture"}
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "server.py"
            script.write_text("pass\n", encoding="utf-8")
            digest = regular_file_sha256(script)
            identity = describe_renderer_identity(
                runtime,
                server_script=script,
                expected_server_script_sha256=digest,
            )
            self.assertEqual(
                validate_renderer_identity(
                    identity,
                    runtime_identity=runtime,
                    expected_server_script_sha256=digest,
                ),
                identity,
            )
            with self.assertRaisesRegex(RuntimeError, "planning and startup"):
                describe_renderer_identity(
                    runtime,
                    server_script=script,
                    expected_server_script_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
