import unittest

from scripts.server.runtime_identity import describe_runtime_identity


class FakeSession:
    def __init__(self):
        self.config = {
            "release_id": "release-1",
            "adapter": {"sha256": "a" * 64},
        }
        self.engine_manifest = {
            "engine": {"sha256": "b" * 64},
            "engine_key": "engine-key",
            "export_key": "export-key",
            "adapter_sha256": "a" * 64,
            "status": "built_microbenchmarked_and_correctness_checked",
            "precision": "bf16_fp16",
            "cuda_graph": {"enabled_for_runtime": False},
            "gpu": "RTX PRO 6000",
            "compute_capability": "12.0",
            "tensorrt": "10.13",
            "cuda": "13.0",
            "torch": "2.7.1",
            "driver": "580.65",
        }

    def describe(self):
        return {
            "gpu": "RTX PRO 6000",
            "compute_capability": "12.0",
            "engine_path": "/engine/model.engine",
            "steps": 5,
            "cfg_scale": 1.0,
            "tea_cache_threshold": 0.3,
            "cuda_graph": False,
        }


class RuntimeIdentityTest(unittest.TestCase):
    def test_extracts_verified_runtime_identity(self):
        identity = describe_runtime_identity(FakeSession())
        self.assertEqual(identity["release_id"], "release-1")
        self.assertEqual(identity["generation"]["steps"], 5)
        self.assertEqual(identity["runtime"]["backend"], "tensorrt")
        self.assertTrue(identity["runtime"]["engine_verified_at_load"])
        self.assertEqual(identity["engine"]["sha256"], "b" * 64)

    def test_fails_closed_on_engine_adapter_or_hardware_mismatch(self):
        session = FakeSession()
        session.engine_manifest["adapter_sha256"] = "c" * 64
        with self.assertRaisesRegex(RuntimeError, "different adapter"):
            describe_runtime_identity(session)

        session = FakeSession()
        session.engine_manifest["compute_capability"] = "8.9"
        with self.assertRaisesRegex(RuntimeError, "hardware identity"):
            describe_runtime_identity(session)

    def test_fails_closed_when_engine_provenance_is_missing(self):
        session = FakeSession()
        del session.engine_manifest["engine_key"]
        with self.assertRaisesRegex(RuntimeError, "engine_key"):
            describe_runtime_identity(session)


if __name__ == "__main__":
    unittest.main()
