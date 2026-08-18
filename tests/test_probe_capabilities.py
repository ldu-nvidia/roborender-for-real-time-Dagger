import copy
import unittest

from scripts.scaling.image3f_probe import capability_error


class ProbeCapabilityTest(unittest.TestCase):
    @staticmethod
    def production_response():
        adapter_sha = "a" * 64
        return {
            "protocol_version": 2,
            "capabilities": {
                "lagged_reference_history": True,
                "reference_lag_max": 512,
            },
            "runtime_identity": {
                "schema_version": 1,
                "release_id": "release-1",
                "adapter_sha256": adapter_sha,
                "generation": {
                    "steps": 5,
                    "tea_cache_threshold": 0.3,
                    "cfg_scale": 1.0,
                    "cuda_graph": False,
                },
                "runtime": {
                    "backend": "tensorrt",
                    "engine_loaded": True,
                    "engine_verified_at_load": True,
                    "gpu": "RTX PRO 6000",
                    "compute_capability": "12.0",
                    "engine_path": "/engine/model.engine",
                },
                "engine": {
                    "sha256": "b" * 64,
                    "engine_key": "engine-key",
                    "export_key": "export-key",
                    "adapter_sha256": adapter_sha,
                    "status": "built_microbenchmarked_and_correctness_checked",
                    "precision": "bf16_fp16",
                    "cuda_graph_enabled_for_runtime": False,
                    "gpu": "RTX PRO 6000",
                    "compute_capability": "12.0",
                    "tensorrt": "10.13",
                    "cuda": "13.0",
                    "torch": "2.7.1",
                    "driver": "580.65",
                },
            },
        }

    @staticmethod
    def production_error(response):
        return capability_error(
            response,
            minimum_protocol_version=2,
            required_reference_lag=256,
            require_reference_history=True,
            expected_release_id="release-1",
            expected_adapter_sha256="a" * 64,
            expected_inference_steps=5,
            expected_teacache_threshold=0.3,
            expected_cfg_scale=1.0,
            expected_cuda_graph=False,
            require_tensorrt_engine=True,
        )

    def test_accepts_protocol_two_and_sufficient_lag(self):
        response = {
            "protocol_version": 2,
            "capabilities": {
                "lagged_reference_history": True,
                "reference_lag_max": 512,
            },
        }
        self.assertIsNone(
            capability_error(
                response,
                minimum_protocol_version=2,
                required_reference_lag=256,
                require_reference_history=True,
            )
        )

    def test_rejects_insufficient_or_malformed_lag(self):
        base = {"protocol_version": 2, "capabilities": {"lagged_reference_history": True}}
        base["capabilities"]["reference_lag_max"] = 128
        self.assertIn(
            "< required 256",
            capability_error(
                base,
                minimum_protocol_version=2,
                required_reference_lag=256,
                require_reference_history=True,
            ),
        )
        base["capabilities"]["reference_lag_max"] = "invalid"
        self.assertIn(
            "malformed",
            capability_error(
                base,
                minimum_protocol_version=2,
                required_reference_lag=256,
                require_reference_history=True,
            ),
        )

    def test_accepts_exact_model_sampler_and_engine_identity(self):
        self.assertIsNone(self.production_error(self.production_response()))

    def test_rejects_every_model_and_sampler_mismatch(self):
        mutations = (
            (("runtime_identity", "release_id"), "other", "release"),
            (("runtime_identity", "adapter_sha256"), "c" * 64, "adapter SHA"),
            (("runtime_identity", "generation", "steps"), 4, "inference steps"),
            (
                ("runtime_identity", "generation", "tea_cache_threshold"),
                0.4,
                "TeaCache",
            ),
            (("runtime_identity", "generation", "cfg_scale"), 5.0, "CFG"),
            (("runtime_identity", "generation", "cuda_graph"), True, "CUDA graph"),
        )
        for path, value, expected_error in mutations:
            with self.subTest(path=path):
                response = copy.deepcopy(self.production_response())
                target = response
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assertIn(expected_error, self.production_error(response))

    def test_rejects_unverified_or_inconsistent_engine_identity(self):
        response = self.production_response()
        response["runtime_identity"]["runtime"]["engine_verified_at_load"] = False
        self.assertIn("not verified", self.production_error(response))

        response = self.production_response()
        response["runtime_identity"]["engine"]["adapter_sha256"] = "d" * 64
        self.assertIn("does not match", self.production_error(response))

        response = self.production_response()
        response["runtime_identity"]["engine"]["compute_capability"] = "8.9"
        self.assertIn("compute_capability", self.production_error(response))


if __name__ == "__main__":
    unittest.main()
