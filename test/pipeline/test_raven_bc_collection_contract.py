"""Regression tests for the byte-identical Raven BC collection path."""

from __future__ import annotations
from test.deployment.source_paths import REPO_ROOT

import ast
import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from test.reference.bc_collection.validate_raven_bc_collection_contract import validate


ROOT = REPO_ROOT
# The mutable development checkout is not the reference used by collection.
# Keep semantic extraction on the same isolated, hash-pinned source as validate.
GR00T = ROOT / "runtime/raven_gr00t_collection_94d7985"
CONFIG = GR00T / "gr00t/rl/config/exp/loco_manip/wsdpt_videogen_collect.yaml"
TRAINER = GR00T / "gr00t/rl/trl/trainer/distill_trainer_videogen.py"
CONTRACT = ROOT / "test/reference/bc_collection/raven_bc_collection_contract.v1.json"
REFERENCE_COMMIT = "94d79851d9c471b1210be81c75116d9bea835a4d"
TRAINER_SHA256 = "cc7f8aa3de2a9d60324b9294ba5db7ce15f339d59fa168ef1ccab7484d11890b"
CONFIG_SHA256 = "c483a32b5a1810dd7abcedfb425b86c939d45e02a0ea43d448f8b0b226f21e5c"


def _reference_text(path: Path, expected_sha256: str) -> str:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise AssertionError(f"pinned Raven reference hash differs: {path}")
    return raw.decode("utf-8")


def _load_class_method(path: Path, expected_sha256: str, class_name: str, method_name: str):
    tree = ast.parse(_reference_text(path, expected_sha256), filename=str(path))
    owner = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    namespace = {"np": np}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[method_name]


@unittest.skipUnless(
    (GR00T / ".git").exists(),
    "requires the ignored, pinned Raven checkout",
)
class RavenBCCollectionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # This also checks the full pinned commit and clean gr00t worktree.
        cls.reference_validation = validate(ROOT, gr00t_root=GR00T)
        if cls.reference_validation["active_gr00t_head"] != REFERENCE_COMMIT:
            raise AssertionError("pinned Raven reference commit differs")

    def test_active_tree_and_yaml_are_exact_raven(self) -> None:
        result = self.reference_validation
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["reference_commit"], "94d7985")
        self.assertEqual(result["active_gr00t_root"], str(GR00T.resolve()))
        self.assertEqual(result["active_trainer_sha256"], TRAINER_SHA256)
        self.assertTrue(result["active_yaml_byte_identical"])
        self.assertEqual(result["allowed_yaml_differences"], [])
        self.assertEqual(result["only_launch_time_scientific_override"], "num_envs")

    def test_contract_and_yaml_pin_three_view_collection(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(contract["source"]["trainer_sha256"], TRAINER_SHA256)
        self.assertEqual(contract["source"]["config_sha256"], CONFIG_SHA256)
        config = yaml.safe_load(_reference_text(CONFIG, CONFIG_SHA256))
        cameras = config["simulator"]["config"]["cameras"]
        videogen = config["algo"]["config"]["videogen"]
        self.assertEqual(
            cameras["conditioning_views"],
            contract["cameras"]["conditioning_views_in_order"],
        )
        self.assertEqual(
            videogen["depth_h5v4_per_view"],
            contract["conditioning"]["depth_h5v4_per_view"],
        )
        self.assertEqual(videogen["seed"], 42)
        self.assertEqual(videogen["rollout_action_source"], "teacher")
        self.assertEqual(config["algo"]["config"]["num_steps_per_env"], 81)

    def test_raven_collection_control_flow_is_unchanged(self) -> None:
        source = _reference_text(TRAINER, TRAINER_SHA256)
        for fragment in (
            "VIDEO_MODEL_CHUNK = 81",
            'policy_state_dict["actions"] = policy_state_dict["gt_actions"].clone()',
            "if len(self._ep_buf[b][\"gt\"]) == VIDEO_MODEL_CHUNK:",
            "_emit(b)",
            "if bool(dones[t, b, 0]):",
            "self._ep_seq_base[b] += 2",
        ):
            self.assertIn(fragment, source)

    def test_h5v4_conversion_matches_fixed_numeric_fixture(self) -> None:
        method = _load_class_method(
            TRAINER,
            TRAINER_SHA256,
            "TRLDistillTrainerVideoGen",
            "_h5v4_control",
        )
        params = json.loads(CONTRACT.read_text(encoding="utf-8"))[
            "conditioning"
        ]["depth_h5v4_per_view"]
        rng = np.random.RandomState(123)
        depth = rng.uniform(0.01, 12.0, size=(81, 2, 7, 9)).astype(np.float32)
        depth[0, 0, 0, 0] = 0.0
        depth[1, 0, 0, 0] = np.inf
        depth[2, 0, 0, 0] = np.nan
        owner = SimpleNamespace(
            vg_h5v4_per_view=params,
            vg_h5v4_default={"raw_clip": [0.0, 10.0], "inv_clip": [1.0, 20.0]},
            vg_invalid_fill="far",
        )
        expected_hashes = {
            "hand_left": "a7f8c8dc28b8fd2d3aaf5b308ab14cecae5a6e79de57bc67c51766932219b0ce",
            "hand_right": "a7f8c8dc28b8fd2d3aaf5b308ab14cecae5a6e79de57bc67c51766932219b0ce",
            "head_color": "7df63618410d089e10ab07254490bb455a7903a413019ba215497573e194b051",
        }
        for view, expected in expected_hashes.items():
            value = method(owner, depth, view)
            self.assertEqual(hashlib.sha256(value.tobytes()).hexdigest(), expected)

    def test_semantic_extraction_rejects_reference_hash_drift(self) -> None:
        with self.assertRaisesRegex(AssertionError, "reference hash differs"):
            _load_class_method(
                TRAINER, "0" * 64, "TRLDistillTrainerVideoGen", "_h5v4_control"
            )


if __name__ == "__main__":
    unittest.main()
