"""CPU-only tests: no Lepton dependency, network access, or GPU allocation."""
from test.deployment.source_paths import H200_ROOT
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class Spec(SimpleNamespace):
    def model_copy(self, deep=False):
        return copy.deepcopy(self) if deep else copy.copy(self)


class PlanningTest(unittest.TestCase):
    def run_plan(self, dry_run, gpus=1):
        created = []
        source = SimpleNamespace(
            metadata=SimpleNamespace(name="leo-cosmos-h200-access"),
            spec=Spec(
                resource_requirement=SimpleNamespace(resource_shape="cpu.large"),
                reservation_config=SimpleNamespace(), container=SimpleNamespace(),
                envs=[SimpleNamespace(name="existing", value="must-not-copy")],
            ),
        )
        client = SimpleNamespace(
            _header={},
            deployment=SimpleNamespace(
                list_all=lambda: [], get=lambda name: source, create=created.append,
            ),
        )
        sdk = SimpleNamespace(WorkspaceRecord=SimpleNamespace(client=lambda: client))
        path = (H200_ROOT / "create_migration_smoke_pod.py")
        spec = importlib.util.spec_from_file_location("smoke_pod_under_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"leptonai.api.v1.workspace_record": sdk}):
            spec.loader.exec_module(module)
        argv = [str(path), "--name", "leo-roborender-beta-smoke-test",
                "--public-key", "ssh-ed25519 test-public-key test-only", "--gpus", str(gpus)]
        if dry_run:
            argv.append("--dry-run")
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            module.main()
        self.assertEqual(source.spec.resource_requirement.resource_shape, "cpu.large")
        return created

    def test_dry_run_cannot_mutate_api(self):
        # No raw _post method exists on this fake client: a regression fails too.
        self.assertEqual(self.run_plan(True), [])

    def test_real_create_is_one_reserved_gpu_and_public_keys_only(self):
        created = self.run_plan(False)
        self.assertEqual(len(created), 1)
        value = created[0]
        self.assertEqual(value.metadata.name, "leo-roborender-beta-smoke-test")
        self.assertEqual(value.spec.resource_requirement.resource_shape, "custom.gpu.1xh200")
        self.assertEqual(value.spec.resource_requirement.min_replicas, 1)
        self.assertEqual(value.spec.reservation_config.reservation_id,
                         "cosmos-solution-architect-xcwtffiq")
        self.assertFalse(value.spec.reservation_config.allow_burst_to_other_reservations)
        self.assertEqual([x.name for x in value.spec.envs],
                         ["ROBORENDER_SMOKE_PUBLIC_KEYS", "ROBORENDER_SMOKE_ID"])
        self.assertEqual(value.spec.envs[1].value, value.metadata.name)

    def test_four_gpu_sweep_stays_reserved(self):
        value = self.run_plan(False, 4)[0]
        self.assertEqual(value.spec.resource_requirement.resource_shape, "custom.gpu.4xh200")
        self.assertFalse(value.spec.reservation_config.allow_burst_to_other_reservations)


if __name__ == "__main__":
    unittest.main()
